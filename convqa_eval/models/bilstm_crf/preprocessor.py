"""
Preprocessor for SIP with Custom Input Format

Input Format:
{
  "prompt": "user instruction",
  "context": "background context",
  "can_retrieve": bool,
  "tools": "available tools description",
  "ambiguous": "true/false",
  "ambiguous_type": 1-4,  # Ground truth label
  "conversations": [...],
  "system": "system prompt"
}

Output Format:
{
  "ambiguous_utterance": bool,
  "total_candidates": int,
  "explanation": "why ambiguous",
  "conditions": [(condition, answer), ...],
  "metadata": {CRF details, confidence, etc.}
}
"""

import json
import torch
from typing import List, Dict, Tuple
from transformers import BertTokenizer


class SIPPreprocessor:
    """
    Preprocesses conversation data with configurable 2-4 class prediction.
    
    Label Definitions:
    - 2 classes: 0=clear, 1=ambiguous
    - 3 classes: 0=clear, 1=needs_clarification, 2=highly_ambiguous
    - 4 classes: 0=clear, 1=slightly_ambiguous, 2=needs_clarification, 3=highly_ambiguous
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        max_len: int = 128,
        num_classes: int = 2  # 2, 3, or 4
    ):
        print(f"[SIPPreprocessor] Initializing...")
        print(f"  BERT: {bert_model}")
        print(f"  Max length: {max_len}")
        print(f"  Num classes: {num_classes}")
        
        assert num_classes in [2, 3, 4], "num_classes must be 2, 3, or 4"
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_len = max_len
        self.num_classes = num_classes
        
        # Class names for each configuration
        self.class_names = {
            2: ['clear', 'ambiguous'],
            3: ['clear', 'needs_clarification', 'highly_ambiguous'],
            4: ['clear', 'slightly_ambiguous', 'needs_clarification', 'highly_ambiguous']
        }
        
        print(f"  Classes: {self.class_names[num_classes]}")
        print()
    
    def parse_conversations(self, data: Dict) -> Tuple[List[str], List[str], List[int]]:
        """
        Parse conversations into user-system pairs with labels.
        
        Returns:
            user_utterances: List of user inputs
            system_utterances: List of system responses
            labels: List of ambiguity labels (from gpt req_clarification or ambiguous_type)
        """
        conversations = data.get('conversations', [])
        
        user_utterances = []
        system_utterances = []
        labels = []
        
        # Use ambiguous_type from data as global label if available
        global_label = None
        if 'ambiguous_type' in data:
            global_label = int(data['ambiguous_type'])
            # Map to num_classes
            global_label = min(global_label, self.num_classes - 1)
        elif data.get('ambiguous', 'false').lower() == 'true':
            global_label = 1  # Default to "ambiguous"
        
        i = 0
        while i < len(conversations):
            # User input (human)
            user_parts = []
            if i < len(conversations) and conversations[i]['from'] == 'human':
                user_parts.append(conversations[i]['value'])
                i += 1
            
            # System response (function_call + observation + gpt)
            system_parts = []
            
            # Function call
            if i < len(conversations) and conversations[i]['from'] == 'function_call':
                system_parts.append(f"[TOOL] {conversations[i]['value']}")
                i += 1
            
            # Observation
            if i < len(conversations) and conversations[i]['from'] == 'observation':
                user_parts.append(f"[RESULT] {conversations[i]['value']}")
                i += 1
            
            # GPT response (may contain req_clarification)
            label = 0  # Default clear
            if i < len(conversations) and conversations[i]['from'] == 'gpt':
                gpt_value = conversations[i]['value']
                system_parts.append(gpt_value)
                
                # Extract label from req_clarification(N) if present
                if 'req_clarification' in gpt_value:
                    import re
                    match = re.search(r'req_clarification\((\d+)\)', gpt_value)
                    if match:
                        label = int(match.group(1))
                        label = min(label, self.num_classes - 1)  # Clip to valid range
                elif global_label is not None:
                    label = global_label
                
                i += 1
            
            # Create pair
            if user_parts and system_parts:
                user_utterances.append(' '.join(user_parts))
                system_utterances.append(' '.join(system_parts))
                labels.append(label)
        
        return user_utterances, system_utterances, labels
    
    def process_conversation(self, data: Dict, use_ground_truth: bool = True) -> Dict:
        """
        Process a single conversation.
        
        Args:
            data: Input data dict
            use_ground_truth: Use provided labels (training) vs auto-detect (inference)
        
        Returns:
            Processed data ready for model
        """
        print(f"\n[Process] Processing conversation...")
        
        # Parse
        user_utterances, system_utterances, labels = self.parse_conversations(data)
        num_pairs = len(user_utterances)
        
        print(f"  Found {num_pairs} user-system pairs")
        if use_ground_truth:
            print(f"  Labels: {labels}")
        
        # Tokenize
        user_tokens = []
        system_tokens = []
        
        for user_text, system_text in zip(user_utterances, system_utterances):
            # User
            user_enc = self.tokenizer(
                user_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            user_tokens.append(user_enc['input_ids'].squeeze(0))
            
            # System
            system_enc = self.tokenizer(
                system_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            system_tokens.append(system_enc['input_ids'].squeeze(0))
        
        # Stack
        user_tokens_tensor = torch.stack(user_tokens) if user_tokens else torch.empty(0, self.max_len, dtype=torch.long)
        system_tokens_tensor = torch.stack(system_tokens) if system_tokens else torch.empty(0, self.max_len, dtype=torch.long)
        
        # Labels
        if use_ground_truth:
            system_labels = torch.tensor(labels, dtype=torch.long)
        else:
            system_labels = torch.zeros(num_pairs, dtype=torch.long)
        
        # Metadata
        metadata = {
            'num_pairs': num_pairs,
            'prompt': data.get('prompt', ''),
            'context': data.get('context', ''),
            'can_retrieve': data.get('can_retrieve', False),
            'tools': data.get('tools', ''),
            'num_classes': self.num_classes,
            'class_names': self.class_names[self.num_classes]
        }
        
        result = {
            'user_utterance': user_tokens_tensor,
            'system_utterance': system_tokens_tensor,
            'user_I_label': torch.zeros(num_pairs, dtype=torch.long),  # Users don't have initiative
            'system_I_label': system_labels,
            'metadata': metadata
        }
        
        print(f"  Metadata: num_pairs={num_pairs}, num_classes={self.num_classes}")
        
        return result
    
    def get_class_name(self, class_idx: int) -> str:
        """Get human-readable class name."""
        return self.class_names[self.num_classes][class_idx]


if __name__ == "__main__":
    # Test
    sample_data = {
        "prompt": "Help user book travel",
        "context": "Travel booking assistant",
        "can_retrieve": True,
        "tools": "search_flights, book_hotel, req_clarification(2-4)",
        "ambiguous": "true",
        "ambiguous_type": 2,
        "conversations": [
            {"from": "human", "value": "Book flight"},
            {"from": "gpt", "value": "req_clarification(2)"}
        ]
    }
    
    # Test 2-class
    preprocessor = SIPPreprocessor(num_classes=2)
    processed = preprocessor.process_conversation(sample_data)
    print(f"\n2-class: Labels={processed['system_I_label']}")
    
    # Test 4-class
    preprocessor = SIPPreprocessor(num_classes=4)
    processed = preprocessor.process_conversation(sample_data)
    print(f"4-class: Labels={processed['system_I_label']}")
