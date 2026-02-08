"""
Revised Preprocessor for SIP with Per-GPT-Utterance Labels

CRITICAL CHANGE: Each GPT utterance has its own ambiguous_type label!

Input Format:
{
  "conversations": [
    {"from": "human", "value": "...", "turn_id": 1},
    {"from": "gpt", "value": "...", "turn_id": 1, "ambiguous_type": 2, "metadata": {...}},
    {"from": "human", "value": "...", "turn_id": 2},
    {"from": "gpt", "value": "...", "turn_id": 2, "ambiguous_type": 0, "metadata": {...}}
  ]
}
"""

import json
import torch
from typing import List, Dict, Tuple
from transformers import BertTokenizer


class SIPPreprocessor:
    """
    Preprocesses conversation data with PER-GPT-UTTERANCE labels.
    
    Key Features:
    - Extracts label from each GPT utterance
    - Handles 2-4 class configurations
    - BERT tokenization
    - Metadata preservation
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        max_len: int = 128,
        num_classes: int = 2
    ):
        print(f"[SIPPreprocessor] Initializing...")
        print(f"  BERT: {bert_model}")
        print(f"  Max length: {max_len}")
        print(f"  Num classes: {num_classes}")
        
        assert num_classes in [2, 3, 4], "num_classes must be 2, 3, or 4"
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_len = max_len
        self.num_classes = num_classes
        
        # Class names
        self.class_names = {
            2: ['clear', 'ambiguous'],
            3: ['clear', 'needs_clarification', 'highly_ambiguous'],
            4: ['clear', 'slightly_ambiguous', 'needs_clarification', 'highly_ambiguous']
        }
        
        print(f"  Classes: {self.class_names[num_classes]}")
        print()
    
    def parse_conversations(
        self,
        data: Dict
    ) -> Tuple[List[str], List[str], List[int], List[Dict]]:
        """
        Parse conversations and extract PER-TURN labels.
        
        CRITICAL: Each GPT utterance must have its own ambiguous_type!
        
        Returns:
            user_utterances: List of user inputs
            system_utterances: List of system responses  
            system_labels: List of labels (ONE PER SYSTEM UTTERANCE)
            turn_metadata: List of metadata per system turn
        """
        conversations = data.get('conversations', [])
        
        user_utterances = []
        system_utterances = []
        system_labels = []
        turn_metadata = []
        
        i = 0
        turn_id = 0
        
        while i < len(conversations):
            conv = conversations[i]
            
            # Skip function_call and observation
            if conv['from'] in ['function_call', 'observation']:
                i += 1
                continue
            
            # Human utterance
            if conv['from'] == 'human':
                user_text = conv['value']
                user_utterances.append(user_text)
                i += 1
                turn_id += 1
                
                # Next should be GPT (after possible function_call/observation)
                # Skip intermediate steps
                while i < len(conversations) and conversations[i]['from'] in ['function_call', 'observation']:
                    i += 1
                
                # Now should be GPT
                if i < len(conversations) and conversations[i]['from'] == 'gpt':
                    gpt_conv = conversations[i]
                    system_text = gpt_conv['value']
                    
                    # CRITICAL: Extract label from THIS specific GPT utterance
                    label = gpt_conv.get('ambiguous_type', 0)
                    
                    # Ensure label is in valid range
                    label = max(0, min(label, self.num_classes - 1))
                    
                    system_utterances.append(system_text)
                    system_labels.append(label)
                    
                    # Extract metadata
                    metadata = gpt_conv.get('metadata', {})
                    metadata['turn_id'] = gpt_conv.get('turn_id', turn_id)
                    turn_metadata.append(metadata)
                    
                    i += 1
        
        return user_utterances, system_utterances, system_labels, turn_metadata
    
    def process_conversation(
        self,
        data: Dict,
        use_ground_truth: bool = True
    ) -> Dict:
        """
        Process a single conversation.
        
        Args:
            data: Input data dict
            use_ground_truth: Use provided labels (training) vs zeros (inference)
        
        Returns:
            Processed data ready for model
        """
        print(f"\n[Process] Processing conversation...")
        
        # Parse
        user_utterances, system_utterances, system_labels, turn_metadata = \
            self.parse_conversations(data)
        
        num_pairs = len(user_utterances)
        
        print(f"  Found {num_pairs} user-system pairs")
        if use_ground_truth:
            print(f"  Labels (per GPT): {system_labels}")
        
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
            system_labels_tensor = torch.tensor(system_labels, dtype=torch.long)
        else:
            system_labels_tensor = torch.zeros(num_pairs, dtype=torch.long)
        
        # Metadata
        metadata = {
            'num_pairs': num_pairs,
            'num_classes': self.num_classes,
            'class_names': self.class_names[self.num_classes],
            'turn_metadata': turn_metadata,
            'label_distribution': {
                self.class_names[self.num_classes][i]: system_labels.count(i)
                for i in range(self.num_classes)
            } if use_ground_truth else {}
        }
        
        result = {
            'user_utterance': user_tokens_tensor,
            'system_utterance': system_tokens_tensor,
            'user_I_label': torch.zeros(num_pairs, dtype=torch.long),
            'system_I_label': system_labels_tensor,
            'metadata': metadata
        }
        
        print(f"  Processed: {num_pairs} pairs, {self.num_classes} classes")
        
        return result
    
    def get_class_name(self, class_idx: int) -> str:
        """Get human-readable class name"""
        return self.class_names[self.num_classes][class_idx]


if __name__ == "__main__":
    # Test with per-GPT labels
    sample_data = {
        "conversations": [
            {"from": "human", "value": "What is X?", "turn_id": 1},
            {
                "from": "gpt",
                "value": "Which year? req_clarification(2)",
                "turn_id": 1,
                "ambiguous_type": 2,
                "metadata": {
                    "answers": ["Which year?"],
                    "original_question": "What is X?"
                }
            },
            {"from": "human", "value": "2019", "turn_id": 2},
            {
                "from": "gpt",
                "value": "The value is 100",
                "turn_id": 2,
                "ambiguous_type": 0,
                "metadata": {
                    "answers": ["100"],
                    "original_question": "What is X in 2019?"
                }
            }
        ]
    }
    
    preprocessor = SIPPreprocessor(num_classes=4)
    processed = preprocessor.process_conversation(sample_data)
    
    print("\n=== Output ===")
    print(f"User utterances: {processed['user_utterance'].shape}")
    print(f"System utterances: {processed['system_utterance'].shape}")
    print(f"System labels: {processed['system_I_label']}")
    print(f"Metadata: {processed['metadata']}")
