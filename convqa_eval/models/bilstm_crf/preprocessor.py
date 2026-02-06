"""
Preprocessor for MuSIc-style SIP with Function Calls
Handles human/observation (odd) and gpt/function (even) positions
"""

import json
import torch
from typing import List, Dict, Tuple, Optional
from collections import Counter
from transformers import BertTokenizer
import numpy as np


class MuSIcPreprocessor:
    """
    Preprocesses conversation data for MuSIc SIP model.
    
    Data Format:
        - human/observation: Odd positions (1, 3, 5, ...)
        - gpt/function_call: Even positions (2, 4, 6, ...)
        - Model learns to predict gpt/function responses
    
    Turn Structure:
        Turn 1: human (odd) → gpt/function (even)
        Turn 2: observation (odd) → gpt/function (even)
        ...
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        max_len: int = 128,
        num_tags: int = 2
    ):
        """
        Initialize preprocessor.
        
        Args:
            bert_model: BERT model name
            max_len: Maximum sequence length
            num_tags: Number of SIP tags (2=binary initiative)
        """
        print(f"[MuSIcPreprocessor] Initializing...")
        print(f"  BERT model: {bert_model}")
        print(f"  Max length: {max_len}")
        print(f"  Num tags: {num_tags}")
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_len = max_len
        self.num_tags = num_tags
        
        print(f"[MuSIcPreprocessor] Initialized successfully")
    
    def parse_conversations(self, data: Dict) -> Tuple[List[str], List[str], List[str]]:
        """
        Parse conversation data into odd/even positions.
        
        Args:
            data: Conversation dictionary
        
        Returns:
            odd_utterances: human/observation (input context)
            even_utterances: gpt/function (model predictions)
            turn_types: Type of each turn (human/observation/gpt/function)
        """
        conversations = data.get('conversations', [])
        
        odd_utterances = []  # human/observation
        even_utterances = []  # gpt/function
        turn_types = []
        
        for i, turn in enumerate(conversations):
            role = turn.get('from', '')
            text = turn.get('value', '')
            
            if i % 2 == 0:  # Odd position (0-indexed, so even index = position 1, 3, 5...)
                # Should be human or observation
                if role not in ['human', 'observation']:
                    print(f"  Warning: Expected human/observation at position {i+1}, got {role}")
                odd_utterances.append(text)
                turn_types.append(role)
            else:  # Even position (position 2, 4, 6...)
                # Should be gpt or function_call
                if role not in ['gpt', 'function_call']:
                    print(f"  Warning: Expected gpt/function_call at position {i+1}, got {role}")
                even_utterances.append(text)
                turn_types.append(role)
        
        return odd_utterances, even_utterances, turn_types
    
    def detect_initiative(
        self,
        even_utterance: str,
        turn_type: str,
        context: List[str]
    ) -> int:
        """
        Detect if system takes initiative at this turn.
        
        Initiative indicators:
        - Function calls (proactive tool use)
        - Clarification questions
        - Suggestions/recommendations
        - Asking for more information
        
        Args:
            even_utterance: gpt/function response
            turn_type: 'gpt' or 'function_call'
            context: Previous utterances
        
        Returns:
            Initiative label (0=no initiative, 1=takes initiative)
        """
        text_lower = even_utterance.lower().strip()
        
        # Function calls always indicate initiative
        if turn_type == 'function_call':
            return 1
        
        # For gpt responses, check for initiative patterns
        initiative_score = 0
        
        # 1. Clarification questions
        clarification_patterns = [
            'could you clarify', 'can you specify', 'do you mean',
            'which one', 'what do you mean by', 'to clarify'
        ]
        if any(pattern in text_lower for pattern in clarification_patterns):
            initiative_score += 2
        
        # 2. Proactive suggestions
        suggestion_patterns = [
            'i suggest', 'i recommend', 'you might want',
            'have you considered', 'you could also', 'let me help'
        ]
        if any(pattern in text_lower for pattern in suggestion_patterns):
            initiative_score += 2
        
        # 3. Questions (proactive engagement)
        if '?' in even_utterance:
            initiative_score += 1
        
        # 4. Tool mention (even without calling)
        tool_patterns = ['use a tool', 'call', 'function', 'search', 'fetch']
        if any(pattern in text_lower for pattern in tool_patterns):
            initiative_score += 1
        
        return 1 if initiative_score >= 2 else 0
    
    def extract_multiturn_features(
        self,
        turn_idx: int,
        initiative_history: List[int],
        turn_types: List[str]
    ) -> Dict[str, int]:
        """
        Extract multi-turn features for CRF.
        
        Features:
        1. Who2Who: odd→even (human/observation→gpt/function) vs even→odd
        2. Position: Turn position in conversation (1→2, 2→3, etc.)
        3. Initiative Count (Intime): Number of prior system initiatives
        4. Initiative Distance: Distance from last system initiative
        
        Args:
            turn_idx: Current turn index (0-based)
            initiative_history: History of system initiatives
            turn_types: Types of each turn
        
        Returns:
            Feature dictionary
        """
        features = {
            'who2who': -1,
            'position': -1,
            'intime': -1,
            'distance': -1
        }
        
        # First turn has no transitions
        if turn_idx == 0:
            return features
        
        # Who2Who: Check if previous turn is odd or even
        # Odd positions (human/observation) are at even indices
        # Even positions (gpt/function) are at odd indices
        prev_is_odd = (turn_idx - 1) % 2 == 0
        curr_is_odd = turn_idx % 2 == 0
        
        if prev_is_odd and not curr_is_odd:
            features['who2who'] = 0  # odd→even (user→system)
        elif not prev_is_odd and curr_is_odd:
            features['who2who'] = 1  # even→odd (system→user)
        else:
            features['who2who'] = 2  # Same type (shouldn't happen)
        
        # Position (0-indexed, up to 19)
        position_idx = turn_idx - 1
        if position_idx < 20:
            features['position'] = position_idx
        
        # Only compute initiative features for even positions (system turns)
        if turn_idx % 2 == 1:  # Even position (gpt/function)
            # Count prior system initiatives
            system_turn_idx = turn_idx // 2
            prior_initiatives = sum(initiative_history[:system_turn_idx])
            
            if prior_initiatives == 0:
                features['intime'] = 0
            elif prior_initiatives == 1:
                features['intime'] = 1
            else:
                features['intime'] = 2
            
            # Distance from last initiative
            if prior_initiatives > 0:
                # Find last initiative position
                last_init_pos = -1
                for i in range(system_turn_idx - 1, -1, -1):
                    if initiative_history[i] == 1:
                        last_init_pos = i * 2 + 1  # Convert to turn index
                        break
                
                if last_init_pos >= 0:
                    distance = turn_idx - last_init_pos
                    if distance == 2:
                        features['distance'] = 0  # Consecutive
                    else:
                        features['distance'] = 1  # Non-consecutive
        
        return features
    
    def tokenize_bert(self, text: str) -> Tuple[List[int], List[int]]:
        """Tokenize text using BERT."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0).tolist()
        attention_mask = encoding['attention_mask'].squeeze(0).tolist()
        
        return input_ids, attention_mask
    
    def process_conversation(
        self,
        data: Dict,
        auto_label: bool = True
    ) -> Dict:
        """
        Process a single conversation.
        
        Args:
            data: Conversation dictionary
            auto_label: Automatically generate initiative labels
        
        Returns:
            Processed data with:
                - odd_utterances: human/observation texts
                - even_utterances: gpt/function texts
                - odd_tokens/masks: Tokenized odd utterances
                - even_tokens/masks: Tokenized even utterances
                - system_I_labels: Initiative labels for even positions
                - multiturn_features: Features for each turn
                - metadata: Conversation metadata
        """
        print(f"\n[MuSIcPreprocessor] Processing conversation...")
        
        # Parse conversations
        odd_utterances, even_utterances, turn_types = self.parse_conversations(data)
        
        num_pairs = min(len(odd_utterances), len(even_utterances))
        odd_utterances = odd_utterances[:num_pairs]
        even_utterances = even_utterances[:num_pairs]
        
        print(f"  Found {num_pairs} turn pairs")
        print(f"  Odd positions (input): {len(odd_utterances)}")
        print(f"  Even positions (system): {len(even_utterances)}")
        
        # Tokenize
        odd_tokens = []
        even_tokens = []
        odd_masks = []
        even_masks = []
        
        for odd_text, even_text in zip(odd_utterances, even_utterances):
            odd_ids, odd_mask = self.tokenize_bert(odd_text)
            even_ids, even_mask = self.tokenize_bert(even_text)
            
            odd_tokens.append(odd_ids)
            even_tokens.append(even_ids)
            odd_masks.append(odd_mask)
            even_masks.append(even_mask)
        
        # Generate initiative labels (only for even positions - system turns)
        system_I_labels = []
        odd_I_labels = []  # Always 0 for user turns
        
        if auto_label:
            context = []
            for i in range(num_pairs):
                # Odd position (user/observation) - no initiative
                odd_I_labels.append(0)
                context.append(odd_utterances[i])
                
                # Even position (system) - detect initiative
                turn_type = turn_types[i*2 + 1] if i*2 + 1 < len(turn_types) else 'gpt'
                initiative = self.detect_initiative(even_utterances[i], turn_type, context)
                system_I_labels.append(initiative)
                context.append(even_utterances[i])
                
                print(f"  Turn {i+1}: {turn_types[i*2] if i*2 < len(turn_types) else 'unknown'} → {turn_type}, Initiative={initiative}")
        else:
            odd_I_labels = [0] * num_pairs
            system_I_labels = [0] * num_pairs
        
        # Extract multi-turn features
        multiturn_features = []
        total_turns = num_pairs * 2
        
        for turn_idx in range(total_turns):
            system_turn_idx = turn_idx // 2
            features = self.extract_multiturn_features(
                turn_idx,
                system_I_labels[:system_turn_idx],
                turn_types
            )
            multiturn_features.append(features)
        
        # Metadata
        metadata = {
            'num_pairs': num_pairs,
            'num_turns': total_turns,
            'system_initiatives': sum(system_I_labels),
            'initiative_rate': sum(system_I_labels) / num_pairs if num_pairs > 0 else 0,
            'system_prompt': data.get('system', ''),
            'tools': data.get('tools', '')
        }
        
        result = {
            'odd_utterances': odd_utterances,
            'even_utterances': even_utterances,
            'odd_tokens': odd_tokens,
            'even_tokens': even_tokens,
            'odd_masks': odd_masks,
            'even_masks': even_masks,
            'odd_I_labels': odd_I_labels,
            'system_I_labels': system_I_labels,
            'multiturn_features': multiturn_features,
            'turn_types': turn_types,
            'metadata': metadata
        }
        
        print(f"[MuSIcPreprocessor] Processing complete")
        print(f"  Metadata: {metadata}")
        
        return result
    
    def to_tensor_batch(self, processed_data: Dict) -> Dict:
        """Convert processed data to PyTorch tensors."""
        return {
            'odd_utterance': torch.tensor([processed_data['odd_tokens']], dtype=torch.long),
            'even_utterance': torch.tensor([processed_data['even_tokens']], dtype=torch.long),
            'odd_mask': torch.tensor([processed_data['odd_masks']], dtype=torch.long),
            'even_mask': torch.tensor([processed_data['even_masks']], dtype=torch.long),
            'odd_I_label': torch.tensor([processed_data['odd_I_labels']], dtype=torch.long),
            'system_I_label': torch.tensor([processed_data['system_I_labels']], dtype=torch.long),
            'multiturn_features': processed_data['multiturn_features'],
            'metadata': processed_data['metadata']
        }


if __name__ == "__main__":
    # Test with new data format
    sample_data = {
        "conversations": [
            {"from": "human", "value": "Search for flights to Paris"},
            {"from": "function_call", "value": "search_flights(destination='Paris')"},
            {"from": "observation", "value": "Found 5 flights to Paris"},
            {"from": "gpt", "value": "I found 5 flights to Paris. Would you like to see options?"},
            {"from": "human", "value": "Yes, show me the cheapest one"},
            {"from": "gpt", "value": "The cheapest flight is $450 with Air France."}
        ],
        "system": "You are a helpful travel assistant",
        "tools": "search_flights, book_flight"
    }
    
    preprocessor = MuSIcPreprocessor(num_tags=2)
    processed = preprocessor.process_conversation(sample_data, auto_label=True)
    
    print("\n=== Processed Output ===")
    print(f"Num pairs: {processed['metadata']['num_pairs']}")
    print(f"System initiative labels: {processed['system_I_labels']}")
    print(f"Initiative rate: {processed['metadata']['initiative_rate']:.1%}")
    
    # Show multi-turn features
    print(f"\nMulti-turn features:")
    for i, feat in enumerate(processed['multiturn_features'][:6]):
        print(f"  Turn {i+1}: {feat}")
