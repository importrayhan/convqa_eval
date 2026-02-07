"""
Preprocessor for MuSIc SIP Model

Key Points from Paper:
- Process utterances SEQUENTIALLY (no odd/even distinction)
- During training: Use X1:T (all utterances including current system)
- During inference: Use X1:T-1 (hide the unobservable system utterance at turn T)
- User and system utterances alternate: [user1, sys1, user2, sys2, ...]
"""

import json
import torch
from typing import List, Dict, Tuple
from transformers import BertTokenizer


class SIPPreprocessor:
    """
    Preprocesses conversation data for SIP (System Initiative Prediction).
    
    Format: Alternating user-system utterances
    - Turn 1: user_utterance_1, system_utterance_1
    - Turn 2: user_utterance_2, system_utterance_2
    - ...
    
    Each turn has:
    - user_utterance: Input from user
    - system_utterance: System response  
    - system_I_label: Whether system takes initiative (0/1)
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        max_len: int = 128
    ):
        print(f"[SIPPreprocessor] Initializing...")
        print(f"  BERT model: {bert_model}")
        print(f"  Max length: {max_len}")
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_len = max_len
        
        print(f"[SIPPreprocessor] Initialized\n")
    
    def parse_conversations(self, data: Dict) -> Tuple[List[str], List[str]]:
        """
        Parse conversation into user-system pairs.
        
        From your format:
        [human, function_call, observation, gpt, human, gpt, ...]
        
        We convert to alternating user-system:
        - Merge (human + observation) -> user context
        - Merge (function_call + gpt) -> system response
        
        Args:
            data: Conversation dict with 'conversations' list
        
        Returns:
            user_utterances: List of user utterances
            system_utterances: List of system responses
        """
        conversations = data.get('conversations', [])
        
        user_utterances = []
        system_utterances = []
        
        i = 0
        while i < len(conversations):
            # Collect user context (human + possibly observation)
            user_parts = []
            
            # First should be human
            if i < len(conversations) and conversations[i]['from'] == 'human':
                user_parts.append(conversations[i]['value'])
                i += 1
            
            # System response (function_call + gpt, or just gpt)
            system_parts = []
            
            # Might be function_call
            if i < len(conversations) and conversations[i]['from'] == 'function_call':
                system_parts.append(f"[TOOL] {conversations[i]['value']}")
                i += 1
            
            # Then observation (becomes part of next user context)
            if i < len(conversations) and conversations[i]['from'] == 'observation':
                user_parts.append(f"[RESULT] {conversations[i]['value']}")
                i += 1
            
            # Then gpt response
            if i < len(conversations) and conversations[i]['from'] == 'gpt':
                system_parts.append(conversations[i]['value'])
                i += 1
            
            # Create turn pair
            if user_parts and system_parts:
                user_utterances.append(' '.join(user_parts))
                system_utterances.append(' '.join(system_parts))
        
        return user_utterances, system_utterances
    
    def detect_initiative(
        self,
        system_utterance: str,
        user_utterance: str,
        context: List[str]
    ) -> int:
        """
        Detect if system takes initiative.
        
        Initiative indicators:
        - Contains [TOOL] (function call)
        - Asks clarification questions
        - Makes proactive suggestions
        
        Args:
            system_utterance: System's response
            user_utterance: User's input
            context: Previous utterances
        
        Returns:
            1 if takes initiative, 0 otherwise
        """
        # Function calls always indicate initiative
        if '[TOOL]' in system_utterance:
            return 1
        
        system_lower = system_utterance.lower()
        
        # Clarification questions
        clarification_patterns = [
            'could you clarify', 'can you specify', 'do you mean',
            'which one', 'what do you mean', 'could you tell me more'
        ]
        if any(p in system_lower for p in clarification_patterns):
            return 1
        
        # Proactive suggestions
        suggestion_patterns = [
            'i suggest', 'i recommend', 'you might want',
            'have you considered', 'would you like', 'shall i'
        ]
        if any(p in system_lower for p in suggestion_patterns):
            return 1
        
        # Questions asking for more info
        if '?' in system_utterance and any(
            word in system_lower for word in ['would', 'could', 'should', 'do you']
        ):
            return 1
        
        return 0
    
    def process_conversation(
        self,
        data: Dict,
        auto_label: bool = True
    ) -> Dict:
        """
        Process a single conversation.
        
        Returns:
            user_utterance: [num_pairs, max_len] - tokenized user utterances
            system_utterance: [num_pairs, max_len] - tokenized system utterances  
            user_I_label: [num_pairs] - always 0 (users don't take initiative)
            system_I_label: [num_pairs] - initiative labels for system
            metadata: conversation metadata
        """
        print(f"\n[Process] Processing conversation...")
        
        # Parse into user-system pairs
        user_utterances, system_utterances = self.parse_conversations(data)
        num_pairs = len(user_utterances)
        
        print(f"  Found {num_pairs} user-system pairs")
        
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
        
        # Generate labels
        user_I_labels = [0] * num_pairs  # Users never take initiative
        system_I_labels = []
        
        if auto_label:
            context = []
            for i in range(num_pairs):
                # Detect system initiative
                initiative = self.detect_initiative(
                    system_utterances[i],
                    user_utterances[i],
                    context
                )
                system_I_labels.append(initiative)
                
                # Update context
                context.extend([user_utterances[i], system_utterances[i]])
                
                print(f"  Pair {i+1}: Initiative={initiative}")
        else:
            system_I_labels = [0] * num_pairs
        
        # Stack tensors
        user_tokens_tensor = torch.stack(user_tokens)  # [num_pairs, max_len]
        system_tokens_tensor = torch.stack(system_tokens)
        
        metadata = {
            'num_pairs': num_pairs,
            'system_initiatives': sum(system_I_labels),
            'initiative_rate': sum(system_I_labels) / num_pairs if num_pairs > 0 else 0
        }
        
        result = {
            'user_utterance': user_tokens_tensor,
            'system_utterance': system_tokens_tensor,
            'user_I_label': torch.tensor(user_I_labels, dtype=torch.long),
            'system_I_label': torch.tensor(system_I_labels, dtype=torch.long),
            'metadata': metadata
        }
        
        print(f"  Metadata: {metadata}")
        
        return result


if __name__ == "__main__":
    # Test
    sample_data = {
        "conversations": [
            {"from": "human", "value": "Book a hotel in Paris"},
            {"from": "function_call", "value": "search_hotels(city='Paris')"},
            {"from": "observation", "value": "Found 10 hotels"},
            {"from": "gpt", "value": "I found 10 hotels. Would you like to see options?"},
            {"from": "human", "value": "Yes please"},
            {"from": "gpt", "value": "Here are the top 3: Hotel A, B, C"}
        ]
    }
    
    preprocessor = SIPPreprocessor()
    processed = preprocessor.process_conversation(sample_data)
    
    print("\n=== Output ===")
    print(f"User utterances: {processed['user_utterance'].shape}")
    print(f"System utterances: {processed['system_utterance'].shape}")
    print(f"System I labels: {processed['system_I_label']}")
