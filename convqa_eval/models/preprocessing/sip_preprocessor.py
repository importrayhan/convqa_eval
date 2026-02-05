"""Preprocessor to convert QuAC format to BiLSTM-CRF input format."""
import json
import torch
from collections import Counter
from typing import List, Dict, Tuple
import re


class SIPPreprocessor:
    """
    Preprocess QuAC data for BiLSTM-CRF.
    
    Converts:
        QuAC format → BiLSTM-CRF tensors
    """
    
    def __init__(self, vocab_size: int = 10000, max_turn_len: int = 50, max_utterance_len: int = 30):
        """
        Initialize preprocessor.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_turn_len: Maximum conversation turns
            max_utterance_len: Maximum tokens per utterance
        """
        self.vocab_size = vocab_size
        self.max_turn_len = max_turn_len
        self.max_utterance_len = max_utterance_len
        
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
        print(f"\n[SIPPreprocessor] Initialized")
        print(f"[SIPPreprocessor] Vocab size: {vocab_size}")
        print(f"[SIPPreprocessor] Max turns: {max_turn_len}")
        print(f"[SIPPreprocessor] Max utterance length: {max_utterance_len}")
    
    def build_vocab(self, data: List[Dict]):
        """Build vocabulary from data."""
        print(f"\n[SIPPreprocessor] Building vocabulary from {len(data)} examples...")
        
        word_counts = Counter()
        
        for item in data:
            # Tokenize question and context
            question_tokens = self._tokenize(item.get("question", ""))
            context_tokens = self._tokenize(item.get("context", ""))
            
            word_counts.update(question_tokens)
            word_counts.update(context_tokens)
            
            # Process conversation history
            for turn in item.get("conversation", []):
                text_tokens = self._tokenize(turn.get("text", ""))
                word_counts.update(text_tokens)
        
        # Select top vocab_size-2 words (excluding <PAD> and <UNK>)
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"[SIPPreprocessor] Vocabulary built: {len(self.word2idx)} words")
        print(f"[SIPPreprocessor] Top 10 words: {list(self.word2idx.keys())[2:12]}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def preprocess_batch(self, data: List[Dict]) -> Tuple[torch.Tensor, ...]:
        """
        Convert batch of QuAC examples to BiLSTM-CRF format.
        
        Args:
            data: List of QuAC examples
        
        Returns:
            Tuple of (user_utterance, system_utterance, system_I_label, user_I_label, mask)
        """
        print(f"\n[SIPPreprocessor] Preprocessing batch of {len(data)} examples...")
        
        batch_size = len(data)
        
        # Initialize tensors
        user_utterance = torch.zeros(batch_size, self.max_turn_len, self.max_utterance_len, dtype=torch.long)
        system_utterance = torch.zeros(batch_size, self.max_turn_len, self.max_utterance_len, dtype=torch.long)
        system_I_label = torch.zeros(batch_size, self.max_turn_len, dtype=torch.long)
        user_I_label = torch.zeros(batch_size, self.max_turn_len, dtype=torch.long)
        mask = torch.zeros(batch_size, self.max_turn_len, dtype=torch.long)
        
        for batch_idx, item in enumerate(data):
            # Extract conversation turns
            conversation = item.get("conversation", [])
            
            # Current question (treated as latest user utterance)
            current_question = item.get("question", "")
            
            # Build conversation history
            user_turns = []
            system_turns = []
            
            for turn in conversation:
                speaker = turn.get("speaker", "USER")
                text = turn.get("text", "")
                
                if speaker == "USER":
                    user_turns.append(text)
                else:
                    system_turns.append(text)
            
            # Add current question as final user turn
            user_turns.append(current_question)
            
            # Derive initiative labels (heuristic based on ambiguity)
            is_ambiguous = item.get("ambiguous_utterance", False)
            num_conditions = len(item.get("conditions", {}))
            
            # Simple heuristic: ambiguous questions suggest system should take initiative
            system_initiative = 1 if is_ambiguous else 0
            user_initiative = 1 if num_conditions > 0 else 0
            
            print(f"  [Example {batch_idx}] Turns: {len(user_turns)} user, {len(system_turns)} system")
            print(f"  [Example {batch_idx}] Ambiguous: {is_ambiguous}, Initiative: {system_initiative}")
            
            # Fill tensors (limit to max_turn_len)
            actual_turns = min(len(user_turns), len(system_turns), self.max_turn_len)
            
            for turn_idx in range(actual_turns):
                # User utterance
                user_tokens = self._tokenize(user_turns[turn_idx])
                user_ids = [self.word2idx.get(w, 1) for w in user_tokens[:self.max_utterance_len]]
                user_utterance[batch_idx, turn_idx, :len(user_ids)] = torch.tensor(user_ids)
                
                # System utterance
                if turn_idx < len(system_turns):
                    system_tokens = self._tokenize(system_turns[turn_idx])
                    system_ids = [self.word2idx.get(w, 1) for w in system_tokens[:self.max_utterance_len]]
                    system_utterance[batch_idx, turn_idx, :len(system_ids)] = torch.tensor(system_ids)
                
                # Labels (same for all turns in this simple version)
                system_I_label[batch_idx, turn_idx] = system_initiative
                user_I_label[batch_idx, turn_idx] = user_initiative
                
                # Mask (valid turns)
                mask[batch_idx, turn_idx] = 1
        
        print(f"[SIPPreprocessor] Batch tensors created:")
        print(f"  User utterance: {user_utterance.shape}")
        print(f"  System utterance: {system_utterance.shape}")
        print(f"  System labels: {system_I_label.shape}")
        print(f"  Mask: {mask.shape}")
        
        return user_utterance, system_utterance, system_I_label, user_I_label, mask
