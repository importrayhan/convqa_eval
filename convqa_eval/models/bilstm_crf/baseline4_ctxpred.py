"""
Baseline 4: CtxPred (BERT)
BERT encoder to predict initiative at next turn based on context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from music_base import BERTUtteranceEncoder


class CtxPred(nn.Module):
    """
    Baseline 4: CtxPred (BERT)
    
    From paper:
    "CtxPred (BERT) uses a BERT encoder to encode the context and predict
    whether to take the initiative at the next turn."
    
    Simple classifier: BERT(context) → binary prediction
    No CRF, no prior-posterior framework.
    
    Use case: Simple baseline, fast inference
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-multilingual-cased',
        hidden_size: int = 256,
        dropout: float = 0.5,
        num_tags: int = 2
    ):
        super().__init__()
        
        self.num_tags = num_tags
        
        print(f"\n[CtxPred] Initializing...")
        
        # BERT encoder
        self.utterance_encoder = BERTUtteranceEncoder(bert_model_name)
        bert_hidden = self.utterance_encoder.hidden_size
        
        # Context aggregation (mean pooling over utterances)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags)
        )
        
        print(f"[CtxPred] Initialized")
        print("  Features: BERT context encoding → classification")
        print("  No CRF, no prior-posterior")
        print("  Best for: Fast, simple baseline\n")
    
    def forward(
        self,
        odd_input_ids: torch.Tensor,
        even_input_ids: torch.Tensor,
        odd_masks: torch.Tensor = None,
        even_masks: torch.Tensor = None,
        odd_I_labels: torch.Tensor = None,
        system_I_labels: torch.Tensor = None,
        multiturn_features: list = None,
        mode: str = 'train'
    ) -> dict:
        """
        Forward pass.
        
        For each turn, encodes all previous context and predicts initiative.
        """
        batch_size, num_pairs, max_len = odd_input_ids.shape
        assert batch_size == 1
        
        # Flatten and encode all utterances
        odd_flat = odd_input_ids.view(-1, max_len)
        even_flat = even_input_ids.view(-1, max_len)
        
        odd_mask_flat = odd_masks.view(-1, max_len) if odd_masks is not None else None
        even_mask_flat = even_masks.view(-1, max_len) if even_masks is not None else None
        
        odd_repr = self.utterance_encoder(odd_flat, odd_mask_flat)
        even_repr = self.utterance_encoder(even_flat, even_mask_flat)
        
        odd_repr = odd_repr.view(1, num_pairs, -1).squeeze(0)
        even_repr = even_repr.view(1, num_pairs, -1).squeeze(0)
        
        if mode == 'train':
            return self._forward_train(odd_repr, even_repr, system_I_labels)
        else:
            return self._forward_inference(odd_repr, even_repr)
    
    def _forward_train(self, odd_repr, even_repr, system_I_labels):
        """Training: predict for each turn independently."""
        system_I_labels = system_I_labels.squeeze(0)
        num_pairs = odd_repr.size(0)
        
        print(f"\n[CtxPred Train] Processing {num_pairs} pairs...")
        
        all_logits = []
        all_labels = []
        
        for pair_idx in range(num_pairs):
            # Context: all previous utterances
            context_reprs = []
            for i in range(pair_idx + 1):
                context_reprs.append(odd_repr[i])
                if i < pair_idx:  # Don't include current even yet
                    context_reprs.append(even_repr[i])
            
            # Mean pool context
            context = torch.stack(context_reprs).mean(dim=0)
            
            # Predict
            logits = self.classifier(context)
            all_logits.append(logits)
            all_labels.append(system_I_labels[pair_idx])
        
        # Stack and compute loss
        logits_tensor = torch.stack(all_logits)
        labels_tensor = torch.stack(all_labels)
        
        loss = F.cross_entropy(logits_tensor, labels_tensor)
        
        print(f"[CtxPred Train] Loss: {loss:.4f}")
        
        return {
            'loss_crf': loss,  # For compatibility
            'loss_mle_e': torch.tensor(0.0),
            'total_loss': loss
        }
    
    def _forward_inference(self, odd_repr, even_repr):
        """Inference: predict for each turn."""
        num_pairs = odd_repr.size(0)
        
        print(f"\n[CtxPred Inference] Processing {num_pairs} pairs...")
        
        predictions = []
        initiative_scores = []
        
        for pair_idx in range(num_pairs):
            # Context
            context_reprs = []
            for i in range(pair_idx + 1):
                context_reprs.append(odd_repr[i])
                if i < pair_idx:
                    context_reprs.append(even_repr[i])
            
            context = torch.stack(context_reprs).mean(dim=0)
            
            # Predict
            logits = self.classifier(context)
            pred = torch.argmax(logits).item()
            
            predictions.append([pred])
            initiative_scores.append(pred)
        
        return {
            'predictions': predictions,
            'initiative_scores': initiative_scores,
            'has_initiative': [score > 0 for score in initiative_scores]
        }
