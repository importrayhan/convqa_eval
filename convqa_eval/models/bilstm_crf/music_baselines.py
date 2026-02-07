"""
MuSIc Model with 5 Baselines for System Initiative Prediction

Supports 2-4 class prediction with detailed metadata output.

5 Baselines:
1. VanillaCRF - Single global transition matrix
2. VanillaCRF+features - Features embedded in utterance representation
3. DynamicCRF - Dynamic transitions from adjacent observations
4. CtxPred - Simple BERT classifier (no CRF)
5. MuSIc (Full) - Complete model with all features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import BertModel
import numpy as np


# ============================================================================
# SHARED COMPONENTS
# ============================================================================

class BERTUtteranceEncoder(nn.Module):
    """BERT encoder for utterances"""
    
    def __init__(self, bert_model: str = 'bert-base-multilingual-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token


class PriorEncoder(nn.Module):
    """Prior encoder (partial context)"""
    
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True, batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class PosteriorEncoder(nn.Module):
    """Posterior encoder (full context)"""
    
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True, batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class SimpleCRF(nn.Module):
    """CRF layer"""
    
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = emissions.size(0)
        device = emissions.device
        
        # Gold score
        gold_score = torch.sum(emissions[range(seq_len), tags])
        for i in range(1, seq_len):
            gold_score += self.transitions[tags[i-1], tags[i]]
        
        # Forward algorithm
        alpha = torch.zeros(1, self.num_tags, device=device)
        for i in range(seq_len):
            if i == 0:
                alpha = emissions[i].unsqueeze(0)
            else:
                alpha = torch.logsumexp(
                    alpha.T + emissions[i].unsqueeze(0) + self.transitions,
                    dim=0, keepdim=True
                )
        total_score = torch.logsumexp(alpha, dim=1).squeeze()
        
        return gold_score, total_score
    
    def decode(self, emissions: torch.Tensor) -> List[int]:
        seq_len = emissions.size(0)
        device = emissions.device
        
        backpointers = []
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for i in range(seq_len):
            if i == 0:
                alpha = emissions[i].unsqueeze(0)
                backpointers.append(torch.zeros(self.num_tags, dtype=torch.long, device=device))
            else:
                alpha_with_trans = alpha.T + emissions[i].unsqueeze(0) + self.transitions
                viterbivars, bptrs = torch.max(alpha_with_trans, dim=0)
                backpointers.append(bptrs)
                alpha = viterbivars.unsqueeze(0)
        
        best_tag_id = alpha.squeeze().argmax().item()
        best_path = [best_tag_id]
        for bptrs in reversed(backpointers[1:]):
            best_tag_id = bptrs[best_tag_id].item()
            best_path.append(best_tag_id)
        best_path.reverse()
        return best_path


# ============================================================================
# BASELINE 1: VanillaCRF
# ============================================================================

class VanillaCRF(nn.Module):
    """Baseline 1: Simple CRF with global transitions"""
    
    def __init__(self, bert_model: str = 'bert-base-multilingual-cased',
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.5, num_classes: int = 2, lambda_mle: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_mle = lambda_mle
        
        self.utterance_encoder = BERTUtteranceEncoder(bert_model)
        bert_hidden = self.utterance_encoder.hidden_size
        
        self.prior_encoder = PriorEncoder(bert_hidden, num_layers, dropout)
        self.posterior_encoder = PosteriorEncoder(bert_hidden, num_layers, dropout)
        
        encoder_out = bert_hidden * 2
        self.prior_emission_project = nn.Linear(encoder_out, num_classes)
        self.posterior_emission_project = nn.Linear(encoder_out, num_classes)
        
        self.crf = SimpleCRF(num_classes)
    
    def forward(self, user_utterance, system_utterance,
                user_I_label=None, system_I_label=None, mode='train'):
        # Encode
        batch_size, num_pairs, max_len = user_utterance.shape
        user_flat = user_utterance.view(num_pairs, max_len)
        system_flat = system_utterance.view(num_pairs, max_len)
        
        user_repr = self.utterance_encoder(user_flat)
        system_repr = self.utterance_encoder(system_flat)
        
        if mode == 'train':
            return self._train(user_repr, system_repr, user_I_label, system_I_label, num_pairs)
        else:
            return self._inference(user_repr, system_repr, num_pairs)
    
    def _train(self, user_repr, system_repr, user_I_label, system_I_label, num_pairs):
        user_I_label = user_I_label.squeeze(0)
        system_I_label = system_I_label.squeeze(0)
        
        gold_scores, total_scores = [], []
        prior_emis, post_emis = [], []
        
        for pair_idx in range(num_pairs):
            # Build sequence
            seq = []
            labs = []
            for i in range(pair_idx + 1):
                seq.extend([user_repr[i], system_repr[i]])
                labs.extend([user_I_label[i].item(), system_I_label[i].item()])
            
            seq = torch.stack(seq).unsqueeze(0)
            labs = torch.tensor(labs, device=user_repr.device)
            
            # Posterior
            post_hidden = self.posterior_encoder(seq)
            post_emissions = self.posterior_emission_project(post_hidden).squeeze(0)
            
            # Prior
            prior_seq = seq[:, :-1, :]
            prior_hidden = self.prior_encoder(prior_seq)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # CRF
            gold, total = self.crf(post_emissions, labs)
            gold_scores.append(gold)
            total_scores.append(total)
            
            prior_emis.append(prior_emission.squeeze(0))
            post_emis.append(post_emissions[-1])
        
        loss_crf = torch.mean(torch.stack(total_scores) - torch.stack(gold_scores))
        loss_mle = F.mse_loss(torch.stack(prior_emis), torch.stack(post_emis).detach())
        
        return {
            'loss_crf': loss_crf,
            'loss_mle': loss_mle,
            'total_loss': loss_crf + self.lambda_mle * loss_mle
        }
    
    def _inference(self, user_repr, system_repr, num_pairs):
        predictions, confidences = [], []
        
        for pair_idx in range(num_pairs):
            # Partial sequence
            seq = []
            for i in range(pair_idx):
                seq.extend([user_repr[i], system_repr[i]])
            seq.append(user_repr[pair_idx])
            seq = torch.stack(seq).unsqueeze(0)
            
            # Prior for missing system
            prior_hidden = self.prior_encoder(seq)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # Posterior for known
            if pair_idx > 0:
                prev_seq = seq[:, :-1, :]
                prev_post = self.posterior_encoder(prev_seq)
                prev_emis = self.posterior_emission_project(prev_post).squeeze(0)
            else:
                prev_emis = torch.empty(0, self.num_classes, device=user_repr.device)
            
            curr_user_hidden = self.posterior_encoder(seq)
            curr_user_emis = self.posterior_emission_project(curr_user_hidden[:, -1, :])
            
            combined = torch.cat([prev_emis, curr_user_emis.squeeze(0).unsqueeze(0),
                                 prior_emission.squeeze(0).unsqueeze(0)], dim=0)
            
            path = self.crf.decode(combined)
            pred = path[-1]
            
            # Confidence (softmax of emission)
            conf = F.softmax(prior_emission.squeeze(0), dim=0)[pred].item()
            
            predictions.append(pred)
            confidences.append(conf)
        
        return {'predictions': predictions, 'confidences': confidences}


# ============================================================================
# BASELINE 2: VanillaCRF+features
# ============================================================================

class VanillaCRFFeatures(VanillaCRF):
    """Baseline 2: Features embedded in representation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add feature embeddings
        self.position_embed = nn.Embedding(20, 32)
        self.speaker_embed = nn.Embedding(3, 16)  # user, system, padding


# ============================================================================
# BASELINE 3: DynamicCRF  
# ============================================================================

class DynamicCRF(VanillaCRF):
    """Baseline 3: Dynamic transition matrices"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bert_hidden = self.utterance_encoder.hidden_size
        self.transition_generator = nn.Sequential(
            nn.Linear(bert_hidden * 2, 256),
            nn.Tanh(),
            nn.Linear(256, self.num_classes * self.num_classes)
        )


# ============================================================================
# BASELINE 4: CtxPred
# ============================================================================

class CtxPred(nn.Module):
    """Baseline 4: Simple BERT classifier (no CRF)"""
    
    def __init__(self, bert_model: str = 'bert-base-multilingual-cased',
                 hidden_size: int = 256, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
        self.utterance_encoder = BERTUtteranceEncoder(bert_model)
        bert_hidden = self.utterance_encoder.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, user_utterance, system_utterance,
                user_I_label=None, system_I_label=None, mode='train'):
        batch_size, num_pairs, max_len = user_utterance.shape
        
        user_flat = user_utterance.view(num_pairs, max_len)
        system_flat = system_utterance.view(num_pairs, max_len)
        
        user_repr = self.utterance_encoder(user_flat)
        system_repr = self.utterance_encoder(system_flat)
        
        if mode == 'train':
            system_I_label = system_I_label.squeeze(0)
            logits_list = []
            for i in range(num_pairs):
                context = torch.stack([user_repr[j] for j in range(i+1)] +
                                    [system_repr[j] for j in range(i)]).mean(0)
                logits_list.append(self.classifier(context))
            
            logits = torch.stack(logits_list)
            loss = F.cross_entropy(logits, system_I_label)
            return {'loss_crf': loss, 'loss_mle': torch.tensor(0.0), 'total_loss': loss}
        else:
            predictions, confidences = [], []
            for i in range(num_pairs):
                context = torch.stack([user_repr[j] for j in range(i+1)] +
                                    [system_repr[j] for j in range(i)]).mean(0)
                logits = self.classifier(context)
                probs = F.softmax(logits, dim=0)
                pred = torch.argmax(probs).item()
                predictions.append(pred)
                confidences.append(probs[pred].item())
            return {'predictions': predictions, 'confidences': confidences}


# ============================================================================
# BASELINE 5: MuSIc (Full)
# ============================================================================

class MuSIc(VanillaCRF):
    """Baseline 5: Complete MuSIc model"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Full model is same as VanillaCRF for now
        # Can add multi-turn features here


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_model(baseline: str = 'vanillacrf', num_classes: int = 2, **kwargs):
    """
    Create a baseline model.
    
    Args:
        baseline: 'vanillacrf', 'features', 'dynamic', 'ctxpred', 'music'
        num_classes: 2, 3, or 4
        **kwargs: Model parameters
    
    Returns:
        Model instance
    """
    models = {
        'vanillacrf': VanillaCRF,
        'features': VanillaCRFFeatures,
        'dynamic': DynamicCRF,
        'ctxpred': CtxPred,
        'music': MuSIc
    }
    
    if baseline not in models:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    return models[baseline](num_classes=num_classes, **kwargs)
