"""
MuSIc: Complete Implementation with All 6 CRF Variants

EXACT SIP PAPER ALIGNMENT:
==========================
Architecture:
1. BERT Utterance Encoder
2. Prior/Posterior Inter-Utterance Encoders (BiLSTM)
3. Multi-Turn Feature-Aware CRF (6 variants)

Training: P(Y1:T | X1:T) - Has full context including system_T
Inference: P(Y1:T | X1:T-1) - Missing system_T (what we predict)

Key Guarantee: System utterance at turn T is HIDDEN during inference!

6 CRF Variants (from SIP paper):
1. VanillaCRF - Single global transition matrix
2. Who2WhoCRF - User↔System specific transitions
3. PositionCRF - Position-specific (turn 0→1, 1→2, ..., 19→20)
4. Who2Who_PositionCRF - Combination of Who2Who + Position
5. IntimeCRF - Initiative count-based (0, 1, 2+ prior initiatives)
6. DistanceCRF - Distance from last initiative (consecutive vs not)
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
    """
    Component 1: BERT Utterance Encoder
    
    Encodes each utterance independently using BERT [CLS] token.
    """
    
    def __init__(self, bert_model: str = 'bert-base-multilingual-cased'):
        super().__init__()
        print(f"[BERT] Loading {bert_model}...")
        self.bert = BertModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
        print(f"[BERT] Loaded, hidden_size={self.hidden_size}")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch * num_pairs, max_len]
        Returns:
            [CLS] representations: [batch * num_pairs, hidden_size]
        """
        outputs = self.bert(input_ids=input_ids)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token


class PriorInterUtteranceEncoder(nn.Module):
    """
    Component 2a: Prior Inter-Utterance Encoder
    
    CRITICAL: Used during INFERENCE when system utterance is UNKNOWN.
    Processes X1:T-1 (without system_T) to predict emission for system_T.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        print(f"[Prior BiLSTM] {hidden_size} → {hidden_size*2} (bidirectional)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [1, seq_len, hidden_size] - PARTIAL sequence (missing system_T)
        Returns:
            [1, seq_len, 2*hidden_size]
        """
        output, _ = self.lstm(x)
        return output


class PosteriorInterUtteranceEncoder(nn.Module):
    """
    Component 2b: Posterior Inter-Utterance Encoder
    
    CRITICAL: Used during TRAINING when we HAVE system utterance.
    Processes X1:T (with system_T) to generate "gold" emissions.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        print(f"[Posterior BiLSTM] {hidden_size} → {hidden_size*2} (bidirectional)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [1, seq_len, hidden_size] - FULL sequence (has system_T)
        Returns:
            [1, seq_len, 2*hidden_size]
        """
        output, _ = self.lstm(x)
        return output


class MultiTurnCRF(nn.Module):
    """
    Component 3: Multi-Turn Feature-Aware CRF
    
    Implements all 6 variants from SIP paper with proper feature extraction.
    
    Multi-Turn Features:
    - who2who: 0=user→sys, 1=sys→user
    - position: 0-19 for turn position
    - intime: 0/1/2+ prior system initiatives  
    - distance: 0=consecutive, 1=non-consecutive
    """
    
    def __init__(self, num_tags: int, crf_variant: str = 'VanillaCRF'):
        super().__init__()
        self.num_tags = num_tags
        self.variant = crf_variant
        
        print(f"[MultiTurnCRF] Variant: {crf_variant}, Tags: {num_tags}")
        
        # Variant 1: VanillaCRF - Single global transition
        self.trans_global = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        
        # Variant 2 & 4 & 5 & 6: Who2Who transitions
        if crf_variant in ['Who2WhoCRF', 'Who2Who_PositionCRF', 'IntimeCRF', 'DistanceCRF']:
            self.trans_u2s = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # User→System
            self.trans_s2u = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # System→User
            print(f"  Added Who2Who transitions")
        
        # Variant 3 & 4: Position-specific transitions
        if crf_variant in ['PositionCRF', 'Who2Who_PositionCRF']:
            self.trans_position = nn.ParameterList([
                nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
                for _ in range(20)  # First 20 transitions
            ])
            print(f"  Added 20 position-specific transitions")
        
        # Variant 5 & 6: Initiative count-based
        if crf_variant in ['IntimeCRF', 'DistanceCRF']:
            self.trans_I0 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)       # No prior
            self.trans_I1 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)       # 1 prior
            self.trans_I2_plus = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # 2+ prior
            print(f"  Added initiative count transitions")
        
        # Variant 6: Distance-based
        if crf_variant == 'DistanceCRF':
            self.trans_consecutive = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
            self.trans_nonconsecutive = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
            print(f"  Added distance-based transitions")
    
    def get_transition(self, turn_idx: int, features: Dict[str, int]) -> torch.Tensor:
        """
        Get transition matrix based on variant and multi-turn features.
        
        Args:
            turn_idx: Current turn index
            features: {'who2who', 'position', 'intime', 'distance'}
        
        Returns:
            Transition matrix [num_tags, num_tags]
        """
        who2who = features.get('who2who', -1)
        position = features.get('position', -1)
        intime = features.get('intime', -1)
        distance = features.get('distance', -1)
        
        if self.variant == 'VanillaCRF':
            return self.trans_global
        
        elif self.variant == 'Who2WhoCRF':
            if who2who == 0:  # User→System
                return self.trans_u2s
            elif who2who == 1:  # System→User
                return self.trans_s2u
            else:
                return self.trans_global
        
        elif self.variant == 'PositionCRF':
            if 0 <= position < 20:
                return self.trans_position[position]
            else:
                return self.trans_global
        
        elif self.variant == 'Who2Who_PositionCRF':
            # Priority: Position > Who2Who > Global
            if 0 <= position < 20:
                return self.trans_position[position]
            elif who2who == 0:
                return self.trans_u2s
            elif who2who == 1:
                return self.trans_s2u
            else:
                return self.trans_global
        
        elif self.variant == 'IntimeCRF':
            if who2who == 1:  # System→User
                return self.trans_s2u
            else:  # User→System
                if intime == 0:
                    return self.trans_I0
                elif intime == 1:
                    return self.trans_I1
                else:
                    return self.trans_I2_plus
        
        elif self.variant == 'DistanceCRF':
            # Most complex: Uses all features
            if who2who == 1:  # System→User
                return self.trans_s2u
            elif intime == 0:  # No prior initiatives
                return self.trans_I0
            else:  # Has prior initiatives
                if distance == 0:  # Consecutive
                    return self.trans_consecutive
                else:  # Non-consecutive
                    return self.trans_nonconsecutive
        
        return self.trans_global
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        features_list: List[Dict[str, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CRF forward (training).
        
        Args:
            emissions: [seq_len, num_tags]
            tags: [seq_len] - Ground truth
            features_list: List of feature dicts per turn
        
        Returns:
            gold_score: Score of gold path
            total_score: Log partition function
        """
        seq_len = emissions.size(0)
        device = emissions.device
        
        # Gold score
        gold_score = torch.sum(emissions[range(seq_len), tags])
        
        for idx in range(1, seq_len):
            trans = self.get_transition(idx, features_list[idx])
            gold_score += trans[tags[idx-1], tags[idx]]
        
        # Forward algorithm
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for idx in range(seq_len):
            trans = self.get_transition(idx, features_list[idx])
            
            if idx == 0:
                alpha = emissions[idx].unsqueeze(0)
            else:
                alpha = torch.logsumexp(
                    alpha.T + emissions[idx].unsqueeze(0) + trans,
                    dim=0,
                    keepdim=True
                )
        
        total_score = torch.logsumexp(alpha, dim=1).squeeze()
        
        return gold_score, total_score
    
    def decode(
        self,
        emissions: torch.Tensor,
        features_list: List[Dict[str, int]]
    ) -> Tuple[List[int], List[float]]:
        """
        Viterbi decoding (inference).
        
        Args:
            emissions: [seq_len, num_tags]
            features_list: Feature dicts per turn
        
        Returns:
            best_path: Predicted tags
            probabilities: Softmax probabilities per position
        """
        seq_len = emissions.size(0)
        device = emissions.device
        
        backpointers = []
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for idx in range(seq_len):
            trans = self.get_transition(idx, features_list[idx])
            
            if idx == 0:
                alpha = emissions[idx].unsqueeze(0)
                backpointers.append(torch.zeros(self.num_tags, dtype=torch.long, device=device))
            else:
                alpha_with_trans = alpha.T + emissions[idx].unsqueeze(0) + trans
                viterbivars, bptrs = torch.max(alpha_with_trans, dim=0)
                backpointers.append(bptrs)
                alpha = viterbivars.unsqueeze(0)
        
        # Backtrack
        best_tag_id = alpha.squeeze().argmax().item()
        best_path = [best_tag_id]
        
        for bptrs in reversed(backpointers[1:]):
            best_tag_id = bptrs[best_tag_id].item()
            best_path.append(best_tag_id)
        
        best_path.reverse()
        
        # Get probabilities
        probs = []
        for idx in range(seq_len):
            prob = F.softmax(emissions[idx], dim=0).cpu().numpy()
            probs.append(prob)
        
        return best_path, probs


def extract_multiturn_features(
    turn_idx: int,
    system_labels: List[int]
) -> Dict[str, int]:
    """
    Extract multi-turn features for transition at turn_idx-1 → turn_idx.
    
    Sequence structure: [user0, sys0, user1, sys1, user2, sys2, ...]
    - Even indices: User utterances
    - Odd indices: System utterances
    
    Args:
        turn_idx: Current turn index (0-based)
        system_labels: Labels for all system turns SO FAR (for intime/distance)
    
    Returns:
        Feature dict with who2who, position, intime, distance
    """
    if turn_idx == 0:
        return {'who2who': -1, 'position': -1, 'intime': -1, 'distance': -1}
    
    # Who2Who: Check speaker transition
    # Even → Odd: User → System (0)
    # Odd → Even: System → User (1)
    prev_is_user = (turn_idx - 1) % 2 == 0
    curr_is_user = turn_idx % 2 == 0
    
    if prev_is_user and not curr_is_user:
        who2who = 0  # User→System
    elif not prev_is_user and curr_is_user:
        who2who = 1  # System→User
    else:
        who2who = 2  # Same (shouldn't happen in normal flow)
    
    # Position (clip to 19)
    position = min(turn_idx - 1, 19)
    
    # Intime & Distance (only for System turns)
    if turn_idx % 2 == 1:  # System turn (odd index)
        system_turn_idx = turn_idx // 2
        
        # Count prior system initiatives
        if system_turn_idx > 0 and len(system_labels) >= system_turn_idx:
            prior_initiatives = sum(
                1 for i in range(system_turn_idx)
                if i < len(system_labels) and system_labels[i] > 0
            )
        else:
            prior_initiatives = 0
        
        # Intime
        if prior_initiatives == 0:
            intime = 0
        elif prior_initiatives == 1:
            intime = 1
        else:
            intime = 2  # 2+
        
        # Distance from last initiative
        if prior_initiatives > 0:
            last_init_turn_idx = -1
            for i in range(system_turn_idx - 1, -1, -1):
                if i < len(system_labels) and system_labels[i] > 0:
                    last_init_turn_idx = i
                    break
            
            if last_init_turn_idx >= 0:
                last_init_seq_idx = last_init_turn_idx * 2 + 1
                dist = turn_idx - last_init_seq_idx
                distance = 0 if dist == 2 else 1  # 0=consecutive, 1=non-consecutive
            else:
                distance = -1
        else:
            distance = -1
    else:
        intime = -1
        distance = -1
    
    return {
        'who2who': who2who,
        'position': position,
        'intime': intime,
        'distance': distance
    }


# ============================================================================
# BASE MODEL CLASS
# ============================================================================

class MuSIcBase(nn.Module):
    """
    Base class for all MuSIc variants.
    
    Implements core Prior-Posterior framework with CRF.
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2,
        lambda_mle: float = 0.1,
        crf_variant: str = 'VanillaCRF'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda_mle = lambda_mle
        self.crf_variant = crf_variant
        
        print(f"\n{'='*70}")
        print(f"Initializing MuSIc Model")
        print(f"{'='*70}")
        print(f"CRF Variant: {crf_variant}")
        print(f"Num classes: {num_classes}")
        print(f"Hidden size: {hidden_size}")
        print(f"Lambda MLE: {lambda_mle}\n")
        
        # Component 1: BERT
        self.utterance_encoder = BERTUtteranceEncoder(bert_model)
        bert_hidden = self.utterance_encoder.hidden_size
        
        # Component 2: Prior & Posterior
        self.prior_encoder = PriorInterUtteranceEncoder(bert_hidden, num_layers, dropout)
        self.posterior_encoder = PosteriorInterUtteranceEncoder(bert_hidden, num_layers, dropout)
        
        # Emission projections
        encoder_output_size = bert_hidden * 2  # BiLSTM bidirectional
        self.prior_emission_project = nn.Linear(encoder_output_size, num_classes)
        self.posterior_emission_project = nn.Linear(encoder_output_size, num_classes)
        
        # Component 3: CRF
        self.crf = MultiTurnCRF(num_classes, crf_variant)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"{'='*70}")
        print(f"Model initialized successfully")
        print(f"{'='*70}\n")
    
    def encode_utterances(
        self,
        user_utterance: torch.Tensor,
        system_utterance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all user and system utterances with BERT.
        
        Args:
            user_utterance: [1, num_pairs, max_len]
            system_utterance: [1, num_pairs, max_len]
        
        Returns:
            user_repr: [num_pairs, bert_hidden]
            system_repr: [num_pairs, bert_hidden]
        """
        batch_size, num_pairs, max_len = user_utterance.shape
        
        # Flatten
        user_flat = user_utterance.view(num_pairs, max_len)
        system_flat = system_utterance.view(num_pairs, max_len)
        
        # BERT encode
        user_repr = self.utterance_encoder(user_flat)
        system_repr = self.utterance_encoder(system_flat)
        
        return user_repr, system_repr
    
    def forward(
        self,
        user_utterance: torch.Tensor,
        system_utterance: torch.Tensor,
        user_I_label: Optional[torch.Tensor] = None,
        system_I_label: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> Dict:
        """Forward pass - delegates to train or inference mode"""
        user_repr, system_repr = self.encode_utterances(user_utterance, system_utterance)
        num_pairs = user_repr.size(0)
        
        if mode == 'train':
            return self._train_forward(user_repr, system_repr, user_I_label, system_I_label, num_pairs)
        else:
            return self._inference_forward(user_repr, system_repr, num_pairs)
    
    def _train_forward(
        self,
        user_repr: torch.Tensor,
        system_repr: torch.Tensor,
        user_I_label: torch.Tensor,
        system_I_label: torch.Tensor,
        num_pairs: int
    ) -> Dict:
        """Training forward - IMPLEMENTATION IN SUBCLASSES"""
        raise NotImplementedError
    
    def _inference_forward(
        self,
        user_repr: torch.Tensor,
        system_repr: torch.Tensor,
        num_pairs: int
    ) -> Dict:
        """Inference forward - IMPLEMENTATION IN SUBCLASSES"""
        raise NotImplementedError


# ============================================================================
# CONCRETE BASELINE IMPLEMENTATIONS
# ============================================================================

class VanillaCRF(MuSIcBase):
    """Baseline 1: VanillaCRF - Single global transition matrix"""
    
    def __init__(self, **kwargs):
        kwargs['crf_variant'] = 'VanillaCRF'
        super().__init__(**kwargs)
    
    def _train_forward(self, user_repr, system_repr, user_I_label, system_I_label, num_pairs):
        """Training: Process incrementally, predict each turn"""
        user_I_label = user_I_label.squeeze(0)
        system_I_label = system_I_label.squeeze(0)
        
        gold_scores, total_scores = [], []
        prior_emis_list, post_emis_list = [], []
        
        for pair_idx in range(num_pairs):
            # Build sequence: [u0, s0, u1, s1, ..., u_pair_idx, s_pair_idx]
            seq, labs = [], []
            for i in range(pair_idx + 1):
                seq.extend([user_repr[i], system_repr[i]])
                labs.extend([user_I_label[i].item(), system_I_label[i].item()])
            
            seq_tensor = torch.stack(seq).unsqueeze(0)
            labs_tensor = torch.tensor(labs, device=user_repr.device)
            
            # Extract features
            features = [extract_multiturn_features(idx, system_I_label[:pair_idx+1].tolist())
                       for idx in range(len(seq))]
            
            # Posterior (HAS system at pair_idx)
            post_hidden = self.posterior_encoder(seq_tensor)
            post_emissions = self.posterior_emission_project(post_hidden).squeeze(0)
            
            # Prior (MISSING system at pair_idx)
            prior_seq = seq_tensor[:, :-1, :]
            prior_hidden = self.prior_encoder(prior_seq)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # CRF
            gold, total = self.crf(post_emissions, labs_tensor, features)
            gold_scores.append(gold)
            total_scores.append(total)
            
            # MLE
            prior_emis_list.append(prior_emission.squeeze(0))
            post_emis_list.append(post_emissions[-1])
        
        loss_crf = torch.mean(torch.stack(total_scores) - torch.stack(gold_scores))
        loss_mle = F.mse_loss(torch.stack(prior_emis_list), torch.stack(post_emis_list).detach())
        
        return {
            'loss_crf': loss_crf,
            'loss_mle': loss_mle,
            'total_loss': loss_crf + self.lambda_mle * loss_mle
        }
    
    def _inference_forward(self, user_repr, system_repr, num_pairs):
        """Inference: System utterance HIDDEN at each turn"""
        predictions, all_probs = [], []
        
        for pair_idx in range(num_pairs):
            # Build partial: [u0, s0, ..., s_{pair_idx-1}, u_pair_idx]
            # MISSING: s_pair_idx
            seq = []
            for i in range(pair_idx):
                seq.extend([user_repr[i], system_repr[i]])
            seq.append(user_repr[pair_idx])
            
            seq_tensor = torch.stack(seq).unsqueeze(0)
            
            # Prior for missing system
            prior_hidden = self.prior_encoder(seq_tensor)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # Posterior for known
            if pair_idx > 0:
                prev_seq = seq_tensor[:, :-1, :]
                prev_post = self.posterior_encoder(prev_seq)
                prev_emis = self.posterior_emission_project(prev_post).squeeze(0)
            else:
                prev_emis = torch.empty(0, self.num_classes, device=user_repr.device)
            
            # Current user
            curr_post = self.posterior_encoder(seq_tensor)
            curr_user_emis = self.posterior_emission_project(curr_post[:, -1, :])
            
            # Combine
            combined = torch.cat([
                prev_emis,
                curr_user_emis.squeeze(0).unsqueeze(0),
                prior_emission.squeeze(0).unsqueeze(0)
            ], dim=0)
            
            # Features
            features = [extract_multiturn_features(idx, []) for idx in range(len(combined))]
            
            # Decode
            path, probs = self.crf.decode(combined, features)
            predictions.append(path[-1])
            all_probs.append(probs[-1])
        
        return {
            'predictions': predictions,
            'probabilities': np.array(all_probs),
            'confidences': [probs[pred] for pred, probs in zip(predictions, all_probs)]
        }


class Who2WhoCRF(MuSIcBase):
    """Baseline 2: Who2WhoCRF - User↔System specific transitions"""
    def __init__(self, **kwargs):
        kwargs['crf_variant'] = 'Who2WhoCRF'
        super().__init__(**kwargs)
    _train_forward = VanillaCRF._train_forward
    _inference_forward = VanillaCRF._inference_forward


class PositionCRF(MuSIcBase):
    """Baseline 3: PositionCRF - Position-specific transitions"""
    def __init__(self, **kwargs):
        kwargs['crf_variant'] = 'PositionCRF'
        super().__init__(**kwargs)
    _train_forward = VanillaCRF._train_forward
    _inference_forward = VanillaCRF._inference_forward


class Who2Who_PositionCRF(MuSIcBase):
    """Baseline 4: Combined Who2Who + Position"""
    def __init__(self, **kwargs):
        kwargs['crf_variant'] = 'Who2Who_PositionCRF'
        super().__init__(**kwargs)
    _train_forward = VanillaCRF._train_forward
    _inference_forward = VanillaCRF._inference_forward


class IntimeCRF(MuSIcBase):
    """Baseline 5: IntimeCRF - Initiative count-based"""
    def __init__(self, **kwargs):
        kwargs['crf_variant'] = 'IntimeCRF'
        super().__init__(**kwargs)
    _train_forward = VanillaCRF._train_forward
    _inference_forward = VanillaCRF._inference_forward


class DistanceCRF(MuSIcBase):
    """Baseline 6: DistanceCRF - Complete with all features (MuSIc Full)"""
    def __init__(self, **kwargs):
        kwargs['crf_variant'] = 'DistanceCRF'
        super().__init__(**kwargs)
    _train_forward = VanillaCRF._train_forward
    _inference_forward = VanillaCRF._inference_forward


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_model(
    baseline: str = 'vanillacrf',
    num_classes: int = 2,
    **kwargs
) -> MuSIcBase:
    """
    Factory function to create baseline models.
    
    Args:
        baseline: One of: vanillacrf, who2who, position, who2who_position, intime, distance
        num_classes: 2, 3, or 4
        **kwargs: Model parameters
    
    Returns:
        Model instance
    """
    models = {
        'vanillacrf': VanillaCRF,
        'who2who': Who2WhoCRF,
        'position': PositionCRF,
        'who2who_position': Who2Who_PositionCRF,
        'intime': IntimeCRF,
        'distance': DistanceCRF,  # MuSIc Full
    }
    
    baseline = baseline.lower()
    if baseline not in models:
        raise ValueError(f"Unknown baseline: {baseline}. Choose from: {list(models.keys())}")
    
    return models[baseline](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    print("Testing MuSIc Baselines\n")
    
    # Test each baseline
    for baseline in ['vanillacrf', 'who2who', 'position', 'who2who_position', 'intime', 'distance']:
        print(f"\n{'='*70}")
        print(f"Testing: {baseline}")
        print(f"{'='*70}")
        
        model = create_model(baseline, num_classes=2, hidden_size=128)
        
        # Dummy data
        user = torch.randint(0, 1000, (1, 2, 64))
        system = torch.randint(0, 1000, (1, 2, 64))
        u_labels = torch.zeros(1, 2, dtype=torch.long)
        s_labels = torch.tensor([[0, 1]], dtype=torch.long)
        
        # Train
        model.train()
        out = model(user, system, u_labels, s_labels, mode='train')
        print(f"✓ Training: Loss={out['total_loss'].item():.4f}")
        
        # Inference
        model.eval()
        with torch.no_grad():
            out = model(user, system, mode='inference')
        print(f"✓ Inference: Predictions={out['predictions']}")
    
    print(f"\n{'='*70}")
    print("All baselines tested successfully!")
    print(f"{'='*70}\n")
