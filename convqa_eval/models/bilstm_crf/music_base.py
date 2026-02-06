"""
MuSIc Architecture for System Initiative Prediction (SIP)

Three Components:
1. BERT Utterance Encoder
2. Prior-Posterior Inter-Utterance Encoders
3. Multi-Turn Feature-Aware CRF Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import BertModel


class BERTUtteranceEncoder(nn.Module):
    """
    Component 1: BERT Utterance Encoder
    Encodes individual utterances using BERT [CLS] token.
    """
    
    def __init__(self, bert_model_name: str = 'bert-base-multilingual-cased'):
        super().__init__()
        print(f"[BERTUtteranceEncoder] Loading {bert_model_name}...")
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        print(f"[BERTUtteranceEncoder] Loaded, hidden_size={self.hidden_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch * num_pairs, max_len]
            attention_mask: [batch * num_pairs, max_len]
        
        Returns:
            pooled_output: [batch * num_pairs, hidden_size] - [CLS] representations
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return pooled_output


class PriorInterUtteranceEncoder(nn.Module):
    """
    Component 2a: Prior Inter-Utterance Encoder
    Encodes conversation BEFORE current system turn (incomplete context).
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
        print(f"[PriorEncoder] BiLSTM: {hidden_size} → {hidden_size*2}")
    
    def forward(self, utterance_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            utterance_sequence: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, 2*hidden_size]
        """
        output, _ = self.lstm(utterance_sequence)
        return output


class PosteriorInterUtteranceEncoder(nn.Module):
    """
    Component 2b: Posterior Inter-Utterance Encoder
    Encodes conversation INCLUDING current system turn (complete context).
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
        print(f"[PosteriorEncoder] BiLSTM: {hidden_size} → {hidden_size*2}")
    
    def forward(self, utterance_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            utterance_sequence: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, 2*hidden_size]
        """
        output, _ = self.lstm(utterance_sequence)
        return output


class MultiTurnFeatureAwareCRF(nn.Module):
    """
    Component 3: Multi-Turn Feature-Aware CRF Layer
    
    Uses conversation-specific transition matrices based on:
    - Who2Who: odd→even vs even→odd transitions
    - Position: Turn position (1→2, 2→3, ...)
    - Initiative Count (Intime): Number of prior system initiatives
    - Initiative Distance: Distance from last system initiative
    """
    
    def __init__(self, num_tags: int = 2):
        super().__init__()
        self.num_tags = num_tags
        
        # Base transition matrix (VanillaCRF uses this alone)
        self.matrice_all = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        
        # Who2Who transitions
        self.matrice_odd2even = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # user→system
        self.matrice_even2odd = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # system→user
        
        # Initiative count-based (for odd→even transitions)
        self.matrice_I0 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # No prior initiative
        self.matrice_I1 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # 1 prior initiative
        self.matrice_I2_more = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # 2+ initiatives
        
        # Distance-based (for odd→even transitions)
        self.matrice_D_consecutive = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # distance=2
        self.matrice_D_nonconsecutive = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)  # distance>=4
        
        # Position-specific (up to 20 transitions)
        self.position_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(num_tags, num_tags) * 0.1) 
            for _ in range(20)
        ])
        
        print(f"[MultiTurnCRF] Initialized with {num_tags} tags")
    
    def _get_transition_matrix(
        self,
        features: Dict[str, int],
        crf_type: str = 'VanillaCRF'
    ) -> torch.Tensor:
        """
        Get transition matrix based on conversation features and CRF type.
        
        Args:
            features: Multi-turn features (who2who, position, intime, distance)
            crf_type: Type of CRF variant
        
        Returns:
            Transition matrix [num_tags, num_tags]
        """
        device = self.matrice_all.device
        padding = torch.zeros(self.num_tags, self.num_tags, device=device)
        
        # First turn - no transitions
        if features['who2who'] == -1:
            return padding
        
        if crf_type == 'VanillaCRF':
            # Single global transition matrix
            return self.matrice_all
        
        elif crf_type == 'VanillaCRF+features':
            # Features are encoded in utterance representation, use global matrix
            return self.matrice_all
        
        elif crf_type == 'Who2WhoCRF':
            # Role-based transitions
            who2who = features['who2who']
            if who2who == 0:
                return self.matrice_odd2even
            elif who2who == 1:
                return self.matrice_even2odd
            else:
                return padding
        
        elif crf_type == 'PositionCRF':
            # Position-specific transitions
            position = features['position']
            if 0 <= position < 20:
                return self.position_matrices[position]
            else:
                return self.matrice_all
        
        elif crf_type == 'IntimeCRF':
            # Initiative count-based
            who2who = features['who2who']
            intime = features['intime']
            
            if who2who == 1:  # even→odd (system→user)
                return self.matrice_even2odd
            else:  # odd→even (user→system)
                if intime == 0:
                    return self.matrice_I0
                elif intime == 1:
                    return self.matrice_I1
                else:
                    return self.matrice_I2_more
        
        elif crf_type == 'DistanceCRF':
            # Distance from last initiative
            who2who = features['who2who']
            intime = features['intime']
            distance = features['distance']
            
            if who2who == 1:  # system→user
                return self.matrice_even2odd
            elif intime == 0:  # No prior initiatives
                return self.matrice_I0
            else:  # Has prior initiatives
                if distance == 0:  # Consecutive
                    return self.matrice_D_consecutive
                else:  # Non-consecutive
                    return self.matrice_D_nonconsecutive
        
        return self.matrice_all
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        multiturn_features: List[Dict[str, int]],
        crf_type: str = 'VanillaCRF',
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CRF loss.
        
        Args:
            emissions: [seq_len, num_tags]
            tags: [seq_len]
            multiturn_features: List of feature dicts for each turn
            crf_type: CRF variant type
            mask: [seq_len] (optional)
        
        Returns:
            gold_score: Score of gold path
            total_score: Log partition function
        """
        seq_len = emissions.size(0)
        device = emissions.device
        
        # Gold score
        gold_score = torch.sum(emissions[range(seq_len), tags])
        
        for idx in range(1, seq_len):
            trans_matrix = self._get_transition_matrix(
                multiturn_features[idx],
                crf_type
            )
            gold_score += trans_matrix[tags[idx-1], tags[idx]]
        
        # Forward algorithm
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for idx in range(seq_len):
            trans_matrix = self._get_transition_matrix(
                multiturn_features[idx],
                crf_type
            )
            
            alpha = torch.logsumexp(
                alpha.T + emissions[idx].unsqueeze(0) + trans_matrix,
                dim=0,
                keepdim=True
            )
        
        total_score = torch.logsumexp(alpha, dim=1).squeeze()
        
        return gold_score, total_score
    
    def decode(
        self,
        emissions: torch.Tensor,
        multiturn_features: List[Dict[str, int]],
        crf_type: str = 'VanillaCRF'
    ) -> List[int]:
        """
        Viterbi decoding.
        
        Args:
            emissions: [seq_len, num_tags]
            multiturn_features: Feature dicts for each turn
            crf_type: CRF variant type
        
        Returns:
            best_path: List of tag indices
        """
        seq_len = emissions.size(0)
        device = emissions.device
        
        backpointers = []
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for idx in range(seq_len):
            trans_matrix = self._get_transition_matrix(
                multiturn_features[idx],
                crf_type
            )
            
            alpha_with_trans = alpha.T + emissions[idx].unsqueeze(0) + trans_matrix
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
        return best_path


class MuSIcBase(nn.Module):
    """
    MuSIc: Multi-Turn System Initiative Prediction with CRF
    
    Architecture:
        1. BERT Utterance Encoder
        2. Prior-Posterior Inter-Utterance Encoders (BiLSTM)
        3. Multi-Turn Feature-Aware CRF Layer
    
    Training:
        - CRF loss on full sequence (posterior)
        - MLE loss aligning prior and posterior emissions
    
    Inference:
        - Incremental turn-by-turn prediction (prior)
        - Viterbi decoding with multi-turn features
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-multilingual-cased',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_tags: int = 2,
        lambda_mle: float = 0.1,
        crf_type: str = 'VanillaCRF'
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.lambda_mle = lambda_mle
        self.crf_type = crf_type
        
        print(f"\n[MuSIc] Initializing...")
        print(f"  CRF type: {crf_type}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num tags: {num_tags}")
        print(f"  Lambda MLE: {lambda_mle}")
        
        # Component 1: BERT Utterance Encoder
        self.utterance_encoder = BERTUtteranceEncoder(bert_model_name)
        bert_hidden = self.utterance_encoder.hidden_size
        
        # Component 2: Prior-Posterior Inter-Utterance Encoders
        self.prior_encoder = PriorInterUtteranceEncoder(bert_hidden, num_layers, dropout)
        self.posterior_encoder = PosteriorInterUtteranceEncoder(bert_hidden, num_layers, dropout)
        
        # Emission projections
        encoder_output_size = bert_hidden * 2  # BiLSTM output
        self.prior_emission_project = nn.Linear(encoder_output_size, num_tags)
        self.posterior_emission_project = nn.Linear(encoder_output_size, num_tags)
        
        # Component 3: Multi-Turn Feature-Aware CRF
        self.crf = MultiTurnFeatureAwareCRF(num_tags)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"[MuSIc] Initialized successfully\n")
    
    def encode_utterances(
        self,
        odd_input_ids: torch.Tensor,
        even_input_ids: torch.Tensor,
        odd_masks: Optional[torch.Tensor] = None,
        even_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode odd (human/observation) and even (gpt/function) utterances.
        
        Args:
            odd_input_ids: [batch, num_pairs, max_len]
            even_input_ids: [batch, num_pairs, max_len]
            odd_masks: [batch, num_pairs, max_len]
            even_masks: [batch, num_pairs, max_len]
        
        Returns:
            odd_repr: [batch, num_pairs, bert_hidden]
            even_repr: [batch, num_pairs, bert_hidden]
        """
        batch_size, num_pairs, max_len = odd_input_ids.shape
        
        print(f"\n[Encoding] Batch={batch_size}, Pairs={num_pairs}, MaxLen={max_len}")
        
        # Flatten
        odd_flat = odd_input_ids.view(batch_size * num_pairs, max_len)
        even_flat = even_input_ids.view(batch_size * num_pairs, max_len)
        
        if odd_masks is not None:
            odd_mask_flat = odd_masks.view(batch_size * num_pairs, max_len)
            even_mask_flat = even_masks.view(batch_size * num_pairs, max_len)
        else:
            odd_mask_flat = None
            even_mask_flat = None
        
        # BERT encoding
        odd_repr = self.utterance_encoder(odd_flat, odd_mask_flat)
        even_repr = self.utterance_encoder(even_flat, even_mask_flat)
        
        # Reshape
        odd_repr = odd_repr.view(batch_size, num_pairs, -1)
        even_repr = even_repr.view(batch_size, num_pairs, -1)
        
        print(f"[Encoding] Odd repr: {odd_repr.shape}, Even repr: {even_repr.shape}")
        
        return odd_repr, even_repr
    
    def forward(
        self,
        odd_input_ids: torch.Tensor,
        even_input_ids: torch.Tensor,
        odd_masks: Optional[torch.Tensor] = None,
        even_masks: Optional[torch.Tensor] = None,
        odd_I_labels: Optional[torch.Tensor] = None,
        system_I_labels: Optional[torch.Tensor] = None,
        multiturn_features: Optional[List[Dict]] = None,
        mode: str = 'train'
    ) -> Dict:
        """Forward pass - see implementation in specific baseline classes."""
        raise NotImplementedError("Use specific baseline classes (VanillaCRF, etc.)")
