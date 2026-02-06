"""
Baseline 2: VanillaCRF+features
Feeds multi-turn features into prior-posterior encoders.
"""

import torch
import torch.nn as nn
from baseline1_vanillacrf import VanillaCRF


class VanillaCRFFeatures(VanillaCRF):
    """
    Baseline 2: VanillaCRF+features
    
    From paper:
    "VanillaCRF+features feeding the three multi-turn features into the
    prior-posterior inter-utterance encoders by encoding the multi-turn
    features as one-hot vectors at each turn and concatenating the vectors
    with the BERT utterance representation."
    
    Features:
    - Who2Who (3 values: odd→even, even→odd, padding)
    - Position (20 values: turn 0-19 + padding)
    - Intime (4 values: I0, I1, I2+, padding)
    
    Use case: Leverage conversation structure without complex transitions
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-multilingual-cased',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_tags: int = 2,
        lambda_mle: float = 0.1
    ):
        super().__init__(
            bert_model_name, hidden_size, num_layers, dropout, num_tags, lambda_mle
        )
        
        # Feature embedding dimensions
        self.who2who_embed = nn.Embedding(3, 16)  # 3 values
        self.position_embed = nn.Embedding(21, 32)  # 20 positions + padding
        self.intime_embed = nn.Embedding(4, 16)  # 4 values
        
        # Total feature dim: 16 + 32 + 16 = 64
        feature_dim = 64
        bert_hidden = self.utterance_encoder.hidden_size
        
        # Update encoder input size
        self.prior_encoder = nn.LSTM(
            bert_hidden + feature_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.posterior_encoder = nn.LSTM(
            bert_hidden + feature_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        print(f"[VanillaCRF+features] Initialized")
        print(f"  Feature encoding: One-hot → embedding (dim={feature_dim})")
        print(f"  Best for: Leveraging conversation structure\n")
    
    def _encode_features(self, multiturn_features: list, device) -> torch.Tensor:
        """
        Encode multi-turn features as embeddings.
        
        Args:
            multiturn_features: List of feature dicts
            device: Torch device
        
        Returns:
            feature_vectors: [seq_len, feature_dim]
        """
        seq_len = len(multiturn_features)
        
        who2who_ids = []
        position_ids = []
        intime_ids = []
        
        for feat in multiturn_features:
            # Map -1 to padding index (last index)
            who2who_ids.append(feat['who2who'] if feat['who2who'] >= 0 else 2)
            position_ids.append(feat['position'] if feat['position'] >= 0 else 20)
            intime_ids.append(feat['intime'] if feat['intime'] >= 0 else 3)
        
        who2who_tensor = torch.tensor(who2who_ids, device=device)
        position_tensor = torch.tensor(position_ids, device=device)
        intime_tensor = torch.tensor(intime_ids, device=device)
        
        # Embed
        who2who_emb = self.who2who_embed(who2who_tensor)  # [seq_len, 16]
        position_emb = self.position_embed(position_tensor)  # [seq_len, 32]
        intime_emb = self.intime_embed(intime_tensor)  # [seq_len, 16]
        
        # Concatenate
        feature_vectors = torch.cat([who2who_emb, position_emb, intime_emb], dim=-1)
        
        return feature_vectors  # [seq_len, 64]
    
    def _forward_train(self, odd_repr, even_repr, odd_I_labels, system_I_labels, multiturn_features):
        """Training with feature encoding."""
        odd_I_labels = odd_I_labels.squeeze(0)
        system_I_labels = system_I_labels.squeeze(0)
        num_pairs = odd_repr.size(0)
        
        print(f"\n[VanillaCRF+features Train] Processing {num_pairs} pairs...")
        
        gold_scores = []
        total_scores = []
        prior_emissions_list = []
        posterior_emissions_list = []
        
        for pair_idx in range(num_pairs):
            # Build sequence
            utterance_sequence = []
            I_label_sequence = []
            
            for i in range(pair_idx + 1):
                utterance_sequence.append(odd_repr[i])
                utterance_sequence.append(even_repr[i])
                I_label_sequence.append(odd_I_labels[i].item())
                I_label_sequence.append(system_I_labels[i].item())
            
            utterance_sequence = torch.stack(utterance_sequence)
            I_label_tensor = torch.tensor(I_label_sequence, device=odd_repr.device)
            turn_features = multiturn_features[:(pair_idx + 1) * 2]
            
            # Encode features
            feature_vectors = self._encode_features(turn_features, odd_repr.device)
            
            # Concatenate with utterances
            utterance_with_features = torch.cat([
                utterance_sequence, feature_vectors
            ], dim=-1).unsqueeze(0)
            
            # Prior
            prior_sequence = utterance_with_features[:, :-1, :]
            prior_hidden = self.prior_encoder(prior_sequence)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # Posterior
            posterior_hidden = self.posterior_encoder(utterance_with_features)
            posterior_emissions = self.posterior_emission_project(posterior_hidden)
            posterior_emissions = posterior_emissions.squeeze(0)
            
            # CRF (still uses VanillaCRF - single matrix)
            gold_score, total_score = self.crf(
                posterior_emissions, I_label_tensor, turn_features, 'VanillaCRF'
            )
            
            gold_scores.append(gold_score)
            total_scores.append(total_score)
            prior_emissions_list.append(prior_emission.squeeze(0))
            posterior_emissions_list.append(posterior_emissions[-1])
        
        # Losses
        gold_scores = torch.stack(gold_scores)
        total_scores = torch.stack(total_scores)
        prior_emissions = torch.stack(prior_emissions_list)
        posterior_emissions = torch.stack(posterior_emissions_list)
        
        loss_crf = torch.mean(total_scores - gold_scores)
        loss_mle = nn.functional.mse_loss(prior_emissions, posterior_emissions.detach())
        
        return {
            'loss_crf': loss_crf,
            'loss_mle_e': loss_mle,
            'total_loss': loss_crf + self.lambda_mle * loss_mle
        }
