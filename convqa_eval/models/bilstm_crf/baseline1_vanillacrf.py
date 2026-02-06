"""
Baseline 1: VanillaCRF
Uses MuSIc architecture with single global transition matrix.
"""

import torch
import torch.nn.functional as F
from typing import Dict
from music_base import MuSIcBase


class VanillaCRF(MuSIcBase):
    """
    Baseline 1: VanillaCRF
    
    MuSIc components:
    ✓ BERT Utterance Encoder
    ✓ Prior-Posterior Inter-Utterance Encoders
    ✓ CRF Layer with SINGLE global transition matrix
    
    From paper:
    "VanillaCRF only uses a unique transition matrix"
    
    Use case: Simple baseline, fastest training
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
            bert_model_name=bert_model_name,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_tags=num_tags,
            lambda_mle=lambda_mle,
            crf_type='VanillaCRF'
        )
        
        print(f"[VanillaCRF] Initialized")
        print("  Features: Single global transition matrix")
        print("  Best for: Fast baseline, simple structure\n")
    
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
    ) -> Dict:
        """
        Forward pass with incremental processing.
        
        Args:
            odd_input_ids: [batch=1, num_pairs, max_len] - human/observation
            even_input_ids: [batch=1, num_pairs, max_len] - gpt/function
            odd_masks: [batch=1, num_pairs, max_len]
            even_masks: [batch=1, num_pairs, max_len]
            odd_I_labels: [batch=1, num_pairs] - Always 0 for user
            system_I_labels: [batch=1, num_pairs] - System initiative labels
            multiturn_features: List of feature dicts for each turn
            mode: 'train' or 'inference'
        """
        # Encode utterances
        odd_repr, even_repr = self.encode_utterances(
            odd_input_ids, even_input_ids, odd_masks, even_masks
        )
        
        batch_size, num_pairs, _ = odd_repr.shape
        assert batch_size == 1, "Batch size must be 1"
        
        # Squeeze batch dimension
        odd_repr = odd_repr.squeeze(0)
        even_repr = even_repr.squeeze(0)
        
        if mode == 'train':
            return self._forward_train(
                odd_repr, even_repr, odd_I_labels, system_I_labels, multiturn_features
            )
        else:
            return self._forward_inference(odd_repr, even_repr, multiturn_features)
    
    def _forward_train(
        self,
        odd_repr: torch.Tensor,
        even_repr: torch.Tensor,
        odd_I_labels: torch.Tensor,
        system_I_labels: torch.Tensor,
        multiturn_features: list
    ) -> Dict:
        """Training mode forward pass."""
        odd_I_labels = odd_I_labels.squeeze(0)
        system_I_labels = system_I_labels.squeeze(0)
        num_pairs = odd_repr.size(0)
        
        print(f"\n[VanillaCRF Train] Processing {num_pairs} pairs...")
        
        gold_scores = []
        total_scores = []
        prior_emissions_list = []
        posterior_emissions_list = []
        
        # Process each turn incrementally
        for pair_idx in range(num_pairs):
            print(f"\n  Pair {pair_idx+1}/{num_pairs}")
            
            # Build conversation sequence: [odd1, even1, odd2, even2, ...]
            utterance_sequence = []
            I_label_sequence = []
            
            for i in range(pair_idx + 1):
                utterance_sequence.append(odd_repr[i])
                utterance_sequence.append(even_repr[i])
                I_label_sequence.append(odd_I_labels[i].item())
                I_label_sequence.append(system_I_labels[i].item())
            
            utterance_sequence = torch.stack(utterance_sequence).unsqueeze(0)
            I_label_tensor = torch.tensor(I_label_sequence, device=odd_repr.device)
            
            # Get features for this subsequence
            turn_features = multiturn_features[:(pair_idx + 1) * 2]
            
            print(f"    Sequence: {utterance_sequence.shape}, Labels: {I_label_tensor}")
            
            # Prior: partial sequence (without current system)
            prior_sequence = utterance_sequence[:, :-1, :]
            prior_hidden = self.prior_encoder(prior_sequence)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # Posterior: full sequence
            posterior_hidden = self.posterior_encoder(utterance_sequence)
            posterior_emissions = self.posterior_emission_project(posterior_hidden)
            posterior_emissions = posterior_emissions.squeeze(0)
            
            print(f"    Prior: {prior_emission.shape}, Posterior: {posterior_emissions.shape}")
            
            # CRF forward
            gold_score, total_score = self.crf(
                posterior_emissions,
                I_label_tensor,
                turn_features,
                crf_type=self.crf_type
            )
            
            print(f"    Gold: {gold_score:.4f}, Total: {total_score:.4f}")
            
            gold_scores.append(gold_score)
            total_scores.append(total_score)
            prior_emissions_list.append(prior_emission.squeeze(0))
            posterior_emissions_list.append(posterior_emissions[-1])
        
        # Compute losses
        gold_scores = torch.stack(gold_scores)
        total_scores = torch.stack(total_scores)
        prior_emissions = torch.stack(prior_emissions_list)
        posterior_emissions = torch.stack(posterior_emissions_list)
        
        loss_crf = torch.mean(total_scores - gold_scores)
        loss_mle = F.mse_loss(prior_emissions, posterior_emissions.detach())
        
        print(f"\n[VanillaCRF Train] CRF: {loss_crf:.4f}, MLE: {loss_mle:.4f}")
        
        return {
            'loss_crf': loss_crf,
            'loss_mle_e': loss_mle,
            'total_loss': loss_crf + self.lambda_mle * loss_mle
        }
    
    def _forward_inference(
        self,
        odd_repr: torch.Tensor,
        even_repr: torch.Tensor,
        multiturn_features: list
    ) -> Dict:
        """Inference mode forward pass."""
        num_pairs = odd_repr.size(0)
        
        print(f"\n[VanillaCRF Inference] Processing {num_pairs} pairs...")
        
        predictions = []
        initiative_scores = []
        
        for pair_idx in range(num_pairs):
            print(f"\n  Pair {pair_idx+1}/{num_pairs}")
            
            # Build partial sequence
            utterance_sequence = []
            for i in range(pair_idx):
                utterance_sequence.append(odd_repr[i])
                utterance_sequence.append(even_repr[i])
            utterance_sequence.append(odd_repr[pair_idx])
            
            utterance_sequence = torch.stack(utterance_sequence).unsqueeze(0)
            
            # Posterior encoding of partial
            partial_posterior_hidden = self.posterior_encoder(utterance_sequence)
            partial_posterior_emissions = self.posterior_emission_project(
                partial_posterior_hidden.squeeze(0)
            )
            
            # Prior for next system
            prior_hidden = self.prior_encoder(utterance_sequence)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # Combine
            combined_emissions = torch.cat([
                partial_posterior_emissions,
                prior_emission
            ], dim=0)
            
            # Features for this subsequence
            turn_features = multiturn_features[:len(combined_emissions)]
            
            print(f"    Emissions: {combined_emissions.shape}, Features: {len(turn_features)}")
            
            # Decode
            predicted_path = self.crf.decode(
                combined_emissions,
                turn_features,
                crf_type=self.crf_type
            )
            predictions.append(predicted_path)
            
            # System initiative is at last even position
            system_initiative = predicted_path[-1]
            initiative_scores.append(system_initiative)
            
            print(f"    Path: {predicted_path}, System initiative: {system_initiative}")
        
        return {
            'predictions': predictions,
            'initiative_scores': initiative_scores,
            'has_initiative': [score > 0 for score in initiative_scores]
        }


if __name__ == "__main__":
    print("=== Testing VanillaCRF ===\n")
    
    model = VanillaCRF(hidden_size=256, num_tags=2)
    
    # Dummy data
    odd_ids = torch.randint(0, 1000, (1, 3, 128))
    even_ids = torch.randint(0, 1000, (1, 3, 128))
    odd_labels = torch.zeros(1, 3, dtype=torch.long)
    system_labels = torch.tensor([[0, 1, 0]], dtype=torch.long)
    
    # Dummy features
    features = [
        {'who2who': -1, 'position': -1, 'intime': -1, 'distance': -1},
        {'who2who': 0, 'position': 0, 'intime': 0, 'distance': -1},
        {'who2who': 1, 'position': 1, 'intime': -1, 'distance': -1},
        {'who2who': 0, 'position': 2, 'intime': 0, 'distance': -1},
        {'who2who': 1, 'position': 3, 'intime': -1, 'distance': -1},
        {'who2who': 0, 'position': 4, 'intime': 1, 'distance': 0},
    ]
    
    # Training
    model.train()
    output = model(
        odd_ids, even_ids,
        odd_I_labels=odd_labels,
        system_I_labels=system_labels,
        multiturn_features=features,
        mode='train'
    )
    print(f"\nTraining output: {output['total_loss']:.4f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(
            odd_ids, even_ids,
            multiturn_features=features,
            mode='inference'
        )
        print(f"\nInference: {output['has_initiative']}")
