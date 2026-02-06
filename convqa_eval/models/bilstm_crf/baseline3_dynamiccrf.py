"""
Baseline 3: DynamicCRF
Uses adjacent observations to generate dynamic transition matrices.
"""

import torch
import torch.nn as nn
from baseline1_vanillacrf import VanillaCRF


class DynamicCRF(VanillaCRF):
    """
    Baseline 3: DynamicCRF
    
    From paper:
    "DynamicCRF uses adjacent input observations x_t, x_{t+1} to generate
    a dynamic transition matrix G_{x_t, x_{t+1}} to model the dependence
    between the corresponding output decisions y_t, y_{t+1}."
    
    Instead of fixed transition matrices, computes dynamic matrices
    from adjacent utterance representations.
    
    Use case: Capture context-dependent transition patterns
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
        
        # Dynamic transition generator
        # Input: concatenate two adjacent utterances
        bert_hidden = self.utterance_encoder.hidden_size
        self.transition_generator = nn.Sequential(
            nn.Linear(bert_hidden * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags * num_tags)
        )
        
        print(f"[DynamicCRF] Initialized")
        print(f"  Features: Dynamic transitions from adjacent observations")
        print(f"  Best for: Context-dependent transition modeling\n")
    
    def _compute_dynamic_transition(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic transition matrix from adjacent observations.
        
        Args:
            x_t: Current observation [hidden_size]
            x_next: Next observation [hidden_size]
        
        Returns:
            transition_matrix: [num_tags, num_tags]
        """
        # Concatenate adjacent observations
        combined = torch.cat([x_t, x_next], dim=0)
        
        # Generate transition scores
        trans_scores = self.transition_generator(combined)
        
        # Reshape to matrix
        trans_matrix = trans_scores.view(self.num_tags, self.num_tags)
        
        return trans_matrix
    
    def _forward_train_dynamic(
        self,
        utterance_sequence: torch.Tensor,
        I_label_tensor: torch.Tensor,
        posterior_emissions: torch.Tensor
    ) -> tuple:
        """CRF forward with dynamic transitions."""
        seq_len = utterance_sequence.size(0)
        device = utterance_sequence.device
        
        # Gold score
        gold_score = torch.sum(posterior_emissions[range(seq_len), I_label_tensor])
        
        for idx in range(1, seq_len):
            # Dynamic transition from adjacent observations
            trans_matrix = self._compute_dynamic_transition(
                utterance_sequence[idx-1],
                utterance_sequence[idx]
            )
            gold_score += trans_matrix[I_label_tensor[idx-1], I_label_tensor[idx]]
        
        # Forward algorithm
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for idx in range(seq_len):
            if idx == 0:
                trans_matrix = torch.zeros(self.num_tags, self.num_tags, device=device)
            else:
                trans_matrix = self._compute_dynamic_transition(
                    utterance_sequence[idx-1],
                    utterance_sequence[idx]
                )
            
            alpha = torch.logsumexp(
                alpha.T + posterior_emissions[idx].unsqueeze(0) + trans_matrix,
                dim=0,
                keepdim=True
            )
        
        total_score = torch.logsumexp(alpha, dim=1).squeeze()
        
        return gold_score, total_score
    
    def _forward_train(self, odd_repr, even_repr, odd_I_labels, system_I_labels, multiturn_features):
        """Training with dynamic transitions."""
        odd_I_labels = odd_I_labels.squeeze(0)
        system_I_labels = system_I_labels.squeeze(0)
        num_pairs = odd_repr.size(0)
        
        print(f"\n[DynamicCRF Train] Processing {num_pairs} pairs...")
        
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
            
            # Prior
            prior_seq = utterance_sequence[:-1].unsqueeze(0)
            prior_hidden = self.prior_encoder(prior_seq)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])
            
            # Posterior
            post_seq = utterance_sequence.unsqueeze(0)
            posterior_hidden = self.posterior_encoder(post_seq)
            posterior_emissions = self.posterior_emission_project(posterior_hidden)
            posterior_emissions = posterior_emissions.squeeze(0)
            
            # CRF with dynamic transitions
            gold_score, total_score = self._forward_train_dynamic(
                utterance_sequence,
                I_label_tensor,
                posterior_emissions
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
