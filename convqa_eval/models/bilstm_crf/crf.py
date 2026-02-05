"""Conditional Random Field layer for sequence labeling."""
import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Linear-chain CRF layer.
    
    Based on pytorch-crf implementation with Viterbi decoding.
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Initialize CRF.
        
        Args:
            num_tags: Number of tags (2 for binary: Initiative/Non-initiative)
            batch_first: Whether batch dimension is first
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition matrix: [num_tags, num_tags]
        # transitions[i][j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start/End transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        print(f"[CRF] Initialized with {num_tags} tags")
        print(f"[CRF] Transition matrix shape: {self.transitions.shape}")
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            emissions: [batch, seq_len, num_tags] - BiLSTM outputs
            tags: [batch, seq_len] - Ground truth tags
            mask: [batch, seq_len] - Padding mask (1=valid, 0=padding)
        
        Returns:
            Negative log-likelihood loss (scalar)
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)  # [seq_len, batch, num_tags]
            tags = tags.transpose(0, 1)            # [seq_len, batch]
            if mask is not None:
                mask = mask.transpose(0, 1)        # [seq_len, batch]
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        
        # Compute log-likelihood
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        
        print(f"[CRF Forward] Emissions shape: {emissions.shape}, Loss: {-llh.mean().item():.4f}")
        return -llh.mean()  # Negative log-likelihood
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor = None) -> list:
        """
        Viterbi decoding to find best tag sequence.
        
        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len]
        
        Returns:
            List of best tag sequences for each batch
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:-1], dtype=torch.uint8, device=emissions.device)
        
        best_paths = self._viterbi_decode(emissions, mask)
        
        print(f"[CRF Decode] Decoded {len(best_paths)} sequences")
        return best_paths
    
    def _compute_score(self, emissions, tags, mask):
        """Compute score of given tag sequence."""
        seq_len, batch_size = tags.shape
        
        # Start transition
        score = self.start_transitions[tags[0]]
        
        # Emission scores
        score += emissions[0, range(batch_size), tags[0]]
        
        # Transition + emission scores
        for i in range(1, seq_len):
            score += self.transitions[tags[i], tags[i - 1]] * mask[i]
            score += emissions[i, range(batch_size), tags[i]] * mask[i]
        
        # End transition (only for last valid position)
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, range(batch_size)]
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_normalizer(self, emissions, mask):
        """Compute partition function (normalization constant)."""
        seq_len, batch_size, num_tags = emissions.shape
        
        # Initialize forward variables
        score = self.start_transitions + emissions[0]
        
        # Forward pass
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # [batch, num_tags, 1]
            broadcast_emissions = emissions[i].unsqueeze(1)  # [batch, 1, num_tags]
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)  # [batch, num_tags]
            
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
        
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)
    
    def _viterbi_decode(self, emissions, mask):
        """Viterbi algorithm for finding best path."""
        seq_len, batch_size, num_tags = emissions.shape
        
        # Initialize
        score = self.start_transitions + emissions[0]
        history = []
        
        # Forward pass
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)
        
        # End transition
        score += self.end_transitions
        
        # Backward pass (backtracking)
        best_paths = []
        best_tags_list = []
        
        for batch_idx in range(batch_size):
            seq_end = mask[:, batch_idx].long().sum() - 1
            _, best_last_tag = score[batch_idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            for hist in reversed(history[:seq_end]):
                best_last_tag = hist[batch_idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            
            best_tags.reverse()
            best_paths.append(best_tags)
        
        return best_paths
