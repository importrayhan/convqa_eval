"""
MuSIc: Multi-turn System Initiative prediction with CRF

Paper Reference: "System Initiative Prediction for Multi-turn Conversations"

KEY INSIGHT FROM PAPER:
---------------------
Training vs Inference Asymmetry:

TRAINING (Fig 5a):
- We have access to ALL utterances X1:T (including unobservable system utterance at T)
- Pass X1:T through BERT + Posterior BiLSTM → posterior emission scores
- Pass X1:T-1 through BERT + Prior BiLSTM → prior emission scores
- CRF uses posterior emissions
- MLE loss: forces prior emissions to approximate posterior emissions at turn T

INFERENCE (Fig 5b):
- We do NOT have system utterance at turn T (it's what we're predicting!)
- Pass X1:T-1 through BERT + Prior BiLSTM → prior emission scores
- Use prior emission score as the "missing" emission at turn T
- CRF decodes using these approximate emissions

This is WHY we need both Prior and Posterior encoders!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import BertModel


class BERTUtteranceEncoder(nn.Module):
    """
    Component 1: BERT Utterance Encoder
    
    Encodes each utterance independently using BERT [CLS] token.
    Used for both user and system utterances.
    """
    
    def __init__(self, bert_model: str = 'bert-base-multilingual-cased'):
        super().__init__()
        print(f"[BERT] Loading {bert_model}...")
        self.bert = BertModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
        print(f"[BERT] Loaded, hidden_size={self.hidden_size}\n")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch * num_utterances, max_len]
        Returns:
            [CLS] representations: [batch * num_utterances, hidden_size]
        """
        outputs = self.bert(input_ids=input_ids)
        # Use [CLS] token (first token) as utterance representation
        return outputs.last_hidden_state[:, 0, :]


class PosteriorInterUtteranceEncoder(nn.Module):
    """
    Component 2a: Posterior Inter-Utterance Encoder
    
    USED DURING TRAINING with FULL context X1:T
    - Processes the complete conversation INCLUDING the current system utterance
    - Learns the "ideal" representation when all information is available
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size,  # BERT output size
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        print(f"[Posterior BiLSTM] {hidden_size} → {hidden_size*2} (bidirectional)\n")
    
    def forward(self, utterance_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            utterance_sequence: [1, seq_len, hidden_size]
                               seq_len = 2*num_pairs (user1, sys1, user2, sys2, ...)
        Returns:
            BiLSTM outputs: [1, seq_len, 2*hidden_size]
        """
        output, _ = self.lstm(utterance_sequence)
        return output


class PriorInterUtteranceEncoder(nn.Module):
    """
    Component 2b: Prior Inter-Utterance Encoder
    
    USED DURING INFERENCE with PARTIAL context X1:T-1
    - Processes conversation WITHOUT the current system utterance
    - Must predict based on incomplete information (realistic scenario)
    - During training, its output is forced to match posterior via MLE loss
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
        print(f"[Prior BiLSTM] {hidden_size} → {hidden_size*2} (bidirectional)\n")
    
    def forward(self, utterance_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            utterance_sequence: [1, seq_len, hidden_size]
                               seq_len = 2*num_pairs - 1 (missing last system utterance)
        Returns:
            BiLSTM outputs: [1, seq_len, 2*hidden_size]
        """
        output, _ = self.lstm(utterance_sequence)
        return output


class SimpleCRF(nn.Module):
    """
    Component 3: CRF Layer (Simplified version for VanillaCRF)
    
    Uses a single learned transition matrix for all transitions.
    More complex variants use conversation-specific transition matrices.
    """
    
    def __init__(self, num_tags: int = 2):
        super().__init__()
        self.num_tags = num_tags
        
        # Single transition matrix [num_tags, num_tags]
        # Element [i,j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        
        print(f"[CRF] Initialized with {num_tags} tags\n")
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CRF loss.
        
        Args:
            emissions: [seq_len, num_tags] - Emission scores from BiLSTM
            tags: [seq_len] - Ground truth tags
        
        Returns:
            gold_score: Score of the gold tag sequence
            total_score: Log partition function (log sum of all possible sequences)
        
        Loss = total_score - gold_score (we want to minimize this)
        """
        seq_len = emissions.size(0)
        device = emissions.device
        
        # === Compute Gold Score ===
        # Score of the actual ground truth sequence
        gold_score = torch.sum(emissions[range(seq_len), tags])  # Emission scores
        
        for i in range(1, seq_len):
            # Add transition scores
            gold_score += self.transitions[tags[i-1], tags[i]]
        
        # === Compute Total Score (Forward Algorithm) ===
        # Log-sum-exp of scores of all possible tag sequences
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for i in range(seq_len):
            # For first position, no transitions
            if i == 0:
                alpha = emissions[i].unsqueeze(0)
            else:
                # alpha: [1, num_tags] representing scores ending in each tag
                # We expand and add transitions and emissions
                alpha = torch.logsumexp(
                    alpha.T + emissions[i].unsqueeze(0) + self.transitions,
                    dim=0,
                    keepdim=True
                )
        
        total_score = torch.logsumexp(alpha, dim=1).squeeze()
        
        return gold_score, total_score
    
    def decode(self, emissions: torch.Tensor) -> List[int]:
        """
        Viterbi decoding to find best tag sequence.
        
        Args:
            emissions: [seq_len, num_tags]
        Returns:
            best_path: List of predicted tags
        """
        seq_len = emissions.size(0)
        device = emissions.device
        
        # Viterbi algorithm with backpointers
        backpointers = []
        alpha = torch.zeros(1, self.num_tags, device=device)
        
        for i in range(seq_len):
            if i == 0:
                alpha = emissions[i].unsqueeze(0)
                backpointers.append(torch.zeros(self.num_tags, dtype=torch.long, device=device))
            else:
                # Find max instead of logsumexp
                alpha_with_trans = alpha.T + emissions[i].unsqueeze(0) + self.transitions
                viterbivars, bptrs = torch.max(alpha_with_trans, dim=0)
                backpointers.append(bptrs)
                alpha = viterbivars.unsqueeze(0)
        
        # Backtrack to find best path
        best_tag_id = alpha.squeeze().argmax().item()
        best_path = [best_tag_id]
        
        for bptrs in reversed(backpointers[1:]):
            best_tag_id = bptrs[best_tag_id].item()
            best_path.append(best_tag_id)
        
        best_path.reverse()
        return best_path


class MuSIc(nn.Module):
    """
    MuSIc: Multi-turn System Initiative prediction with CRF
    
    Complete architecture with 3 components:
    1. BERT Utterance Encoder
    2. Prior & Posterior Inter-Utterance Encoders  
    3. CRF Layer
    
    Training Strategy:
    ------------------
    For each turn t in conversation:
        1. Build full sequence: [user1, sys1, ..., user_t, sys_t]
        2. Posterior path: Full sequence → Posterior BiLSTM → emissions
        3. Prior path: Partial sequence [:-1] → Prior BiLSTM → emission at t
        4. CRF loss on posterior emissions
        5. MLE loss: MSE(prior_emission_t, posterior_emission_t)
    
    Inference Strategy:
    ------------------
    For turn t:
        1. Build partial sequence: [user1, sys1, ..., user_t] (no sys_t!)
        2. Prior path: Partial sequence → Prior BiLSTM → emission at t
        3. Use this emission as the "missing" emission for sys_t
        4. CRF decode to get predictions
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-multilingual-cased',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_tags: int = 2,
        lambda_mle: float = 0.1
    ):
        super().__init__()
        
        self.num_tags = num_tags
        self.lambda_mle = lambda_mle
        
        print(f"\n{'='*60}")
        print(f"Initializing MuSIc Model")
        print(f"{'='*60}")
        print(f"Hidden size: {hidden_size}")
        print(f"Num layers: {num_layers}")
        print(f"Num tags: {num_tags}")
        print(f"Lambda MLE: {lambda_mle}\n")
        
        # Component 1: BERT
        self.utterance_encoder = BERTUtteranceEncoder(bert_model)
        bert_hidden = self.utterance_encoder.hidden_size
        
        # Component 2: Prior & Posterior
        self.prior_encoder = PriorInterUtteranceEncoder(bert_hidden, num_layers, dropout)
        self.posterior_encoder = PosteriorInterUtteranceEncoder(bert_hidden, num_layers, dropout)
        
        # Emission projections (BiLSTM output → tag scores)
        encoder_output_size = bert_hidden * 2  # Bidirectional
        self.prior_emission_project = nn.Linear(encoder_output_size, num_tags)
        self.posterior_emission_project = nn.Linear(encoder_output_size, num_tags)
        
        # Component 3: CRF
        self.crf = SimpleCRF(num_tags)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"{'='*60}")
        print(f"MuSIc Initialized Successfully")
        print(f"{'='*60}\n")
    
    def encode_utterances(
        self,
        user_utterance: torch.Tensor,
        system_utterance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all user and system utterances using BERT.
        
        Args:
            user_utterance: [batch=1, num_pairs, max_len]
            system_utterance: [batch=1, num_pairs, max_len]
        
        Returns:
            user_repr: [num_pairs, bert_hidden]
            system_repr: [num_pairs, bert_hidden]
        """
        batch_size, num_pairs, max_len = user_utterance.shape
        
        print(f"\n[Encode] Encoding {num_pairs} user-system pairs...")
        
        # Flatten for BERT
        user_flat = user_utterance.view(num_pairs, max_len)
        system_flat = system_utterance.view(num_pairs, max_len)
        
        # BERT encode
        user_repr = self.utterance_encoder(user_flat)  # [num_pairs, bert_hidden]
        system_repr = self.utterance_encoder(system_flat)
        
        print(f"[Encode] User: {user_repr.shape}, System: {system_repr.shape}")
        
        return user_repr, system_repr
    
    def forward(
        self,
        user_utterance: torch.Tensor,
        system_utterance: torch.Tensor,
        user_I_label: torch.Tensor = None,
        system_I_label: torch.Tensor = None,
        mode: str = 'train'
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            user_utterance: [1, num_pairs, max_len]
            system_utterance: [1, num_pairs, max_len]
            user_I_label: [1, num_pairs] (always 0)
            system_I_label: [1, num_pairs] (0 or 1)
            mode: 'train' or 'inference'
        """
        # Encode all utterances with BERT
        user_repr, system_repr = self.encode_utterances(user_utterance, system_utterance)
        num_pairs = user_repr.size(0)
        
        if mode == 'train':
            return self._train_forward(
                user_repr, system_repr, user_I_label, system_I_label, num_pairs
            )
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
        """
        Training forward pass.
        
        Key: Process INCREMENTALLY, one turn at a time!
        Why? Because at inference, we also process incrementally.
        
        For each turn t:
        1. Build full sequence up to t: [u1, s1, u2, s2, ..., u_t, s_t]
        2. Posterior: Process full sequence
        3. Prior: Process partial [u1, s1, ..., u_t] (no s_t)
        4. Compare: Force prior to match posterior at position s_t
        """
        user_I_label = user_I_label.squeeze(0)
        system_I_label = system_I_label.squeeze(0)
        
        print(f"\n{'='*60}")
        print(f"TRAINING MODE - Processing {num_pairs} turns incrementally")
        print(f"{'='*60}\n")
        
        gold_scores = []
        total_scores = []
        prior_emissions_list = []
        posterior_emissions_list = []
        
        for pair_idx in range(num_pairs):
            print(f"--- Turn {pair_idx+1}/{num_pairs} ---")
            
            # Build conversation sequence up to this turn
            # Format: [user1, system1, user2, system2, ..., user_t, system_t]
            utterance_sequence = []
            I_label_sequence = []
            
            for i in range(pair_idx + 1):
                utterance_sequence.append(user_repr[i])
                utterance_sequence.append(system_repr[i])
                I_label_sequence.append(user_I_label[i].item())
                I_label_sequence.append(system_I_label[i].item())
            
            utterance_sequence = torch.stack(utterance_sequence).unsqueeze(0)  # [1, 2*(pair_idx+1), hidden]
            I_label_tensor = torch.tensor(I_label_sequence, device=user_repr.device)
            
            print(f"  Sequence length: {utterance_sequence.shape[1]}")
            print(f"  Labels: {I_label_sequence}")
            
            # === POSTERIOR PATH (full context) ===
            # Use ALL utterances including current system response
            # This represents the "ideal" case where we have complete information
            posterior_hidden = self.posterior_encoder(utterance_sequence)  # [1, seq_len, 2*hidden]
            posterior_emissions = self.posterior_emission_project(posterior_hidden)
            posterior_emissions = posterior_emissions.squeeze(0)  # [seq_len, num_tags]
            
            # === PRIOR PATH (partial context) ===
            # Remove the last utterance (current system response)
            # This simulates inference where we don't have system_t yet
            prior_sequence = utterance_sequence[:, :-1, :]  # [1, seq_len-1, hidden]
            prior_hidden = self.prior_encoder(prior_sequence)  # [1, seq_len-1, 2*hidden]
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])  # [1, num_tags]
            
            print(f"  Posterior emissions: {posterior_emissions.shape}")
            print(f"  Prior emission (for missing system): {prior_emission.shape}")
            
            # === CRF LOSS ===
            # Use posterior emissions (they have full context)
            gold_score, total_score = self.crf(posterior_emissions, I_label_tensor)
            
            print(f"  CRF - Gold: {gold_score:.4f}, Total: {total_score:.4f}")
            
            gold_scores.append(gold_score)
            total_scores.append(total_score)
            
            # === MLE LOSS TERMS ===
            # We want prior emission to match posterior emission at the last position (system_t)
            prior_emissions_list.append(prior_emission.squeeze(0))
            posterior_emissions_list.append(posterior_emissions[-1])  # Last position = system_t
            
            print()
        
        # Aggregate losses
        gold_scores = torch.stack(gold_scores)
        total_scores = torch.stack(total_scores)
        prior_emissions = torch.stack(prior_emissions_list)  # [num_pairs, num_tags]
        posterior_emissions = torch.stack(posterior_emissions_list)  # [num_pairs, num_tags]
        
        # CRF Loss: Negative log-likelihood
        loss_crf = torch.mean(total_scores - gold_scores)
        
        # MLE Loss: Force prior to match posterior
        # This teaches the prior encoder to predict what posterior would output
        # even without seeing the system utterance!
        loss_mle = F.mse_loss(prior_emissions, posterior_emissions.detach())
        
        total_loss = loss_crf + self.lambda_mle * loss_mle
        
        print(f"{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"CRF Loss: {loss_crf:.4f}")
        print(f"MLE Loss: {loss_mle:.4f}")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'loss_crf': loss_crf,
            'loss_mle': loss_mle,
            'total_loss': total_loss
        }
    
    def _inference_forward(
        self,
        user_repr: torch.Tensor,
        system_repr: torch.Tensor,
        num_pairs: int
    ) -> Dict:
        """
        Inference forward pass.
        
        Key: At turn t, we do NOT have system_t yet!
        
        For each turn t:
        1. Build partial sequence: [u1, s1, ..., s_{t-1}, u_t] (no s_t!)
        2. Prior: Process this partial sequence
        3. Use prior's output as the "missing" emission for s_t
        4. Decode to get predictions
        """
        print(f"\n{'='*60}")
        print(f"INFERENCE MODE - Processing {num_pairs} turns")
        print(f"{'='*60}\n")
        
        all_predictions = []
        system_predictions = []
        
        for pair_idx in range(num_pairs):
            print(f"--- Turn {pair_idx+1}/{num_pairs} ---")
            
            # Build partial sequence (NO current system utterance!)
            # [user1, system1, ..., system_{t-1}, user_t]
            utterance_sequence = []
            
            for i in range(pair_idx):
                utterance_sequence.append(user_repr[i])
                utterance_sequence.append(system_repr[i])
            
            # Add current user
            utterance_sequence.append(user_repr[pair_idx])
            
            utterance_sequence = torch.stack(utterance_sequence).unsqueeze(0)  # [1, 2*pair_idx+1, hidden]
            
            print(f"  Partial sequence length: {utterance_sequence.shape[1]}")
            print(f"  Missing: system utterance at position {pair_idx+1}")
            
            # === PRIOR PATH (used in inference) ===
            # Encode partial sequence with Prior encoder
            prior_hidden = self.prior_encoder(utterance_sequence)
            prior_emission = self.prior_emission_project(prior_hidden[:, -1, :])  # Last position
            
            # === ALSO need emissions for all previous positions ===
            # For Viterbi decoding, we need emissions for the entire sequence
            # But for previous turns, we can use Posterior (we have those utterances)
            if pair_idx > 0:
                # Get emissions for [u1, s1, ..., s_{t-1}]
                prev_sequence = utterance_sequence[:, :-1, :]  # Remove current user
                prev_posterior_hidden = self.posterior_encoder(prev_sequence)
                prev_emissions = self.posterior_emission_project(prev_posterior_hidden).squeeze(0)
            else:
                prev_emissions = torch.empty(0, self.num_tags, device=user_repr.device)
            
            # Emissions for current user (just added)
            # We can use Posterior for this since we have the utterance
            curr_user_hidden = self.posterior_encoder(utterance_sequence)
            curr_user_emission = self.posterior_emission_project(curr_user_hidden[:, -1, :])
            
            # Combine: [prev emissions, current user emission, PREDICTED system emission]
            combined_emissions = torch.cat([
                prev_emissions,
                curr_user_emission.squeeze(0).unsqueeze(0),
                prior_emission.squeeze(0).unsqueeze(0)
            ], dim=0)  # [2*(pair_idx+1), num_tags]
            
            print(f"  Combined emissions: {combined_emissions.shape}")
            
            # Viterbi decode
            predicted_path = self.crf.decode(combined_emissions)
            
            # System prediction is at the last position (even index)
            system_pred = predicted_path[-1]
            
            print(f"  Predicted path: {predicted_path}")
            print(f"  System initiative: {system_pred}\n")
            
            all_predictions.append(predicted_path)
            system_predictions.append(system_pred)
        
        print(f"{'='*60}")
        print(f"INFERENCE SUMMARY")
        print(f"{'='*60}")
        print(f"System predictions: {system_predictions}")
        print(f"Initiative rate: {sum(system_predictions)/len(system_predictions):.1%}")
        print(f"{'='*60}\n")
        
        return {
            'predictions': all_predictions,
            'system_predictions': system_predictions,
            'initiative_rate': sum(system_predictions) / len(system_predictions) if system_predictions else 0
        }


if __name__ == "__main__":
    print("\nTesting MuSIc Model\n")
    
    model = MuSIc(hidden_size=128, num_tags=2)
    
    # Dummy data
    user_utt = torch.randint(0, 1000, (1, 3, 128))
    system_utt = torch.randint(0, 1000, (1, 3, 128))
    user_labels = torch.zeros(1, 3, dtype=torch.long)
    system_labels = torch.tensor([[0, 1, 0]], dtype=torch.long)
    
    # Training
    model.train()
    output = model(user_utt, system_utt, user_labels, system_labels, mode='train')
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(user_utt, system_utt, mode='inference')
