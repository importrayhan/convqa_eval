"""
BiLSTM-CRF for System Initiative Prediction (MuSIc, CIKM'23).

Faithful re-implementation of Meng et al. "System Initiative Prediction
for Multi-turn Conversational Information Seeking" following Figures 4-5
and Equations 1-10 of the paper.

ARCHITECTURE  (Figure 4):
  1. BERT utterance encoder   — encodes each utterance via average pooling
  2. Prior / Posterior BiLSTM  — inter-utterance encoders
  3. Multi-turn feature-aware CRF layer

TRAINING  (Figure 5a):
  - Posterior encoder sees  X_{1:T}    (including unobservable system turn T)
  - Prior    encoder sees  X_{1:T-1}  (context only)
  - CRF runs on posterior emissions  for turns 1..T
  - MSE loss forces prior emission at T-1 → posterior emission at T
  - L = L_crf + lambda * L_mse

INFERENCE (Figure 5b):
  - Posterior encoder sees  X_{1:T-1}  (context only)  →  emissions 1..T-1
  - Prior    encoder sees  X_{1:T-1}  (context only)  →  emission at T
  - Combine: [posterior_1, ..., posterior_{T-1}, prior_T]
  - Viterbi decode → only y_T is used for evaluation

KEY DESIGN:
  - Single-device execution per conversation (batch_size=1).
  - Conversation is a FLAT interleaved sequence:
      [user_1, sys_1, user_2, sys_2, ..., user_N, sys_N]
    where sys_N is the "unobservable" turn T.
  - Odd indices (0,2,4,...) = user turns.  Even indices (1,3,5,...) = system turns.
  - Turn T-1 = user_N (last user utterance), Turn T = sys_N (to predict).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoConfig

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Utterance Encoder — pluggable transformer with AVERAGE pooling
# ══════════════════════════════════════════════════════════════════════════════
class TransformerUtteranceEncoder(nn.Module):
    """
    Encode each utterance into a single vector via average pooling (Sec 4.2.1).

    Loading priority:
      1. encoder_path  — local directory  (e.g. /custom_path/ModernBERT-base)
      2. encoder_name  — HuggingFace hub  (e.g. answerdotai/ModernBERT-base)
    A BiLSTM-CRF uses exactly ONE encoder at a time, never both.
    """

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        encoder_path: str = None,
        gradient_checkpointing: bool = False,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        if encoder_path and os.path.isdir(encoder_path):
            source = encoder_path
            self._source_desc = f"local:{encoder_path}"
        else:
            source = encoder_name
            self._source_desc = f"hub:{encoder_name}"
            if encoder_path:
                log.warning(
                    f"encoder_path='{encoder_path}' not a valid directory, "
                    f"falling back to hub name '{encoder_name}'")

        cfg = AutoConfig.from_pretrained(source)
        self.encoder = AutoModel.from_pretrained(source, config=cfg)
        self.hidden_size = cfg.hidden_size
        self.resolved_source = source

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids:      [B, L]
        attention_mask:  [B, L]  (optional — computed from input_ids if None)
        Returns:         [B, hidden_size]  via average pooling (paper Sec 4.2.1)
        """
        if attention_mask is None:
            # non-padding tokens (pad_token_id = 0 for BERT)
            attention_mask = (input_ids != 0).long()
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, d]
        # Average pooling over non-padding tokens
        mask_f = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (hidden * mask_f).sum(dim=1)          # [B, d]
        lengths = mask_f.sum(dim=1).clamp(min=1)       # [B, 1]
        return summed / lengths                        # [B, d]

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BiLSTM inter-utterance encoders (prior / posterior)
#     Paper Sec 4.2.2, footnote 1: "implemented by BiLSTMs"
# ══════════════════════════════════════════════════════════════════════════════
class InterUtteranceEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [1, T, input_size] → [1, T, 2*hidden_size]"""
        out, _ = self.lstm(x)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Multi-turn feature extraction (Paper Sec 4.2.3)
# ══════════════════════════════════════════════════════════════════════════════
def compute_multiturn_features(
    num_turns: int,
    system_labels: List[int],
) -> List[Dict[str, int]]:
    """
    Compute features S = {s^r, s^n, s^d} for transitions between all
    adjacent pairs (t, t+1) in the interleaved sequence.

    The interleaved sequence has indices 0..num_turns-1 where:
      - even indices (0,2,4,...) are user utterances
      - odd  indices (1,3,5,...) are system utterances

    For transition from turn t to t+1:
      s^r_{t:t+1}: role transition direction
        = "u2s" (0→1, 2→3, ...) or "s2u" (1→2, 3→4, ...)
      s^n_{t:t+1}: number of times system took initiative before next system turn
        Only defined when s^r = u2s.  Binary: 0 or >0
      s^d_{t:t+1}: distance to last system initiative turn
        Only defined when s^r = u2s and s^n > 0.  Binary: =2 or >2

    Returns list of T-1 feature dicts for transitions (0→1), (1→2), ..., (T-2→T-1)
    """
    features = []
    # Track system initiative history for s^n and s^d computation
    # system_labels[i] is the initiative label for the i-th user-system pair
    # In the interleaved sequence, system turn for pair i is at index 2*i+1

    for t in range(num_turns - 1):
        feat = {}

        # s^r: role transition direction
        if t % 2 == 0:  # even→odd = user→system
            feat["who2who"] = 0  # u2s
        else:            # odd→even = system→user
            feat["who2who"] = 1  # s2u

        # s^n and s^d: only meaningful for u2s transitions
        if feat["who2who"] == 0:  # u2s
            # Which system pair index is the target?
            # t is even (user), t+1 is odd (system), pair_idx = t//2
            pair_idx = t // 2
            # Count system initiatives BEFORE this system turn
            init_count = 0
            last_init_pair = -1
            for j in range(pair_idx):
                if j < len(system_labels) and system_labels[j] > 0:
                    init_count += 1
                    last_init_pair = j

            feat["init_count"] = 1 if init_count > 0 else 0  # binary

            # Distance: only when init_count > 0
            if init_count > 0:
                # Distance in terms of turns between last initiative system
                # turn and current system turn
                # last initiative system turn is at interleaved index 2*last_init_pair+1
                # current system turn is at interleaved index 2*pair_idx+1
                distance = (2 * pair_idx + 1) - (2 * last_init_pair + 1)
                feat["distance"] = 0 if distance <= 2 else 1  # 0 = d==2, 1 = d>2
            else:
                feat["distance"] = -1  # not applicable
        else:
            feat["init_count"] = -1  # not applicable for s2u
            feat["distance"] = -1

        features.append(feat)

    return features


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Multi-turn feature-aware CRF layer (Paper Sec 4.2.3, Eqs 4-7)
# ══════════════════════════════════════════════════════════════════════════════
class MultiTurnCRF(nn.Module):
    """
    6 CRF variants from the paper:
      VanillaCRF          — Eq. 4: single global transition matrix
      Who2WhoCRF          — Eq. 5: G^{u2s}, G^{s2u}
      PositionCRF         — position-specific transitions (up to 20)
      Who2Who_PositionCRF — combined
      IntimeCRF           — Eq. 6: G^{s2u}, G^{u2s,n=0}, G^{u2s,n>0}
      DistanceCRF         — Eq. 7: G^{s2u}, G^{u2s,n=0}, G^{u2s,n>0,d=2}, G^{u2s,n>0,d>2}
    """
    VARIANTS = ("VanillaCRF", "Who2WhoCRF", "PositionCRF",
                "Who2Who_PositionCRF", "IntimeCRF", "DistanceCRF")

    def __init__(self, num_tags: int, variant: str = "VanillaCRF"):
        super().__init__()
        assert variant in self.VARIANTS, f"Unknown variant {variant}"
        self.num_tags = num_tags
        self.variant = variant

        # All variants can fall back to a global transition matrix
        self.trans_global = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)

        # Start scores (transition into the first tag)
        self.start_scores = nn.Parameter(torch.randn(num_tags) * 0.1)

        if variant in ("Who2WhoCRF", "Who2Who_PositionCRF",
                        "IntimeCRF", "DistanceCRF"):
            self.trans_u2s = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
            self.trans_s2u = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)

        if variant in ("PositionCRF", "Who2Who_PositionCRF"):
            self.trans_position = nn.ParameterList(
                [nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
                 for _ in range(20)])

        if variant in ("IntimeCRF", "DistanceCRF"):
            # Eq. 6: G^{u2s,n=0} and G^{u2s,n>0}
            self.trans_u2s_n0 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
            self.trans_u2s_n1 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)

        if variant == "DistanceCRF":
            # Eq. 7: G^{u2s,n>0,d=2} and G^{u2s,n>0,d>2}
            self.trans_u2s_n1_d2 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
            self.trans_u2s_n1_dgt2 = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)

    def _get_trans(self, feat: Dict[str, int], position: int) -> torch.Tensor:
        """Select the correct transition matrix for the (t → t+1) transition."""
        v = self.variant
        w = feat["who2who"]  # 0=u2s, 1=s2u

        if v == "VanillaCRF":
            return self.trans_global

        if v == "Who2WhoCRF":
            return self.trans_u2s if w == 0 else self.trans_s2u

        if v == "PositionCRF":
            p = min(position, 19)
            return self.trans_position[p] if 0 <= p < 20 else self.trans_global

        if v == "Who2Who_PositionCRF":
            p = min(position, 19)
            if 0 <= p < 20:
                return self.trans_position[p]
            return self.trans_u2s if w == 0 else self.trans_s2u

        # IntimeCRF: Eq. 6
        if v == "IntimeCRF":
            if w == 1:  # s2u
                return self.trans_s2u
            # u2s: select by init_count
            if feat["init_count"] == 0:
                return self.trans_u2s_n0
            else:
                return self.trans_u2s_n1

        # DistanceCRF: Eq. 7 — the FULL model (MuSIc Full)
        if v == "DistanceCRF":
            if w == 1:  # s2u
                return self.trans_s2u
            # u2s: first check init_count
            if feat["init_count"] == 0:
                return self.trans_u2s_n0
            else:
                # init_count > 0: select by distance
                if feat["distance"] == 0:  # d == 2
                    return self.trans_u2s_n1_d2
                else:  # d > 2
                    return self.trans_u2s_n1_dgt2

        return self.trans_global  # fallback

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        features: List[Dict[str, int]],
        class_weights: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CRF negative log-likelihood with optional class weighting.

        emissions:     [T, num_tags]  — emission scores for all T turns
        tags:          [T]            — gold label sequence
        features:      list of T-1 feature dicts
        class_weights: [num_tags]     — per-class weight multiplier (optional).
                       Scales the gold path score so minority-class positions
                       contribute more to the loss gradient.

        Returns: (gold_score, log_Z)  so that NLL = log_Z - gold_score
        """
        T = emissions.size(0)

        # Gold score: start + emissions + transitions
        # When class_weights is provided, each position's emission+transition
        # contribution is scaled by the weight of its gold tag.
        w0 = class_weights[tags[0]] if class_weights is not None else 1.0
        gold = w0 * (self.start_scores[tags[0]] + emissions[0, tags[0]])
        for t in range(1, T):
            tr = self._get_trans(features[t - 1], position=t - 1)
            wt = class_weights[tags[t]] if class_weights is not None else 1.0
            gold = gold + wt * (tr[tags[t - 1], tags[t]] + emissions[t, tags[t]])

        # Forward algorithm for log Z (partition function — NOT weighted,
        # because it must sum over ALL possible paths, not just the gold one)
        alphas = self.start_scores + emissions[0]  # [num_tags]
        for t in range(1, T):
            tr = self._get_trans(features[t - 1], position=t - 1)
            scores = alphas.unsqueeze(1) + tr + emissions[t].unsqueeze(0)
            alphas = torch.logsumexp(scores, dim=0)

        log_Z = torch.logsumexp(alphas, dim=0)
        return gold, log_Z

    def decode(
        self,
        emissions: torch.Tensor,
        features: List[Dict[str, int]],
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Viterbi decoding.

        emissions: [T, num_tags]
        features:  list of T-1 feature dicts

        Returns: (best_path, probs_per_step)
        """
        T, K = emissions.shape
        dp = self.start_scores + emissions[0]
        bp = []

        for t in range(1, T):
            tr = self._get_trans(features[t - 1], position=t - 1)
            scores = dp.unsqueeze(1) + tr + emissions[t].unsqueeze(0)
            dp, idx = scores.max(dim=0)
            bp.append(idx)

        # Backtrack
        path = [dp.argmax().item()]
        for ptr in reversed(bp):
            path.insert(0, ptr[path[0]].item())

        # Softmax probs per step (for confidence scores)
        probs = [F.softmax(emissions[t], dim=-1).detach().cpu().numpy()
                 for t in range(T)]
        return path, probs


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Full Model — BiLSTMCRFModel (single-device, batch_size=1)
# ══════════════════════════════════════════════════════════════════════════════
class BiLSTMCRFModel(nn.Module):
    """
    MuSIc: BERT-encoder → BiLSTM (prior/posterior) → multi-turn CRF.

    Processes one conversation at a time.

    Data format:
      user_ids:    [1, N, L]  — N user utterance token-id sequences
      system_ids:  [1, N, L]  — N system utterance token-id sequences
      user_labels: [1, N]     — initiative labels for user turns
      system_labels: [1, N]   — initiative labels for system turns

    The interleaved sequence is:
      [user_0, sys_0, user_1, sys_1, ..., user_{N-1}, sys_{N-1}]
    which has 2N turns.  Turn T = sys_{N-1} (the last system turn).
    Turn T-1 = user_{N-1} (the last user utterance).
    """

    def __init__(
        self,
        num_classes: int = 2,
        encoder_name: str = "bert-base-uncased",
        encoder_path: str = None,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        lambda_mle: float = 1.0,
        crf_variant: str = "VanillaCRF",
        gradient_checkpointing: bool = False,
        freeze_encoder_epochs: int = 0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_mle = lambda_mle
        self.freeze_encoder_epochs = freeze_encoder_epochs

        # Class weights for CRF loss (None = unweighted)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # 1. BERT Utterance Encoder (Sec 4.2.1)
        freeze = freeze_encoder_epochs > 0
        self.utterance_encoder = TransformerUtteranceEncoder(
            encoder_name=encoder_name,
            encoder_path=encoder_path,
            gradient_checkpointing=gradient_checkpointing,
            freeze_encoder=freeze,
        )
        enc_dim = self.utterance_encoder.hidden_size

        # 2. Prior / Posterior BiLSTM inter-utterance encoders (Sec 4.2.2)
        self.prior_encoder = InterUtteranceEncoder(
            enc_dim, hidden_size, num_layers, dropout)
        self.posterior_encoder = InterUtteranceEncoder(
            enc_dim, hidden_size, num_layers, dropout)

        # 3. Emission MLPs (Eqs 2-3, parameters NOT shared)
        bidir = hidden_size * 2
        self.posterior_emission = nn.Sequential(
            nn.Linear(bidir, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes))
        self.prior_emission = nn.Sequential(
            nn.Linear(bidir, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes))

        # 4. Multi-turn feature-aware CRF (Sec 4.2.3)
        self.crf = MultiTurnCRF(num_classes, crf_variant)

    # ── helpers ───────────────────────────────────────────────────────────
    def maybe_unfreeze_encoder(self, epoch: int):
        if epoch >= self.freeze_encoder_epochs:
            self.utterance_encoder.unfreeze()

    def _encode_utterances(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [1, N, L] → [N, enc_dim]  via BERT + average pooling."""
        N, L = ids.shape[1], ids.shape[2]
        flat_ids = ids.view(N, L)
        return self.utterance_encoder(flat_ids)  # [N, enc_dim]

    def _interleave(
        self, user_repr: torch.Tensor, sys_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Interleave user and system representations into a flat sequence.
        user_repr: [N, d], sys_repr: [N, d]
        Returns: [2N, d]  = [u_0, s_0, u_1, s_1, ..., u_{N-1}, s_{N-1}]
        """
        N, d = user_repr.shape
        combined = torch.stack([user_repr, sys_repr], dim=1)  # [N, 2, d]
        return combined.view(2 * N, d)  # [2N, d]

    def _interleave_labels(
        self, user_labels: torch.Tensor, sys_labels: torch.Tensor
    ) -> torch.Tensor:
        """Interleave labels: [N] + [N] → [2N]"""
        N = user_labels.size(0)
        combined = torch.stack([user_labels, sys_labels], dim=1)  # [N, 2]
        return combined.view(2 * N)  # [2N]

    # ── forward dispatch ─────────────────────────────────────────────────
    def forward(self, user_ids, system_ids,
                user_labels=None, system_labels=None, mode="train",
                per_turn=True):
        """
        user_ids:      [1, N, L]
        system_ids:    [1, N, L]
        user_labels:   [1, N]  (only for training)
        system_labels: [1, N]  (only for training)
        mode:          "train" or "inference"
        per_turn:      True  → predict at every system turn (pacific/simmic)
                       False → predict only at the last system turn (claqua);
                               all prior turns are treated as observed context
        """
        if mode == "train":
            return self._train_forward(
                user_ids, system_ids,
                user_labels.squeeze(0), system_labels.squeeze(0),
                per_turn=per_turn)
        else:
            return self._inference_forward(user_ids, system_ids,
                                           per_turn=per_turn)

    # ──────────────────────────────────────────────────────────────────────
    # TRAINING  (Figure 5a) — evaluate EVERY system turn
    # ──────────────────────────────────────────────────────────────────────
    def _train_forward(self, user_ids, system_ids, user_labels, system_labels,
                       per_turn=True):
        """
        For each pair index k in the training range, treat sys_k as the
        "next turn T" whose label we are predicting.

        per_turn=True  (pacific/simmic): k loops over 0..N-1, so every
            system turn produces a training signal.
        per_turn=False (claqua): k = N-1 only.  All prior turns are
            treated as fully observed context.  The only target label is
            the last system turn's ambiguous_type.
        """
        # 1. BERT encode all utterances once (shared across all k)
        user_repr = self._encode_utterances(user_ids)   # [N, d]
        sys_repr  = self._encode_utterances(system_ids) # [N, d]
        N = user_repr.size(0)

        if N == 0:
            zero = torch.tensor(0.0, device=user_repr.device, requires_grad=True)
            return {"loss_crf": zero, "loss_mle": zero, "total_loss": zero}

        losses_crf, losses_mle = [], []

        # per_turn=True: train on every system turn k=0..N-1
        # per_turn=False: train only on the last system turn k=N-1
        k_range = range(N) if per_turn else range(N - 1, N)

        for k in k_range:
            # Build full subsequence including sys_k:
            #   [u_0, s_0, u_1, s_1, ..., u_k, s_k]  = 2(k+1) turns
            full_parts = []
            for i in range(k + 1):
                full_parts.append(user_repr[i])
                full_parts.append(sys_repr[i])
            full_seq = torch.stack(full_parts)  # [2(k+1), d]
            T = full_seq.size(0)                # T = 2(k+1)

            # Context = full_seq minus the last element (sys_k)
            context_seq = full_seq[:T - 1]      # [T-1, d]

            # Posterior encoder: sees full [u_0,s_0,...,u_k,s_k]
            po_hidden = self.posterior_encoder(full_seq.unsqueeze(0))
            po_emissions = self.posterior_emission(po_hidden.squeeze(0))  # [T, C]

            # Prior encoder: sees context [u_0,s_0,...,u_k] (missing sys_k)
            pr_hidden = self.prior_encoder(context_seq.unsqueeze(0))
            pr_emission = self.prior_emission(pr_hidden[:, -1, :])  # [1, C]

            # Labels: interleave user/system labels up to pair k
            labs_u = user_labels[:k + 1]
            labs_s = system_labels[:k + 1]
            labels_flat = self._interleave_labels(labs_u, labs_s)  # [2(k+1)]

            # Multi-turn features
            sys_labels_list = system_labels[:k + 1].tolist()
            features = compute_multiturn_features(T, sys_labels_list)

            # CRF NLL (with optional class weighting for imbalanced data)
            gold_score, log_Z = self.crf(po_emissions, labels_flat, features,
                                         class_weights=self.class_weights)
            losses_crf.append(log_Z - gold_score)

            # MSE: prior at T-1 ≈ posterior at T
            losses_mle.append(F.mse_loss(
                pr_emission.squeeze(0), po_emissions[-1].detach()))

        loss_crf = torch.stack(losses_crf).mean()
        loss_mle = torch.stack(losses_mle).mean()
        total_loss = loss_crf + self.lambda_mle * loss_mle

        return {
            "loss_crf": loss_crf,
            "loss_mle": loss_mle,
            "total_loss": total_loss,
        }

    # ──────────────────────────────────────────────────────────────────────
    # INFERENCE  (Figure 5b) — predict at EVERY system turn
    # ──────────────────────────────────────────────────────────────────────
    def _inference_forward(self, user_ids, system_ids, per_turn=True):
        """
        Predict system initiative labels.

        per_turn=True  (pacific/simmic): predict at every system turn k=0..N-1.
        per_turn=False (claqua): predict only at the last system turn k=N-1.
            All prior turns are observed context.

        Returns:
          predictions:   [M]         labels (M=N if per_turn, else M=1)
          probabilities: [M, C]      softmax probs
          confidences:   [M]         max-class probability
          full_paths:    [M][varies] full Viterbi path per prediction
          transition_info: [M]       which CRF transition matrix was used
        """
        user_repr = self._encode_utterances(user_ids)   # [N, d]
        sys_repr  = self._encode_utterances(system_ids) # [N, d]
        N = user_repr.size(0)

        if N == 0:
            return {"predictions": [], "probabilities": np.empty((0, self.num_classes)),
                    "confidences": [], "full_paths": [], "transition_info": []}

        predictions = []
        probabilities = []
        confidences = []
        full_paths = []
        transition_info = []

        # per_turn=True: predict at every system turn k=0..N-1
        # per_turn=False: predict only at k=N-1; prior turns use gold labels
        #   for CRF features (since they are observed, not predicted)
        k_range = range(N) if per_turn else range(N - 1, N)

        # Collect decoded labels for CRF feature computation at later turns.
        # When per_turn=False, prior system labels are known (observed turns),
        # so we seed with zeros (non-initiative) for turns 0..N-2 because
        # we don't have gold labels at inference time — but the model treats
        # them as fixed context, not predictions to evaluate.
        decoded_sys_labels = [] if per_turn else [0] * (N - 1)

        for k in k_range:
            # Context = [u_0, s_0, ..., u_{k-1}, s_{k-1}, u_k]
            context_parts = []
            for i in range(k):
                context_parts.append(user_repr[i])
                context_parts.append(sys_repr[i])
            context_parts.append(user_repr[k])  # current user turn

            context_seq = torch.stack(context_parts)  # [2k+1, d]
            T_minus_1 = context_seq.size(0)
            T = T_minus_1 + 1

            # Posterior encoder on context (turns 1..T-1)
            po_hidden = self.posterior_encoder(context_seq.unsqueeze(0))
            po_emissions = self.posterior_emission(
                po_hidden.squeeze(0))  # [T-1, C]

            # Prior encoder on context → approximated emission at T
            pr_hidden = self.prior_encoder(context_seq.unsqueeze(0))
            pr_emission_T = self.prior_emission(
                pr_hidden[:, -1, :]).squeeze(0)  # [C]

            # Combined: posterior for 1..T-1, prior for T
            combined_emissions = torch.cat(
                [po_emissions, pr_emission_T.unsqueeze(0)], dim=0)  # [T, C]

            # Multi-turn features using decoded labels from prior turns
            features = compute_multiturn_features(T, decoded_sys_labels + [0])

            # Record which transition matrix is selected for last transition
            last_feat = features[-1] if features else {}
            last_trans_name = self._describe_transition(last_feat)

            # Viterbi decode
            path, probs = self.crf.decode(combined_emissions, features)

            # y_T = last element of path
            pred_T = path[-1]
            prob_T = probs[-1]

            predictions.append(pred_T)
            probabilities.append(prob_T)
            confidences.append(float(prob_T[pred_T]))
            full_paths.append(path)
            transition_info.append({
                "pair_index": k,
                "transition_matrix": last_trans_name,
                "features": last_feat,
            })

            # Update decoded labels for use in subsequent turns' features
            decoded_sys_labels.append(pred_T)

        return {
            "predictions": predictions,
            "probabilities": np.array(probabilities),
            "confidences": confidences,
            "full_paths": full_paths,
            "transition_info": transition_info,
        }

    def _describe_transition(self, feat: Dict) -> str:
        """Return human-readable name of the transition matrix selected."""
        v = self.crf.variant
        if v == "VanillaCRF":
            return "G_global"
        w = feat.get("who2who", -1)
        if v == "Who2WhoCRF":
            return "G_u2s" if w == 0 else "G_s2u"
        if v in ("IntimeCRF", "DistanceCRF"):
            if w == 1:
                return "G_s2u"
            n = feat.get("init_count", 0)
            if n == 0:
                return "G_u2s_n0"
            if v == "IntimeCRF":
                return "G_u2s_n>0"
            d = feat.get("distance", 0)
            return "G_u2s_n>0_d=2" if d == 0 else "G_u2s_n>0_d>2"
        return "G_global"
# ══════════════════════════════════════════════════════════════════════════════
CRF_VARIANTS = {
    "vanillacrf": "VanillaCRF",
    "who2who": "Who2WhoCRF",
    "position": "PositionCRF",
    "who2who_position": "Who2Who_PositionCRF",
    "intime": "IntimeCRF",
    "distance": "DistanceCRF",
}

def create_model(
    baseline: str = "vanillacrf",
    num_classes: int = 2,
    encoder_name: str = "bert-base-uncased",
    encoder_path: str = None,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    lambda_mle: float = 1.0,
    gradient_checkpointing: bool = False,
    freeze_encoder_epochs: int = 0,
    class_weights: torch.Tensor = None,
) -> BiLSTMCRFModel:
    """
    Build a BiLSTM-CRF model with one encoder (BERT or ModernBERT, not both).

    Args:
        baseline: CRF variant name (vanillacrf, who2who, position,
                  who2who_position, intime, distance).
        encoder_name: HuggingFace hub identifier.
        encoder_path: Local filesystem path (takes precedence over encoder_name).
        class_weights: [num_classes] tensor of per-class weights for CRF loss.
                       Use compute_class_weights() to compute from label counts.
    """
    variant = CRF_VARIANTS.get(baseline.lower())
    if variant is None:
        raise ValueError(
            f"Unknown baseline '{baseline}'. Choose from {list(CRF_VARIANTS)}")
    return BiLSTMCRFModel(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_path=encoder_path,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lambda_mle=lambda_mle,
        crf_variant=variant,
        gradient_checkpointing=gradient_checkpointing,
        freeze_encoder_epochs=freeze_encoder_epochs,
        class_weights=class_weights,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Class weight computation
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(
    label_counts: Dict[int, int],
    num_classes: int,
    mode: str = "balanced",
) -> torch.Tensor:
    """
    Compute per-class weights for the CRF loss.

    Args:
        label_counts: {class_id: count} from Counter(all_labels)
        num_classes:   number of classes
        mode:
          "balanced"  — inverse frequency, normalized so weights sum to num_classes.
                        Equivalent to sklearn's compute_class_weight('balanced').
                        Example: clear=2869, ambiguous=394 → [0.57, 4.14]
          "sqrt"      — square-root of balanced weights (less aggressive).
                        Example: clear=2869, ambiguous=394 → [0.75, 2.03]

    Returns: [num_classes] tensor
    """
    total = sum(label_counts.get(c, 0) for c in range(num_classes))
    weights = []
    for c in range(num_classes):
        count = label_counts.get(c, 1)  # avoid division by zero
        if mode == "balanced":
            # w_c = total / (num_classes * count)
            w = total / (num_classes * count)
        elif mode == "sqrt":
            w = (total / (num_classes * count)) ** 0.5
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
