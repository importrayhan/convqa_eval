"""
Defense baselines for comparative evaluation.

  1. Unguarded — no defense, always passes
  2. PromptGuard-2 — DeBERTa-v2 jailbreak classifier (Meta)
  3. GammaGuard — lightweight residual adapter on frozen LLM

All implement: detect(text) -> {is_adversarial, score, latency_ms}
"""

import time
import logging
from typing import Dict, List
import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Unguarded baseline
# ══════════════════════════════════════════════════════════════════════════════
class UnguardedDefense:
    """No defense — always passes. Baseline for ASR measurement."""

    def __init__(self):
        self.name = "unguarded"

    def detect(self, text: str, **kwargs) -> Dict:
        return {"is_adversarial": False, "score": 0.0, "latency_ms": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# PromptGuard-2 (DeBERTa-v2 sequence classifier)
# ══════════════════════════════════════════════════════════════════════════════
class PromptGuardDefense:
    """
    Meta's Prompt-Guard-2 (86M DeBERTa-v2 for jailbreak detection).

    The model classifies text into: benign (0), injection (1), jailbreak (2).
    We flag injection OR jailbreak as adversarial.

    Limitation: 512 token context → we feed only the user's query text
    (the 'human' field), not the full conversation prompt.

    Config: DeBERTa-v2, hidden=384, heads=6, layers=12, vocab=128100.
    """

    def __init__(self, model_path: str):
        self.name = "prompt_guard_2"
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._load(model_path)

    def _load(self, path: str):
        try:
            from transformers import (
                AutoModelForSequenceClassification, AutoTokenizer,
            )
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.eval()
            self.device = next(self.model.parameters()).device
            log.info(f"PromptGuard-2 loaded from {path}")
        except Exception as e:
            log.warning(f"Could not load PromptGuard-2 from {path}: {e}")
            self.model = None

    def detect(self, text: str, **kwargs) -> Dict:
        import torch

        if self.model is None:
            return {"is_adversarial": False, "score": 0.0, "latency_ms": 0.0}

        t0 = time.time()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        latency = (time.time() - t0) * 1000

        # Labels: 0=benign, 1=injection, 2=jailbreak
        benign_prob = float(probs[0]) if len(probs) > 0 else 1.0
        injection_prob = float(probs[1]) if len(probs) > 1 else 0.0
        jailbreak_prob = float(probs[2]) if len(probs) > 2 else 0.0
        adv_score = injection_prob + jailbreak_prob

        return {
            "is_adversarial": adv_score > 0.5,
            "score": adv_score,
            "latency_ms": latency,
            "benign_prob": benign_prob,
            "injection_prob": injection_prob,
            "jailbreak_prob": jailbreak_prob,
        }


# ══════════════════════════════════════════════════════════════════════════════
# GammaGuard-style: Embedding-Level Residual Denoising
# ══════════════════════════════════════════════════════════════════════════════
class GammaGuardDefense:
    """
    Embedding-Level Residual Denoising & Attention-Correction, inspired
    by Gamma-Guard (EMNLP 2025).

    ARCHITECTURE:
      The defense learns a residual branch R(h) that estimates the
      adversarial perturbation in the hidden state:

        h_clean_hat = h_input - R(h_input)    (denoised hidden state)

      Training objective: given pairs (h_clean, h_adversarial) from the
      TRAINING set, minimize:

        L_denoise = ||R(h_adv) - (h_adv - h_clean)||²   (reconstruction)
        L_class   = CE(classifier(h - R(h)), y_clean)    (classification)
        L_total   = L_denoise + λ·L_class

      The denoiser learns what adversarial perturbation looks like in
      the hidden space.  At inference, the residual norm ||R(h)||
      serves as the adversarial score — large residuals indicate
      the input was perturbed.

    DETECTION:
      score = ||R(h_input)|| / ||h_input||   (relative perturbation)
      If score > threshold → adversarial

    WHY THIS WORKS:
      Clean inputs produce small R(h) ≈ 0 (no perturbation to remove).
      Adversarial inputs produce large R(h) (denoiser detects the
      perturbation pattern learned from training pairs).

    TRAINING DATA:
      Must use TRAINING set only.  Requires pairs of (h_clean, h_adv)
      from the same input run with and without the adversarial suffix.
    """
    import numpy as np
    def __init__(self, hidden_dim: int = 4096):
        self.name = "gamma_guard"
        self.denoiser = None
        self.classifier = None
        self.threshold = 0.0
        self.is_trained = False

    def build(self, input_dim: int, d_hidden: int = 128):
        """Build the denoiser and classifier."""
        import torch.nn as nn

        class ResidualDenoiser(nn.Module):
            """Estimates the adversarial perturbation vector."""
            def __init__(self, d_in, d_h):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_h),
                    nn.GELU(),
                    nn.LayerNorm(d_h),
                    nn.Linear(d_h, d_h),
                    nn.GELU(),
                    nn.Linear(d_h, d_in),  # output same dim as input
                )
            def forward(self, h):
                return self.net(h)  # R(h): estimated perturbation

        class BinaryClassifier(nn.Module):
            """Classifies denoised hidden state."""
            def __init__(self, d_in, d_h):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_h),
                    nn.GELU(),
                    nn.Linear(d_h, 2),
                )
            def forward(self, h):
                return self.net(h)

        self.denoiser = ResidualDenoiser(input_dim, d_hidden)
        self.classifier = BinaryClassifier(input_dim, d_hidden)

    def train_on_pairs(
        self,
        h_clean_list: List[np.ndarray],
        h_adv_list: List[np.ndarray],
        epochs: int = 100,
        lr: float = 5e-4,
        lambda_class: float = 0.5,
    ):
        """
        Train on paired (h_clean, h_adv) from TRAINING set.

        Loss = L_denoise + λ·L_class where:
          L_denoise = MSE(R(h_adv), h_adv - h_clean)
          L_class   = CE(classifier(h_adv - R(h_adv)), label=0)
                    + CE(classifier(h_clean), label=0)
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        if not h_clean_list or not h_adv_list:
            log.warning("GammaGuard: no training pairs"); return
        if len(h_clean_list) != len(h_adv_list):
            n = min(len(h_clean_list), len(h_adv_list))
            h_clean_list, h_adv_list = h_clean_list[:n], h_adv_list[:n]

        d = h_clean_list[0].shape[-1] if hasattr(h_clean_list[0], 'shape') else len(h_clean_list[0])
        self.build(d)

        H_clean = torch.tensor(np.vstack(h_clean_list), dtype=torch.float32)
        H_adv = torch.tensor(np.vstack(h_adv_list), dtype=torch.float32)
        target_perturbation = H_adv - H_clean  # what the suffix added

        params = list(self.denoiser.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        self.denoiser.train()
        self.classifier.train()

        for epoch in range(epochs):
            # Denoising loss: R(h_adv) should recover the perturbation
            R_adv = self.denoiser(H_adv)
            loss_denoise = F.mse_loss(R_adv, target_perturbation)

            # Classification loss: denoised adv should look clean
            h_denoised = H_adv - R_adv
            logits_denoised = self.classifier(h_denoised)
            logits_clean = self.classifier(H_clean)
            # Both should classify as "clean" (label=0)
            labels_clean = torch.zeros(len(H_clean), dtype=torch.long)
            loss_class = (F.cross_entropy(logits_denoised, labels_clean) +
                          F.cross_entropy(logits_clean, labels_clean)) / 2

            loss = loss_denoise + lambda_class * loss_class
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                log.info(f"  GammaGuard epoch {epoch}: "
                         f"denoise={loss_denoise:.4f} class={loss_class:.4f}")

        self.denoiser.eval()
        self.classifier.eval()

        # Calibrate threshold on training data
        with torch.no_grad():
            R_clean = self.denoiser(H_clean)
            R_adv = self.denoiser(H_adv)
            clean_norms = torch.norm(R_clean, dim=1) / torch.norm(H_clean, dim=1).clamp(min=1e-8)
            adv_norms = torch.norm(R_adv, dim=1) / torch.norm(H_adv, dim=1).clamp(min=1e-8)
            # Threshold: 95th percentile of clean residual norms
            self.threshold = float(torch.quantile(clean_norms, 0.95))

        self.is_trained = True
        log.info(f"GammaGuard trained: {epochs} epochs, "
                 f"threshold={self.threshold:.6f}, "
                 f"params={sum(p.numel() for p in params):,}")

    def detect(self, hidden_state=None, **kwargs) -> Dict:
        """Detect adversarial input by residual norm."""
        import torch
        import numpy as np
        if not self.is_trained or hidden_state is None:
            return {"is_adversarial": False, "score": 0.0, "latency_ms": 0.0}

        t0 = time.time()
        if isinstance(hidden_state, np.ndarray):
            x = torch.tensor(hidden_state.astype(np.float32))
        else:
            x = hidden_state.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            R = self.denoiser(x)
            residual_norm = float(torch.norm(R, dim=1)[0])
            input_norm = float(torch.norm(x, dim=1)[0])
            score = residual_norm / max(input_norm, 1e-8)

        latency = (time.time() - t0) * 1000

        return {
            "is_adversarial": score > self.threshold,
            "score": score,
            "latency_ms": latency,
            "residual_norm": residual_norm,
        }


