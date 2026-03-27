"""
GPU utilities for dual-A100 80 GB setup.

STRATEGY (addressing the original issue):
─────────────────────────────────────────
BiLSTM-CRF processes one conversation at a time (batch_size=1) and builds
an incremental sequence for each turn.  Splitting a single conversation
across two GPUs then merging is ERRONEOUS because:
  • CRF forward/backward requires the full sequence on one device.
  • Prior/Posterior encoders share hidden states turn-by-turn.

CORRECT dual-GPU approaches:
  A)  **Experiment-parallel**: GPU 0 trains encoder-X, GPU 1 trains encoder-Y
      (e.g., bert-base on gpu:0, modernbert on gpu:1 simultaneously).
  B)  **Fold-parallel**: For k-fold or multi-seed, run fold-0 on gpu:0 and
      fold-1 on gpu:1 concurrently.
  C)  **Pipeline within conversation** (future): encoder on gpu:0, CRF head
      on gpu:1 — but only when batch>1 with padded conversations.

Memory budget per A100 80 GB:
  bert-base-uncased   ≈  0.4 GB  →  plenty of room
  modernbert-base     ≈  0.6 GB
  modernbert-large    ≈  1.4 GB
  + BiLSTM + CRF      ≈  0.1 GB
  + activations (long conversations, max_len=8192) ≈ up to 40 GB with grad-ckpt
  ⇒ single GPU can handle modernbert-large with gradient_checkpointing + fp16.
"""

import os, torch, logging
logger = logging.getLogger(__name__)


def get_device_map(prefer_gpu: int = 0) -> torch.device:
    """Return a single device for this training run."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA — falling back to CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    gpu = min(prefer_gpu, n - 1)
    dev = torch.device(f"cuda:{gpu}")
    logger.info(f"Using {torch.cuda.get_device_name(gpu)} "
                f"({torch.cuda.get_device_properties(gpu).total_memory / 1e9:.1f} GB)")
    return dev


def log_gpu_memory(tag: str = ""):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        res   = torch.cuda.memory_reserved(i) / 1e9
        logger.info(f"[GPU {i}] {tag}  alloc={alloc:.2f} GB  reserved={res:.2f} GB")


def estimate_memory(encoder_name: str, max_length: int, num_turns: int) -> dict:
    """Rough memory estimate for a single conversation."""
    # Approximate model sizes (params × 4 bytes for fp32, ×2 for grads)
    PARAM_ESTIMATES = {
        "bert-base-uncased": 110e6,
        "bert-base-multilingual-cased": 178e6,
        "bert-large-uncased": 340e6,
        "answerdotai/ModernBERT-base": 150e6,
        "answerdotai/ModernBERT-large": 395e6,
	"/mnt/scratch/users/40645696/ModernBERT-base": 150e6

    }
    params = PARAM_ESTIMATES.get(encoder_name, 150e6)
    model_mem = params * 4 * 2 / 1e9  # fp32 + grads
    # Activation memory: rough estimate per token per layer
    hidden = 768 if "base" in encoder_name else 1024
    layers = 12 if "base" in encoder_name else 24
    act_per_token = hidden * layers * 4 / 1e9  # fp32
    act_total = act_per_token * max_length * num_turns * 2  # ×2 for bidir
    return {
        "model_gb": model_mem,
        "activation_gb": act_total,
        "total_gb": model_mem + act_total,
        "fits_80gb": (model_mem + act_total) < 72,  # leave 8 GB headroom
        "recommend_grad_ckpt": (model_mem + act_total) > 40,
        "recommend_fp16": (model_mem + act_total) > 30,
    }
