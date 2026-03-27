#!/usr/bin/env python3
"""
train.py — Unified training for BiLSTM-CRF with pluggable encoders.

ENCODER LOADING:
  --encoder bert-base-uncased                       # from HuggingFace hub
  --encoder_path /custom_path/ModernBERT-base       # from local directory
  When --encoder_path is a valid directory it takes precedence.
  A model uses ONE encoder at a time, never both BERT and ModernBERT together.

DUAL-GPU NOTE:
  This script runs on ONE GPU.  To use both A100s simultaneously, launch
  two processes:
    python scripts/train.py --gpu 0 --encoder bert-base-uncased  ...
    python scripts/train.py --gpu 1 --encoder_path /models/ModernBERT-base ...
"""

import os, sys, json, time, argparse, logging
from pathlib import Path
from collections import Counter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convqa_eval.models.bilstm_crf.model_final_imbalance import create_model, compute_class_weights
from convqa_eval.data.final_loader import (
    SIPDataset, sip_collate_single, tokenize_conversation,
    train_val_split, sample_train_fraction, load_benchmark,
)
from final_evaluator import compute_metrics, CLASS_NAMES
from convqa_eval.utils.seed import set_seed
from convqa_eval.utils.gpu import get_device_map, log_gpu_memory, estimate_memory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Resolve which path the tokenizer should load from
# ══════════════════════════════════════════════════════════════════════════════
def resolve_encoder_source(encoder_name: str, encoder_path: str = None) -> str:
    """Return the path/name that both the model and tokenizer should use."""
    if encoder_path and os.path.isdir(encoder_path):
        return encoder_path
    return encoder_name


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, device, scaler, use_amp,
                    per_turn=True):
    model.train()
    total_loss, crf_loss, mle_loss, n = 0, 0, 0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        u  = batch["user_utterance"].unsqueeze(0).to(device)
        s  = batch["system_utterance"].unsqueeze(0).to(device)
        ul = batch["user_I_label"].unsqueeze(0).to(device)
        sl = batch["system_I_label"].unsqueeze(0).to(device)
        if u.shape[1] == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                out = model(u, s, ul, sl, mode="train", per_turn=per_turn)
            scaler.scale(out["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(u, s, ul, sl, mode="train", per_turn=per_turn)
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += out["total_loss"].item()
        crf_loss   += out["loss_crf"].item()
        mle_loss   += out["loss_mle"].item()
        n += 1

    d = max(n, 1)
    return {"total": total_loss/d, "crf": crf_loss/d, "mle": mle_loss/d}


@torch.no_grad()
def evaluate(model, loader, device, num_classes, per_turn=True):
    """
    Return metric dict plus average val loss.

    per_turn=True:  predict at every system turn, compare all labels.
    per_turn=False: predict only at last system turn, compare last label.
    """
    model.eval()
    all_p, all_l, all_prob = [], [], []
    cum_loss, n_loss = 0.0, 0
    for batch in tqdm(loader, desc="  eval ", leave=False):
        u  = batch["user_utterance"].unsqueeze(0).to(device)
        s  = batch["system_utterance"].unsqueeze(0).to(device)
        ul = batch["user_I_label"].unsqueeze(0).to(device)
        sl = batch["system_I_label"].unsqueeze(0).to(device)
        if u.shape[1] == 0:
            continue
        # ── val loss (train-mode forward, no gradient) ───────────────────
        out_train = model(u, s, ul, sl, mode="train", per_turn=per_turn)
        cum_loss += out_train["total_loss"].item()
        n_loss   += 1
        # ── val predictions (inference mode) ─────────────────────────────
        out_inf = model(u, s, mode="inference", per_turn=per_turn)
        all_p.extend(out_inf["predictions"])
        if per_turn:
            all_l.extend(batch["system_I_label"].tolist())
        else:
            # Only the last system label is the target
            all_l.append(batch["system_I_label"][-1].item())
        if out_inf.get("probabilities") is not None and len(out_inf["probabilities"]) > 0:
            all_prob.append(out_inf["probabilities"])

    probs   = np.vstack(all_prob) if all_prob else None
    metrics = compute_metrics(all_l, all_p, probs, num_classes)
    metrics["val_loss"] = cum_loss / max(n_loss, 1)
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, args, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("roc_data", "classification_report",
                                 "confusion_matrix", "per_class",
                                 "false_positive_rates")},
        "args": vars(args),
    }, path)


# ══════════════════════════════════════════════════════════════════════════════
#  Training curves — 2×2 publication-quality figure
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history: dict, run_dir: Path, run_name: str):
    """
    2×2 figure:
      top-left:  train loss vs val loss
      top-right: val F1 + val accuracy
      bot-left:  val precision + val recall
      bot-right: val AUC-ROC
    Best-epoch markers on each panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=13, fontweight="bold")

    # ── (0,0) Loss ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", color="#2196F3",
            label="Train loss", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_loss"], "s--", color="#F44336",
            label="Val loss", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    best_idx = int(np.argmin(history["val_loss"]))
    ax.axvline(epochs[best_idx], color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"best={history['val_loss'][best_idx]:.4f}",
                xy=(epochs[best_idx], history["val_loss"][best_idx]),
                fontsize=8, color="#F44336")

    # ── (0,1) F1 + Accuracy ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["val_f1"], "o-", color="#4CAF50",
            label="Val F1 (weighted)", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_acc"], "s--", color="#FF9800",
            label="Val Accuracy", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation F1 & Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)
    best_f1_idx = int(np.argmax(history["val_f1"]))
    ax.axvline(epochs[best_f1_idx], color="gray", linestyle=":", alpha=0.5)
    ax.annotate(f"best F1={history['val_f1'][best_f1_idx]:.4f}",
                xy=(epochs[best_f1_idx], history["val_f1"][best_f1_idx]),
                fontsize=8, color="#4CAF50")

    # ── (1,0) Precision + Recall ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, history["val_precision"], "o-", color="#9C27B0",
            label="Val Precision", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_recall"], "s--", color="#00BCD4",
            label="Val Recall", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Precision & Recall")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── (1,1) AUC-ROC ───────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, history["val_auc"], "o-", color="#E91E63",
            label="Val AUC-ROC", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC")
    ax.set_title("Validation AUC-ROC")
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)
    if max(history["val_auc"]) > 0:
        best_auc_idx = int(np.argmax(history["val_auc"]))
        ax.annotate(f"best={history['val_auc'][best_auc_idx]:.4f}",
                    xy=(epochs[best_auc_idx], history["val_auc"][best_auc_idx]),
                    fontsize=8, color="#E91E63")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = run_dir / "training_curves.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Training curves saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Train BiLSTM-CRF baseline")

    # ── data ─────────────────────────────────────────────────────────────
    ap.add_argument("--benchmark", type=str, default="pacific",
                    help="Benchmark name (pacific / simmic / claqua)")
    ap.add_argument("--train_data", type=str, default=None,
                    help="Explicit train JSON (overrides --benchmark)")
    ap.add_argument("--val_data", type=str, default=None,
                    help="Explicit val JSON (else split from train)")
    ap.add_argument("--data_dir", type=str, default="benchmarks")
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="Fraction of train data to use (0.01-1.0)")
    ap.add_argument("--val_split", type=float, default=0.10,
                    help="Val fraction when no explicit val file")
    ap.add_argument("--per_turn", type=str, default=None,
                    choices=["true", "false"],
                    help="Per-turn evaluation (true=every system turn, "
                         "false=last turn only).  Default: auto from benchmark "
                         "(pacific/simmic=true, claqua=false)")

    # ── encoder ──────────────────────────────────────────────────────────
    ap.add_argument("--encoder", type=str, default="bert-base-uncased",
                    help="HF hub name: bert-base-uncased, "
                         "answerdotai/ModernBERT-base, etc.")
    ap.add_argument("--encoder_path", type=str, default=None,
                    help="LOCAL directory path, e.g. /custom_path/ModernBERT-base. "
                         "Takes precedence over --encoder when it is a valid dir.")
    ap.add_argument("--max_length", type=int, default=512)

    # ── model ────────────────────────────────────────────────────────────
    ap.add_argument("--baseline", type=str, default="vanillacrf",
                    choices=["vanillacrf", "who2who", "position",
                             "who2who_position", "intime", "distance"])
    ap.add_argument("--num_classes", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2,
                    help="BiLSTM layers (1 is faster and often sufficient)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lambda_mle", type=float, default=1.0,
                    help="Weight for MSE loss (paper Eq.8 uses 1.0)")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=0,
                    help="Freeze transformer encoder for first N epochs")
    ap.add_argument("--class_weight", type=str, default="none",
                    choices=["none", "balanced", "sqrt"],
                    help="Class weighting for CRF loss to handle imbalanced data. "
                         "'balanced' = inverse frequency (strong), "
                         "'sqrt' = sqrt of inverse frequency (moderate)")

    # ── training ─────────────────────────────────────────────────────────
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=2e-5,
                    help="Separate LR for encoder (when unfrozen)")
    ap.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "adamw", "adamax", "sgd"])
    ap.add_argument("--fp16", action="store_true",
                    help="Mixed-precision training")
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing in encoder")
    ap.add_argument("--seed", type=int, default=42)

    # ── hardware ─────────────────────────────────────────────────────────
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index (0 or 1 for dual-A100; -1 for CPU)")
    ap.add_argument("--fast_cpu", action="store_true",
                    help="Apply CPU-optimized defaults: max_length=128, "
                         "hidden_size=128, num_layers=1, encoder_lr=5e-5, "
                         "freeze_encoder_epochs=2, lr=2e-3")

    # ── output ───────────────────────────────────────────────────────────
    ap.add_argument("--output_dir", type=str, default="checkpoints")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--run_name", type=str, default=None,
                    help="Sub-folder name (auto-generated if None)")

    args = ap.parse_args()

    # ── apply --fast_cpu preset (override only unset/default values) ──────
    if args.fast_cpu:
        # These override defaults but NOT explicit user flags.
        # We detect "user set it" by checking if it differs from the
        # argparse default.  Since argparse doesn't track this natively,
        # we apply unconditionally — fast_cpu is meant as a full preset.
        args.gpu = -1
        if args.max_length == 512:      # user didn't set it
            args.max_length = 128
        if args.hidden_size == 256:
            args.hidden_size = 128
        if args.num_layers == 2:
            args.num_layers = 1
        if args.lr == 1e-3:
            args.lr = 2e-3
        if args.encoder_lr == 2e-5:
            args.encoder_lr = 5e-5
        if args.freeze_encoder_epochs == 0:
            args.freeze_encoder_epochs = 2
        log.info("--fast_cpu applied: max_length=%d hidden=%d layers=%d "
                 "lr=%.1e enc_lr=%.1e freeze=%d",
                 args.max_length, args.hidden_size, args.num_layers,
                 args.lr, args.encoder_lr, args.freeze_encoder_epochs)

    set_seed(args.seed)

    # ── resolve per_turn mode ────────────────────────────────────────────
    if args.per_turn is not None:
        args.per_turn = args.per_turn.lower() == "true"
    else:
        # Auto-detect from benchmark registry
        from convqa_eval.data.loader import BENCHMARK_REGISTRY
        bm = BENCHMARK_REGISTRY.get(args.benchmark, {})
        args.per_turn = bm.get("per_turn_default", True)
    log.info(f"Per-turn mode: {args.per_turn}  "
             f"({'every system turn' if args.per_turn else 'last system turn only'})")

    # ── resolve encoder source ───────────────────────────────────────────
    enc_source = resolve_encoder_source(args.encoder, args.encoder_path)
    enc_short  = Path(enc_source).name
    log.info(f"Encoder source: {enc_source}  (short={enc_short})")

    # ── run name & dirs ──────────────────────────────────────────────────
    if args.run_name is None:
        args.run_name = (f"{args.benchmark}_{enc_short}_{args.baseline}"
                         f"_cls{args.num_classes}_frac{args.train_fraction}"
                         f"_ml{args.max_length}_seed{args.seed}")
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── device ───────────────────────────────────────────────────────────
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = get_device_map(args.gpu)

    # ── memory estimate ──────────────────────────────────────────────────
    mem = estimate_memory(enc_source, args.max_length, num_turns=10)
    log.info(f"Memory estimate: {mem['total_gb']:.1f} GB  "
             f"(fits_80gb={mem['fits_80gb']}, "
             f"grad_ckpt_recommended={mem['recommend_grad_ckpt']})")
    if mem["recommend_grad_ckpt"] and not args.gradient_checkpointing:
        log.warning("Large model detected -- consider --gradient_checkpointing")
    if mem["recommend_fp16"] and not args.fp16:
        log.warning("Large model detected -- consider --fp16")

    # ── tokenizer (same source as encoder) ───────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(enc_source)
    log.info(f"Tokenizer loaded from: {enc_source}  max_length={args.max_length}")

    # ── data loading ─────────────────────────────────────────────────────
    if args.train_data:
        with open(args.train_data) as f:
            raw_train = json.load(f)
        if isinstance(raw_train, dict):
            raw_train = [raw_train]
    else:
        raw_train = load_benchmark(args.benchmark, "train", args.data_dir)
    log.info(f"Raw train conversations: {len(raw_train)}")

    raw_train = sample_train_fraction(raw_train, args.train_fraction, args.seed)
    log.info(f"After fraction={args.train_fraction}: {len(raw_train)} conversations")

    if args.val_data:
        with open(args.val_data) as f:
            raw_val = json.load(f)
        if isinstance(raw_val, dict):
            raw_val = [raw_val]
    else:
        raw_train, raw_val = train_val_split(raw_train, args.val_split, args.seed)
    log.info(f"Train: {len(raw_train)}  Val: {len(raw_val)}")

    log.info("Tokenizing...")
    train_proc = [tokenize_conversation(c, tokenizer, args.max_length,
                                        args.num_classes) for c in tqdm(raw_train)]
    val_proc   = [tokenize_conversation(c, tokenizer, args.max_length,
                                        args.num_classes) for c in tqdm(raw_val)]

    train_loader = DataLoader(SIPDataset(train_proc), batch_size=1,
                              shuffle=True, collate_fn=sip_collate_single)
    val_loader   = DataLoader(SIPDataset(val_proc), batch_size=1,
                              shuffle=False, collate_fn=sip_collate_single)

    # label stats
    all_labels = []
    for p in train_proc:
        all_labels.extend(p["system_I_label"].tolist())
    cnt = Counter(all_labels)
    names = CLASS_NAMES[args.num_classes]
    log.info("Train label distribution: " +
             ", ".join(f"{names[k]}={v}" for k, v in sorted(cnt.items())))

    # class weights for imbalanced data
    cw = None
    if args.class_weight != "none":
        cw = compute_class_weights(cnt, args.num_classes, mode=args.class_weight)
        log.info(f"Class weights ({args.class_weight}): " +
                 ", ".join(f"{names[i]}={cw[i]:.3f}" for i in range(args.num_classes)))

    # ── model ────────────────────────────────────────────────────────────
    model = create_model(
        baseline=args.baseline,
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        encoder_path=args.encoder_path,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lambda_mle=args.lambda_mle,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        class_weights=cw,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,} total, {train_params:,} trainable")
    log.info(f"Encoder hidden_size={model.utterance_encoder.hidden_size}  "
             f"source={model.utterance_encoder._source_desc}")

    # ── optimizer (differential LR) ──────────────────────────────────────
    encoder_params = list(model.utterance_encoder.parameters())
    other_params   = [p for n, p in model.named_parameters()
                      if "utterance_encoder" not in n]
    param_groups = [
        {"params": other_params,   "lr": args.lr},
        {"params": encoder_params, "lr": args.encoder_lr},
    ]
    OPT = {"adam": optim.Adam, "adamw": optim.AdamW,
           "adamax": optim.Adamax, "sgd": optim.SGD}
    opt_cls = OPT[args.optimizer]
    optimizer = (opt_cls(param_groups, momentum=0.9) if args.optimizer == "sgd"
                 else opt_cls(param_groups))

    scaler = GradScaler(enabled=args.fp16)

    # ── history ──────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_crf_loss": [], "train_mle_loss": [],
        "val_loss": [], "val_f1": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_auc": [],
    }
    best_f1 = 0.0

    log.info(f"\n{'='*72}")
    log.info(f"  Training: {args.run_name}")
    log.info(f"  Encoder:  {enc_source}")
    log.info(f"  Baseline: {args.baseline}  Classes: {args.num_classes}")
    log.info(f"  Epochs: {args.epochs}  LR: {args.lr}  Enc-LR: {args.encoder_lr}")
    log.info(f"  FP16: {args.fp16}  GradCkpt: {args.gradient_checkpointing}")
    log.info(f"{'='*72}\n")

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── training loop ────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.maybe_unfreeze_encoder(epoch)

        train_m = train_one_epoch(model, train_loader, optimizer,
                                  device, scaler, args.fp16,
                                  per_turn=args.per_turn)
        val_m   = evaluate(model, val_loader, device, args.num_classes,
                           per_turn=args.per_turn)
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["total"])
        history["train_crf_loss"].append(train_m["crf"])
        history["train_mle_loss"].append(train_m["mle"])
        history["val_loss"].append(val_m["val_loss"])
        history["val_f1"].append(val_m["f1"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_precision"].append(val_m["precision"])
        history["val_recall"].append(val_m["recall"])
        history["val_auc"].append(val_m["auc_roc"])

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_m['total']:.4f}  "
            f"val_loss={val_m['val_loss']:.4f}  "
            f"val_f1={val_m['f1']:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  "
            f"val_P={val_m['precision']:.4f}  "
            f"val_R={val_m['recall']:.4f}  "
            f"auc={val_m['auc_roc']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            save_checkpoint(model, optimizer, epoch, val_m, args,
                            run_dir / "best_f1_model.pt")
            log.info(f"  -> new best F1 = {best_f1:.4f}  (epoch {epoch})")

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_m, args,
                            run_dir / f"epoch_{epoch}.pt")

        log_gpu_memory(f"epoch {epoch}")

    save_checkpoint(model, optimizer, args.epochs, val_m, args,
                    run_dir / "last_model.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── plot training curves ─────────────────────────────────────────────
    try:
        plot_training_curves(history, run_dir, args.run_name)
    except Exception as e:
        log.warning(f"Could not plot training curves: {e}")

    log.info(f"\n{'='*72}")
    log.info(f"  Training complete.")
    log.info(f"  Best val F1     = {best_f1:.4f}")
    log.info(f"  Final val loss  = {history['val_loss'][-1]:.4f}")
    log.info(f"  Final val F1    = {history['val_f1'][-1]:.4f}")
    log.info(f"  Final val AUC   = {history['val_auc'][-1]:.4f}")
    log.info(f"  Encoder         = {enc_source}")
    log.info(f"  Checkpoints in  : {run_dir}")
    log.info(f"  Training curves : {run_dir / 'training_curves.png'}")
    log.info(f"{'='*72}\n")


if __name__ == "__main__":
    main()
