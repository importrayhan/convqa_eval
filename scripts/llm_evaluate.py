#!/usr/bin/env python3
"""
llm_evaluate.py — Zero-shot LLM evaluation for conversational ambiguity.

Modes:
  evaluate  — Run benchmark evaluation (batch inference + metrics)
  demo      — Interactive multi-turn prediction display
  gcg       — Adversarial robustness test with GCG suffix optimization

Usage:
  # Zero-shot evaluation
  python scripts/llm_evaluate.py evaluate \
      --model_path Qwen/Qwen2.5-0.5B-Instruct \
      --benchmark pacific --num_classes 2

  # Interactive demo
  python scripts/llm_evaluate.py demo \
      --model_path Qwen/Qwen2.5-0.5B-Instruct \
      --benchmark pacific --max_display 3

  # GCG adversarial test
  python scripts/llm_evaluate.py gcg \
      --model_path Qwen/Qwen2.5-0.5B-Instruct \
      --benchmark pacific --gcg_steps 50
"""

import sys, os, json, argparse, logging, time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convqa_eval.models.llm.prompts import (
    build_per_turn_prompts, parse_llm_label, LABEL_NAMES_2, LABEL_NAMES_4,
)
from convqa_eval.data.final_loader import (
    load_benchmark, train_val_split, sample_train_fraction,
    BENCHMARK_REGISTRY,
)
from final_evaluator import compute_metrics, CLASS_NAMES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_DIM = "\033[2m"
C_MAGENTA = "\033[95m"


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_tokenizer(model_path: str, device_map: str = "auto",
                              torch_dtype: str = "auto"):
    """Load a HuggingFace causal LM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Model loaded: {model.config._name_or_path}  "
             f"params={sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def generate_response(model, tokenizer, messages: list,
                      max_new_tokens: int = 16) -> str:
    """Generate a response from chat messages."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0)

    # Strip input tokens
    gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate mode
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluate(args, model, tokenizer):
    """Batch evaluation on benchmark."""
    names = CLASS_NAMES[args.num_classes]

    # Load data
    if args.test_data:
        with open(args.test_data) as f:
            raw_test = json.load(f)
        if isinstance(raw_test, dict): raw_test = [raw_test]
    else:
        try:
            raw_test = load_benchmark(args.benchmark, "test", args.data_dir)
        except FileNotFoundError:
            raw_all = load_benchmark(args.benchmark, "train", args.data_dir)
            _, raw_test = train_val_split(raw_all, 0.1, args.seed)

    log.info(f"Test conversations: {len(raw_test)}")

    all_preds, all_labels, all_raw_outputs = [], [], []
    unparseable = 0

    for conv in tqdm(raw_test, desc="Evaluating"):
        prompts = build_per_turn_prompts(
            conv, args.num_classes, args.per_turn)

        for messages, gpt_idx, gold_label in prompts:
            raw_output = generate_response(model, tokenizer, messages,
                                           max_new_tokens=args.max_new_tokens)
            pred = parse_llm_label(raw_output, args.num_classes)
            if pred == -1:
                unparseable += 1
                pred = 0  # default to clear
            all_preds.append(pred)
            all_labels.append(gold_label)
            all_raw_outputs.append(raw_output)

    log.info(f"Total predictions: {len(all_preds)}  Unparseable: {unparseable}")

    # Metrics
    metrics = compute_metrics(all_labels, all_preds, None, args.num_classes)

    log.info(f"\n{'='*72}")
    log.info(f"  LLM Zero-Shot Results — {args.benchmark}")
    log.info(f"  Model: {args.model_path}")
    log.info(f"{'='*72}")
    log.info(f"  Accuracy  = {metrics['accuracy']:.4f}")
    log.info(f"  Precision = {metrics['precision']:.4f}")
    log.info(f"  Recall    = {metrics['recall']:.4f}")
    log.info(f"  F1 (wt)   = {metrics['f1']:.4f}")
    log.info(f"  AUC-ROC   = {metrics['auc_roc']:.4f}")
    log.info(f"\n{metrics['classification_report']}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "num_classes": args.num_classes,
        "per_turn": args.per_turn,
        "total_predictions": len(all_preds),
        "unparseable": unparseable,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("roc_data", "classification_report")},
    }
    out_path = out_dir / f"llm_{args.benchmark}_{Path(args.model_path).name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Results saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Demo mode — interactive multi-turn prediction display
# ══════════════════════════════════════════════════════════════════════════════
def run_demo(args, model, tokenizer):
    """Interactive display showing multi-turn predictions."""
    names = CLASS_NAMES[args.num_classes]

    raw_data = load_benchmark(args.benchmark, "test", args.data_dir)
    if len(raw_data) > args.max_display:
        raw_data = raw_data[:args.max_display]

    for conv_idx, conv in enumerate(raw_data):
        convs = conv.get("conversations", conv.get("turns", []))
        print(f"\n{'━'*78}")
        print(f"{C_BOLD}  Conversation {conv_idx+1}  {C_DIM}(model: {args.model_path}){C_RESET}")
        print(f"{'━'*78}")

        prompts = build_per_turn_prompts(
            conv, args.num_classes, args.per_turn)

        prompt_map = {gpt_idx: (msgs, label) for msgs, gpt_idx, label in prompts}

        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", "")[:120]
            turn_idx = convs.index(turn)

            if role == "human":
                print(f"  {C_CYAN}👤 USER:{C_RESET}  {value}")
            elif role == "observation":
                print(f"  {C_DIM}📋 OBS:{C_RESET}  {value[:80]}")
            elif role == "gpt":
                if turn_idx in prompt_map:
                    messages, gold = prompt_map[turn_idx]

                    # Normal prediction
                    raw_out = generate_response(model, tokenizer, messages,
                                                max_new_tokens=args.max_new_tokens)
                    pred = parse_llm_label(raw_out, args.num_classes)
                    if pred == -1: pred = 0

                    match = "✓" if pred == gold else "✗"
                    match_col = C_GREEN if pred == gold else C_RED
                    pred_name = names[pred] if 0 <= pred < len(names) else "?"
                    gold_name = names[gold] if 0 <= gold < len(names) else "?"

                    print(f"  {C_BOLD}🤖 SYS:{C_RESET}   {value}")
                    print(f"        pred={C_YELLOW}{pred_name}{C_RESET}  "
                          f"gold={gold_name}  "
                          f"raw=\"{raw_out[:30]}\"  "
                          f"{match_col}{match}{C_RESET}")

                    # Adversarial prediction (if --gcg_suffix provided)
                    if args.gcg_suffix:
                        from convqa_eval.models.llm.prompts import build_classification_prompt
                        adv_messages = build_classification_prompt(
                            conv, turn_idx, args.num_classes,
                            adversarial_suffix=args.gcg_suffix)
                        adv_out = generate_response(model, tokenizer, adv_messages,
                                                    max_new_tokens=args.max_new_tokens)
                        adv_pred = parse_llm_label(adv_out, args.num_classes)
                        if adv_pred == -1: adv_pred = 0
                        adv_name = names[adv_pred] if 0 <= adv_pred < len(names) else "?"
                        flipped = pred != adv_pred
                        flip_str = f"{C_RED}FLIPPED{C_RESET}" if flipped else f"{C_GREEN}stable{C_RESET}"
                        print(f"        {C_MAGENTA}+GCG:{C_RESET} pred={adv_name}  "
                              f"raw=\"{adv_out[:30]}\"  {flip_str}")
                else:
                    print(f"  {C_DIM}🤖 SYS (context):{C_RESET} {value}")
            elif role == "function_call":
                print(f"  {C_DIM}⚙️  CALL{C_RESET}")

        print()


# ══════════════════════════════════════════════════════════════════════════════
# GCG attack mode
# ══════════════════════════════════════════════════════════════════════════════
def run_gcg(args, model, tokenizer):
    """Run GCG adversarial suffix optimization — batch or single mode."""
    from convqa_eval.models.llm.gcg_attack import run_gcg_attack, run_gcg_batch
    from convqa_eval.models.llm.prompts import build_classification_prompt

    names = CLASS_NAMES[args.num_classes]
    raw_data = load_benchmark(args.benchmark, "train", args.data_dir)

    if args.gcg_n_samples > 1:
        # ── Batch mode: universal suffix across multiple prompts ──────
        import random as rand_mod
        rand_mod.seed(args.seed)
        if len(raw_data) > args.gcg_n_samples:
            raw_data = rand_mod.sample(raw_data, args.gcg_n_samples)

        all_messages = []
        all_labels = []
        for conv in raw_data:
            prompts = build_per_turn_prompts(conv, args.num_classes, args.per_turn)
            for messages, gpt_idx, label in prompts:
                all_messages.append(messages)
                all_labels.append(label)

        # Target: flip all to the adversarial class
        # Use the majority class's opposite as target
        from collections import Counter as Ctr
        cnt = Ctr(all_labels)
        majority = cnt.most_common(1)[0][0]
        target_label = 1 - majority if args.num_classes == 2 else (majority + 2) % 4
        target_output = str(target_label)

        log.info(f"GCG Batch Attack:")
        log.info(f"  Prompts: {len(all_messages)}")
        log.info(f"  Target label: {target_label} ({names[target_label]})")
        log.info(f"  Steps: {args.gcg_steps}  Suffix length: {args.gcg_suffix_len}")

        t0 = time.time()
        best_suffix, loss_history, successes = run_gcg_batch(
            model, tokenizer, all_messages, target_output,
            num_steps=args.gcg_steps,
            suffix_length=args.gcg_suffix_len,
            batch_size=args.gcg_batch_size,
            top_k=args.gcg_top_k)
        elapsed = time.time() - t0

        log.info(f"\nGCG batch completed in {elapsed:.1f}s")
        log.info(f"  Best suffix: \"{best_suffix}\"")
        log.info(f"  Final loss: {loss_history[-1]:.4f}")
        log.info(f"  Final success: {successes[-1]}/{len(all_messages)}")

        result = {
            "mode": "batch",
            "benchmark": args.benchmark,
            "model_path": args.model_path,
            "n_prompts": len(all_messages),
            "target_label": target_label,
            "suffix": best_suffix,
            "gcg_steps": args.gcg_steps,
            "loss_history": loss_history,
            "successes_history": successes,
            "elapsed_seconds": elapsed,
        }

    else:
        # ── Single mode: attack one conversation ─────────────────────
        conv = raw_data[min(args.gcg_conv_idx, len(raw_data) - 1)]
        prompts = build_per_turn_prompts(conv, args.num_classes, args.per_turn)
        if not prompts:
            log.error("No turns to attack"); return

        messages, gpt_idx, gold_label = prompts[-1]
        target_label = (1 - gold_label) if args.num_classes == 2 else (gold_label + 2) % 4
        target_output = str(target_label)

        clean_out = generate_response(model, tokenizer, messages,
                                      max_new_tokens=args.max_new_tokens)
        clean_pred = parse_llm_label(clean_out, args.num_classes)
        log.info(f"  Clean: {clean_pred}  Gold: {gold_label}  Target: {target_label}")

        t0 = time.time()
        best_suffix, loss_history = run_gcg_attack(
            model, tokenizer, messages, target_output,
            num_steps=args.gcg_steps, suffix_length=args.gcg_suffix_len,
            batch_size=args.gcg_batch_size, top_k=args.gcg_top_k)
        elapsed = time.time() - t0

        adv_messages = build_classification_prompt(
            conv, gpt_idx, args.num_classes, adversarial_suffix=best_suffix)
        adv_out = generate_response(model, tokenizer, adv_messages,
                                    max_new_tokens=args.max_new_tokens)
        adv_pred = parse_llm_label(adv_out, args.num_classes)

        log.info(f"  Clean={clean_pred}  Adv={adv_pred}  "
                 f"Flipped={clean_pred != adv_pred}")

        result = {
            "mode": "single",
            "benchmark": args.benchmark, "model_path": args.model_path,
            "gold_label": gold_label, "target_label": target_label,
            "clean_pred": clean_pred, "adversarial_pred": adv_pred,
            "flipped": clean_pred != adv_pred,
            "suffix": best_suffix, "loss_history": loss_history,
            "elapsed_seconds": elapsed,
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gcg_{args.benchmark}_{Path(args.model_path).name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Results saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Defend mode — RGRC defense evaluation
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Defend mode — suffix-sensitivity defense
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Defend mode — anomaly detection on suffix-induced distribution shifts
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Defend mode — suffix-sensitivity anomaly detection
# ══════════════════════════════════════════════════════════════════════════════
def _load_suffix_choices(suffix_file: str, single_suffix: str) -> List[str]:
    """Load suffix choices from file or use single suffix."""
    suffixes = []
    if suffix_file and Path(suffix_file).exists():
        p = Path(suffix_file)
        if p.suffix == ".json":
            import json as _json
            data = _json.load(open(p))
            if isinstance(data, list):
                suffixes = [str(s) for s in data if s]
            elif isinstance(data, dict):
                suffixes = [str(v) for v in data.values() if v]
        else:
            suffixes = [l.strip() for l in open(p) if l.strip()]
    if single_suffix and single_suffix not in suffixes:
        suffixes.append(single_suffix)
    return suffixes if suffixes else [single_suffix or ""]


def _compute_shift_old(h_clean, h_adv, logits_clean, logits_adv,
                   d_hat, r_features, num_classes):
    """Compute the 16-dim shift vector between clean and suffix runs."""
    proj_c = float(np.dot(h_clean, d_hat))
    proj_a = float(np.dot(h_adv, d_hat))
    hn_c, hn_a = np.linalg.norm(h_clean), np.linalg.norm(h_adv)
    cos_c = proj_c / max(hn_c, 1e-8)
    cos_a = proj_a / max(hn_a, 1e-8)
    h_sim = float(np.dot(h_clean, h_adv) / (max(hn_c, 1e-8) * max(hn_a, 1e-8)))

    n_lab = max(num_classes, 4)
    cp = np.clip([logits_clean.get(f"label_{i}_prob", 1.0/n_lab)
                  for i in range(n_lab)], 1e-8, 1)
    ap = np.clip([logits_adv.get(f"label_{i}_prob", 1.0/n_lab)
                  for i in range(n_lab)], 1e-8, 1)
    cp, ap = cp/cp.sum(), ap/ap.sum()
    kl = float(np.sum(cp * np.log(cp / ap)))
    js = float(0.5*np.sum(cp*np.log(2*cp/(cp+ap))) +
               0.5*np.sum(ap*np.log(2*ap/(cp+ap))))
    pred_c, pred_a = int(np.argmax(cp)), int(np.argmax(ap))
    flipped = 1.0 if pred_c != pred_a else 0.0

    return {
        "cos_delta": cos_c - cos_a,
        "proj_delta": proj_c - proj_a,
        "h_similarity": h_sim,
        "logit_kl": kl,
        "logit_js": js,
        "pred_flipped": flipped,
        "entropy_clean": logits_clean.get("label_entropy", 0),
        "entropy_adv": logits_adv.get("label_entropy", 0),
        "entropy_delta": logits_clean.get("label_entropy", 0) - logits_adv.get("label_entropy", 0),
        "top1_prob_clean": logits_clean.get("top1_prob", 0),
        "top1_prob_adv": logits_adv.get("top1_prob", 0),
        "logit_gap_clean": logits_clean.get("logit_gap", 0),
        "logit_gap_adv": logits_adv.get("logit_gap", 0),
        "r_avg_idf": r_features.get("avg_idf", 0),
        "r_coverage": r_features.get("coverage", 0),
    }, flipped > 0.5, pred_c, pred_a


def _compute_shift(h_clean, h_adv, logits_clean, logits_adv,
                   d_hat, r_features, num_classes):
    """Compute 14-dim shift vector. pred_flipped EXCLUDED (reviewer fix).

    Returns: (shift_dict, is_flipped: bool, pred_clean: int, pred_adv: int)
    The bool is_flipped is used ONLY for partitioning, never as a feature.
    """
    proj_c = float(np.dot(h_clean, d_hat))
    proj_a = float(np.dot(h_adv, d_hat))
    hn_c, hn_a = np.linalg.norm(h_clean), np.linalg.norm(h_adv)
    cos_c = proj_c / max(hn_c, 1e-8)
    cos_a = proj_a / max(hn_a, 1e-8)
    h_sim = float(np.dot(h_clean, h_adv) / (max(hn_c, 1e-8) * max(hn_a, 1e-8)))

    n_lab = max(num_classes, 4)
    cp = np.clip([logits_clean.get(f"label_{i}_prob", 1.0/n_lab)
                  for i in range(n_lab)], 1e-8, 1)
    ap = np.clip([logits_adv.get(f"label_{i}_prob", 1.0/n_lab)
                  for i in range(n_lab)], 1e-8, 1)
    cp, ap = cp/cp.sum(), ap/ap.sum()
    kl = float(np.sum(cp * np.log(cp / ap)))
    js = float(0.5*np.sum(cp*np.log(2*cp/(cp+ap))) +
               0.5*np.sum(ap*np.log(2*ap/(cp+ap))))
    pred_c, pred_a = int(np.argmax(cp)), int(np.argmax(ap))
    flipped = pred_c != pred_a  # bool, NOT included in shift dict

    shift = {
        "cos_delta": cos_c - cos_a,
        "proj_delta": proj_c - proj_a,
        "h_similarity": h_sim,
        "logit_kl": kl,
        "logit_js": js,
        # *** pred_flipped REMOVED — reviewer fix #1 ***
        "entropy_clean": logits_clean.get("label_entropy", 0),
        "entropy_adv": logits_adv.get("label_entropy", 0),
        "entropy_delta": logits_clean.get("label_entropy", 0) - logits_adv.get("label_entropy", 0),
        "top1_prob_clean": logits_clean.get("top1_prob", 0),
        "top1_prob_adv": logits_adv.get("top1_prob", 0),
        "logit_gap_clean": logits_clean.get("logit_gap", 0),
        "logit_gap_adv": logits_adv.get("logit_gap", 0),
        "r_avg_idf": r_features.get("avg_idf", 0),
        "r_coverage": r_features.get("coverage", 0),
    }
    return shift, flipped, pred_c, pred_a

class ShiftDenoiser:
    """Learns D(h_suffix) → h_clean. Includes train/val visualization."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.train_history = {"loss": [], "cos_train": [], "cos_val": []}

    def train(self, h_clean_list, h_suffix_list, epochs=100, lr=5e-4,
              val_fraction=0.15):
        """Train with held-out validation for denoiser quality check."""
        import torch
        import torch.nn as nn

        n = min(len(h_clean_list), len(h_suffix_list))
        if n < 4:
            log.warning("Denoiser: too few pairs"); return

        # Train/val split
        n_val = max(2, int(n * val_fraction))
        n_train = n - n_val
        d = h_clean_list[0].shape[-1]

        H_c = torch.tensor(np.vstack(h_clean_list[:n]), dtype=torch.float32)
        H_s = torch.tensor(np.vstack(h_suffix_list[:n]), dtype=torch.float32)
        H_c_train, H_c_val = H_c[:n_train], H_c[n_train:]
        H_s_train, H_s_val = H_s[:n_train], H_s[n_train:]

        bottleneck = min(d, 256)
        self.model = nn.Sequential(
            nn.Linear(d, bottleneck), nn.GELU(), nn.LayerNorm(bottleneck),
            nn.Linear(bottleneck, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, d))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            R = self.model(H_s_train)
            loss = nn.functional.mse_loss(H_s_train - R, H_c_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            self.train_history["loss"].append(float(loss.item()))
            with torch.no_grad():
                cos_tr = nn.functional.cosine_similarity(
                    H_s_train - self.model(H_s_train), H_c_train, dim=1).mean()
                cos_vl = nn.functional.cosine_similarity(
                    H_s_val - self.model(H_s_val), H_c_val, dim=1).mean()
                self.train_history["cos_train"].append(float(cos_tr))
                self.train_history["cos_val"].append(float(cos_vl))

            if epoch % 25 == 0:
                log.info(f"  Denoiser epoch {epoch}: MSE={loss.item():.6f}  "
                         f"cos_train={cos_tr:.4f}  cos_val={cos_vl:.4f}")

        self.model.eval()
        self.is_trained = True

        final_cos_tr = self.train_history["cos_train"][-1]
        final_cos_vl = self.train_history["cos_val"][-1]
        log.info(f"  Denoiser trained: n_train={n_train}, n_val={n_val}  "
                 f"cos_train={final_cos_tr:.4f}  cos_val={final_cos_vl:.4f}")

    def denoise(self, h_input):
        """Estimate h_clean from h_input."""
        import torch
        if not self.is_trained:
            return h_input
        x = torch.tensor(h_input, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return (x - self.model(x)).squeeze(0).numpy()

    def plot_training(self, output_dir="llm_results", model_name=""):
        """Visualize denoiser training: loss curve + train/val cosine."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from pathlib import Path

        rcParams.update({"font.family": "serif", "font.size": 11})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        epochs = range(len(self.train_history["loss"]))

        # Panel 1: Loss curve
        ax = axes[0]
        ax.plot(epochs, self.train_history["loss"], color="#1565C0", lw=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title("(a) Denoiser Training Loss", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.15)
        ax.set_yscale("log")

        # Panel 2: Train vs Val cosine similarity
        ax = axes[1]
        ax.plot(epochs, self.train_history["cos_train"],
                color="#2E7D32", lw=2, label="Train")
        ax.plot(epochs, self.train_history["cos_val"],
                color="#C62828", lw=2, label="Validation", linestyle="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("cos(D(h_suffix), h_clean)", fontsize=12)
        ax.set_title("(b) Denoiser Recovery Quality", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(0, 1.05)

        # Annotate final values
        final_tr = self.train_history["cos_train"][-1]
        final_vl = self.train_history["cos_val"][-1]
        ax.axhline(final_vl, color="#C62828", alpha=0.3, ls=":")
        ax.text(len(epochs)*0.7, final_vl + 0.03,
                f"val={final_vl:.3f}", fontsize=10, color="#C62828")

        gap = final_tr - final_vl
        if gap > 0.05:
            ax.text(len(epochs)*0.7, final_tr - 0.05,
                    f"gap={gap:.3f} (overfit risk)",
                    fontsize=9, color="#FF6F00")

        fig.suptitle(f"Shift Denoiser — {model_name}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "denoiser_training.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info(f"Denoiser plot: {p}")
        return str(p)


def run_defend(args, model, tokenizer):
    """Suffix-sensitivity anomaly detection with multi-suffix support
    and interactive demo mode."""
    import re as re_mod
    from convqa_eval.models.llm.ambiguity_direction import (
        extract_hidden_states, compute_ambiguity_direction,
        extract_prediction_logits,
    )
    from convqa_eval.models.llm.rgrc_defense import (
        compute_channel_r, evaluate_defense, plot_triangulated_defense,
        TriangulatedSeparator,
    )
    from convqa_eval.models.llm.prompts import (
        build_classification_prompt, parse_llm_label, LABEL_NAMES_2, LABEL_NAMES_4,
    )
    from QPP_measures import (
        PseudoCollection, QPPScorer, parse_sip_for_qpp,
    )
    import random as rand_mod
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler

    rand_mod.seed(args.seed)
    names = CLASS_NAMES[args.num_classes]

    # Load suffix choices
    suffix_choices = _load_suffix_choices(
        getattr(args, "suffix_file", ""), args.adv_suffix)
    train_suffix = args.adv_suffix or suffix_choices[0]
    if not train_suffix:
        log.error("Need --adv_suffix or --suffix_file. Run gcg first.")
        return
    log.info(f"Train suffix: \"{train_suffix[:50]}...\"")
    log.info(f"Test suffix pool: {len(suffix_choices)} choices")

    # ── Load data ────────────────────────────────────────────────────
    raw_train = load_benchmark(args.benchmark, "train", args.data_dir)
    try:
        raw_test = load_benchmark(args.benchmark, "test", args.data_dir)
    except FileNotFoundError:
        raw_train, raw_test = train_val_split(raw_train, 0.1, args.seed)
    if len(raw_train) > args.max_samples:
        raw_train = rand_mod.sample(raw_train, args.max_samples)
    if len(raw_test) > args.max_samples:
        raw_test = rand_mod.sample(raw_test, args.max_samples)

    collection = PseudoCollection()
    for conv in raw_train + raw_test:
        for turn in conv.get("conversations", conv.get("turns", [])):
            if turn.get("from", turn.get("role", "")) == "observation":
                t = turn.get("value", "")
                if t.strip():
                    collection.add_document(t)
    scorer = QPPScorer(collection)

    # Direction
    if args.direction_path and Path(args.direction_path).exists():
        direction = np.load(args.direction_path)
        layer = args.direction_layer
        if layer < 0:
            m = re_mod.search(r"layer(\d+)", args.direction_path)
            layer = int(m.group(1)) if m else model.config.num_hidden_layers // 2
    else:
        log.info("Computing direction...")
        cl_m, am_m = [], []
        for conv in raw_train[:30]:
            for msgs, _, lab in build_per_turn_prompts(conv, args.num_classes, args.per_turn):
                (cl_m if lab == 0 else am_m).append(msgs)
        cl_m, am_m = cl_m[:25], am_m[:25]
        layer = model.config.num_hidden_layers * 2 // 3
        hc = extract_hidden_states(model, tokenizer, cl_m, layers=[layer], pool=args.pool)
        ha = extract_hidden_states(model, tokenizer, am_m, layers=[layer], pool=args.pool)
        direction, sep = compute_ambiguity_direction(hc[layer], ha[layer])
    d_hat = direction / max(np.linalg.norm(direction), 1e-8)

    # ── Helper: process one turn ─────────────────────────────────────
    def process_turn(conv, rec, suffix):
        convs_list = conv.get("conversations", conv.get("turns", []))
        gpt_indices = [i for i, c in enumerate(convs_list)
                       if c.get("from", c.get("role", "")) == "gpt"]
        if rec["turn_idx"] >= len(gpt_indices):
            return None
        gpt_idx = gpt_indices[rec["turn_idx"]]
        clean_msgs = build_classification_prompt(conv, gpt_idx, args.num_classes)
        hc_d = extract_hidden_states(model, tokenizer, [clean_msgs],
                                      layers=[layer], pool=args.pool)
        if layer not in hc_d or not len(hc_d[layer]):
            return None
        h_clean = hc_d[layer][0]
        logits_clean = extract_prediction_logits(model, tokenizer, clean_msgs,
                                                  args.num_classes)
        adv_msgs = build_classification_prompt(conv, gpt_idx, args.num_classes,
                                                adversarial_suffix=suffix)
        ha_d = extract_hidden_states(model, tokenizer, [adv_msgs],
                                      layers=[layer], pool=args.pool)
        if layer not in ha_d or not len(ha_d[layer]):
            return None
        h_adv = ha_d[layer][0]
        logits_adv = extract_prediction_logits(model, tokenizer, adv_msgs,
                                                args.num_classes)
        r = compute_channel_r(rec["query"], rec["observations"], scorer=scorer)
        shift, flipped, pred_c, pred_a = _compute_shift(
            h_clean, h_adv, logits_clean, logits_adv,
            d_hat, r, args.num_classes)
        return {
            "shift": shift, "flipped": flipped,
            "pred_clean": pred_c, "pred_adv": pred_a,
            "query": rec["query"], "label": rec["label"],
            "logits_clean": logits_clean, "logits_adv": logits_adv,
            "h_clean": h_clean, "h_adv": h_adv,
            "turn_idx": rec["turn_idx"],
            "observations": rec.get("observations", []),
        }

    # ── Phase 1: Train on TRAIN set (single suffix) ─────────────────
    log.info(f"\nPhase 1: Train shifts ({len(raw_train)} convs)...")
    train_shifts, train_flipped = [], []
    train_h_clean_pairs, train_h_adv_pairs = [], []  # for GammaGuard
    for conv in tqdm(raw_train, desc="Train"):
        records = parse_sip_for_qpp(conv, args.num_classes)
        if not records: continue
        targets = [records[-1]] if not args.per_turn else records
        for rec in targets:
            out = process_turn(conv, rec, train_suffix)
            if out:
                train_shifts.append(out["shift"])
                train_flipped.append(out["flipped"])
                train_h_clean_pairs.append(out["h_clean"])
                train_h_adv_pairs.append(out["h_adv"])

    #log.info("\\nTraining shift denoiser D(h_suffix) → h_clean...")
    denoiser = ShiftDenoiser()
    denoiser.train(train_h_clean_pairs, train_h_adv_pairs, epochs=100)

    shift_keys = sorted(train_shifts[0].keys()) if train_shifts else []
    X_train = np.array([[s[k] for k in shift_keys] for s in train_shifts])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    normal_mask = np.array([not f for f in train_flipped])
    log.info(f"Train: {len(train_shifts)} shifts, "
             f"normal={normal_mask.sum()}, flipped={sum(train_flipped)}")

    if args.model_type == "isolation_forest":
        contam = max(0.01, sum(train_flipped) / max(len(train_flipped), 1))
        detector = IsolationForest(contamination=contam, random_state=42,
                                    n_estimators=200)
        detector.fit(X_train_s[normal_mask] if normal_mask.any() else X_train_s)
    elif args.model_type == "elliptic":
        contam = max(0.01, min(0.49, sum(train_flipped)/max(len(train_flipped),1)))
        detector = EllipticEnvelope(contamination=contam, random_state=42)
        detector.fit(X_train_s[normal_mask] if normal_mask.any() else X_train_s)
    else:
        from sklearn.svm import SVC
        detector = SVC(kernel="rbf", probability=True, class_weight="balanced")
        detector.fit(X_train_s, np.array([int(f) for f in train_flipped]))

    # ── Phase 2: Test (random suffix from pool) ──────────────────────
    log.info(f"\nPhase 2: Test ({len(raw_test)} convs, "
             f"{len(suffix_choices)} suffix choices)...")
    def process_turn_inference(conv, rec, suffix, denoiser):
        '''Single-pass: run LLM on prompt+suffix, denoise to get h_clean.

        Args:
            conv: conversation dict
            rec: turn record from parse_sip_for_qpp
            suffix: the test adversarial suffix to probe with
            denoiser: trained ShiftDenoiser
        '''
        convs_list = conv.get("conversations", conv.get("turns", []))
        gpt_indices = [i for i, c in enumerate(convs_list)
                       if c.get("from", c.get("role", "")) == "gpt"]
        if rec["turn_idx"] >= len(gpt_indices):
            return None
        gpt_idx = gpt_indices[rec["turn_idx"]]

        # Build prompt WITH the test suffix (simulates attacked input)
        adv_msgs = build_classification_prompt(
            conv, gpt_idx, args.num_classes,
            adversarial_suffix=suffix)

        # Single LLM pass on the suffix-appended prompt
        h_dict = extract_hidden_states(model, tokenizer, [adv_msgs],
                                        layers=[layer], pool=args.pool)
        if layer not in h_dict or not len(h_dict[layer]):
            return None
        h_input = h_dict[layer][0]
        logits_input = extract_prediction_logits(
            model, tokenizer, adv_msgs, args.num_classes)

        # Estimate h_clean via denoiser (replaces second LLM pass)
        h_clean_est = denoiser.denoise(h_input)

        # Build "denoised logits" — use the input logits for both
        # (the denoiser operates in hidden space, not logit space;
        #  logit-level features compare input vs denoised-estimate)
        logits_clean_est = logits_input  # approximation

        r = compute_channel_r(rec["query"], rec["observations"],
                               scorer=scorer)
        shift, flipped, pred_c, pred_a = _compute_shift(
            h_clean_est, h_input, logits_clean_est, logits_input,
            d_hat, r, args.num_classes)

        return {
            "shift": shift, "flipped": flipped,
            "pred_clean": pred_c, "pred_adv": pred_a,
            "query": rec["query"], "label": rec["label"],
            "h_clean": h_clean_est, "h_adv": h_input,
            "turn_idx": rec["turn_idx"],
            "observations": rec.get("observations", []),
        }

    test_results = []
    for ci, conv in enumerate(tqdm(raw_test, desc="Test")):
        conv_id = conv.get("id", f"{args.benchmark}_{ci:04d}")
        records = parse_sip_for_qpp(conv, args.num_classes)
        if not records: continue
        targets = [records[-1]] if not args.per_turn else records
        for rec in targets:
            test_suf = rand_mod.choice(suffix_choices)
            out = process_turn(conv, rec, test_suf)
            if not out: continue
            x = np.array([[out["shift"][k] for k in shift_keys]])
            x_s = scaler.transform(x)
            if args.model_type in ("isolation_forest", "elliptic"):
                raw_sc = -detector.decision_function(x_s)[0]
                is_adv = detector.predict(x_s)[0] == -1
            else:
                probs = detector.predict_proba(x_s)[0]
                raw_sc = probs[1]
                is_adv = raw_sc > 0.5
            out["is_adversarial"] = bool(is_adv)
            out["anomaly_score"] = float(raw_sc)
            out["suffix_used"] = test_suf[:50]
            out["conv_id"] = conv_id
            out["conv_idx"] = ci
            test_results.append(out)

    # Normalize scores
    all_raw = [r["anomaly_score"] for r in test_results]
    s_min, s_max = min(all_raw), max(all_raw)
    for r in test_results:
        r["adversarial_prob"] = (r["anomaly_score"]-s_min) / max(s_max-s_min, 1e-8)
        r["clean_prob"] = 1.0 - r["adversarial_prob"]
        r["merged"] = r["shift"]

    clean_res = [r for r in test_results if not r["flipped"]]
    adv_res = [r for r in test_results if r["flipped"]]
    log.info(f"Test: {len(clean_res)} normal, {len(adv_res)} flipped")

    # Metrics
    if clean_res and adv_res:
        metrics = evaluate_defense(clean_res, adv_res)
        log.info(f"\n{'='*72}")
        log.info(f"  Defense — {args.benchmark} ({args.model_type})")
        log.info(f"{'='*72}")
        for k, v in metrics.items():
            log.info(f"  {k:12s} = {v:.4f}")

    # ── Baseline comparison ──────────────────────────────────────────
    comparison_metrics = None
    if getattr(args, "compare", False) and clean_res and adv_res:
        from convqa_eval.models.llm.defense_baselines import (
            UnguardedDefense, PromptGuardDefense, GammaGuardDefense,
        )
        comparison_metrics = {}
        n_adv = len(adv_res)
        n_clean = len(clean_res)

        def _method_metrics(n_detected, n_fp, n_adv_total, n_clean_total):
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            labels = [0]*n_clean_total + [1]*n_adv_total
            preds = ([0]*(n_clean_total - n_fp) + [1]*n_fp +
                     [1]*n_detected + [0]*(n_adv_total - n_detected))
            tp, fn = n_detected, n_adv_total - n_detected
            tn = n_clean_total - n_fp
            asr = 1.0 - tp / max(n_adv_total, 1)
            return {"asr": asr, "fp": n_fp, "tp": tp, "fn": fn, "tn": tn,
                    "precision": precision_score(labels, preds, zero_division=0),
                    "recall": recall_score(labels, preds, zero_division=0),
                    "f1": f1_score(labels, preds, zero_division=0),
                    "accuracy": accuracy_score(labels, preds) if labels else 0}

        n_det = sum(1 for r in adv_res if r["is_adversarial"])
        n_fp = sum(1 for r in clean_res if r["is_adversarial"])
        comparison_metrics["Ours"] = _method_metrics(n_det, n_fp, n_adv, n_clean)
        comparison_metrics["Unguarded"] = _method_metrics(0, 0, n_adv, n_clean)

        pg_path = getattr(args, "prompt_guard_path", "")
        if pg_path:
            log.info("Running PromptGuard-2...")
            pg = PromptGuardDefense(pg_path)
            if pg.model is not None:
                pg_fp, pg_det = 0, 0
                for r in clean_res:
                    d = pg.detect(r.get("query", ""))
                    if d["is_adversarial"]: pg_fp += 1
                for r in adv_res:
                    suf = r.get("suffix_used", args.adv_suffix or "")
                    d = pg.detect(r.get("query", "") + " " + suf)
                    if d["is_adversarial"]: pg_det += 1
                comparison_metrics["PromptGuard-2"] = _method_metrics(
                    pg_det, pg_fp, n_adv, n_clean)

        log.info("Running GammaGuard (trained on train set)...")
        gg = GammaGuardDefense()
        if train_h_clean_pairs and train_h_adv_pairs:
            gg.train_on_pairs(train_h_clean_pairs, train_h_adv_pairs, epochs=100)
            gg_fp, gg_det = 0, 0
            for r in test_results:
                h = r.get("h_adv") if r["flipped"] else r.get("h_clean")
                if h is not None:
                    d = gg.detect(hidden_state=h)
                    if r["flipped"]:
                        if d["is_adversarial"]: gg_det += 1
                    else:
                        if d["is_adversarial"]: gg_fp += 1
            n_gg_adv = sum(1 for r in test_results if r["flipped"])
            n_gg_clean = sum(1 for r in test_results if not r["flipped"])
            comparison_metrics["GammaGuard"] = _method_metrics(
                gg_det, gg_fp, n_gg_adv, n_gg_clean)

        # Cascaded: PromptGuard-2 first pass → our method second pass
        if pg_path:
            log.info("Running PG2+Ours (union — either flags)...")
            cascade_fp, cascade_det = 0, 0
            cascade_times = []

            for r in test_results:
                t0_c = time.time()
                query = r.get("query", "")
                suf = r.get("suffix_used", args.adv_suffix or "")
                is_flipped = r["flipped"]

                # Pass 1: PromptGuard-2
                text_to_check = (query + " " + suf) if is_flipped else query
                pg_flag = pg.detect(text_to_check)["is_adversarial"]

                # Pass 2: our suffix-sensitivity anomaly score
                x = np.array([[r["shift"][k] for k in shift_keys]])
                x_s = scaler.transform(x)
                if args.model_type in ("isolation_forest", "elliptic"):
                    our_flag = detector.predict(x_s)[0] == -1
                else:
                    our_flag = detector.predict_proba(x_s)[0][1] > 0.5

                # Union: flag if EITHER detects
                flagged = pg_flag or our_flag

                elapsed_c = (time.time() - t0_c) * 1000
                cascade_times.append(elapsed_c)

                if is_flipped:
                    if flagged: cascade_det += 1
                else:
                    if flagged: cascade_fp += 1

            n_casc_adv = sum(1 for r in test_results if r["flipped"])
            n_casc_clean = sum(1 for r in test_results if not r["flipped"])
            casc_metrics = _method_metrics(
                cascade_det, cascade_fp, n_casc_adv, n_casc_clean)
            casc_metrics["avg_latency_ms"] = float(np.mean(cascade_times))
            casc_metrics["p50_latency_ms"] = float(np.percentile(cascade_times, 50))
            casc_metrics["p95_latency_ms"] = float(np.percentile(cascade_times, 95))
            comparison_metrics["PG2+Ours"] = casc_metrics

            log.info(f"  Cascade latency: avg={casc_metrics['avg_latency_ms']:.1f}ms  "
                     f"p50={casc_metrics['p50_latency_ms']:.1f}ms  "
                     f"p95={casc_metrics['p95_latency_ms']:.1f}ms")

        log.info("\nComparison:")
        hdr = f'  {"Method":16s} {"ASR":>6s} {"FP":>4s} {"Prec":>6s} {"Rec":>6s} {"F1":>6s} {"Acc":>6s} {"Lat(ms)":>8s}'
        log.info(hdr)
        for method, m in comparison_metrics.items():
            lat = f'{m["avg_latency_ms"]:.1f}' if "avg_latency_ms" in m else "-"
            log.info(f'  {method:16s} {m["asr"]*100:5.1f}% {m["fp"]:4d} '
                     f'{m["precision"]*100:5.1f}% {m["recall"]*100:5.1f}% '
                     f'{m["f1"]*100:5.1f}% {m["accuracy"]*100:5.1f}% '
                     f'{lat:>8s}')

    # ── Visualization ────────────────────────────────────────────────
    if clean_res and adv_res:
        plot_sep = TriangulatedSeparator(args.model_type)
        plot_sep.feature_keys = shift_keys
        plot_sep.scaler = scaler
        plot_sep.model = detector
        plot_triangulated_defense(clean_res, adv_res, plot_sep,
                                  output_dir=args.output_dir,
                                  model_name=Path(args.model_path).name,
                                  comparison_metrics=comparison_metrics)
    else:
        metrics = {"note": f"clean={len(clean_res)}, adv={len(adv_res)}"}

    # ── Demo display ─────────────────────────────────────────────────
    if getattr(args, "show_demo", False):
        n_demo = min(getattr(args, "demo_n", 10), len(test_results))
        print(f"\n{'━'*78}")
        print(f"  DEFENSE DEMO — {args.benchmark}")
        print(f"{'━'*78}")
        for r in test_results[:n_demo]:
            label_name = names[r["label"]] if r["label"] < len(names) else "?"
            pred_name = names[r["pred_clean"]] if r["pred_clean"] < len(names) else "?"
            adv_pred = names[r["pred_adv"]] if r["pred_adv"] < len(names) else "?"
            det = f"{C_RED}BLOCKED{C_RESET}" if r["is_adversarial"] else f"{C_GREEN}PASSED{C_RESET}"
            flip = f"{C_RED}FLIPPED{C_RESET}" if r["flipped"] else f"{C_GREEN}stable{C_RESET}"
            print(f"\n  {C_CYAN}Query:{C_RESET} {r['query'][:80]}")
            print(f"  Gold: {label_name}  "
                  f"Clean pred: {C_YELLOW}{pred_name}{C_RESET}  "
                  f"Suffix pred: {adv_pred}  {flip}")
            print(f"  KL={r['shift']['logit_kl']:.4f}  "
                  f"cos_Δ={r['shift']['cos_delta']:.4f}  "
                  f"score={r['adversarial_prob']:.3f}  "
                  f"→ {det}")
            if r["is_adversarial"]:
                print(f"  {C_MAGENTA}↳ Defense action: reject suffix, "
                      f"use clean prediction '{pred_name}'{C_RESET}")

    # ── Save in leaderboard submission format ─────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_path).name

    # Group test_results by conversation
    from collections import defaultdict
    conv_groups = defaultdict(list)
    conv_contexts = {}
    for r in test_results:
        cid = r.get("conv_id", f"{args.benchmark}_{r.get('conv_idx', 0):04d}")
        turn_entry = {
            "turn_idx": r.get("turn_idx", len(conv_groups[cid])),
            "query": r["query"],
            "pred": int(r["pred_clean"]),
            "confidence": 1.0 - r.get("adversarial_prob", 0.0),
            # Defense fields
            "pred_clean": int(r["pred_clean"]),
            "pred_adv": int(r["pred_adv"]),
            "defense_flag": r["is_adversarial"],
            "defended_pred": int(r["pred_clean"]) if r["is_adversarial"] else int(r["pred_adv"]),
            "anomaly_score": round(r.get("adversarial_prob", 0.0), 4),
        }
        conv_groups[cid].append(turn_entry)
        # Store context (observations) separately
        if cid not in conv_contexts and r.get("observations"):
            conv_contexts[cid] = {
                "observations": {str(turn_entry["turn_idx"]): obs[:500]
                                 for obs in r.get("observations", [])}
            }

    # 1. Predictions file (compact, SIP-aligned)
    submission = {
        "metadata": {
            "team": "",
            "model": args.model_path,
            "method": "triangulated_defense",
            "defense": "anomaly_shift",
            "model_type": args.model_type,
            "features": shift_keys,
            "config": {
                "benchmark": args.benchmark,
                "per_turn": args.per_turn,
                "num_classes": args.num_classes,
                "train_suffix": train_suffix[:80],
                "n_suffix_choices": len(suffix_choices),
            },
            "per_turn": args.per_turn,
            "num_classes": args.num_classes,
        },
        "predictions": [
            {"conv_id": cid, "turns": turns}
            for cid, turns in conv_groups.items()
        ],
    }

    pred_path = out_dir / f"predictions_{args.benchmark}_{model_name}.json"
    with open(pred_path, "w") as f:
        json.dump(submission, f, indent=2)
    log.info(f"Predictions: {pred_path} "
             f"({len(conv_groups)} convs, {len(test_results)} turns)")

    # 2. Context file (optional, for reproducibility)
    if conv_contexts:
        ctx_path = out_dir / f"context_{args.benchmark}_{model_name}.json"
        with open(ctx_path, "w") as f:
            json.dump(conv_contexts, f, indent=2)
        log.info(f"Context: {ctx_path}")

    # 3. Summary
    summary = {
        "benchmark": args.benchmark, "model_path": args.model_path,
        "defense": "anomaly_shift", "model_type": args.model_type,
        "n_train": len(train_shifts), "n_train_flipped": sum(train_flipped),
        "n_test": len(test_results),
        "n_test_normal": len(clean_res), "n_test_flipped": len(adv_res),
        "metrics": metrics,
    }
    if comparison_metrics:
        summary["comparison"] = comparison_metrics
    summary_path = out_dir / f"defend_{args.benchmark}_{model_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary: {summary_path}")

# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="LLM zero-shot evaluation for ambiguity classification")
    sub = ap.add_subparsers(dest="mode", help="Operation mode")

    # Common args
    for name in ["evaluate", "demo", "gcg", "direction", "defend"]:
        p = sub.add_parser(name)
        p.add_argument("--model_path", type=str, required=True,
                       help="HF model name or local path")
        p.add_argument("--benchmark", type=str, default="pacific")
        p.add_argument("--test_data", type=str, default=None)
        p.add_argument("--data_dir", type=str, default="benchmarks")
        p.add_argument("--num_classes", type=int, default=2, choices=[2, 4])
        p.add_argument("--per_turn", type=str, default=None,
                       choices=["true", "false"])
        p.add_argument("--max_new_tokens", type=int, default=16)
        p.add_argument("--output_dir", type=str, default="llm_results")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device_map", type=str, default="auto")
        p.add_argument("--torch_dtype", type=str, default="auto")

    # Demo-specific
    demo_p = sub.choices["demo"]
    demo_p.add_argument("--max_display", type=int, default=3)
    demo_p.add_argument("--gcg_suffix", type=str, default="",
                        help="Pre-computed GCG suffix for adversarial comparison")

    # GCG-specific
    gcg_p = sub.choices["gcg"]
    gcg_p.add_argument("--gcg_steps", type=int, default=50)
    gcg_p.add_argument("--gcg_suffix_len", type=int, default=20)
    gcg_p.add_argument("--gcg_batch_size", type=int, default=64)
    gcg_p.add_argument("--gcg_top_k", type=int, default=256)
    gcg_p.add_argument("--gcg_conv_idx", type=int, default=0,
                       help="Conversation index to attack (single mode)")
    gcg_p.add_argument("--gcg_n_samples", type=int, default=1,
                       help="Number of samples for batch GCG (1=single mode, >1=batch)")

    # Direction-specific
    dir_p = sub.choices["direction"]
    dir_p.add_argument("--max_samples", type=int, default=100,
                       help="Max conversations to sample for direction extraction")
    dir_p.add_argument("--adv_suffix", type=str, default="",
                       help="Pre-computed GCG adversarial suffix (from gcg mode)")
    dir_p.add_argument("--pool", type=str, default="last",
                       choices=["last", "mean", "first"])

    # Defend-specific
    def_p = sub.choices["defend"]
    def_p.add_argument("--direction_path", type=str, default=None,
                       help="Path to .npy direction vector (from direction mode)")
    def_p.add_argument("--direction_layer", type=int, default=-1,
                       help="Layer for direction (-1 = auto-detect from filename)")
    def_p.add_argument("--adv_suffix", type=str, default="",
                       help="GCG adversarial suffix (used for training)")
    def_p.add_argument("--suffix_file", type=str, default="",
                       help="JSON/text file with multiple suffix choices "
                            "(one per line or JSON list). Random suffix "
                            "chosen per test input for generalizability.")
    def_p.add_argument("--show_demo", action="store_true",
                       help="Show interactive demo of defense decisions")
    def_p.add_argument("--demo_n", type=int, default=10,
                       help="Number of demo examples to display")
    def_p.add_argument("--prompt_guard_path", type=str, default="",
                       help="Path to PromptGuard-2 DeBERTa model directory")
    def_p.add_argument("--compare", action="store_true",
                       help="Run all baselines (unguarded, PG-2, GammaGuard, ours)")
    def_p.add_argument("--max_samples", type=int, default=50)
    def_p.add_argument("--alpha", type=float, default=0.05,
                       help="Target FPR for threshold calibration")
    def_p.add_argument("--model_type", type=str, default="isolation_forest",
                       choices=["isolation_forest", "elliptic", "rbf_svm"],
                       help="Anomaly detector: isolation_forest (unsupervised), "
                            "elliptic (Mahalanobis), rbf_svm (supervised)")
    def_p.add_argument("--pool", type=str, default="last",
                       choices=["last", "mean", "first"])

    args = ap.parse_args()
    if args.mode is None:
        ap.print_help()
        return

    # Resolve per_turn
    if args.per_turn is not None:
        args.per_turn = args.per_turn.lower() == "true"
    else:
        bm = BENCHMARK_REGISTRY.get(args.benchmark, {})
        args.per_turn = bm.get("per_turn_default", True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.device_map, args.torch_dtype)

    if args.mode == "evaluate":
        run_evaluate(args, model, tokenizer)
    elif args.mode == "demo":
        run_demo(args, model, tokenizer)
    elif args.mode == "gcg":
        run_gcg(args, model, tokenizer)
    elif args.mode == "direction":
        run_direction(args, model, tokenizer)
    elif args.mode == "defend":
        run_defend(args, model, tokenizer)


# ══════════════════════════════════════════════════════════════════════════════
# Direction mode — find the ambiguity direction vector
# ══════════════════════════════════════════════════════════════════════════════
def run_direction(args, model, tokenizer):
    """Extract ambiguity direction, compute 4-scenario cosine similarity plot."""
    from convqa_eval.models.llm.ambiguity_direction import (
        extract_hidden_states, compute_ambiguity_direction,
        probe_accuracy, compute_layerwise_cosine_similarity,
        plot_cosine_similarity_layers,
    )
    from convqa_eval.models.llm.prompts import build_classification_prompt
    import random as rand_mod

    names = CLASS_NAMES[args.num_classes]
    rand_mod.seed(args.seed)

    # Load data
    raw_data = load_benchmark(args.benchmark, "test", args.data_dir)
    if len(raw_data) > args.max_samples:
        raw_data = rand_mod.sample(raw_data, args.max_samples)

    log.info(f"Building prompts from {len(raw_data)} conversations...")

    # Separate clear and ambiguous prompts
    clear_messages, ambig_messages = [], []
    clear_convs, ambig_convs = [], []  # (conv, gpt_idx) for suffix injection

    for conv in raw_data:
        prompts = build_per_turn_prompts(conv, args.num_classes, args.per_turn)
        for messages, gpt_idx, label in prompts:
            if label == 0:
                clear_messages.append(messages)
            else:
                ambig_messages.append(messages)
                ambig_convs.append((conv, gpt_idx))

    log.info(f"Clear: {len(clear_messages)}  Ambiguous: {len(ambig_messages)}")
    if not clear_messages or not ambig_messages:
        log.error("Need both classes"); return

    # Cap for memory
    max_per_class = min(args.max_samples // 2, 50)
    clear_messages = clear_messages[:max_per_class]
    ambig_messages = ambig_messages[:max_per_class]
    ambig_convs = ambig_convs[:max_per_class]

    # ── Extract hidden states for clear and ambiguous ────────────────
    log.info("Extracting hidden states for clear prompts...")
    h_clear = extract_hidden_states(model, tokenizer, clear_messages,
                                     pool=args.pool)
    log.info("Extracting hidden states for ambiguous prompts...")
    h_ambig = extract_hidden_states(model, tokenizer, ambig_messages,
                                     pool=args.pool)

    # ── Compute ambiguity direction at each layer ────────────────────
    layers = sorted(set(h_clear.keys()) & set(h_ambig.keys()))
    direction_per_layer = {}
    for l in layers:
        d, sep = compute_ambiguity_direction(h_clear[l], h_ambig[l])
        direction_per_layer[l] = d

    # ── Linear probe ─────────────────────────────────────────────────
    all_h = {l: np.concatenate([h_clear[l], h_ambig[l]], axis=0) for l in layers}
    all_labels = np.array([0]*len(clear_messages) + [1]*len(ambig_messages))
    log.info("\nLinear probe accuracy:")
    probe_acc = probe_accuracy(all_h, all_labels)
    best_layer = max(probe_acc, key=probe_acc.get)
    for l, acc in sorted(probe_acc.items()):
        marker = " ← BEST" if l == best_layer else ""
        log.info(f"  Layer {l:3d}: {acc:.4f}{marker}")

    # ── Build suffix-injected prompts ────────────────────────────────
    # Scenario 3: ambiguous + GCG adversarial suffix
    adv_suffix = args.adv_suffix if args.adv_suffix else ""
    random_suffix = "".join(rand_mod.choices(
        "abcdefghijklmnopqrstuvwxyz !?.,;:", k=40))

    adv_messages = []
    rand_messages = []
    for conv, gpt_idx in ambig_convs:
        if adv_suffix:
            adv_msgs = build_classification_prompt(
                conv, gpt_idx, args.num_classes,
                adversarial_suffix=adv_suffix)
            adv_messages.append(adv_msgs)
        rand_msgs = build_classification_prompt(
            conv, gpt_idx, args.num_classes,
            adversarial_suffix=random_suffix)
        rand_messages.append(rand_msgs)

    # Extract hidden states for suffix scenarios
    log.info("Extracting hidden states for random_suffix prompts...")
    h_rand = extract_hidden_states(model, tokenizer, rand_messages,
                                    pool=args.pool)

    h_adv = None
    if adv_messages:
        log.info("Extracting hidden states for adv_suffix prompts...")
        h_adv = extract_hidden_states(model, tokenizer, adv_messages,
                                       pool=args.pool)

    # ── Compute layerwise cosine similarity for all 4 scenarios ──────
    log.info("\nComputing layerwise cosine similarities...")

    scenarios = {}
    scenarios["ambiguous"] = compute_layerwise_cosine_similarity(
        h_ambig, direction_per_layer)
    scenarios["clear"] = compute_layerwise_cosine_similarity(
        h_clear, direction_per_layer)
    scenarios["ambiguous+random_suffix"] = compute_layerwise_cosine_similarity(
        h_rand, direction_per_layer)
    if h_adv:
        scenarios["ambiguous+adv_suffix"] = compute_layerwise_cosine_similarity(
            h_adv, direction_per_layer)

    # Print summary at best layer
    log.info(f"\nCosine similarity at best probe layer {best_layer}:")
    for name, layer_cos in scenarios.items():
        if best_layer in layer_cos:
            vals = layer_cos[best_layer]
            log.info(f"  {name:30s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # ── Plot ─────────────────────────────────────────────────────────
    model_name = Path(args.model_path).name
    viz_path = plot_cosine_similarity_layers(
        scenarios, output_dir=args.output_dir,
        title=f"{model_name} / {args.benchmark} — "
              f"Cosine similarity with ambiguity direction")
    log.info(f"Plot saved: {viz_path}")

    # ── Save results ─────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialize cosine stats
    cos_stats = {}
    for name, layer_cos in scenarios.items():
        cos_stats[name] = {
            str(l): {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for l, v in sorted(layer_cos.items())
        }

    result = {
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "n_clear": len(clear_messages),
        "n_ambiguous": len(ambig_messages),
        "best_probe_layer": best_layer,
        "best_probe_accuracy": probe_acc[best_layer],
        "probe_accuracy_per_layer": {str(k): v for k, v in probe_acc.items()},
        "cosine_similarity": cos_stats,
        "adv_suffix_used": adv_suffix or None,
        "random_suffix_used": random_suffix,
    }
    out_path = out_dir / f"direction_{args.benchmark}_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Results saved to: {out_path}")

    # Save direction vector at best layer
    dir_path = out_dir / f"ambiguity_direction_layer{best_layer}.npy"
    np.save(dir_path, direction_per_layer[best_layer])
    log.info(f"Direction vector saved to: {dir_path}")


if __name__ == "__main__":
    main()
