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
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="LLM zero-shot evaluation for ambiguity classification")
    sub = ap.add_subparsers(dest="mode", help="Operation mode")

    # Common args
    for name in ["evaluate", "demo", "gcg", "direction"]:
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
