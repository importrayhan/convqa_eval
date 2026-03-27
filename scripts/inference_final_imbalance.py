#!/usr/bin/env python3
"""
inference.py — Run predictions on test data with rich turn-by-turn display.

Shows a simulation-style conversation with system predictions per turn.
"""

import os, sys, json, argparse, logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convqa_eval.models.bilstm_crf.model_final_imbalance import create_model
from convqa_eval.data.final_loader import (
    SIPDataset, sip_collate_single, tokenize_conversation,
    load_benchmark, parse_sip_conversation,
)
from evaluator import CLASS_NAMES
from convqa_eval.utils.gpu import get_device_map

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Colour codes for terminal ────────────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD  = "\033[1m"
C_GREEN = "\033[92m"
C_RED   = "\033[91m"
C_CYAN  = "\033[96m"
C_YELLOW = "\033[93m"
C_DIM   = "\033[2m"

LABEL_COLOUR = {0: C_GREEN, 1: C_YELLOW, 2: C_RED, 3: C_RED + C_BOLD}


def display_conversation(
    raw_conv, predictions, confidences, ground_truth, names, num_classes,
    conv_idx: int, per_turn: bool = True,
):
    """Pretty-print one conversation turn-by-turn.

    per_turn=True:  predictions[i] corresponds to the i-th gpt turn.
    per_turn=False: predictions has length 1, for the LAST gpt turn only.
                    All prior gpt turns are displayed as context.
    """
    convs = raw_conv.get("conversations", raw_conv.get("turns", []))
    print(f"\n{'━'*78}")
    print(f"{C_BOLD}  Conversation {conv_idx+1}{C_RESET}")
    print(f"{'━'*78}")

    # Count total gpt turns to know which is last
    gpt_indices = [j for j, c in enumerate(convs)
                   if c.get("from", c.get("role", "")) == "gpt"]
    total_gpt = len(gpt_indices)
    last_gpt_idx = gpt_indices[-1] if gpt_indices else -1

    gpt_count = 0   # tracks which gpt turn we're on
    pred_idx = 0     # tracks which prediction to consume

    for c in convs:
        role = c.get("from", c.get("role", ""))
        val  = str(c.get("value", ""))[:120]

        if role == "human":
            print(f"  {C_CYAN}👤 USER:{C_RESET}  {val}")
        elif role == "observation":
            ot = c.get("observation_type", "ctx")
            print(f"  {C_DIM}📋 OBS [{ot}]:{C_RESET} {val[:80]}…" if len(val) > 80 else
                  f"  {C_DIM}📋 OBS [{ot}]:{C_RESET} {val}")
        elif role == "gpt":
            gt_raw = c.get("ambiguous_type", "?")

            # Decide whether this gpt turn has a prediction
            has_pred = False
            if per_turn:
                has_pred = pred_idx < len(predictions)
            else:
                # Only the last gpt turn gets a prediction
                has_pred = (gpt_count == total_gpt - 1) and pred_idx < len(predictions)

            if has_pred:
                p = predictions[pred_idx]
                conf = confidences[pred_idx] if pred_idx < len(confidences) else 0
                # Ground truth: for per_turn=True, use gpt_count index;
                # for per_turn=False, the target is always the last label
                if per_turn:
                    gt = ground_truth[gpt_count] if gpt_count < len(ground_truth) else -1
                else:
                    gt = ground_truth[-1] if ground_truth else -1
                col = LABEL_COLOUR.get(p, "")
                match = "✓" if p == gt else "✗"
                match_col = C_GREEN if p == gt else C_RED
                print(f"  {C_BOLD}🤖 SYS:{C_RESET}   {val}")
                print(f"        {col}pred={names[p]}{C_RESET}  "
                      f"conf={conf:.2f}  "
                      f"gold={names[gt] if 0<=gt<num_classes else '?'}  "
                      f"{match_col}{match}{C_RESET}")
                pred_idx += 1
            else:
                # Context turn (no prediction) — show as dimmed
                if not per_turn and gpt_count < total_gpt - 1:
                    print(f"  {C_DIM}🤖 SYS (context):{C_RESET} {val}")
                else:
                    print(f"  {C_BOLD}🤖 SYS:{C_RESET}   {val}")

            gpt_count += 1
        elif role == "function_call":
            print(f"  {C_DIM}⚙️  CALL: {val}{C_RESET}")

    # summary
    if predictions:
        n_correct = sum(1 for p, g in zip(predictions,
                        ground_truth[-len(predictions):] if not per_turn
                        else ground_truth) if p == g)
        total = len(predictions)
        print(f"\n  {'─'*40}")
        print(f"  Turns: {total}  Correct: {n_correct}/{total}  "
              f"({100*n_correct/total:.0f}%)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--benchmark", type=str, default="pacific")
    ap.add_argument("--test_data", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="benchmarks")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_display", type=int, default=10,
                    help="Max conversations to display")
    ap.add_argument("--output", type=str, default=None,
                    help="Save predictions JSON")
    ap.add_argument("--per_turn", type=str, default=None,
                    choices=["true", "false"],
                    help="Per-turn inference (auto from benchmark if not set)")
    args = ap.parse_args()

    # Resolve per_turn
    if args.per_turn is not None:
        args.per_turn = args.per_turn.lower() == "true"
    else:
        from convqa_eval.data.loader import BENCHMARK_REGISTRY
        bm = BENCHMARK_REGISTRY.get(args.benchmark, {})
        args.per_turn = bm.get("per_turn_default", True)

    device = get_device_map(args.gpu) if args.gpu >= 0 else torch.device("cpu")
    names = CLASS_NAMES[args.num_classes]

    # load data
    if args.test_data:
        with open(args.test_data) as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict): raw_data = [raw_data]
    else:
        raw_data = load_benchmark(args.benchmark, "test", args.data_dir)

    # load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    a = ckpt.get("args", {})
    enc_name = a.get("encoder", "bert-base-uncased")
    enc_path = a.get("encoder_path", None)
    ml  = a.get("max_length", 512)

    # Resolve: local path takes precedence over hub name
    enc_source = enc_path if (enc_path and os.path.isdir(enc_path)) else enc_name
    log.info(f"Encoder source (from checkpoint): {enc_source}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(enc_source)

    model = create_model(
        baseline=a.get("baseline", "vanillacrf"),
        num_classes=args.num_classes,
        encoder_name=enc_name,
        encoder_path=enc_path,
        hidden_size=a.get("hidden_size", 256),
        num_layers=a.get("num_layers", 2),
        dropout=a.get("dropout", 0.3),
        lambda_mle=a.get("lambda_mle", 0.1),
    ).to(device)
    # strict=False: old checkpoints may lack class_weights buffer;
    # new checkpoints will overwrite the default ones(num_classes).
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    log.info(f"Model loaded: {enc_source} / {a.get('baseline','vanillacrf')}")

    # run inference
    all_results = []
    for i, raw_conv in enumerate(raw_data):
        proc = tokenize_conversation(raw_conv, tokenizer, ml,
                                     args.num_classes, use_ground_truth=True)
        u = proc["user_utterance"].unsqueeze(0).to(device)
        s = proc["system_utterance"].unsqueeze(0).to(device)
        if u.shape[1] == 0:
            continue
        with torch.no_grad():
            out = model(u, s, mode="inference", per_turn=args.per_turn)

        preds = out["predictions"]
        confs = out.get("confidences", [0.0] * len(preds))
        golds = proc["system_I_label"].tolist()
        trans = out.get("transition_info", [])

        if i < args.max_display:
            display_conversation(raw_conv, preds, confs, golds,
                                 names, args.num_classes, i,
                                 per_turn=args.per_turn)

        # Per-turn detail
        turn_details = []
        for t_idx in range(len(preds)):
            # When per_turn=False, preds has 1 element for the LAST system turn.
            # The gold label is golds[-1], not golds[0].
            if args.per_turn:
                gt = golds[t_idx] if t_idx < len(golds) else -1
            else:
                gt = golds[-1] if golds else -1

            detail = {
                "turn_index": t_idx if args.per_turn else len(golds) - 1,
                "prediction": preds[t_idx],
                "prediction_label": names[preds[t_idx]] if preds[t_idx] < len(names) else "?",
                "ground_truth": gt,
                "ground_truth_label": names[gt] if 0 <= gt < len(names) else "?",
                "confidence": float(confs[t_idx]),
                "correct": preds[t_idx] == gt,
            }
            if t_idx < len(trans):
                detail["transition_matrix"] = trans[t_idx].get("transition_matrix", "")
                detail["features"] = trans[t_idx].get("features", {})
            if out.get("probabilities") is not None and t_idx < len(out["probabilities"]):
                detail["class_probabilities"] = {
                    names[c]: float(out["probabilities"][t_idx][c])
                    for c in range(min(len(names), len(out["probabilities"][t_idx])))
                }
            turn_details.append(detail)

        n_correct = sum(1 for d in turn_details if d.get("correct", False))
        all_results.append({
            "conversation_idx": i,
            "num_turns": len(preds),
            "correct": n_correct,
            "accuracy": n_correct / max(len(preds), 1),
            "turns": turn_details,
        })

    # global summary
    total_turns = sum(r["num_turns"] for r in all_results)
    total_correct = sum(r["correct"] for r in all_results)
    log.info(f"\nGlobal: {total_correct}/{total_turns} system turns correct "
             f"({100*total_correct/max(total_turns,1):.1f}%)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": all_results,
                       "global_accuracy": total_correct / max(total_turns, 1)}, f, indent=2)
        log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
