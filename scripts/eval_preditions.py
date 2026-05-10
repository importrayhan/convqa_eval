#!/usr/bin/env python3
"""
eval_predictions.py — ConvQA-Eval Leaderboard Evaluation Suite.

Evaluates submissions against gold labels for clarification-need
prediction and adversarial defense benchmarks.

SUBMISSION FORMAT:
  A submission consists of TWO files:

  1. predictions.json — compact, per-turn predictions in SIP structure:
     {
       "metadata": {
         "team": "TeamName",
         "model": "Llama-3.1-8B-Instruct",
         "method": "triangulated_defense",
         "features": ["logit_kl", "cos_delta", ...],
         "config": {"model_type": "isolation_forest", ...},
         "per_turn": true,
         "num_classes": 2
       },
       "predictions": [
         {
           "conv_id": "pacific_001",
           "turns": [
             {"turn_idx": 0, "query": "What is the rate?",
              "pred": 0, "confidence": 0.92},
             {"turn_idx": 1, "query": "What about the other?",
              "pred": 1, "confidence": 0.78}
           ]
         }
       ]
     }

  2. context.json (optional, for reproducibility) — observations and
     conversation context, keyed by conv_id:
     {
       "pacific_001": {
         "observations": {
           "0": "Savings rate is 3.5%...",
           "1": "Fixed deposit rate..."
         }
       }
     }

  For defense submissions, each turn includes additional fields:
     {"turn_idx": 0, "query": "...", "pred": 0,
      "pred_clean": 0, "pred_adv": 1, "defense_flag": true,
      "defended_pred": 0, "anomaly_score": 0.782}

EVALUATION:
  python scripts/eval_predictions.py \
      --predictions submission/predictions.json \
      --gold benchmarks/pacific/data/test.json \
      --benchmark pacific

  Produces:
    - Clarification F1 (weighted, macro), Accuracy, per-class report
    - Defense metrics (if defense fields present): ASR, Prec, Rec, F1
    - Leaderboard-ready JSON with all scores
"""

import json, argparse, logging, sys
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_gold_labels(gold_path: str, benchmark: str, per_turn: bool,
                     num_classes: int) -> dict:
    """Load gold labels from benchmark data, keyed by (conv_id, turn_idx)."""
    from convqa_eval.data.loader import remap_label

    with open(gold_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]

    gold = {}
    for conv_idx, conv in enumerate(raw):
        conv_id = conv.get("id", f"{benchmark}_{conv_idx:04d}")
        convs = conv.get("conversations", conv.get("turns", []))
        turn_idx = 0
        for c in convs:
            role = c.get("from", c.get("role", ""))
            if role == "gpt":
                raw_label = int(c.get("ambiguous_type", 0))
                label = remap_label(raw_label, num_classes)
                gold[(conv_id, turn_idx)] = {
                    "label": label, "raw_label": raw_label,
                }
                turn_idx += 1
        # If per_turn=false, keep only last
        if not per_turn and turn_idx > 0:
            last_key = (conv_id, turn_idx - 1)
            last_val = gold[last_key]
            # Remove all but last
            keys_to_remove = [(conv_id, t) for t in range(turn_idx - 1)]
            for k in keys_to_remove:
                gold.pop(k, None)

    return gold


def evaluate_submission(predictions: dict, gold: dict, num_classes: int):
    """Evaluate a submission against gold labels."""
    from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                                  recall_score, classification_report)

    pred_list = predictions.get("predictions", [])
    metadata = predictions.get("metadata", {})
    has_defense = False

    # Collect aligned (gold, pred) pairs
    golds, preds = [], []
    defended_preds = []
    det_labels, det_preds = [], []  # for defense evaluation

    matched, unmatched = 0, 0
    for conv_entry in pred_list:
        conv_id = conv_entry.get("conv_id", "")
        for turn in conv_entry.get("turns", []):
            tidx = turn["turn_idx"]
            key = (conv_id, tidx)
            if key not in gold:
                unmatched += 1
                continue
            matched += 1
            g = gold[key]["label"]
            p = turn["pred"]
            golds.append(g)
            preds.append(p)

            # Defense fields
            if "defended_pred" in turn:
                has_defense = True
                defended_preds.append(turn["defended_pred"])
                flipped = turn.get("pred_clean", p) != turn.get("pred_adv", p)
                flagged = turn.get("defense_flag", False)
                det_labels.append(1 if flipped else 0)
                det_preds.append(1 if flagged else 0)

    log.info(f"Matched: {matched}, Unmatched: {unmatched}")

    if not golds:
        log.error("No matching predictions found")
        return None

    # ── Clarification metrics ────────────────────────────────────────
    names = {0: "clear", 1: "ambiguous"} if num_classes == 2 else \
            {0: "clear", 1: "slight", 2: "needs_clar", 3: "high_ambig"}

    results = {
        "metadata": metadata,
        "n_predictions": matched,
        "n_unmatched": unmatched,
        "clarification": {
            "f1_weighted": f1_score(golds, preds, average="weighted", zero_division=0),
            "f1_macro": f1_score(golds, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(golds, preds),
            "precision_w": precision_score(golds, preds, average="weighted", zero_division=0),
            "recall_w": recall_score(golds, preds, average="weighted", zero_division=0),
        },
    }

    log.info(f"\n{'='*72}")
    log.info(f"  CLARIFICATION PREDICTION")
    log.info(f"{'='*72}")
    log.info(f"  F1 (weighted) = {results['clarification']['f1_weighted']:.4f}")
    log.info(f"  F1 (macro)    = {results['clarification']['f1_macro']:.4f}")
    log.info(f"  Accuracy      = {results['clarification']['accuracy']:.4f}")
    log.info(f"\n{classification_report(golds, preds, zero_division=0)}")

    # ── Defense metrics (if present) ─────────────────────────────────
    if has_defense and det_labels:
        tp = sum(1 for l, p in zip(det_labels, det_preds) if l == 1 and p == 1)
        fp = sum(1 for l, p in zip(det_labels, det_preds) if l == 0 and p == 1)
        fn = sum(1 for l, p in zip(det_labels, det_preds) if l == 1 and p == 0)
        tn = sum(1 for l, p in zip(det_labels, det_preds) if l == 0 and p == 0)
        n_adv = sum(det_labels)

        defense_metrics = {
            "asr": 1.0 - tp / max(n_adv, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision_score(det_labels, det_preds, zero_division=0),
            "recall": recall_score(det_labels, det_preds, zero_division=0),
            "f1": f1_score(det_labels, det_preds, zero_division=0),
            "accuracy": accuracy_score(det_labels, det_preds),
        }
        results["defense"] = defense_metrics

        # Defended clarification
        if defended_preds:
            results["defended_clarification"] = {
                "f1_weighted": f1_score(golds, defended_preds, average="weighted", zero_division=0),
                "f1_macro": f1_score(golds, defended_preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(golds, defended_preds),
            }

        log.info(f"\n{'='*72}")
        log.info(f"  DEFENSE DETECTION")
        log.info(f"{'='*72}")
        log.info(f"  ASR       = {defense_metrics['asr']*100:.1f}%")
        log.info(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
        log.info(f"  Precision = {defense_metrics['precision']*100:.1f}%")
        log.info(f"  Recall    = {defense_metrics['recall']*100:.1f}%")
        log.info(f"  F1        = {defense_metrics['f1']*100:.1f}%")
        log.info(f"  Accuracy  = {defense_metrics['accuracy']*100:.1f}%")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="ConvQA-Eval Leaderboard Evaluation")
    ap.add_argument("--predictions", type=str, required=True)
    ap.add_argument("--gold", type=str, required=True,
                    help="Path to gold test data (benchmark JSON)")
    ap.add_argument("--benchmark", type=str, default="pacific")
    ap.add_argument("--num_classes", type=int, default=2, choices=[2, 4])
    ap.add_argument("--per_turn", type=str, default=None,
                    choices=["true", "false"])
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    # Resolve per_turn from submission metadata or benchmark default
    with open(args.predictions) as f:
        submission = json.load(f)
    meta = submission.get("metadata", {})

    if args.per_turn is not None:
        per_turn = args.per_turn.lower() == "true"
    elif "per_turn" in meta:
        per_turn = meta["per_turn"]
    else:
        from convqa_eval.data.loader import BENCHMARK_REGISTRY
        per_turn = BENCHMARK_REGISTRY.get(args.benchmark, {}).get(
            "per_turn_default", True)

    gold = load_gold_labels(args.gold, args.benchmark, per_turn, args.num_classes)
    log.info(f"Gold labels: {len(gold)} turns from {args.gold}")

    results = evaluate_submission(submission, gold, args.num_classes)

    if results:
        out = args.output or str(
            Path(args.predictions).with_suffix(".results.json"))
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
