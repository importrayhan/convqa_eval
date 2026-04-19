#!/usr/bin/env python3
"""
qpp_evaluate.py — QPP heuristic baseline for conversational ambiguity.

Builds a pseudo-collection from observation texts, computes pre-retrieval
and (optionally) post-retrieval QPP features per turn, and classifies
each system turn as ambiguous/clear by thresholding.

Usage:
  # Tier 1+2 only (no ranked list):
  python scripts/qpp_evaluate.py --benchmark pacific --num_classes 2

  # With mock ranked list (observations + random irrelevant docs):
  python scripts/qpp_evaluate.py --benchmark pacific --mock_ranked_list

  # With T5 query rewriter:
  python scripts/qpp_evaluate.py --benchmark pacific --rewriter_path /models/t5-qr

  # With pyserini index (Tier 3):
  python scripts/qpp_evaluate.py --benchmark pacific --index_path /idx/pacific

  # Evaluate only last turn (claqua style):
  python scripts/qpp_evaluate.py --benchmark claqua --per_turn false
"""

import sys, os, json, argparse, logging, random
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_QPP_measures import (
    PseudoCollection, QPPScorer, parse_sip_for_qpp,
    threshold_classify, find_best_threshold,
)
from convqa_eval.data.final_loader import (
    load_benchmark, train_val_split, sample_train_fraction,
    BENCHMARK_REGISTRY, remap_label,
)
from final_evaluator import compute_metrics, CLASS_NAMES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="QPP heuristic baseline for ambiguity classification")

    ap.add_argument("--benchmark", type=str, default="pacific",
                    help="Benchmark name (pacific / simmic / claqua)")
    ap.add_argument("--data_dir", type=str, default="benchmarks")
    ap.add_argument("--train_data", type=str, default=None)
    ap.add_argument("--test_data", type=str, default=None)
    ap.add_argument("--num_classes", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--per_turn", type=str, default=None,
                    choices=["true", "false"],
                    help="Per-turn evaluation (auto from benchmark)")

    ap.add_argument("--mock_ranked_list", action="store_true",
                    help="Build mock ranked list from observations + random docs")
    ap.add_argument("--n_irrelevant", type=int, default=10,
                    help="Number of random irrelevant docs per turn for mock list")
    ap.add_argument("--index_path", type=str, default=None,
                    help="Path to pyserini Lucene index (enables Tier 3)")
    ap.add_argument("--rewriter_path", type=str, default=None,
                    help="Path to local T5 query rewriter model")
    ap.add_argument("--bert_tokenizer_path", type=str, default=None,
                    help="Path to BERT model directory for subword tokenization")

    ap.add_argument("--threshold_method", type=str, default="otsu",
                    choices=["percentile", "fixed", "otsu", "best_f1"],
                    help="Thresholding method for QPP → binary")
    ap.add_argument("--threshold_percentile", type=float, default=25.0,
                    help="Percentile for 'percentile' method")
    ap.add_argument("--score_key", type=str, default="avg_idf",
                    help="QPP measure to use for thresholding")

    ap.add_argument("--output_dir", type=str, default="qpp_results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--train_fraction", type=float, default=1.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Resolve per_turn
    if args.per_turn is not None:
        args.per_turn = args.per_turn.lower() == "true"
    else:
        bm = BENCHMARK_REGISTRY.get(args.benchmark, {})
        args.per_turn = bm.get("per_turn_default", True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = CLASS_NAMES[args.num_classes]

    log.info(f"Benchmark: {args.benchmark}  Per-turn: {args.per_turn}")
    log.info(f"Score key: {args.score_key}  Threshold: {args.threshold_method}")

    # ── Load data ────────────────────────────────────────────────────────
    if args.train_data:
        with open(args.train_data) as f:
            raw_train = json.load(f)
        if isinstance(raw_train, dict):
            raw_train = [raw_train]
    else:
        raw_train = load_benchmark(args.benchmark, "train", args.data_dir)

    raw_train = sample_train_fraction(raw_train, args.train_fraction, args.seed)

    if args.test_data:
        with open(args.test_data) as f:
            raw_test = json.load(f)
        if isinstance(raw_test, dict):
            raw_test = [raw_test]
    else:
        try:
            raw_test = load_benchmark(args.benchmark, "test", args.data_dir)
        except FileNotFoundError:
            raw_train, raw_test = train_val_split(
                raw_train, args.val_split, args.seed)
            log.info("No test split found — using val split from train")

    log.info(f"Train conversations: {len(raw_train)}")
    log.info(f"Test conversations:  {len(raw_test)}")

    # ── Parse into per-turn records ──────────────────────────────────────
    train_records = []
    for conv in raw_train:
        train_records.extend(parse_sip_for_qpp(conv, args.num_classes))

    test_records = []
    for conv in raw_test:
        recs = parse_sip_for_qpp(conv, args.num_classes)
        if not args.per_turn and recs:
            recs = [recs[-1]]  # only last turn
        test_records.extend(recs)

    log.info(f"Train turns: {len(train_records)}  Test turns: {len(test_records)}")

    # Label distribution
    train_cnt = Counter(r["label"] for r in train_records)
    test_cnt = Counter(r["label"] for r in test_records)
    log.info("Train labels: " + ", ".join(
        f"{names[k]}={v}" for k, v in sorted(train_cnt.items())))
    log.info("Test labels:  " + ", ".join(
        f"{names[k]}={v}" for k, v in sorted(test_cnt.items())))

    # ── Build pseudo-collection from ALL observation texts ───────────────
    collection = PseudoCollection()
    all_observations = []
    for conv in raw_train + raw_test:
        convs = conv.get("conversations", conv.get("turns", []))
        for c in convs:
            role = c.get("from", c.get("role", ""))
            if role == "observation":
                text = c.get("value", "")
                if text.strip():
                    collection.add_document(text)
                    all_observations.append(text)

    log.info(f"Pseudo-collection: {collection}")

    # ── Build scorer ─────────────────────────────────────────────────────
    scorer = QPPScorer(collection,
                       bert_tokenizer_path=args.bert_tokenizer_path,
                       rewriter_path=args.rewriter_path)

    # ── Score all turns ──────────────────────────────────────────────────
    def score_records(records, desc="scoring"):
        all_features = []
        for rec in tqdm(records, desc=desc):
            irr_docs = []
            if args.mock_ranked_list:
                irr_docs = [rng.choice(all_observations)
                            for _ in range(min(args.n_irrelevant,
                                               len(all_observations)))]
                if not irr_docs:
                    irr_docs = []

            feats = scorer.score_turn(
                query=rec["query"],
                observations=rec["observations"],
                irrelevant_docs=irr_docs if args.mock_ranked_list else None,
            )
            all_features.append(feats)
        return all_features

    log.info("Scoring train turns...")
    train_features = score_records(train_records, "train scoring")
    log.info("Scoring test turns...")
    test_features = score_records(test_records, "test scoring")

    # ── Extract score arrays ─────────────────────────────────────────────
    available_keys = sorted(set().union(*(f.keys() for f in test_features)))
    log.info(f"Available QPP measures: {available_keys}")

    if args.score_key not in available_keys:
        log.warning(f"Score key '{args.score_key}' not available. "
                    f"Falling back to 'avg_idf'")
        args.score_key = "avg_idf"

    train_scores = np.array([f.get(args.score_key, 0.0) for f in train_features])
    test_scores = np.array([f.get(args.score_key, 0.0) for f in test_features])
    train_labels = np.array([r["label"] for r in train_records])
    test_labels = np.array([r["label"] for r in test_records])

    # ── Determine threshold ──────────────────────────────────────────────
    if args.threshold_method == "best_f1":
        # Find best threshold on train set, apply to test
        best_t, best_train_f1, lower_is_amb = find_best_threshold(
            train_scores, train_labels)
        log.info(f"Best threshold on train: {best_t:.4f}  "
                 f"(train macro-F1={best_train_f1:.4f}  "
                 f"direction={'lower' if lower_is_amb else 'higher'}=ambiguous)")
        if lower_is_amb:
            test_preds = (test_scores < best_t).astype(int)
        else:
            test_preds = (test_scores > best_t).astype(int)
        threshold_used = best_t
        score_direction = lower_is_amb
    else:
        test_preds, threshold_used = threshold_classify(
            test_scores, method=args.threshold_method,
            percentile=args.threshold_percentile)
        score_direction = True  # default: lower = ambiguous

    log.info(f"Threshold used: {threshold_used:.4f}")
    log.info(f"Predictions: clear={sum(test_preds==0)}, "
             f"ambiguous={sum(test_preds==1)}")

    # ── Compute probs for AUC-ROC ────────────────────────────────────────
    s_min, s_max = test_scores.min(), test_scores.max()
    if s_max > s_min:
        norm = (test_scores - s_min) / (s_max - s_min)
    else:
        norm = np.full_like(test_scores, 0.5)
    # Probability direction depends on which direction means ambiguous
    if score_direction:  # lower = ambiguous
        probs = np.column_stack([norm, 1.0 - norm])
    else:  # higher = ambiguous
        probs = np.column_stack([1.0 - norm, norm])

    # ── Evaluate ─────────────────────────────────────────────────────────
    metrics = compute_metrics(
        test_labels.tolist(), test_preds.tolist(), probs, args.num_classes)

    log.info(f"\n{'='*72}")
    log.info(f"  QPP Baseline Results — {args.benchmark}")
    log.info(f"  Score: {args.score_key}  Threshold: {args.threshold_method}")
    log.info(f"{'='*72}")
    log.info(f"  Accuracy  = {metrics['accuracy']:.4f}")
    log.info(f"  Precision = {metrics['precision']:.4f}")
    log.info(f"  Recall    = {metrics['recall']:.4f}")
    log.info(f"  F1 (wt)   = {metrics['f1']:.4f}")
    log.info(f"  AUC-ROC   = {metrics['auc_roc']:.4f}")
    log.info(f"\n{metrics['classification_report']}")

    # ── Per-measure sweep ────────────────────────────────────────────────
    log.info(f"\n{'─'*72}")
    log.info(f"  Per-measure best-F1 (threshold tuned on train)")
    log.info(f"{'─'*72}")

    sweep_results = {}
    for key in available_keys:
        tr_s = np.array([f.get(key, 0.0) for f in train_features])
        te_s = np.array([f.get(key, 0.0) for f in test_features])
        if tr_s.max() == tr_s.min():
            continue
        bt, bf, lo_amb = find_best_threshold(tr_s, train_labels)
        if lo_amb:
            preds_k = (te_s < bt).astype(int)
        else:
            preds_k = (te_s > bt).astype(int)
        from sklearn.metrics import f1_score, accuracy_score
        f1_w = f1_score(test_labels, preds_k, average="weighted", zero_division=0)
        f1_m = f1_score(test_labels, preds_k, average="macro", zero_division=0)
        acc = accuracy_score(test_labels, preds_k)
        direction_str = "lo" if lo_amb else "hi"
        sweep_results[key] = {"f1_weighted": f1_w, "f1_macro": f1_m,
                              "accuracy": acc, "threshold": bt,
                              "direction": direction_str}
        log.info(f"  {key:20s}  F1w={f1_w:.4f}  F1m={f1_m:.4f}  "
                 f"Acc={acc:.4f}  t={bt:.4f}  dir={direction_str}")

    # ── Save results ─────────────────────────────────────────────────────
    result_obj = {
        "benchmark": args.benchmark,
        "score_key": args.score_key,
        "threshold_method": args.threshold_method,
        "threshold_used": threshold_used,
        "per_turn": args.per_turn,
        "mock_ranked_list": args.mock_ranked_list,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("roc_data", "classification_report")},
        "per_measure_sweep": sweep_results,
        "collection_stats": {
            "N": collection.N,
            "vocab_size": len(collection.df),
            "total_tokens": collection.total_tokens,
        },
        "train_label_dist": dict(train_cnt),
        "test_label_dist": dict(test_cnt),
    }

    out_path = out_dir / f"qpp_{args.benchmark}_{args.score_key}.json"
    with open(out_path, "w") as f:
        json.dump(result_obj, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")

    # ── Per-turn output (for diagnostic comparison with BiLSTM-CRF) ─────
    turn_details = []
    for i, (rec, feats) in enumerate(zip(test_records, test_features)):
        td = {
            "turn_index": rec["turn_idx"],
            "query": rec["query"][:100],
            "prediction": int(test_preds[i]),
            "prediction_label": names[int(test_preds[i])],
            "ground_truth": rec["label"],
            "ground_truth_label": names[rec["label"]],
            "correct": int(test_preds[i]) == rec["label"],
            "qpp_score": float(test_scores[i]),
            "qpp_features": {k: round(v, 4) for k, v in feats.items()},
        }
        turn_details.append(td)

    detail_path = out_dir / f"qpp_{args.benchmark}_turns.json"
    with open(detail_path, "w") as f:
        json.dump(turn_details, f, indent=2)
    log.info(f"Turn details saved to: {detail_path}")


if __name__ == "__main__":
    main()
