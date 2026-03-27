#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained checkpoints with multi-seed variance analysis.

Produces: per-class metrics, confusion matrices, ROC-AUC, FPR, mean±std tables.
"""

import os, sys, json, argparse, logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convqa_eval.models.bilstm_crf.model_final_imbalance import create_model, CRF_VARIANTS
from convqa_eval.data.final_loader import (
    SIPDataset, sip_collate_single, tokenize_conversation, load_benchmark,
)
from evaluator import compute_metrics, CLASS_NAMES
from convqa_eval.utils.seed import set_seed
from convqa_eval.utils.gpu import get_device_map

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_and_evaluate(checkpoint_path, test_loader, device, num_classes,
                      per_turn=True):
    """Load checkpoint, rebuild model, run eval."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt.get("args", {})

    model = create_model(
        baseline=a.get("baseline", "vanillacrf"),
        num_classes=num_classes,
        encoder_name=a.get("encoder", "bert-base-uncased"),
        encoder_path=a.get("encoder_path", None),
        hidden_size=a.get("hidden_size", 256),
        num_layers=a.get("num_layers", 2),
        dropout=a.get("dropout", 0.3),
        lambda_mle=a.get("lambda_mle", 0.1),
    ).to(device)
    # strict=False: old checkpoints may lack class_weights buffer;
    # new checkpoints will overwrite the default ones(num_classes).
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    all_p, all_l, all_prob = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  eval", leave=False):
            u = batch["user_utterance"].unsqueeze(0).to(device)
            s = batch["system_utterance"].unsqueeze(0).to(device)
            if u.shape[1] == 0:
                continue
            out = model(u, s, mode="inference", per_turn=per_turn)
            all_p.extend(out["predictions"])
            if per_turn:
                all_l.extend(batch["system_I_label"].tolist())
            else:
                all_l.append(batch["system_I_label"][-1].item())
            if out.get("probabilities") is not None and len(out["probabilities"]) > 0:
                all_prob.append(out["probabilities"])
    probs = np.vstack(all_prob) if all_prob else None
    return compute_metrics(all_l, all_p, probs, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint (or directory for multi-baseline)")
    ap.add_argument("--benchmark", type=str, default="pacific")
    ap.add_argument("--test_data", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="benchmarks")
    ap.add_argument("--num_classes", type=int, default=2, choices=[2,3,4])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                    help="Seeds for variance analysis")
    ap.add_argument("--output_dir", type=str, default="evaluation_results")
    ap.add_argument("--per_turn", type=str, default=None,
                    choices=["true", "false"],
                    help="Per-turn eval (auto from benchmark if not set)")
    args = ap.parse_args()

    # Resolve per_turn
    if args.per_turn is not None:
        args.per_turn = args.per_turn.lower() == "true"
    else:
        from convqa_eval.data.loader import BENCHMARK_REGISTRY
        bm = BENCHMARK_REGISTRY.get(args.benchmark, {})
        args.per_turn = bm.get("per_turn_default", True)

    device = get_device_map(args.gpu) if args.gpu >= 0 else torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = CLASS_NAMES[args.num_classes]

    # load test data
    if args.test_data:
        with open(args.test_data) as f:
            raw = json.load(f)
        if isinstance(raw, dict): raw = [raw]
    else:
        raw = load_benchmark(args.benchmark, "test", args.data_dir)

    # need tokenizer from checkpoint args
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best_f1_model.pt"
    ckpt_peek = torch.load(ckpt_path, map_location="cpu")
    a_ckpt = ckpt_peek.get("args", {})
    enc_name = a_ckpt.get("encoder", "bert-base-uncased")
    enc_path = a_ckpt.get("encoder_path", None)
    max_len  = a_ckpt.get("max_length", 512)

    # Resolve: local path takes precedence over hub name
    enc_source = enc_path if (enc_path and os.path.isdir(enc_path)) else enc_name
    log.info(f"Encoder source (from checkpoint): {enc_source}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(enc_source)
    test_proc = [tokenize_conversation(c, tokenizer, max_len, args.num_classes)
                 for c in tqdm(raw, desc="tokenize")]
    test_loader = DataLoader(SIPDataset(test_proc), batch_size=1,
                             shuffle=False, collate_fn=sip_collate_single)

    # multi-seed evaluation
    seed_results = []
    for seed in args.seeds:
        set_seed(seed)
        log.info(f"Seed {seed}")
        metrics = load_and_evaluate(ckpt_path, test_loader, device,
                                    args.num_classes, per_turn=args.per_turn)
        seed_results.append(metrics)
        log.info(f"  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
                 f"AUC={metrics['auc_roc']:.4f}")

    # aggregate
    rows = []
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        vals = [r[k] for r in seed_results]
        rows.append({"metric": k, "mean": np.mean(vals), "std": np.std(vals),
                      "min": np.min(vals), "max": np.max(vals)})
    # per-class F1
    for ci, cn in enumerate(names):
        vals = [r["per_class"]["f1"][ci] for r in seed_results]
        rows.append({"metric": f"f1_{cn}", "mean": np.mean(vals), "std": np.std(vals),
                      "min": np.min(vals), "max": np.max(vals)})
    # FPR
    for cn in names:
        vals = [r["false_positive_rates"][cn] for r in seed_results]
        rows.append({"metric": f"fpr_{cn}", "mean": np.mean(vals), "std": np.std(vals),
                      "min": np.min(vals), "max": np.max(vals)})

    df = pd.DataFrame(rows)
    log.info(f"\n{df.to_string(index=False)}")

    df.to_csv(out_dir / "variance_report.csv", index=False)

    # detailed report (first seed)
    best = seed_results[0]
    report = {
        "args": vars(args),
        "seeds": args.seeds,
        "summary": {k: {"mean": float(np.mean([r[k] for r in seed_results])),
                         "std": float(np.std([r[k] for r in seed_results]))}
                     for k in ["accuracy","precision","recall","f1","auc_roc"]},
        "classification_report": best["classification_report"],
        "confusion_matrix": best["confusion_matrix"],
        "false_positive_rates": best["false_positive_rates"],
    }
    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\nResults saved to {out_dir}")
    log.info(f"Classification report:\n{best['classification_report']}")


if __name__ == "__main__":
    main()
