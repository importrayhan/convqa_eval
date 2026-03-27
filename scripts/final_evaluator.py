"""
Evaluation engine — computes classification metrics, generates reports.
"""

import json, os, logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    2: ["clear", "ambiguous"],
    3: ["clear", "needs_clarification", "highly_ambiguous"],
    4: ["clear", "slightly_ambiguous", "needs_clarification", "highly_ambiguous"],
}


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict:
    """Full metric suite: acc, P, R, F1, AUC-ROC, FPR, per-class breakdown."""
    names = CLASS_NAMES[num_classes]
    acc = accuracy_score(labels, preds)
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    pp, pr_, pf, ps = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0, labels=list(range(num_classes)))
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # FPR per class
    fpr_per_class = {}
    for c in range(num_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr_per_class[names[c]] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # AUC-ROC
    auc_roc = 0.0
    roc_data = None
    if probs is not None:
        try:
            if num_classes == 2:
                fpr_arr, tpr_arr, thr = roc_curve(labels, probs[:, 1])
                auc_roc = auc(fpr_arr, tpr_arr)
                roc_data = {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()}
            else:
                oh = np.eye(num_classes)[np.array(labels)]
                auc_roc = roc_auc_score(oh, probs, multi_class="ovr", average="weighted")
        except Exception:
            pass

    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "auc_roc": auc_roc,
        "false_positive_rates": fpr_per_class,
        "per_class": {
            "precision": pp.tolist(), "recall": pr_.tolist(),
            "f1": pf.tolist(), "support": ps.tolist(),
        },
        "confusion_matrix": cm.tolist(),
        "roc_data": roc_data,
        "classification_report": classification_report(
            labels, preds, target_names=names, zero_division=0),
    }


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    """Mean ± std across seeds for each metric."""
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


class ConvQAEval:
    """
    Top-level evaluation runner (CoIR-style interface).

    Usage:
        evaluator = ConvQAEval(tasks=["pacific"], num_classes=2)
        results = evaluator.run(model, tokenizer, output_folder="results/bert")
    """

    def __init__(self, tasks=None, num_classes: int = 2, batch_size: int = 1):
        from convqa_eval.data.loader import get_tasks, load_benchmark
        self.task_names = list((tasks or get_tasks()).keys()) if isinstance(tasks, (dict, type(None))) else tasks
        self.num_classes = num_classes
        self.batch_size = batch_size

    def run(self, model, tokenizer, data_split="test", output_folder="results",
            device="cpu", seeds=None, per_turn=None):
        """Evaluate model on all tasks, return {task: metrics}.

        per_turn: True/False or None (auto from benchmark registry).
        """
        import torch
        from convqa_eval.data.loader import (load_benchmark, tokenize_conversation,
                                              SIPDataset, sip_collate_single,
                                              BENCHMARK_REGISTRY)
        from torch.utils.data import DataLoader

        os.makedirs(output_folder, exist_ok=True)
        all_results = {}

        for task in self.task_names:
            logger.info(f"Evaluating on {task}/{data_split}")

            # Resolve per_turn for this task
            if per_turn is not None:
                task_per_turn = per_turn
            else:
                bm = BENCHMARK_REGISTRY.get(task, {})
                task_per_turn = bm.get("per_turn_default", True)

            raw = load_benchmark(task, data_split)
            processed = [tokenize_conversation(c, tokenizer, num_classes=self.num_classes,
                                               use_ground_truth=True) for c in raw]
            ds = SIPDataset(processed)
            dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=sip_collate_single)

            all_preds, all_labels, all_probs = [], [], []
            model.eval()
            with torch.no_grad():
                for batch in dl:
                    u = batch["user_utterance"].unsqueeze(0).to(device)
                    s = batch["system_utterance"].unsqueeze(0).to(device)
                    if u.shape[1] == 0:
                        continue
                    out = model(u, s, mode="inference", per_turn=task_per_turn)
                    all_preds.extend(out["predictions"])
                    if task_per_turn:
                        all_labels.extend(batch["system_I_label"].tolist())
                    else:
                        all_labels.append(batch["system_I_label"][-1].item())
                    if out.get("probabilities") is not None and len(out["probabilities"]) > 0:
                        all_probs.append(out["probabilities"])

            probs = np.vstack(all_probs) if all_probs else None
            metrics = compute_metrics(all_labels, all_preds, probs, self.num_classes)
            all_results[task] = metrics

            # save per-task
            with open(os.path.join(output_folder, f"{task}_{data_split}_results.json"), "w") as f:
                json.dump({k: v for k, v in metrics.items() if k != "roc_data"}, f, indent=2)

        return all_results
