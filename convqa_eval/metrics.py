"""Metrics for conversational query intent detection."""
from typing import List, Dict
import numpy as np


def compute_metrics(predictions: List[Dict], ground_truth: List[Dict], config: Dict) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of model predictions
        ground_truth: List of ground truth annotations
        config: Task configuration with metric specifications
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Ambiguity detection accuracy
    if "ambiguous_utterance" in config.get("output_fields", []):
        amb_acc = _compute_ambiguity_accuracy(predictions, ground_truth)
        metrics["ambiguity_accuracy"] = amb_acc
    
    # Condition prediction F1
    if "conditions" in config.get("output_fields", []):
        cond_f1 = _compute_condition_f1(predictions, ground_truth)
        metrics["condition_f1"] = cond_f1
    
    # Candidate count MAE
    if "total_candidates" in config.get("output_fields", []):
        count_mae = _compute_candidate_count_mae(predictions, ground_truth)
        metrics["candidate_count_mae"] = count_mae
    
    # Explanation BLEU (if available)
    if "explanation" in config.get("output_fields", []):
        expl_bleu = _compute_explanation_bleu(predictions, ground_truth)
        metrics["explanation_bleu"] = expl_bleu
    
    return metrics


def _compute_ambiguity_accuracy(preds: List[Dict], gt: List[Dict]) -> float:
    """Compute accuracy for ambiguous utterance detection."""
    correct = 0
    total = 0
    
    for pred, ref in zip(preds, gt):
        if "ambiguous_utterance" in pred and "ambiguous_utterance" in ref:
            if pred["ambiguous_utterance"] == ref["ambiguous_utterance"]:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def _compute_condition_f1(preds: List[Dict], gt: List[Dict]) -> float:
    """Compute F1 for condition prediction."""
    precision_scores = []
    recall_scores = []
    
    for pred, ref in zip(preds, gt):
        pred_conds = set(pred.get("conditions", {}).keys())
        ref_conds = set(ref.get("conditions", {}).keys())
        
        if len(pred_conds) == 0 and len(ref_conds) == 0:
            continue
        
        if len(pred_conds) > 0:
            precision = len(pred_conds & ref_conds) / len(pred_conds)
            precision_scores.append(precision)
        
        if len(ref_conds) > 0:
            recall = len(pred_conds & ref_conds) / len(ref_conds)
            recall_scores.append(recall)
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0.0
    
    return f1


def _compute_candidate_count_mae(preds: List[Dict], gt: List[Dict]) -> float:
    """Compute MAE for candidate count prediction."""
    errors = []
    
    for pred, ref in zip(preds, gt):
        if "total_candidates" in pred and "total_candidates" in ref:
            error = abs(pred["total_candidates"] - ref["total_candidates"])
            errors.append(error)
    
    return np.mean(errors) if errors else 0.0


def _compute_explanation_bleu(preds: List[Dict], gt: List[Dict]) -> float:
    """Compute BLEU score for explanations (simplified)."""
    # Simplified BLEU-1 (unigram overlap)
    scores = []
    
    for pred, ref in zip(preds, gt):
        pred_text = pred.get("explanation", "")
        ref_text = ref.get("explanation", "")
        
        pred_words = set(pred_text.lower().split())
        ref_words = set(ref_text.lower().split())
        
        if len(pred_words) > 0 and len(ref_words) > 0:
            overlap = len(pred_words & ref_words)
            score = overlap / len(pred_words)
            scores.append(score)
    
    return np.mean(scores) if scores else 0.0
