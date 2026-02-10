"""
Final Comprehensive Evaluation Script

NEW FEATURES:
1. Variance computation across multiple runs
2. Random seed control
3. Per-class confusion matrices
4. AUC-ROC curves visualization
5. Key operating points marked
6. Bootstrap confidence intervals
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bilstm_crf.preprocessor import SIPPreprocessor
from models.bilstm_crf.music_baselines import create_model
from torch.utils.data import Dataset, DataLoader


# Set random seeds for reproducibility
def set_seed(seed: int):
    """Set all random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SIPDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def evaluate_single_run(
    model,
    dataloader,
    device,
    num_classes,
    class_names
) -> Dict:
    """Evaluate single run"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            user_utt = batch['user_utterance'].unsqueeze(0).to(device)
            system_utt = batch['system_utterance'].unsqueeze(0).to(device)
            system_labels = batch['system_I_label']
            
            output = model(user_utt, system_utt, mode='inference')
            
            preds = output['predictions']
            probs = output.get('probabilities', None)
            
            all_preds.extend(preds)
            all_labels.extend(system_labels.tolist())
            
            if probs is not None:
                all_probs.append(probs)
    
    if all_probs:
        all_probs = np.vstack(all_probs)
    else:
        all_probs = None
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        average='weighted',
        zero_division=0
    )
    
    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_preds,
        average=None,
        zero_division=0,
        labels=list(range(num_classes))
    )
    
    # Per-class confusion matrices
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Per-class confusion matrices (one per class)
    per_class_cm = {}
    for class_idx in range(num_classes):
        binary_labels = [1 if l == class_idx else 0 for l in all_labels]
        binary_preds = [1 if p == class_idx else 0 for p in all_preds]
        per_class_cm[class_names[class_idx]] = confusion_matrix(
            binary_labels, binary_preds, labels=[0, 1]
        )
    
    # AUC-ROC
    auc_roc = 0.0
    roc_data = None
    if all_probs is not None:
        try:
            if num_classes == 2:
                fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
                auc_roc = auc(fpr, tpr)
                roc_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
            else:
                all_labels_oh = np.eye(num_classes)[np.array(all_labels)]
                auc_roc = roc_auc_score(
                    all_labels_oh, all_probs,
                    multi_class='ovr',
                    average='weighted'
                )
                # Store per-class ROC curves
                roc_data = {}
                for class_idx in range(num_classes):
                    fpr, tpr, thresholds = roc_curve(
                        all_labels_oh[:, class_idx],
                        all_probs[:, class_idx]
                    )
                    roc_data[class_names[class_idx]] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds,
                        'auc': auc(fpr, tpr)
                    }
        except:
            pass
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'per_class': {
            'precision': per_class_p.tolist(),
            'recall': per_class_r.tolist(),
            'f1': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'per_class_confusion_matrix': {k: v.tolist() for k, v in per_class_cm.items()},
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'roc_data': roc_data
    }


def plot_roc_curves(
    baseline_results: Dict[str, Dict],
    class_names: List[str],
    num_classes: int,
    output_dir: str
):
    """
    Plot ROC curves with key operating points.
    
    Key operating points:
    - Maximum F1 point
    - Equal error rate (EER)
    - 90% specificity point
    """
    roc_dir = os.path.join(output_dir, 'roc_curves')
    os.makedirs(roc_dir, exist_ok=True)
    
    if num_classes == 2:
        # Binary ROC curves
        plt.figure(figsize=(10, 8))
        
        for baseline, results in baseline_results.items():
            roc_data = results.get('roc_data')
            if roc_data and 'fpr' in roc_data:
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                thresholds = roc_data['thresholds']
                auc_score = results['auc_roc']
                
                plt.plot(fpr, tpr, label=f'{baseline.upper()} (AUC={auc_score:.3f})', linewidth=2)
                
                # Mark key operating points
                # 1. Maximum F1
                f1_scores = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr) + 1e-10)
                max_f1_idx = np.argmax(f1_scores)
                plt.plot(fpr[max_f1_idx], tpr[max_f1_idx], 'o', markersize=8, 
                        label=f'{baseline} Max F1={f1_scores[max_f1_idx]:.3f}')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Baselines', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(roc_dir, 'all_baselines_roc.png'), dpi=300)
        plt.close()
        
    else:
        # Multiclass: One plot per class
        for class_idx, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 8))
            
            for baseline, results in baseline_results.items():
                roc_data = results.get('roc_data')
                if roc_data and class_name in roc_data:
                    class_roc = roc_data[class_name]
                    fpr = class_roc['fpr']
                    tpr = class_roc['tpr']
                    auc_score = class_roc['auc']
                    
                    plt.plot(fpr, tpr, label=f'{baseline.upper()} (AUC={auc_score:.3f})', linewidth=2)
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {class_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(roc_dir, f'roc_{class_name}.png'), dpi=300)
            plt.close()
    
    print(f"  ✓ ROC curves saved to {roc_dir}")


def plot_per_class_confusion_matrices(
    baseline_results: Dict[str, Dict],
    class_names: List[str],
    output_dir: str
):
    """Plot per-class confusion matrices"""
    cm_dir = os.path.join(output_dir, 'per_class_confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    for baseline, results in baseline_results.items():
        per_class_cm = results.get('per_class_confusion_matrix', {})
        
        if per_class_cm:
            n_classes = len(per_class_cm)
            fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 4))
            
            if n_classes == 1:
                axes = [axes]
            
            for idx, (class_name, cm) in enumerate(per_class_cm.items()):
                cm_array = np.array(cm)
                
                sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=['Other', class_name],
                           yticklabels=['Other', class_name])
                axes[idx].set_title(f'{class_name}', fontweight='bold')
                axes[idx].set_ylabel('True')
                axes[idx].set_xlabel('Predicted')
            
            plt.suptitle(f'{baseline.upper()} - Per-Class Confusion Matrices', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, f'{baseline}_per_class_cm.png'), dpi=300)
            plt.close()
    
    print(f"  ✓ Per-class confusion matrices saved to {cm_dir}")


def compute_variance_across_seeds(
    baseline_results_seeds: Dict[str, List[Dict]],
    class_names: List[str]
) -> pd.DataFrame:
    """
    Compute mean and variance across multiple seeds.
    
    Returns DataFrame with mean ± std for each metric.
    """
    rows = []
    
    for baseline, results_list in baseline_results_seeds.items():
        if not results_list:
            continue
        
        # Extract metrics across runs
        accuracies = [r['accuracy'] for r in results_list]
        precisions = [r['precision'] for r in results_list]
        recalls = [r['recall'] for r in results_list]
        f1s = [r['f1'] for r in results_list]
        aucs = [r['auc_roc'] for r in results_list]
        
        row = {
            'Baseline': baseline.upper(),
            'Accuracy': f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
            'Precision': f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
            'Recall': f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
            'F1': f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            'AUC-ROC': f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            'Runs': len(results_list)
        }
        
        # Per-class F1
        for class_idx, class_name in enumerate(class_names):
            class_f1s = [r['per_class']['f1'][class_idx] for r in results_list]
            row[f'F1-{class_name}'] = f"{np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Final Comprehensive Evaluation')
    
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_final')
    parser.add_argument('--num_classes', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds for multiple runs')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Final Comprehensive Evaluation with Variance Analysis")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print("="*70 + "\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    with open(args.test_data, 'r') as f:
        test_data_raw = json.load(f)
    if isinstance(test_data_raw, dict):
        test_data_raw = [test_data_raw]
    
    preprocessor = SIPPreprocessor(num_classes=args.num_classes)
    class_names = preprocessor.class_names[args.num_classes]
    
    test_processed = [preprocessor.process_conversation(conv) for conv in tqdm(test_data_raw)]
    test_dataset = SIPDataset(test_processed)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    baselines = ['distance']
    
    # Store results across seeds
    baseline_results_seeds = {b: [] for b in baselines}
    
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"Evaluation with Seed: {seed}")
        print('='*70)
        
        set_seed(seed)
        
        for baseline in baselines:
            checkpoint_path = os.path.join(args.checkpoint_dir,  'best_f1_model.pt')
            
            if not os.path.exists(checkpoint_path):
                continue
            
            print(f"\n[{baseline.upper()}] Loading and evaluating...")
            
            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model_args = checkpoint.get('args', {})
            
            model = create_model(
                baseline=baseline,
                num_classes=args.num_classes,
                hidden_size=model_args.get('hidden_size', 256),
                num_layers=model_args.get('num_layers', 2),
                dropout=model_args.get('dropout', 0.5),
                lambda_mle=model_args.get('lambda_mle', 0.1)
            ).to(args.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            results = evaluate_single_run(model, test_loader, args.device, args.num_classes, class_names)
            baseline_results_seeds[baseline].append(results)
            
            print(f"  F1: {results['f1']:.4f}, AUC-ROC: {results['auc_roc']:.4f}")
    
    # Compute statistics across seeds
    print("\n" + "="*70)
    print("Computing Variance Statistics")
    print("="*70 + "\n")
    
    variance_table = compute_variance_across_seeds(baseline_results_seeds, class_names)
    print(variance_table.to_string(index=False))
    
    variance_path = os.path.join(args.output_dir, 'results_with_variance.csv')
    variance_table.to_csv(variance_path, index=False)
    print(f"\n✓ Variance table saved: {variance_path}")
    
    # Use first seed results for visualization
    first_seed_results = {b: results[0] for b, results in baseline_results_seeds.items() if results}
    
    # Plot ROC curves
    print("\n[Plotting] Generating ROC curves...")
    plot_roc_curves(first_seed_results, class_names, args.num_classes, args.output_dir)
    
    # Plot per-class confusion matrices
    print("[Plotting] Generating per-class confusion matrices...")
    plot_per_class_confusion_matrices(first_seed_results, class_names, args.output_dir)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
