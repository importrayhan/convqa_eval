"""
Complete Training Script for MuSIc SIP Models

Evaluation Metrics:
- Accuracy
- F1 Score (weighted for multiclass)
- Precision (weighted)
- Recall (weighted)
- AUC-ROC (weighted for multiclass)
- Per-class metrics
- Confusion matrix

Features:
- All 6 baselines supported
- 2-4 class configuration
- Training curves with matplotlib
- Best model checkpointing
- Per-epoch evaluation
"""

import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bilstm_crf.preprocessor import SIPPreprocessor
from models.bilstm_crf.music_baselines import create_model


class SIPDataset(Dataset):
    """Dataset for SIP conversations"""
    def __init__(self, processed_data):
        self.data = processed_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Batch size must be 1 for incremental processing"""
    assert len(batch) == 1, "Batch size must be 1"
    return batch[0]


def load_data(data_path):
    """Load conversations from JSON"""
    print(f"\n[Data] Loading from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    print(f"[Data] Loaded {len(data)} conversations")
    return data


def evaluate_with_auc(model, dataloader, device, num_classes, class_names):
    """
    Comprehensive evaluation with all metrics including AUC-ROC.
    
    Returns:
        Dict with accuracy, precision, recall, f1, auc_roc, per_class, cm
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n[Eval] Evaluating...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            user_utt = batch['user_utterance'].unsqueeze(0).to(device)
            system_utt = batch['system_utterance'].unsqueeze(0).to(device)
            system_labels = batch['system_I_label']
            
            # Inference
            output = model(user_utt, system_utt, mode='inference')
            
            # Collect predictions
            preds = output['predictions']
            probs = output.get('probabilities', None)
            
            all_preds.extend(preds)
            all_labels.extend(system_labels.tolist())
            
            if probs is not None:
                all_probs.append(probs)
    
    # Stack probabilities
    if all_probs:
        all_probs = np.vstack(all_probs)
    
    # Standard metrics
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
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # AUC-ROC
    auc_roc = 0.0
    if all_probs:
        try:
            if num_classes == 2:
                # Binary classification
                auc_roc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multiclass (one-vs-rest)
                # Convert labels to one-hot
                all_labels_array = np.array(all_labels)
                all_labels_oh = np.eye(num_classes)[all_labels_array]
                
                auc_roc = roc_auc_score(
                    all_labels_oh,
                    all_probs,
                    multi_class='ovr',
                    average='weighted'
                )
        except Exception as e:
            print(f"[Warning] Could not compute AUC-ROC: {e}")
            auc_roc = 0.0
    
    # Print results
    print(f"\n[Eval] Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    
    print(f"\n[Eval] Per-Class Metrics:")
    for i in range(num_classes):
        print(f"  Class {i} ({class_names[i]}):")
        print(f"    Precision: {per_class_p[i]:.4f}")
        print(f"    Recall:    {per_class_r[i]:.4f}")
        print(f"    F1:        {per_class_f1[i]:.4f}")
        print(f"    Support:   {per_class_support[i]}")
    
    print(f"\n[Eval] Confusion Matrix:")
    print(cm)
    
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
        'confusion_matrix': cm.tolist()
    }


def plot_training_curves(history, save_path, num_classes, class_names):
    """Plot comprehensive training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # CRF Loss
    axes[0, 1].plot(epochs, history['train_crf_loss'], 'b-', label='Train CRF')
    axes[0, 1].set_title('CRF Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MLE Loss
    axes[0, 2].plot(epochs, history['train_mle_loss'], 'b-', label='Train MLE')
    axes[0, 2].set_title('MLE Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Validation Metrics
    if 'val_f1' in history:
        axes[1, 0].plot(epochs, history['val_accuracy'], 'g-', label='Accuracy')
        axes[1, 0].plot(epochs, history['val_f1'], 'm-', label='F1')
        axes[1, 0].plot(epochs, history['val_precision'], 'b--', label='Precision')
        axes[1, 0].plot(epochs, history['val_recall'], 'r--', label='Recall')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC-ROC
        axes[1, 1].plot(epochs, history['val_auc_roc'], 'c-', label='AUC-ROC')
        axes[1, 1].set_title('AUC-ROC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC-ROC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Per-Class F1 (last epoch)
        if num_classes <= 4:
            last_per_class = history['val_per_class_f1'][-1] if history['val_per_class_f1'] else [0] * num_classes
            axes[1, 2].bar(range(num_classes), last_per_class)
            axes[1, 2].set_title('Per-Class F1 (Final Epoch)')
            axes[1, 2].set_xlabel('Class')
            axes[1, 2].set_ylabel('F1 Score')
            axes[1, 2].set_xticks(range(num_classes))
            axes[1, 2].set_xticklabels([name[:10] for name in class_names], rotation=45)
            axes[1, 2].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Plot] Saved training curves to {save_path}")
    plt.close()


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_crf_loss = 0
    total_mle_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        user_utt = batch['user_utterance'].unsqueeze(0).to(device)
        system_utt = batch['system_utterance'].unsqueeze(0).to(device)
        user_labels = batch['user_I_label'].unsqueeze(0).to(device)
        system_labels = batch['system_I_label'].unsqueeze(0).to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(
            user_utt, system_utt,
            user_labels, system_labels,
            mode='train'
        )
        
        loss = output['total_loss']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        total_crf_loss += output['loss_crf'].item()
        total_mle_loss += output['loss_mle'].item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'crf': f"{output['loss_crf'].item():.4f}",
            'mle': f"{output['loss_mle'].item():.4f}"
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'crf_loss': total_crf_loss / n,
        'mle_loss': total_mle_loss / n
    }


def main():
    parser = argparse.ArgumentParser(description='Train MuSIc SIP Model with AUC-ROC')
    
    # Data
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    
    # Model
    parser.add_argument('--baseline', type=str, default='vanillacrf',
                       choices=['vanillacrf', 'who2who', 'position', 'who2who_position', 'intime', 'distance'])
    parser.add_argument('--num_classes', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lambda_mle', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MuSIc SIP Model Training with AUC-ROC")
    print("="*70)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*70 + "\n")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    train_data_raw = load_data(args.train_data)
    val_data_raw = load_data(args.val_data) if args.val_data else None
    
    # Preprocess
    preprocessor = SIPPreprocessor(num_classes=args.num_classes)
    class_names = preprocessor.class_names[args.num_classes]
    
    print("[Preprocess] Processing training data...")
    train_processed = [
        preprocessor.process_conversation(conv) for conv in tqdm(train_data_raw)
    ]
    
    if val_data_raw:
        print("[Preprocess] Processing validation data...")
        val_processed = [
            preprocessor.process_conversation(conv) for conv in tqdm(val_data_raw)
        ]
    
    # Datasets
    train_dataset = SIPDataset(train_processed)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    if val_data_raw:
        val_dataset = SIPDataset(val_processed)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = create_model(
        baseline=args.baseline,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lambda_mle=args.lambda_mle
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_crf_loss': [],
        'train_mle_loss': []
    }
    
    if val_data_raw:
        history.update({
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc_roc': [],
            'val_per_class_f1': []
        })
    
    best_f1 = 0.0
    best_auc = 0.0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_crf_loss'].append(train_metrics['crf_loss'])
        history['train_mle_loss'].append(train_metrics['mle_loss'])
        
        print(f"\n[Train] Epoch {epoch}:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  CRF:  {train_metrics['crf_loss']:.4f}")
        print(f"  MLE:  {train_metrics['mle_loss']:.4f}")
        
        # Validate
        if val_data_raw:
            val_metrics = evaluate_with_auc(model, val_loader, args.device, args.num_classes, class_names)
            
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_auc_roc'].append(val_metrics['auc_roc'])
            history['val_per_class_f1'].append(val_metrics['per_class']['f1'])
            
            # Save best F1
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_f1_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1': best_f1,
                    'metrics': val_metrics,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"\n[Checkpoint] Saved best F1 model (F1={best_f1:.4f})")
            
            # Save best AUC
            if val_metrics['auc_roc'] > best_auc:
                best_auc = val_metrics['auc_roc']
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_auc_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc_roc': best_auc,
                    'metrics': val_metrics,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"[Checkpoint] Saved best AUC model (AUC={best_auc:.4f})")
        
        # Periodic save
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"\n[Checkpoint] Saved to {checkpoint_path}")
    
    # Save history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_path = os.path.join(args.checkpoint_dir, 'training_curves.png')
    plot_training_curves(history, plot_path, args.num_classes, class_names)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    if val_data_raw:
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Best AUC-ROC:  {best_auc:.4f}")
    print(f"History saved to: {history_path}")
    print(f"Plot saved to: {plot_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
