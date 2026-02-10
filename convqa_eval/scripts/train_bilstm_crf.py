"""
Complete Training Script with Advanced Features

NEW FEATURES:
1. Validation split percentage option
2. Multiple optimizer support (Adam, Adamax, RMSprop, SGD)
3. Train vs Validation loss plotting
4. Best and worst prediction examples
5. Model architecture visualization
"""

import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bilstm_crf.preprocessor import SIPPreprocessor
from models.bilstm_crf.music_baselines import create_model


class SIPDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def get_optimizer(name: str, parameters, lr: float):
    """Create optimizer by name"""
    optimizers = {
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    
    optimizer_class = optimizers[name.lower()]
    
    # SGD needs momentum
    if name.lower() == 'sgd':
        return optimizer_class(parameters, lr=lr, momentum=0.9)
    else:
        return optimizer_class(parameters, lr=lr)


def print_model_architecture(model, baseline: str):
    """Print top-level architecture of the model"""
    print("\n" + "="*70)
    print(f"Model Architecture: {baseline.upper()}")
    print("="*70)
    
    print("\n📊 Component Breakdown:\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Component-wise breakdown
    components = {
        'BERT Encoder': 0,
        'Prior BiLSTM': 0,
        'Posterior BiLSTM': 0,
        'Emission Layers': 0,
        'CRF Transitions': 0
    }
    
    for name, param in model.named_parameters():
        if 'utterance_encoder' in name or 'bert' in name:
            components['BERT Encoder'] += param.numel()
        elif 'prior_encoder' in name:
            components['Prior BiLSTM'] += param.numel()
        elif 'posterior_encoder' in name:
            components['Posterior BiLSTM'] += param.numel()
        elif 'emission' in name:
            components['Emission Layers'] += param.numel()
        elif 'trans' in name or 'crf' in name:
            components['CRF Transitions'] += param.numel()
    
    # Print architecture
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                     MUSIC ARCHITECTURE                          │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│                                                                 │")
    print("│  ┌──────────────────────────────────────────────────────────┐  │")
    print("│  │  1. BERT Utterance Encoder                               │  │")
    print(f"│  │     - Parameters: {components['BERT Encoder']:,}".ljust(65) + "│  │")
    print("│  │     - Encodes each utterance → [768]                     │  │")
    print("│  └──────────────────────────────────────────────────────────┘  │")
    print("│                          ↓                                      │")
    print("│  ┌──────────────────────────────────────────────────────────┐  │")
    print("│  │  2. Prior Inter-Utterance Encoder (BiLSTM)               │  │")
    print(f"│  │     - Parameters: {components['Prior BiLSTM']:,}".ljust(65) + "│  │")
    print("│  │     - Processes X1:T-1 (missing system_T)                │  │")
    print("│  │     - Used for INFERENCE                                 │  │")
    print("│  └──────────────────────────────────────────────────────────┘  │")
    print("│                          ↓                                      │")
    print("│  ┌──────────────────────────────────────────────────────────┐  │")
    print("│  │  3. Posterior Inter-Utterance Encoder (BiLSTM)           │  │")
    print(f"│  │     - Parameters: {components['Posterior BiLSTM']:,}".ljust(65) + "│  │")
    print("│  │     - Processes X1:T (has system_T)                      │  │")
    print("│  │     - Used for TRAINING                                  │  │")
    print("│  └──────────────────────────────────────────────────────────┘  │")
    print("│                          ↓                                      │")
    print("│  ┌──────────────────────────────────────────────────────────┐  │")
    print("│  │  4. Emission Projection Layers                           │  │")
    print(f"│  │     - Parameters: {components['Emission Layers']:,}".ljust(65) + "│  │")
    print("│  │     - BiLSTM hidden → class logits                       │  │")
    print("│  └──────────────────────────────────────────────────────────┘  │")
    print("│                          ↓                                      │")
    print("│  ┌──────────────────────────────────────────────────────────┐  │")
    print(f"│  │  5. Multi-Turn CRF ({model.crf.variant})".ljust(65) + "│  │")
    print(f"│  │     - Parameters: {components['CRF Transitions']:,}".ljust(65) + "│  │")
    print("│  │     - Dynamic transitions based on features              │  │")
    
    # List transition matrices
    if hasattr(model.crf, 'trans_global'):
        print("│  │     - Matrices: global".ljust(65) + "│  │")
    if hasattr(model.crf, 'trans_u2s'):
        print("│  │       + user→system, system→user".ljust(65) + "│  │")
    if hasattr(model.crf, 'trans_position'):
        print("│  │       + 20 position-specific".ljust(65) + "│  │")
    if hasattr(model.crf, 'trans_I0'):
        print("│  │       + I0, I1, I2+ (initiative count)".ljust(65) + "│  │")
    if hasattr(model.crf, 'trans_consecutive'):
        print("│  │       + consecutive, non-consecutive".ljust(65) + "│  │")
    
    print("│  └──────────────────────────────────────────────────────────┘  │")
    print("│                                                                 │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Total Parameters:      {total_params:,}".ljust(67) + "│")
    print(f"│  Trainable Parameters:  {trainable_params:,}".ljust(67) + "│")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()


def evaluate_with_examples(model, dataloader, device, num_classes, class_names, preprocessor):
    """
    Evaluate and return best/worst examples.
    
    Returns:
        metrics: Dict with accuracy, f1, etc.
        best_example: Best predicted conversation
        worst_example: Worst predicted conversation
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            user_utt = batch['user_utterance'].unsqueeze(0).to(device)
            system_utt = batch['system_utterance'].unsqueeze(0).to(device)
            system_labels = batch['system_I_label']
            
            # Inference
            output = model(user_utt, system_utt, mode='inference')
            preds = output['predictions']
            
            # Calculate F1 for this conversation
            if len(preds) > 0 and len(system_labels) > 0:
                conv_f1 = f1_score(system_labels.tolist(), preds, average='weighted', zero_division=0)
            else:
                conv_f1 = 0.0
            
            all_preds.extend(preds)
            all_labels.extend(system_labels.tolist())
            
            # Store example
            all_examples.append({
                'predictions': preds,
                'labels': system_labels.tolist(),
                'f1': conv_f1,
                'batch_idx': batch_idx,
                'metadata': batch.get('metadata', {})
            })
    
    # Compute overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Find best and worst examples
    all_examples.sort(key=lambda x: x['f1'], reverse=True)
    best_example = all_examples[0] if all_examples else None
    worst_example = all_examples[-1] if all_examples else None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, best_example, worst_example


def print_example(example, title, class_names):
    """Print a prediction example"""
    if not example:
        return
    
    print("\n" + "="*70)
    print(title)
    print("="*70)
    
    preds = example['predictions']
    labels = example['labels']
    f1 = example['f1']
    
    print(f"\n📊 Conversation F1: {f1:.4f}")
    print(f"Turns: {len(preds)}\n")
    
    print("Turn | True Label              | Predicted Label         | Match")
    print("-" * 70)
    
    for turn_idx, (pred, label) in enumerate(zip(preds, labels), 1):
        true_name = class_names[label]
        pred_name = class_names[pred]
        match = "✓" if pred == label else "✗"
        
        print(f"{turn_idx:4} | {true_name:23} | {pred_name:23} | {match}")
    
    print()


def plot_loss_curves(history, save_path):
    """Plot train vs validation loss"""
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total Loss
    axes[0].plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    if 'val_total_loss' in history and history['val_total_loss']:
        axes[0].plot(epochs, history['val_total_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('Total Loss (CRF + MLE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1' in history and history['val_f1']:
        axes[1].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
        axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Plot] Loss curves saved to {save_path}")


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_crf_loss = 0
    total_mle_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        user_utt = batch['user_utterance'].unsqueeze(0).to(device)
        system_utt = batch['system_utterance'].unsqueeze(0).to(device)
        user_labels = batch['user_I_label'].unsqueeze(0).to(device)
        system_labels = batch['system_I_label'].unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        
        output = model(
            user_utt, system_utt,
            user_labels, system_labels,
            mode='train'
        )
        
        loss = output['total_loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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
        'total_loss': total_loss / n,
        'crf_loss': total_crf_loss / n,
        'mle_loss': total_mle_loss / n
    }


def main():
    parser = argparse.ArgumentParser(description='Train MuSIc SIP Model (Enhanced)')
    
    # Data
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation data (if not provided, will split from train)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split percentage (default: 0.15)')
    
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
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamax', 'rmsprop', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MuSIc SIP Model Training (Enhanced)")
    print("="*70)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*70 + "\n")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    print("[Data] Loading training data...")
    with open(args.train_data, 'r') as f:
        train_data_raw = json.load(f)
    if isinstance(train_data_raw, dict):
        train_data_raw = [train_data_raw]
    
    # Preprocess
    preprocessor = SIPPreprocessor(num_classes=args.num_classes)
    class_names = preprocessor.class_names[args.num_classes]
    
    print("[Preprocess] Processing data...")
    train_processed = [
        preprocessor.process_conversation(conv) for conv in tqdm(train_data_raw)
    ]
    
    # Split validation if needed
    if args.val_data:
        print(f"[Data] Loading validation data from {args.val_data}...")
        with open(args.val_data, 'r') as f:
            val_data_raw = json.load(f)
        if isinstance(val_data_raw, dict):
            val_data_raw = [val_data_raw]
        
        val_processed = [
            preprocessor.process_conversation(conv) for conv in tqdm(val_data_raw)
        ]
    else:
        print(f"[Data] Splitting {args.val_split*100:.0f}% for validation...")
        train_size = int((1 - args.val_split) * len(train_processed))
        val_size = len(train_processed) - train_size
        
        train_processed, val_processed = random_split(
            train_processed,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Convert back to list
        train_processed = [train_processed[i] for i in range(len(train_processed))]
        val_processed = [val_processed[i] for i in range(len(val_processed))]
    
    print(f"[Data] Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Datasets
    train_dataset = SIPDataset(train_processed)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
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
    
    # Print architecture
    print_model_architecture(model, args.baseline)
    
    # Optimizer
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
    print(f"[Optimizer] Using {args.optimizer.upper()} with lr={args.lr}\n")
    
    # Training history
    history = {
        'train_total_loss': [],
        'train_crf_loss': [],
        'train_mle_loss': [],
        'val_total_loss': [],
        'val_f1': [],
        'val_accuracy': []
    }
    
    best_f1 = 0.0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_crf_loss'].append(train_metrics['crf_loss'])
        history['train_mle_loss'].append(train_metrics['mle_loss'])
        
        print(f"\n[Train] Epoch {epoch}:")
        print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
        print(f"  CRF Loss:   {train_metrics['crf_loss']:.4f}")
        print(f"  MLE Loss:   {train_metrics['mle_loss']:.4f}")
        
        # Validate
        val_metrics, best_ex, worst_ex = evaluate_with_examples(
            model, val_loader, args.device, args.num_classes, class_names, preprocessor
        )
        
        # Compute validation loss (approximate)
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                user_utt = batch['user_utterance'].unsqueeze(0).to(args.device)
                system_utt = batch['system_utterance'].unsqueeze(0).to(args.device)
                user_labels = batch['user_I_label'].unsqueeze(0).to(args.device)
                system_labels = batch['system_I_label'].unsqueeze(0).to(args.device)
                
                output = model(user_utt, system_utt, user_labels, system_labels, mode='train')
                val_total_loss += output['total_loss'].item()
        
        val_total_loss /= len(val_loader)
        history['val_total_loss'].append(val_total_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        print(f"\n[Validation] Epoch {epoch}:")
        print(f"  Total Loss: {val_total_loss:.4f}")
        print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"  Precision:  {val_metrics['precision']:.4f}")
        print(f"  Recall:     {val_metrics['recall']:.4f}")
        print(f"  F1:         {val_metrics['f1']:.4f}")
        
        # Save best
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
            print(f"\n[Checkpoint] New best F1: {best_f1:.4f}")
        
        # Periodic save
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, checkpoint_path)
    
    # Save history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot
    plot_path = os.path.join(args.checkpoint_dir, 'loss_curves.png')
    plot_loss_curves(history, plot_path)
    
    # Final evaluation with examples
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)
    
    final_metrics, best_example, worst_example = evaluate_with_examples(
        model, val_loader, args.device, args.num_classes, class_names, preprocessor
    )
    
    print_example(best_example, "🏆 BEST Prediction Example", class_names)
    print_example(worst_example, "📉 WORST Prediction Example", class_names)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best F1: {best_f1:.4f}")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Final Val F1: {final_metrics['f1']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
