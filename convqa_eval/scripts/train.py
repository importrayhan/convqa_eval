"""Training Script for MuSIc - with F1, Precision, Recall and Plotting"""

import os, sys, json, torch, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.preprocessor import SIPPreprocessor
from model.music_model import MuSIc

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

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            user_utt = batch['user_utterance'].unsqueeze(0).to(device)
            system_utt = batch['system_utterance'].unsqueeze(0).to(device)
            system_labels = batch['system_I_label']
            
            output = model(user_utt, system_utt, mode='inference')
            all_preds.extend(output['system_predictions'])
            all_labels.extend(system_labels.tolist())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history: axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0,0].set_title('Total Loss'); axes[0,0].set_xlabel('Epoch'); axes[0,0].legend(); axes[0,0].grid(True)
    
    axes[0,1].plot(epochs, history['train_crf_loss'], 'b-')
    axes[0,1].set_title('CRF Loss'); axes[0,1].grid(True)
    
    axes[1,0].plot(epochs, history['train_mle_loss'], 'b-')
    axes[1,0].set_title('MLE Loss'); axes[1,0].grid(True)
    
    if 'val_f1' in history:
        axes[1,1].plot(epochs, history['val_f1'], 'g-', label='F1')
        axes[1,1].plot(epochs, history['val_precision'], 'b--', label='Precision')
        axes[1,1].plot(epochs, history['val_recall'], 'r--', label='Recall')
        axes[1,1].set_title('Metrics'); axes[1,1].legend(); axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total, crf, mle = 0, 0, 0
    
    for batch in tqdm(dataloader, desc="Training"):
        user_utt = batch['user_utterance'].unsqueeze(0).to(device)
        system_utt = batch['system_utterance'].unsqueeze(0).to(device)
        user_labels = batch['user_I_label'].unsqueeze(0).to(device)
        system_labels = batch['system_I_label'].unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        output = model(user_utt, system_utt, user_labels, system_labels, mode='train')
        output['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total += output['total_loss'].item()
        crf += output['loss_crf'].item()
        mle += output['loss_mle'].item()
    
    n = len(dataloader)
    return {'loss': total/n, 'crf_loss': crf/n, 'mle_loss': mle/n}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', default=None)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load & preprocess
    preprocessor = SIPPreprocessor()
    train_data = [preprocessor.process_conversation(c) for c in load_data(args.train_data)]
    train_loader = DataLoader(SIPDataset(train_data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if args.val_data:
        val_data = [preprocessor.process_conversation(c) for c in load_data(args.val_data)]
        val_loader = DataLoader(SIPDataset(val_data), batch_size=1, collate_fn=collate_fn)
    
    # Model
    model = MuSIc(hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {'train_loss':[], 'train_crf_loss':[], 'train_mle_loss':[]}
    if val_loader: history.update({'val_f1':[], 'val_precision':[], 'val_recall':[]})
    
    best_f1 = 0
    
    # Train
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        metrics = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(metrics['loss'])
        history['train_crf_loss'].append(metrics['crf_loss'])
        history['train_mle_loss'].append(metrics['mle_loss'])
        print(f"Train - Loss: {metrics['loss']:.4f}")
        
        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            print(f"Val - F1: {val_metrics['f1']:.4f}, P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/best_model.pt")
                print(f"Saved best model (F1={best_f1:.4f})")
    
    # Save
    with open(f"{args.checkpoint_dir}/history.json", 'w') as f:
        json.dump(history, f)
    plot_curves(history, f"{args.checkpoint_dir}/training_curves.png")
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
