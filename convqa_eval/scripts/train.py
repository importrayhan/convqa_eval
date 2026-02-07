import os, sys, json, torch, torch.optim as optim, argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.preprocessor import SIPPreprocessor
from model.music_baselines import create_model

class SIPDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch[0]

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            user = batch['user_utterance'].unsqueeze(0).to(device)
            system = batch['system_utterance'].unsqueeze(0).to(device)
            labels = batch['system_I_label']
            out = model(user, system, mode='inference')
            all_preds.extend(out['predictions'])
            all_labels.extend(labels.tolist())
    
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'cm': cm.tolist()}

def plot_curves(history, path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_f1' in history: axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True)
    axes[0,1].plot(epochs, history['train_crf'], 'b-'); axes[0,1].set_title('CRF Loss'); axes[0,1].grid(True)
    axes[1,0].plot(epochs, history['train_mle'], 'b-'); axes[1,0].set_title('MLE Loss'); axes[1,0].grid(True)
    if 'val_f1' in history:
        axes[1,1].plot(epochs, history['val_f1'], 'g-', label='F1')
        axes[1,1].plot(epochs, history['val_precision'], 'b--', label='P')
        axes[1,1].plot(epochs, history['val_recall'], 'r--', label='R')
        axes[1,1].set_title('Metrics'); axes[1,1].legend(); axes[1,1].grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', default=None)
    parser.add_argument('--baseline', default='vanillacrf', choices=['vanillacrf','features','dynamic','ctxpred','music'])
    parser.add_argument('--num_classes', type=int, default=2, choices=[2,3,4])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    preprocessor = SIPPreprocessor(num_classes=args.num_classes)
    with open(args.train_data) as f: train_raw = json.load(f)
    train_data = [preprocessor.process_conversation(c) for c in train_raw]
    train_loader = DataLoader(SIPDataset(train_data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if args.val_data:
        with open(args.val_data) as f: val_raw = json.load(f)
        val_data = [preprocessor.process_conversation(c) for c in val_raw]
        val_loader = DataLoader(SIPDataset(val_data), batch_size=1, collate_fn=collate_fn)
    
    model = create_model(args.baseline, num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {'train_loss':[], 'train_crf':[], 'train_mle':[]}
    if val_loader: history.update({'val_f1':[], 'val_precision':[], 'val_recall':[], 'val_loss':[]})
    best_f1 = 0
    
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            user = batch['user_utterance'].unsqueeze(0).to(device)
            system = batch['system_utterance'].unsqueeze(0).to(device)
            u_lab = batch['user_I_label'].unsqueeze(0).to(device)
            s_lab = batch['system_I_label'].unsqueeze(0).to(device)
            optimizer.zero_grad()
            out = model(user, system, u_lab, s_lab, mode='train')
            out['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out['total_loss'].item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['train_crf'].append(avg_loss)
        history['train_mle'].append(0)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        
        if val_loader:
            metrics = evaluate(model, val_loader, device, args.num_classes)
            history['val_f1'].append(metrics['f1'])
            history['val_precision'].append(metrics['precision'])
            history['val_recall'].append(metrics['recall'])
            history['val_loss'].append(avg_loss)
            print(f"Val: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/best_model.pt")
                print(f"Saved best (F1={best_f1:.4f})")
    
    with open(f"{args.checkpoint_dir}/history.json", 'w') as f: json.dump(history, f)
    plot_curves(history, f"{args.checkpoint_dir}/curves.png")
    print(f"Done! Best F1: {best_f1:.4f}")

if __name__ == "__main__": main()
