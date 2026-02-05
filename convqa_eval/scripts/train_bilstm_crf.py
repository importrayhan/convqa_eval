"""Training script for BiLSTM-CRF model."""
import json
import torch
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from convqa_eval.models.bilstm_crf.model import BiLSTMCRF
from convqa_eval.models.preprocessing.sip_preprocessor import SIPPreprocessor


def train():
    """Train BiLSTM-CRF model."""
    print("="*60)
    print("BiLSTM-CRF Training Script")
    print("="*60)
    
    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n[Train] Device: {DEVICE}")
    print(f"[Train] Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "quac_sample.json"
    print(f"\n[Train] Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"[Train] Loaded {len(data)} examples")
    
    # Initialize preprocessor and build vocab
    preprocessor = SIPPreprocessor(vocab_size=VOCAB_SIZE)
    preprocessor.build_vocab(data)
    
    # Initialize model
    print(f"\n[Train] Initializing model...")
    model = BiLSTMCRF(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n[Train] Starting training...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Process in batches
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            
            # Preprocess
            user_utt, system_utt, system_label, _, mask = preprocessor.preprocess_batch(batch)
            
            user_utt = user_utt.to(DEVICE)
            system_utt = system_utt.to(DEVICE)
            system_label = system_label.to(DEVICE)
            mask = mask.to(DEVICE)
            
            # Forward pass
            loss = model(user_utt, system_utt, system_label, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {num_batches}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        print("-"*60)
    
    # Save model
    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model_path = checkpoint_dir / "bilstm_crf_model.pt"
    
    torch.save(model.state_dict(), model_path)
    print(f"\n[Train] Model saved to {model_path}")
    
    # Save vocabulary
    vocab_path = checkpoint_dir / "vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(preprocessor.word2idx, f, indent=2)
    
    print(f"[Train] Vocabulary saved to {vocab_path}")
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    train()
