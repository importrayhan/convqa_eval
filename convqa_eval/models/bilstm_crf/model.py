"""BiLSTM-CRF model for utterance initiative detection."""
import torch
import torch.nn as nn
from .crf import CRF


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF for System Initiative Prediction (SIP).
    
    Architecture:
        Embedding → BiLSTM → Linear → CRF
    
    Input:
        - user_utterance: [batch, turns, max_len]
        - system_utterance: [batch, turns, max_len]
    Output:
        - Initiative labels: [batch, turns] - binary (0/1)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_tags: int = 2  # Initiative=1, Non-initiative=0
    ):
        """
        Initialize BiLSTM-CRF.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: BiLSTM hidden dimension (per direction)
            num_layers: Number of BiLSTM layers
            dropout: Dropout rate
            num_tags: Number of output tags (2 for binary)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags
        
        print(f"[BiLSTMCRF] Initializing model...")
        print(f"[BiLSTMCRF] Vocab size: {vocab_size}")
        print(f"[BiLSTMCRF] Embedding dim: {embedding_dim}")
        print(f"[BiLSTMCRF] Hidden dim: {hidden_dim} (bidirectional → {hidden_dim*2})")
        print(f"[BiLSTMCRF] Num layers: {num_layers}")
        print(f"[BiLSTMCRF] Num tags: {num_tags}")
        
        # Shared embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Separate BiLSTMs for user and system utterances
        self.user_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.system_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear projection to tag space
        # Input: 2 * hidden_dim * 2 (user + system, each bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim * 4, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)
        
        print(f"[BiLSTMCRF] Model initialized successfully")
    
    def forward(
        self,
        user_utterance: torch.Tensor,
        system_utterance: torch.Tensor,
        system_I_label: torch.Tensor = None,
        mask: torch.Tensor = None
    ):
        """
        Forward pass.
        
        Args:
            user_utterance: [batch, turns, max_len] - User utterances
            system_utterance: [batch, turns, max_len] - System utterances
            system_I_label: [batch, turns] - Ground truth initiative labels
            mask: [batch, turns] - Valid turn mask
        
        Returns:
            If training: CRF loss (scalar)
            If inference: Predicted tag sequences (list of lists)
        """
        batch_size, num_turns, max_len = user_utterance.shape
        
        print(f"\n[BiLSTMCRF Forward] Batch: {batch_size}, Turns: {num_turns}, Max len: {max_len}")
        
        # Reshape for processing all turns at once
        user_flat = user_utterance.view(batch_size * num_turns, max_len)
        system_flat = system_utterance.view(batch_size * num_turns, max_len)
        
        # Embed
        user_emb = self.embedding(user_flat)  # [batch*turns, max_len, emb_dim]
        system_emb = self.embedding(system_flat)
        
        print(f"[BiLSTMCRF] User embedding shape: {user_emb.shape}")
        print(f"[BiLSTMCRF] System embedding shape: {system_emb.shape}")
        
        # BiLSTM encoding
        user_lstm_out, _ = self.user_lstm(user_emb)  # [batch*turns, max_len, hidden*2]
        system_lstm_out, _ = self.system_lstm(system_emb)
        
        # Pool: Take last hidden state (or mean/max pool)
        # Using last hidden state for each utterance
        user_repr = user_lstm_out[:, -1, :]  # [batch*turns, hidden*2]
        system_repr = system_lstm_out[:, -1, :]
        
        print(f"[BiLSTMCRF] User LSTM output: {user_repr.shape}")
        print(f"[BiLSTMCRF] System LSTM output: {system_repr.shape}")
        
        # Concatenate user and system representations
        combined = torch.cat([user_repr, system_repr], dim=-1)  # [batch*turns, hidden*4]
        combined = self.dropout(combined)
        
        # Project to tag space
        emissions = self.hidden2tag(combined)  # [batch*turns, num_tags]
        
        # Reshape back to [batch, turns, num_tags]
        emissions = emissions.view(batch_size, num_turns, self.num_tags)
        
        print(f"[BiLSTMCRF] Emissions shape: {emissions.shape}")
        
        # CRF layer
        if system_I_label is not None:  # Training mode
            loss = self.crf(emissions, system_I_label, mask)
            print(f"[BiLSTMCRF] Training mode - Loss: {loss.item():.4f}")
            return loss
        else:  # Inference mode
            predictions = self.crf.decode(emissions, mask)
            print(f"[BiLSTMCRF] Inference mode - Decoded {len(predictions)} sequences")
            return predictions
