"""BiLSTM-CRF baseline wrapper for ConvQA-Eval."""
import torch
from typing import List, Dict
from ..base import BaseConvQAModel
from .model import BiLSTMCRF
from ..preprocessing.sip_preprocessor import SIPPreprocessor


class BiLSTMCRFBaseline(BaseConvQAModel):
    """
    BiLSTM-CRF baseline for conversational initiative prediction.
    
    Predicts whether system should take initiative based on conversation history.
    """
    
    def __init__(
        self,
        model_path: str = None,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize BiLSTM-CRF baseline.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            device: Device (cpu/cuda)
        """
        self.device = torch.device(device)
        self.preprocessor = SIPPreprocessor(vocab_size=vocab_size)
        
        print(f"\n[BiLSTMCRFBaseline] Initializing on device: {device}")
        
        # Initialize model
        self.model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Load weights if provided
        if model_path:
            print(f"[BiLSTMCRFBaseline] Loading weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        print(f"[BiLSTMCRFBaseline] Model ready for inference")
    
    def predict(self, inputs: List[Dict]) -> List[Dict]:
        """
        Generate predictions for batch of inputs.
        
        Args:
            inputs: List of input dicts from QuAC format
        
        Returns:
            List of prediction dicts with ambiguity detection
        """
        print(f"\n[BiLSTMCRFBaseline] Predicting for {len(inputs)} inputs...")
        
        # Preprocess
        user_utt, system_utt, _, _, mask = self.preprocessor.preprocess_batch(inputs)
        
        user_utt = user_utt.to(self.device)
        system_utt = system_utt.to(self.device)
        mask = mask.to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(user_utt, system_utt, mask=mask)
        
        # Convert predictions to output format
        outputs = []
        for batch_idx, pred_seq in enumerate(predictions):
            # pred_seq is a list of binary labels [0, 1, 1, 0, ...]
            # Check if system should take initiative (any 1 in sequence)
            system_should_initiate = any(label == 1 for label in pred_seq)
            num_initiative_turns = sum(pred_seq)
            
            print(f"  [Prediction {batch_idx}] Initiative sequence: {pred_seq[:10]}...")
            print(f"  [Prediction {batch_idx}] System should initiate: {system_should_initiate}")
            
            output = {
                "ambiguous_utterance": system_should_initiate,
                "total_candidates": num_initiative_turns,
                "explanation": f"BiLSTM-CRF detected {num_initiative_turns} turns requiring initiative",
                "conditions": {
                    "system_initiative_required": str(system_should_initiate)
                },
                "metadata": {
                    "model": "BiLSTM-CRF",
                    "initiative_sequence": pred_seq,
                    "num_turns": len(pred_seq)
                }
            }
            outputs.append(output)
        
        print(f"[BiLSTMCRFBaseline] Generated {len(outputs)} predictions")
        return outputs
