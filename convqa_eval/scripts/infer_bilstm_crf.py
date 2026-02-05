"""Inference script for BiLSTM-CRF model."""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from convqa_eval.models.bilstm_crf.baseline import BiLSTMCRFBaseline


def infer():
    """Run inference with trained BiLSTM-CRF."""
    print("="*60)
    print("BiLSTM-CRF Inference Script")
    print("="*60)
    
    # Paths
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "bilstm_crf_model.pt"
    data_path = Path(__file__).parent.parent.parent / "data" / "quac_sample.json"
    output_path = Path(__file__).parent.parent.parent / "results" / "bilstm_crf_predictions.json"
    
    print(f"\n[Infer] Model: {checkpoint_path}")
    print(f"[Infer] Data: {data_path}")
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"[Infer] Loaded {len(data)} examples")
    
    # Initialize model
    model = BiLSTMCRFBaseline(
        model_path=str(checkpoint_path),
        device="cpu"
    )
    
    # Build vocab from data
    model.preprocessor.build_vocab(data)
    
    # Run inference
    print(f"\n[Infer] Running inference...")
    predictions = model.predict(data)
    
    # Add input data to predictions for reference
    for i, pred in enumerate(predictions):
        pred["input"] = {
            "id": data[i].get("id"),
            "question": data[i].get("question")
        }
    
    # Save predictions
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n[Infer] Predictions saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Prediction Summary:")
    print("="*60)
    
    num_ambiguous = sum(1 for p in predictions if p["ambiguous_utterance"])
    print(f"Total examples: {len(predictions)}")
    print(f"Ambiguous utterances: {num_ambiguous} ({num_ambiguous/len(predictions)*100:.1f}%)")
    
    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    Question: {pred['input']['question']}")
        print(f"    Ambiguous: {pred['ambiguous_utterance']}")
        print(f"    Explanation: {pred['explanation']}")
        print(f"    Initiative turns: {pred['total_candidates']}")


if __name__ == "__main__":
    infer()
