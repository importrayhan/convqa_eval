"""
Complete Inference Script for MuSIc SIP Models

Features:
- Load trained model
- Run predictions on test data
- Generate rich output with metadata
- Per-turn predictions
- Confidence scores
- Save structured JSON output
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bilstm_crf.preprocessor import SIPPreprocessor
from models.bilstm_crf.music_baselines import create_model
from models.bilstm_crf.output_generator import SIPOutputGenerator


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"\n[Model] Loading from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Create model
    model = create_model(
        baseline=args['baseline'],
        num_classes=args['num_classes'],
        hidden_size=args.get('hidden_size', 256),
        num_layers=args.get('num_layers', 2),
        dropout=args.get('dropout', 0.5),
        lambda_mle=args.get('lambda_mle', 0.1)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[Model] Loaded successfully")
    print(f"  Baseline: {args['baseline']}")
    print(f"  Num classes: {args['num_classes']}")
    
    if 'f1' in checkpoint:
        print(f"  Best F1: {checkpoint['f1']:.4f}")
    if 'auc_roc' in checkpoint:
        print(f"  Best AUC: {checkpoint['auc_roc']:.4f}")
    
    return model, args


def run_inference(model, dataloader, preprocessor, generator, device):
    """
    Run inference and generate rich output.
    
    Returns:
        List of structured output dicts
    """
    model.eval()
    results = []
    
    print("\n[Inference] Running predictions...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            user_utt = batch['user_utterance'].unsqueeze(0).to(device)
            system_utt = batch['system_utterance'].unsqueeze(0).to(device)
            metadata = batch.get('metadata', {})
            
            # Get predictions
            output = model(user_utt, system_utt, mode='inference')
            
            predictions = output['predictions']
            confidences = output.get('confidences', [0.5] * len(predictions))
            probabilities = output.get('probabilities', None)
            
            # Generate rich output
            # Note: We need the original data for this
            # For now, create basic structure
            result = {
                'conversation_id': batch_idx,
                'num_turns': len(predictions),
                'predictions': predictions,
                'confidences': [float(c) for c in confidences],
                'per_turn_results': []
            }
            
            # Per-turn results
            for turn_idx, (pred, conf) in enumerate(zip(predictions, confidences)):
                turn_result = {
                    'turn_id': turn_idx + 1,
                    'predicted_class': int(pred),
                    'predicted_class_name': preprocessor.class_names[preprocessor.num_classes][pred],
                    'confidence': float(conf)
                }
                
                # Add probabilities if available
                if probabilities is not None and turn_idx < len(probabilities):
                    turn_result['class_probabilities'] = {
                        preprocessor.class_names[preprocessor.num_classes][i]: float(probabilities[turn_idx][i])
                        for i in range(preprocessor.num_classes)
                    }
                
                result['per_turn_results'].append(turn_result)
            
            # Overall statistics
            result['statistics'] = {
                'total_turns': len(predictions),
                'clear_turns': sum(1 for p in predictions if p == 0),
                'ambiguous_turns': sum(1 for p in predictions if p > 0),
                'highly_ambiguous_turns': sum(1 for p in predictions if p >= 2),
                'average_confidence': float(np.mean(confidences)) if confidences else 0.0
            }
            
            # Add metadata if available
            if metadata:
                result['input_metadata'] = metadata
            
            results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MuSIc SIP Inference')
    
    # Required
    parser.add_argument('--input_data', type=str, required=True,
                       help='Input JSON file with conversations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for predictions')
    
    # Optional
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (must be 1)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MuSIc SIP Inference")
    print("="*70)
    print(f"  Input: {args.input_data}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Device: {args.device}")
    print("="*70 + "\n")
    
    # Load model
    model, model_args = load_model(args.checkpoint, args.device)
    
    # Load data
    print(f"\n[Data] Loading from {args.input_data}...")
    with open(args.input_data, 'r') as f:
        conversations = json.load(f)
    
    if isinstance(conversations, dict):
        conversations = [conversations]
    
    print(f"[Data] Loaded {len(conversations)} conversations")
    
    # Preprocess
    preprocessor = SIPPreprocessor(num_classes=model_args['num_classes'])
    generator = SIPOutputGenerator(preprocessor, preprocessor.class_names[model_args['num_classes']])
    
    print("\n[Preprocess] Processing conversations...")
    processed_data = []
    for conv in tqdm(conversations):
        try:
            processed = preprocessor.process_conversation(conv, use_ground_truth=False)
            processed_data.append(processed)
        except Exception as e:
            print(f"[Warning] Failed to process conversation: {e}")
            continue
    
    print(f"[Preprocess] Successfully processed {len(processed_data)} conversations")
    
    # Create dataloader
    from torch.utils.data import Dataset, DataLoader
    
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
    
    dataset = SIPDataset(processed_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Run inference
    results = run_inference(model, dataloader, preprocessor, generator, args.device)
    
    # Save results
    print(f"\n[Output] Saving to {args.output}...")
    
    output_data = {
        'metadata': {
            'model': {
                'baseline': model_args['baseline'],
                'num_classes': model_args['num_classes'],
                'checkpoint': args.checkpoint
            },
            'data': {
                'num_conversations': len(results),
                'input_file': args.input_data
            }
        },
        'predictions': results
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    print(f"Total conversations: {len(results)}")
    
    total_turns = sum(r['num_turns'] for r in results)
    total_clear = sum(r['statistics']['clear_turns'] for r in results)
    total_ambiguous = sum(r['statistics']['ambiguous_turns'] for r in results)
    avg_confidence = np.mean([r['statistics']['average_confidence'] for r in results])
    
    print(f"Total turns: {total_turns}")
    print(f"  Clear: {total_clear} ({total_clear/total_turns*100:.1f}%)")
    print(f"  Ambiguous: {total_ambiguous} ({total_ambiguous/total_turns*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"\nResults saved to: {args.output}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
