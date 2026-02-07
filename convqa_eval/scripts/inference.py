"""Inference Script for MuSIc"""

import os, sys, json, torch
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..models.preprocessor import SIPPreprocessor
from ..models.music_model import MuSIc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = MuSIc().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load & preprocess data
    preprocessor = SIPPreprocessor()
    with open(args.input_data, 'r') as f:
        conversations = json.load(f)
    
    results = []
    
    with torch.no_grad():
        for conv in tqdm(conversations, desc="Inference"):
            processed = preprocessor.process_conversation(conv, auto_label=False)
            
            user_utt = processed['user_utterance'].unsqueeze(0).to(device)
            system_utt = processed['system_utterance'].unsqueeze(0).to(device)
            
            output = model(user_utt, system_utt, mode='inference')
            
            results.append({
                'system_predictions': output['system_predictions'],
                'initiative_rate': output['initiative_rate']
            })
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved predictions to {args.output}")
    print(f"Avg initiative rate: {sum(r['initiative_rate'] for r in results)/len(results):.2%}")

if __name__ == "__main__":
    main()
