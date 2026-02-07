import os, sys, json, torch, argparse
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.preprocessor import SIPPreprocessor
from model.music_baselines import create_model
from model.output_generator import SIPOutputGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline', default='vanillacrf')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    preprocessor = SIPPreprocessor(num_classes=args.num_classes)
    model = create_model(args.baseline, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    generator = SIPOutputGenerator(preprocessor, preprocessor.class_names[args.num_classes])
    
    with open(args.input_data) as f: conversations = json.load(f)
    results = []
    
    with torch.no_grad():
        for conv in tqdm(conversations, desc="Inference"):
            processed = preprocessor.process_conversation(conv, use_ground_truth=False)
            user = processed['user_utterance'].unsqueeze(0).to(device)
            system = processed['system_utterance'].unsqueeze(0).to(device)
            out = model(user, system, mode='inference')
            
            result = generator.generate_output(
                conv,
                predictions=out['predictions'],
                confidences=out['confidences']
            )
            results.append(result)
    
    with open(args.output, 'w') as f: json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {args.output}")

if __name__ == "__main__": main()
