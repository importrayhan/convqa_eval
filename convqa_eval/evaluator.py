"""Main evaluation engine for conversational QA intent detection."""
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time

from .metrics import compute_metrics


class ConvQAEvaluator:
    """
    Main evaluator class for conversational query intent detection.
    
    Usage:
        >>> tasks = get_tasks(["quac"])
        >>> evaluator = ConvQAEvaluator(tasks=tasks, batch_size=32)
        >>> results = evaluator.run(model, output_folder="results/my_model")
    """
    
    def __init__(self, tasks: List[Dict], batch_size: int = 32, verbose: bool = True):
        """
        Initialize evaluator.
        
        Args:
            tasks: List of task dictionaries from get_tasks()
            batch_size: Batch size for model inference
            verbose: Print progress information
        """
        self.tasks = tasks
        self.batch_size = batch_size
        self.verbose = verbose
    
    def run(
        self,
        model,
        output_folder: Optional[str] = None,
        save_predictions: bool = True
    ) -> Dict:
        """
        Run evaluation on all tasks.
        
        Args:
            model: Model instance implementing predict() method
            output_folder: Path to save results
            save_predictions: Whether to save individual predictions
        
        Returns:
            Dictionary containing results for all tasks
        """
        all_results = {}
        
        for task in self.tasks:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating on {task['name']}: {task['description']}")
                print(f"{'='*60}")
            
            task_results = self._evaluate_task(task, model, save_predictions)
            all_results[task['name']] = task_results
            
            if self.verbose:
                print(f"\nResults for {task['name']}:")
                for metric, value in task_results['metrics'].items():
                    print(f"  {metric}: {value:.4f}")
        
        # Save results
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / "results.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            
            if self.verbose:
                print(f"\nResults saved to {output_path}")
        
        return all_results
    
    def _evaluate_task(self, task: Dict, model, save_predictions: bool) -> Dict:
        """Evaluate on a single task."""
        data = task['data']
        predictions = []
        
        # Batch inference
        for i in tqdm(range(0, len(data), self.batch_size), desc="Inference", disable=not self.verbose):
            batch = data[i:i + self.batch_size]
            batch_inputs = [self._format_input(item) for item in batch]
            batch_preds = model.predict(batch_inputs)
            predictions.extend(batch_preds)
        
        # Compute metrics
        metrics = compute_metrics(predictions, data, task['config'])
        
        return {
            "task_name": task['name'],
            "num_examples": len(data),
            "metrics": metrics,
            "predictions": predictions if save_predictions else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _format_input(self, item: Dict) -> Dict:
        """Format data item into model input."""
        return {
            "prompt": item.get("question", ""),
            "context": item.get("context", ""),
            "can_retrieve": item.get("can_retrieve", True),
            "tools": item.get("tools", []),
            "conversation": item.get("conversation", [])
        }
