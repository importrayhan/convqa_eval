"""Example script to run evaluation."""
from convqa_eval import get_tasks, ConvQAEvaluator
from convqa_eval.models.pyterrier_baseline import PyTerrierRAGBaseline


def main():
    # Load tasks
    tasks = get_tasks(["quac", "coqa"])
    print(f"Loaded {len(tasks)} tasks")
    
    # Initialize model
    model = PyTerrierRAGBaseline()
    
    # Run evaluation
    evaluator = ConvQAEvaluator(tasks=tasks, batch_size=16)
    results = evaluator.run(
        model=model,
        output_folder="results/pyterrier_baseline",
        save_predictions=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task_name, task_results in results.items():
        print(f"\n{task_name}:")
        for metric, value in task_results['metrics'].items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
