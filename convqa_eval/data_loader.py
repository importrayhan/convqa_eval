"""Task loader for conversational QA benchmarks."""
import json
from pathlib import Path
from typing import List, Dict, Optional

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"
DATA_DIR = Path(__file__).parent.parent / "data"

AVAILABLE_TASKS = {
    "quac": "Question Answering in Context",
    "coqa": "Conversational Question Answering",
    "qrecc": "Question Rewriting in Conversational Context",
}


def list_available_tasks() -> Dict[str, str]:
    """List all available benchmarks."""
    return AVAILABLE_TASKS.copy()


def get_tasks(tasks: Optional[List[str]] = None, data_dir: Optional[str] = None) -> List[Dict]:
    """
    Load specified benchmark tasks.
    
    Args:
        tasks: List of task names. If None, loads all available tasks.
        data_dir: Custom data directory path. If None, uses default.
    
    Returns:
        List of task dictionaries with config and data.
    
    Example:
        >>> tasks = get_tasks(["quac", "coqa"])
        >>> print(tasks[0]['name'])
        'quac'
    """
    if tasks is None:
        tasks = list(AVAILABLE_TASKS.keys())
    
    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = DATA_DIR
    
    loaded_tasks = []
    for task_name in tasks:
        if task_name not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(AVAILABLE_TASKS.keys())}")
        
        # Load benchmark config
        config_file = BENCHMARK_DIR / f"{task_name}.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Load data
        data_file = data_path / f"{task_name}_sample.json"
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        loaded_tasks.append({
            "name": task_name,
            "description": AVAILABLE_TASKS[task_name],
            "config": config,
            "data": data
        })
    
    return loaded_tasks


def load_custom_task(task_name: str, config_path: str, data_path: str) -> Dict:
    """Load a custom benchmark task."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return {
        "name": task_name,
        "description": config.get("description", "Custom task"),
        "config": config,
        "data": data
    }
