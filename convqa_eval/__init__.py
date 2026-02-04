"""ConvQA-Eval: Conversational Query Intent Detection Evaluation Suite."""

__version__ = "0.1.0"

from .data_loader import get_tasks, list_available_tasks
from .evaluator import ConvQAEvaluator
from .metrics import compute_metrics

__all__ = [
    "get_tasks",
    "list_available_tasks",
    "ConvQAEvaluator",
    "compute_metrics",
]
