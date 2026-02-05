"""Abstract base model interface."""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseConvQAModel(ABC):
    """
    Abstract base class for conversational QA intent detection models.
    
    All models must implement the predict() method.
    """
    
    @abstractmethod
    def predict(self, inputs: List[Dict]) -> List[Dict]:
        """
        Generate predictions for a batch of inputs.
        
        Args:
            inputs: List of input dictionaries with keys:
                - prompt: str
                - context: str
                - can_retrieve: bool
                - tools: List[str]
                - conversation: List[Dict]
        
        Returns:
            List of prediction dictionaries with keys:
                - ambiguous_utterance: bool
                - total_candidates: int
                - explanation: str
                - conditions: Dict[str, str]
        """
        pass
    
    def __call__(self, inputs: List[Dict]) -> List[Dict]:
        """Alias for predict()."""
        return self.predict(inputs)
