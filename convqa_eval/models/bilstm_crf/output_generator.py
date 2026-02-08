"""
Output Generator for SIP Predictions

Generates structured output with metadata:
{
  "ambiguous_utterance": bool,
  "total_candidates": int,
  "explanation": str,
  "conditions": [(condition, answer), ...],
  "metadata": {confidence, CRF_layer, token_importance, etc.}
}
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class SIPOutputGenerator:
    """Generates structured output from model predictions"""
    
    def __init__(self, preprocessor, class_names: List[str]):
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def generate_conditions(self, user_text: str, system_text: str, 
                           pred_class: int) -> List[Tuple[str, str]]:
        """
        Generate candidate conditions based on ambiguity type.
        
        For ambiguous cases, generate possible interpretations.
        """
        if pred_class == 0:  # Clear
            return []
        
        conditions = []
        
        # Parse user query for ambiguous terms
        user_lower = user_text.lower()
        
        # Location ambiguity
        if any(word in user_lower for word in ['there', 'it', 'that place']):
            conditions.append(("Location: Paris", "Weather: 22°C sunny"))
            conditions.append(("Location: London", "Weather: 15°C cloudy"))
        
        # Time ambiguity
        if any(word in user_lower for word in ['tomorrow', 'later', 'soon']):
            conditions.append(("Time: Tomorrow morning", "Departure: 8AM"))
            conditions.append(("Time: Tomorrow evening", "Departure: 6PM"))
        
        # Entity ambiguity
        if any(word in user_lower for word in ['flight', 'hotel', 'booking']):
            conditions.append(("Type: One-way flight", "Price: $450"))
            conditions.append(("Type: Round-trip flight", "Price: $780"))
        
        # Generic fallback
        if not conditions:
            conditions.append(("Interpretation 1", "Possible answer 1"))
            conditions.append(("Interpretation 2", "Possible answer 2"))
        
        return conditions[:min(len(conditions), pred_class + 1)]  # More ambiguous = more candidates
    
    def generate_explanation(self, user_text: str, system_text: str,
                            pred_class: int, confidence: float) -> str:
        """Generate human-readable explanation"""
        
        if pred_class == 0:
            return "Query is clear and unambiguous."
        
        explanations = {
            1: f"Query needs minor clarification (confidence: {confidence:.2%}). ",
            2: f"Query requires significant clarification (confidence: {confidence:.2%}). ",
            3: f"Query is highly ambiguous (confidence: {confidence:.2%}). "
        }
        
        base_exp = explanations.get(pred_class, "Query may be ambiguous. ")
        
        # Add specific reasons
        reasons = []
        user_lower = user_text.lower()
        
        if any(word in user_lower for word in ['it', 'that', 'this', 'there']):
            reasons.append("contains ambiguous pronouns")
        if len(user_text.split()) < 5:
            reasons.append("is very short")
        if any(word in user_lower for word in ['what about', 'how about', 'and']):
            reasons.append("uses follow-up patterns")
        
        if reasons:
            base_exp += "Reasons: " + ", ".join(reasons) + "."
        
        return base_exp
    
    def generate_token_importance(self, user_text: str, 
                                  emissions: torch.Tensor) -> Dict[str, float]:
        """
        Generate token-level importance scores.
        
        (Simplified - in practice would use attention weights)
        """
        tokens = user_text.split()
        
        # Mock importance (would use actual attention in production)
        importance = {}
        for i, token in enumerate(tokens):
            # Ambiguous tokens get higher scores
            if token.lower() in ['it', 'that', 'this', 'there', 'what']:
                importance[token] = 0.8 + 0.2 * np.random.rand()
            else:
                importance[token] = 0.3 * np.random.rand()
        
        return importance
    
    def generate_output(
        self,
        data: Dict,
        predictions: List[int],
        confidences: List[float],
        crf_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate complete structured output.
        
        Args:
            data: Original input data
            predictions: List of class predictions per turn
            confidences: List of confidence scores per turn
            crf_metadata: Optional CRF layer metadata
        
        Returns:
            Structured output dictionary
        """
        # Get last prediction (final system response)
        final_pred = predictions[-1] if predictions else 0
        final_conf = confidences[-1] if confidences else 0.0
        
        # Parse conversation for context
        conversations = data.get('conversations', [])
        user_text = ""
        system_text = ""
        
        for conv in conversations:
            if conv['from'] == 'human':
                user_text = conv['value']
            elif conv['from'] == 'gpt':
                system_text = conv['value']
        
        # Generate components
        is_ambiguous = final_pred > 0
        conditions = self.generate_conditions(user_text, system_text, final_pred)
        explanation = self.generate_explanation(user_text, system_text, final_pred, final_conf)
        token_importance = self.generate_token_importance(user_text, None)
        
        # Build output
        output = {
            "ambiguous_utterance": is_ambiguous,
            "ambiguous_class": self.class_names[final_pred],
            "total_candidates": len(conditions),
            "explanation": explanation,
            "conditions": [
                {"condition": cond, "answer": ans}
                for cond, ans in conditions
            ],
            "metadata": {
                "confidence_score": float(final_conf),
                "all_predictions": predictions,
                "all_confidences": [float(c) for c in confidences],
                "num_turns": len(predictions),
                "num_classes": self.num_classes,
                "class_distribution": self._get_class_distribution(predictions),
                "token_importance": token_importance,
                "CRF_layer": crf_metadata or {},
                "input_metadata": {
                    "prompt": data.get('prompt', ''),
                    "context": data.get('context', ''),
                    "can_retrieve": data.get('can_retrieve', False),
                    "tools": data.get('tools', '')
                }
            }
        }
        
        return output
    
    def _get_class_distribution(self, predictions: List[int]) -> Dict[str, int]:
        """Get distribution of predicted classes"""
        dist = {name: 0 for name in self.class_names}
        for pred in predictions:
            dist[self.class_names[pred]] += 1
        return dist


if __name__ == "__main__":
    # Test
    from preprocessor import SIPPreprocessor
    
    preprocessor = SIPPreprocessor(num_classes=3)
    generator = SIPOutputGenerator(preprocessor, preprocessor.class_names[3])
    
    data = {
        "prompt": "Book travel",
        "conversations": [
            {"from": "human", "value": "Book it there"},
            {"from": "gpt", "value": "Where would you like to go?"}
        ]
    }
    
    output = generator.generate_output(
        data,
        predictions=[0, 2],
        confidences=[0.95, 0.78]
    )
    
    import json
    print(json.dumps(output, indent=2))
