"""
Baseline 5: MuSIc (Full Model)
Complete MuSIc with all multi-turn features and feature-aware CRF.
"""

from baseline1_vanillacrf import VanillaCRF


class MuSIc(VanillaCRF):
    """
    Baseline 5: MuSIc (Complete Model)
    
    Full MuSIc architecture combining:
    1. BERT Utterance Encoder
    2. Prior-Posterior Inter-Utterance Encoders
    3. Multi-Turn Feature-Aware CRF with:
       - Who2Who transitions
       - Position-specific transitions  
       - Initiative count (Intime)
       - Initiative distance
    
    CRF Type: DistanceCRF (uses all features)
    
    From paper:
    "MuSIc consists of three parts: (i) a BERT utterance encoder,
    (ii) prior-posterior inter-utterance encoders, and (iii) a
    multi-turn feature-aware CRF layer."
    
    Use case: Best performance, full model
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-multilingual-cased',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_tags: int = 2,
        lambda_mle: float = 0.1
    ):
        # Initialize with VanillaCRF structure
        super().__init__(
            bert_model_name, hidden_size, num_layers, dropout, num_tags, lambda_mle
        )
        
        # Override CRF type to use full feature-aware variant
        self.crf_type = 'DistanceCRF'  # Uses all features
        
        print(f"[MuSIc] Initialized")
        print("  Components:")
        print("    ✓ BERT Utterance Encoder")
        print("    ✓ Prior-Posterior Inter-Utterance Encoders")
        print("    ✓ Multi-Turn Feature-Aware CRF (DistanceCRF)")
        print("  Features: Who2Who, Position, Intime, Distance")
        print("  Best for: Maximum performance\n")
