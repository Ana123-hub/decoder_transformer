from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Configuration class for the Decoder-Only Transformer architecture, comprising phase1 and phase2 model training."""
    block_size: int = 256        # Maximum context length (sequence length N)
    vocab_size: int = 50257      # Vocabulary size (will update after training tokenizer)
    n_layer: int = 6             # Number of Transformer blocks
    n_head: int = 6              # Number of attention heads
    n_embd: int = 384            # Embedding dimension (d_model)
    dropout: float = 0.1         # Dropout probability
    bias: bool = False           # True: bias in Linears and LayerNorms, like GPT-2. False: modern LLM style (LLaMA)

    @classmethod
    def from_preset(cls, phase_name: str) -> "ModelConfig":
        presets: Dict[str, Dict[str, Any]] = {
            "phase1_shakespeare": {
                "vocab_size": 5000,
                "block_size": 128,
                 "n_layer": 4,
                 "n_head": 4,
                "n_embd": 256,
            },

            "phase2_tinystories": {
            "vocab_size": 10000,
            "block_size": 256,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            }
        }          
        if phase_name not in presets:
            raise ValueError(f"Unknown preset: {phase_name}. Choose from {list(presets.keys())}")
        return cls(**presets[phase_name])
        

    # Sanity check validation
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"Embedding dim ({self.n_embd}) must be divisible by head count ({self.n_head})."
        self.head_dim = self.n_embd // self.n_head