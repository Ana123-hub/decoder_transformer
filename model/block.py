import sys
from pathlib import Path
import torch.nn as nn

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.normalization import RMSNorm
from model.attention import CausalSelfAttention
from model.mlp import SwiGLUMLP

class TransformerBlock(nn.Module):
    """Single Transformer Block (Pre-RMSNorm, Causal Attention, SwiGLU MLP)."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, layer_past=None, use_cache=False):
            if use_cache:
                # When KV-Cache is active, receive the updated attention output and present cache key/values
                attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
                x = x + attn_out
                x = x + self.mlp(self.ln_2(x))
                return x, present
            else:
                # Standard forward pass (Training)
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
                return x

   