import math
import sys
from pathlib import Path 
from networkx import config
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.embedding import RotaryEmbedding, apply_rotary_emb

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and FlashAttention."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout

    def forward(self, x, layer_past=None, use_cache=False):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension (n_embd)
        
        # Projects Q, K, V in one go for efficiency
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

         # Glue past keys and values if present
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        present = (k, v) if use_cache else None
        
        # Compute Attention
        if use_cache and T == 1:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        # Return tuple if caching, otherwise return just tensor y
        if use_cache:
            return y, present
        return y  # <--- Ensures y is ALWAYS returned when use_cache=False

        