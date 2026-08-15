import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """Computes and caches Rotary Positional Embeddings (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # Inverse frequencies theta_i = 10000^(-2(i-1)/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute frequency grid across maximum sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # Shape: (max_seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # Shape: (max_seq_len, dim)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: int):
        # Returns cos and sin tensors trimmed to current sequence length
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates spatial dimensions for complex rotation: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies RoPE rotation to query and key tensors."""
    # Reshape cos and sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed