import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network layer."""
    def __init__(self, config):
        super().__init__()
        # Standard LLaMA expansion ratio: 8/3 * d_model
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path with SiLU activation multiplied element-wise with up-projection path
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))