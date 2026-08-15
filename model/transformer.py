import sys
from pathlib import Path
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.normalization import RMSNorm
from model.block import TransformerBlock

class DecoderTransformer(nn.Module):
    """Full Decoder-Only Transformer Language Model."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd)
        ))
        
        # Language Model Head (maps hidden representation -> vocab logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: Share weights between token embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_key_values=None, use_cache=False):
        B, T = idx.size()
        
        # 1. Pass tokens through embedding (wte) and dropout
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        # 2. Pass through Transformer Blocks (with KV-Cache support)
        presents = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                x, present = block(x, layer_past=layer_past, use_cache=use_cache)
                presents.append(present)
            else:
                x = block(x)

        # 3. Final Norm
        x = self.transformer.ln_f(x)

        # 4. Calculate Logits & Loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
        else:
            logits = self.lm_head(x)
            loss = None

        if use_cache:
            return logits, presents
        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 1. Collect all parameters requiring gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    
        # 2. Separate into decay (2D+ matrices) and no-decay (1D vectors like norms/biases)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
    
        # 3. Use fused AdamW if running on CUDA for maximum GPU throughput
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device_type)
        extra_args = dict(fused=True) if use_fused else dict()
    
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
        return optimizer

