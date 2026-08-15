import torch
from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Configuration class for training hyperparameters and hardware settings."""
    # Data paths
    dataset_name: str = "tinyshakespeare"
    data_dir: str = "data_raw/tinyshakespeare"
    
    # Batching & Optimization
    batch_size: int = 16          # Micro-batch size per forward pass
    grad_accum_steps: int = 4     # Gradient accumulation steps (Effective batch size = 16 * 4 = 64)
    learning_rate: float = 1e-3   # Maximum learning rate
    max_lr: float = 1e-3
    min_lr: float = 1e-4          # Minimum learning rate after decay
    weight_decay: float = 0.1     # Weight decay (L2 regularization) for non-bias weights
    max_grad_norm: float = 1.0    # Maximum gradient norm for gradient clipping
    
    # Schedule & Steps
    max_steps: int = 1000         # Total training steps for Phase 1
    warmup_steps: int = 100       # Warmup steps for cosine scheduler
    
    # Hardware & Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
    
    # Checkpointing & Logging
    eval_interval: int = 100      # How often to compute validation loss
    log_interval: int = 10        # How often to log training loss
    save_interval: int = 500      # How often to save model checkpoints
    out_dir: str = "out"          # Output directory for checkpoints

    