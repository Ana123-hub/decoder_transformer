import math

def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    """Calculates learning rate at iteration step using linear warmup + cosine decay."""
    # 1. Linear warmup phase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2. Post-max steps floor
    if step > max_steps:
        return min_lr
    
    # 3. Cosine decay phase
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)