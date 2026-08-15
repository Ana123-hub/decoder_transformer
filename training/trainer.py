import torch

@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters: int, device: str, dtype: torch.dtype):
    """Evaluates cross-entropy loss across multiple batches in evaluation mode."""
    model.eval()
    losses = torch.zeros(eval_iters)
    data_iter = iter(dataloader)
    
    for k in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
            _, loss = model(x, y)
        losses[k] = loss.item()
        
    model.train()
    return losses.mean().item()

def train_step(model, optimizer, dataloader, grad_accum_steps: int, max_grad_norm: float, device: str, dtype: torch.dtype, scaler=None):
    """Executes one effective optimization step across grad_accum_steps micro-batches."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    data_iter = iter(dataloader)

    for micro_step in range(grad_accum_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass under Automatic Mixed Precision
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
            _, loss = model(x, y)
            loss = loss / grad_accum_steps

        accumulated_loss += loss.item()

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # Gradient clipping to prevent exploding gradients
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return accumulated_loss * grad_accum_steps