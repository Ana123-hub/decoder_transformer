import os
import torch

def save_checkpoint(model, optimizer, step: int, loss: float, out_dir: str = "out", filename: str = "ckpt.pt"):
    """Saves model state, optimizer state, and current step index."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved state to {path} (Step {step}, Loss {loss:.4f})")

def load_checkpoint(path: str, model, optimizer=None):
    """Loads weights into model and state into optimizer."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded state from {path} (Resuming at Step {checkpoint.get('step', 0)})")
    return checkpoint.get("step", 0)