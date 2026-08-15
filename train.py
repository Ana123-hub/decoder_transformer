import os
import time
import math
import torch

from config.model_config import ModelConfig
from config.train_config import TrainConfig
from model.transformer import DecoderTransformer
from data.dataset import get_dataloader
from training.scheduler import get_lr
from training.trainer import train_step, estimate_loss
from training.checkpointing import save_checkpoint

def main():
    # 1. Load Configurations
    m_config = ModelConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=False
    )
    
    t_config = TrainConfig(
        batch_size=32,
        max_lr=1e-3,
        min_lr=1e-4,
        max_steps=2000,
        warmup_steps=100,
        weight_decay=0.1,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        eval_interval=100,
        log_interval=20,
        save_interval=500,
        out_dir="out"
    )

    # 2. Setup Device & Mixed Precision Context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"Using Device: {device} | Autocast Precision: {dtype}")

    # 3. Instantiate Data Loaders
    train_bin = os.path.join("data", "tinyshakespeare", "train.bin")
    val_bin = os.path.join("data", "tinyshakespeare", "val.bin")

    train_loader = get_dataloader(train_bin, m_config.block_size, t_config.batch_size, shuffle=True)
    val_loader = get_dataloader(val_bin, m_config.block_size, t_config.batch_size, shuffle=False)

    # 4. Instantiate Model & Optimizer
    torch.manual_seed(1337)
    model = DecoderTransformer(m_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=t_config.weight_decay,
        learning_rate=t_config.max_lr,
        device_type=device
    )

    # AMP GradScaler (used for float16; float32/bfloat16 run cleanly without scaling)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and dtype == torch.float16) else None

    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting Training Loop...\n")

    # 5. Training Loop
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(1, t_config.max_steps + 1):
        # Update learning rate via scheduler
        lr = get_lr(step, t_config.max_steps, t_config.warmup_steps, t_config.max_lr, t_config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Execute 1 training optimization step
        step_loss = train_step(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            grad_accum_steps=t_config.grad_accum_steps,
            max_grad_norm=t_config.max_grad_norm,
            device=device,
            dtype=dtype,
            scaler=scaler
        )

        # Logging
        if step % t_config.log_interval == 0 or step == 1:
            dt = time.time() - start_time
            print(f"Step {step:4d}/{t_config.max_steps} | Loss: {step_loss:.4f} | LR: {lr:.6f} | Time: {dt:.2f}s")
            start_time = time.time()

        # Evaluation & Checkpointing
        if step % t_config.eval_interval == 0 or step == t_config.max_steps:
            val_loss = estimate_loss(model, val_loader, eval_iters=20, device=device, dtype=dtype)
            print(f"\n---> Evaluation at Step {step}: Val Loss = {val_loss:.4f}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, out_dir=t_config.out_dir, filename="best_model.pt")
            print()

        # Periodic checkpointing
        if step % t_config.save_interval == 0:
            save_checkpoint(model, optimizer, step, step_loss, out_dir=t_config.out_dir, filename=f"ckpt_step_{step}.pt")

    print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()