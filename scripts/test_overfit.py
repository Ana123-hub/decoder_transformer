import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from config.model_config import ModelConfig
from model.transformer import DecoderTransformer

def run_overfit_test():
    print("=" * 60)
    print("RUNNING SINGLE-BATCH OVERFIT SANITY TEST")
    print("=" * 60)

    # 1. Instantiate tiny model config
    config = ModelConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,  # Turn off dropout for deterministic overfit test
        bias=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderTransformer(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-3, device_type=device)

    # 2. Create 1 single fake data batch (Batch Size = 2, Sequence Length = 64)
    torch.manual_seed(42)
    x = torch.randint(0, config.vocab_size, (2, config.block_size), device=device)
    y = torch.randint(0, config.vocab_size, (2, config.block_size), device=device)

    print(f"Device: {device}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training on 1 static batch for 150 iterations...\n")

    model.train()
    for step in range(1, 151):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % 15 == 0 or step == 1:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    if loss.item() < 0.1:
        print("TEST PASSED! Model successfully overfitted on single batch.")
        print("Your backpropagation, embeddings, attention, and loss logic are sound!")
    else:
        print("TEST FAILED! Loss did not converge. Check learning rate or gradient flow.")
    print("=" * 60)

if __name__ == "__main__":
    run_overfit_test()