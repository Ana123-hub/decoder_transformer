import os
from pyexpat import model
import argparse
import torch
import torch.nn as nn
from model.transformer import DecoderTransformer
from config.model_config import ModelConfig
from utils.sampling import sample_next_token
import inspect 

# Note: Adjust import based on your tokenizer path
from tokenizers import Tokenizer
# Fallback/helper if sample_next_token is imported from another utility module
try:
    from utils.sampling import sample_next_token
except ImportError:
    # Inline definition if not imported externally
    import torch.nn.functional as F

    def sample_next_token(logits, generated_tokens=None, repetition_penalty=1.2, temperature=0.7, top_k=20):
        logits = logits.clone()
        if generated_tokens and repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        logits = logits / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# Phase Artifact Mapping Configuration
PHASE_MAP = {
    "1": {
        "dir": os.path.join("artifacts", "phase1_tinyshakespeare"),
        "ckpt_file": "best_model.pt",
        "default_prompt": "FIRST CITIZEN:\nBefore we proceed any further, hear me speak.",
        "preset_key": "phase1_shakespeare"
    },
    "2": {
        "dir": os.path.join("artifacts", "phase2_tinystories"),
        "ckpt_file": "best_model2.pt",
        "default_prompt": "Once upon a time, there was a little bird named Tim.",
        "preset_key": "phase2_tinystories"
    }
}

class StandardMLP(nn.Module):
    """Fallback Standard GELU MLP for checkpoints trained without SwiGLU."""
    def __init__(self, n_embd: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

def load_phase_artifacts(phase_key: str, device: str):
    """Loads checkpoint, tokenizer, and config with shape and MLP auto-adaptation."""
    if phase_key not in PHASE_MAP:
        raise ValueError(f"Invalid phase '{phase_key}'. Choose '1' or '2'.")

    phase_info = PHASE_MAP[phase_key]
    artifact_dir = phase_info["dir"]
    ckpt_path = os.path.join(artifact_dir, phase_info["ckpt_file"])
    tokenizer_path = os.path.join(artifact_dir, "tokenizer.json")

    # 1. Load Tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Could not find tokenizer at {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # 2. Load Checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"--> Loading Phase {phase_key} checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    step_info = checkpoint.get('step', checkpoint.get('epoch', 'unknown')) if isinstance(checkpoint, dict) else 'unknown'
    loss_val = checkpoint.get('loss', checkpoint.get('val_loss', 'N/A')) if isinstance(checkpoint, dict) else 'N/A'
    loss_str = f"{loss_val:.4f}" if isinstance(loss_val, float) else str(loss_val)
    print(f"--> Checkpoint loaded successfully (Step: {step_info}, Loss: {loss_str})")

    # 3. Dynamic Config Resolution & Architecture Auto-Detection
    if isinstance(checkpoint, dict) and "config" in checkpoint and isinstance(checkpoint["config"], dict):
        config = ModelConfig(**checkpoint["config"])
    else:
        # Infer tensor dimensions directly from embedding weights
        wte_shape = state_dict["transformer.wte.weight"].shape  # [vocab_size, n_embd]
        inferred_vocab_size = wte_shape[0]
        inferred_n_embd = wte_shape[1]
        
        # Count transformer layer blocks
        layer_indices = [
            int(k.split(".")[2]) for k in state_dict.keys() 
            if k.startswith("transformer.h.") and k.rsplit(".", 1)[0].endswith("ln_1")
        ]
        inferred_n_layer = max(layer_indices) + 1 if layer_indices else 6
        inferred_n_head = inferred_n_embd // 64 if inferred_n_embd % 64 == 0 else 6

        # Detect MLP architecture type from layer key naming
        is_standard_mlp = any("mlp.c_fc" in k for k in state_dict.keys())
        mlp_type_str = "standard" if is_standard_mlp else "swiglu"

        config_kwargs = {
            "vocab_size": inferred_vocab_size,
            "n_embd": inferred_n_embd,
            "n_layer": inferred_n_layer,
            "n_head": inferred_n_head,
        }

        # Inspect ModelConfig constructor parameters dynamically
        init_params = inspect.signature(ModelConfig.__init__).parameters
        for k, v in [("mlp_type", mlp_type_str), ("use_swiglu", not is_standard_mlp), ("swiglu", not is_standard_mlp)]:
            if k in init_params:
                config_kwargs[k] = v

        config = ModelConfig(**config_kwargs)

        # Ensure flags are mirrored on config instance
        for attr, val in [("use_swiglu", not is_standard_mlp), ("swiglu", not is_standard_mlp), ("mlp_type", mlp_type_str)]:
            setattr(config, attr, val)

        print(f"--> Auto-detected architecture: n_embd={inferred_n_embd}, n_layer={inferred_n_layer}, n_head={inferred_n_head}, mlp_type='{mlp_type_str}'")

    # 4. Instantiate Base Model
    model = DecoderTransformer(config).to(device)

    # 5. Adapt MLP blocks dynamically if checkpoint requires standard GELU MLP (c_fc / c_proj)
    is_standard_mlp = any("mlp.c_fc" in k for k in state_dict.keys())
    if is_standard_mlp:
        first_mlp = getattr(model.transformer.h[0], "mlp", None)
        if first_mlp is not None and not hasattr(first_mlp, "c_fc"):
            c_fc_weight = state_dict["transformer.h.0.mlp.c_fc.weight"]
            hidden_dim, n_embd = c_fc_weight.shape
            has_bias = "transformer.h.0.mlp.c_fc.bias" in state_dict

            for block in model.transformer.h:
                block.mlp = StandardMLP(n_embd=n_embd, hidden_dim=hidden_dim, bias=has_bias).to(device)
            print("--> Dynamically adapted blocks to Standard GELU MLP (c_fc / c_proj)")

    # 6. Load State Dict
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, phase_info["default_prompt"]


def generate():
    parser = argparse.ArgumentParser(description="Multi-Phase Text Generation CLI")
    parser.add_argument("--phase", type=str, default="1", choices=["1", "2"],
                        help="Select training phase: '1' for TinyShakespeare, '2' for TinyStories.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom input prompt string.")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-K sampling constraint.")
    parser.add_argument("--penalty", type=float, default=1.2,
                        help="Repetition penalty multiplier.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Execution device.")
    args = parser.parse_args()

    # Hardware detection override if default requested
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    elif device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"\n" + "="*60)
    print(f"🎭 RUNNING GENERATION | PHASE {args.phase} | DEVICE: {device.upper()}")
    print("="*60)

    # Load artifacts dynamically based on phase
    model, tokenizer, default_prompt = load_phase_artifacts(args.phase, device)
    
    prompt = args.prompt if args.prompt is not None else default_prompt

    # Tokenize Prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

    print("\n--- Generated Output ---")
    print(prompt, end="", flush=True)

    # Autoregressive Generation Loop with KV-Cache
    with torch.no_grad():
        past_key_values = None
        current_input = input_ids
        generated_history = input_ids[0].tolist()

        for _ in range(args.max_tokens):
            logits, past_key_values = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True
            )

            next_token_logits = logits[:, -1, :]

            # Sample next token
            next_token = sample_next_token(
                next_token_logits,
                generated_tokens=generated_history,
                repetition_penalty=args.penalty,
                temperature=args.temp,
                top_k=args.top_k
            )

            token_id = next_token.item()
            generated_history.append(token_id)

            # With accumulated decoding to handle multi-byte UTF-8 tokens cleanly:
            full_text = tokenizer.decode(generated_history)
            previous_text = tokenizer.decode(generated_history[:-1])
            new_text = full_text[len(previous_text):]
            print(new_text, end="", flush=True)
            current_input = next_token

    print("\n-----------------------\n")


if __name__ == "__main__":
    generate()