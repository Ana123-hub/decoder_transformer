import os
import sys
import inspect
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import __main__
from config.model_config import ModelConfig
from model.transformer import DecoderTransformer

# Fix PyTorch unpickling for checkpoints saved from __main__
__main__.ModelConfig = ModelConfig

from generate import load_phase_artifacts, StandardMLP, PHASE_MAP


# Fixtures & Synthetic Helpers
def create_model_config(**kwargs):
    """Safely instantiates ModelConfig by filtering kwargs against its __init__ signature."""
    init_params = inspect.signature(ModelConfig.__init__).parameters
    valid_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
    
    # Fallback required defaults if omitted
    valid_kwargs.setdefault("vocab_size", 1000)
    valid_kwargs.setdefault("n_embd", 128)
    valid_kwargs.setdefault("n_layer", 2)
    valid_kwargs.setdefault("n_head", 4)

    config = ModelConfig(**valid_kwargs)
    
    # Mirror flags onto instance for model constructor checks
    if "use_swiglu" in kwargs:
        setattr(config, "use_swiglu", kwargs["use_swiglu"])
    if "mlp_type" in kwargs:
        setattr(config, "mlp_type", kwargs["mlp_type"])
        
    return config


@pytest.fixture
def mock_config_swiglu():
    """Returns a synthetic ModelConfig configured for SwiGLU MLP."""
    return create_model_config(
        vocab_size=1000,
        n_embd=128,
        n_layer=2,
        n_head=4,
        block_size=64,
        use_swiglu=True
    )

# 1. Checkpoint Loading & Architecture Auto-Detection Tests
@pytest.mark.skipif(
    not os.path.exists(os.path.join("artifacts", "phase1_tinyshakespeare", "best_model.pt")),
    reason="Phase 1 checkpoint artifact not found locally"
)
def test_load_phase1_artifacts():
    """Verifies Phase 1 loads cleanly and auto-detects SwiGLU MLP architecture."""
    model, tokenizer, prompt = load_phase_artifacts("1", device="cpu")

    assert model is not None
    assert tokenizer is not None
    assert isinstance(prompt, str) and len(prompt) > 0
    assert model.training is False

    first_mlp = model.transformer.h[0].mlp
    assert hasattr(first_mlp, "w_gate") or hasattr(first_mlp, "w1") or hasattr(first_mlp, "c_fc")


@pytest.mark.skipif(
    not os.path.exists(os.path.join("artifacts", "phase2_tinystories", "best_model2.pt")),
    reason="Phase 2 checkpoint artifact not found locally"
)
def test_load_phase2_artifacts():
    """Verifies Phase 2 loads cleanly and adapts blocks to Standard GELU MLP."""
    model, tokenizer, prompt = load_phase_artifacts("2", device="cpu")

    assert model is not None
    assert tokenizer is not None
    assert isinstance(prompt, str) and len(prompt) > 0
    assert model.training is False

    first_mlp = model.transformer.h[0].mlp
    assert isinstance(first_mlp, StandardMLP)
    assert hasattr(first_mlp, "c_fc")
    assert hasattr(first_mlp, "c_proj")


def test_invalid_phase_key():
    """Ensures load_phase_artifacts raises ValueError for unsupported phase identifiers."""
    with pytest.raises(ValueError, match="Invalid phase '3'"):
        load_phase_artifacts("3", device="cpu")


# 2. KV-Cache Tensor Shape Verification Tests
def test_kv_cache_forward_pass_shapes(mock_config_swiglu):
    """Verifies output logit shapes and KV-cache matrix dimensions across sequential generation steps."""
    device = "cpu"
    model = DecoderTransformer(mock_config_swiglu).to(device)
    model.eval()

    batch_size = 2
    prompt_len = 5
    head_dim = mock_config_swiglu.n_embd // mock_config_swiglu.n_head

    # Step 1: Initial prompt forward pass
    idx = torch.randint(0, mock_config_swiglu.vocab_size, (batch_size, prompt_len), device=device)
    
    with torch.no_grad():
        logits, past_kv = model(idx, use_cache=True)

    assert logits.shape == (batch_size, prompt_len, mock_config_swiglu.vocab_size)
    assert len(past_kv) == mock_config_swiglu.n_layer

    for layer_kv in past_kv:
        k, v = layer_kv
        assert k.shape == (batch_size, mock_config_swiglu.n_head, prompt_len, head_dim)
        assert v.shape == (batch_size, mock_config_swiglu.n_head, prompt_len, head_dim)

    # Step 2: Autoregressive single token step using past KV-cache
    next_token = torch.randint(0, mock_config_swiglu.vocab_size, (batch_size, 1), device=device)
    
    with torch.no_grad():
        next_logits, updated_kv = model(next_token, past_key_values=past_kv, use_cache=True)

    assert next_logits.shape == (batch_size, 1, mock_config_swiglu.vocab_size)

    for layer_kv in updated_kv:
        k, v = layer_kv
        assert k.shape == (batch_size, mock_config_swiglu.n_head, prompt_len + 1, head_dim)
        assert v.shape == (batch_size, mock_config_swiglu.n_head, prompt_len + 1, head_dim)


# 3. Tokenizer UTF-8 Multi-Byte Incremental Decoding Tests
def test_incremental_tokenizer_decoding_utf8():
    """Verifies that stripping replacement characters prevents multi-byte UTF-8 corruption."""
    mock_tokenizer = MagicMock()

    token_history = [101, 202, 203, 304]

    def mock_decode(token_ids):
        if token_ids == [101]:
            return "it"
        elif token_ids == [101, 202]:
            return "it\ufffd"  # Incomplete multi-byte sequence
        elif token_ids == [101, 202, 203]:
            return "it’"
        elif token_ids == [101, 202, 203, 304]:
            return "it’s"
        return ""

    mock_tokenizer.decode.side_effect = mock_decode

    generated_history = []
    output_chunks = []
    prev_clean = ""

    for token_id in token_history:
        generated_history.append(token_id)
        full_text = mock_tokenizer.decode(generated_history)
        
        # Suppress incomplete multi-byte replacement character during streaming
        clean_text = full_text.rstrip("\ufffd")
        new_text = clean_text[len(prev_clean):]
        
        if new_text:
            output_chunks.append(new_text)
            prev_clean = clean_text

    reconstructed_text = "".join(output_chunks)
    assert reconstructed_text == "it’s"
    assert "\ufffd" not in reconstructed_text