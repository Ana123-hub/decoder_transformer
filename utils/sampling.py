import torch
import torch.nn.functional as F

def sample_next_token(logits, generated_tokens=None, repetition_penalty=1.2, temperature=0.7, top_k=20):
    # 1. Apply Repetition Penalty to recently generated tokens
    if generated_tokens is not None and repetition_penalty != 1.0:
        for token_id in set(generated_tokens[-32:]):  # Look back last 32 tokens
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty

    # 2. Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # 2. Top-k filtering
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    # 3. Softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # 4. Multinomial sampling
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

