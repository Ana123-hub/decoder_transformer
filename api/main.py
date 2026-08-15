import os
import sys
import asyncio
from typing import AsyncGenerator
from contextlib import asynccontextmanager

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import __main__
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.model_config import ModelConfig

# Fix PyTorch unpickling for checkpoints saved from __main__
__main__.ModelConfig = ModelConfig

from generate import load_phase_artifacts, PHASE_MAP
from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PhaseDetail,
)

# Global artifacts memory cache
LOADED_ARTIFACTS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_or_load_phase(phase_key: str):
    """Retrieves or lazily loads phase model, tokenizer, and default prompt."""
    if phase_key not in PHASE_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid phase '{phase_key}'. Choose '1' or '2'."
        )

    if phase_key not in LOADED_ARTIFACTS:
        try:
            model, tokenizer, default_prompt = load_phase_artifacts(phase_key, device=DEVICE)
            LOADED_ARTIFACTS[phase_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "default_prompt": default_prompt,
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load artifacts for Phase {phase_key}: {str(e)}"
            )

    return LOADED_ARTIFACTS[phase_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-loads Phase 1 and Phase 2 models at API startup."""
    print(f"--> Initializing API on device: {DEVICE}")
    for p_key in ["1", "2"]:
        try:
            get_or_load_phase(p_key)
        except Exception as e:
            print(f"[Warning] Could not pre-load Phase {p_key}: {e}")
    yield
    LOADED_ARTIFACTS.clear()


app = FastAPI(
    title="Decoder Transformer Generation API",
    description="FastAPI service serving Phase 1 (TinyShakespeare) and Phase 2 (TinyStories) models with streaming support.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Core Autoregressive Inference Generator
def generate_tokens_stream(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
):
    """Autoregressive KV-cached token generator yielding token text chunks safely."""
    encoded = tokenizer.encode(prompt)
    prompt_tokens = encoded.ids
    if not prompt_tokens:
        prompt_tokens = [tokenizer.token_to_id("<|endoftext|>") or 0]

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
    generated_ids = list(prompt_tokens)

    past_kv = None
    prev_clean = ""

    for _ in range(max_tokens):
        with torch.no_grad():
            if past_kv is None:
                logits, past_kv = model(input_ids, use_cache=True)
            else:
                logits, past_kv = model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)

        next_token_logits = logits[:, -1, :] / temperature

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= repetition_penalty
                else:
                    next_token_logits[0, token_id] /= repetition_penalty

        # Top-K filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_token_id)
        input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)

        # Multi-byte UTF-8 incremental string decoding
        full_text = tokenizer.decode(generated_ids)
        clean_text = full_text.rstrip("\ufffd")
        new_text = clean_text[len(prev_clean):]

        if new_text:
            yield new_text
            prev_clean = clean_text


# API Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns runtime environment status and active phase model states."""
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        loaded_phases=list(LOADED_ARTIFACTS.keys())
    )


@app.get("/phases", response_model=list[PhaseDetail])
def list_phases():
    """Returns available model phase details."""
    details = []
    for p_key, p_info in PHASE_MAP.items():
        # Fallback names based on phase key
        default_name = "TinyShakespeare" if p_key == "1" else "TinyStories" if p_key == "2" else f"Phase {p_key}"
        
        phase_name = p_info.get("name", default_name)
        artifact_dir = p_info.get("dir", p_info.get("checkpoint_dir", "Unknown"))

        details.append(
            PhaseDetail(
                phase=p_key,
                name=phase_name,
                description=f"Artifact dir: {artifact_dir}",
                checkpoint_loaded=p_key in LOADED_ARTIFACTS
            )
        )
    return details


@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    """Generates complete text response as a single JSON payload."""
    artifacts = get_or_load_phase(req.phase)
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    prompt = req.prompt or artifacts["default_prompt"]

    chunks = list(
        generate_tokens_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )
    )

    full_generated_text = "".join(chunks)

    return GenerateResponse(
        phase=req.phase,
        prompt=prompt,
        generated_text=full_generated_text,
        tokens_generated=len(chunks)
    )


@app.post("/generate/stream")
async def generate_text_stream(req: GenerateRequest):
    """Streams generated text tokens in real time via Server-Sent Events (SSE)."""
    artifacts = get_or_load_phase(req.phase)
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    prompt = req.prompt or artifacts["default_prompt"]

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        token_stream = generate_tokens_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )

        for chunk in token_stream:
            # Format chunk as Server-Sent Event payload
            clean_chunk = chunk.replace("\n", "\\n")
            yield f"data: {clean_chunk}\n\n"
            await asyncio.sleep(0.01)

        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")