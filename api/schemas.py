from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    phase: str = Field(
        default="2",
        description="Phase model identifier ('1' for TinyShakespeare, '2' for TinyStories)"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Input text prompt. If None or empty, the phase default prompt is used."
    )
    max_tokens: int = Field(
        default=200,
        ge=1,
        le=1024,
        description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=2.0,
        description="Sampling temperature. Higher values yield more creative output."
    )
    top_k: Optional[int] = Field(
        default=40,
        ge=0,
        description="Top-K sampling threshold. Set to 0 or None to disable."
    )
    repetition_penalty: float = Field(
        default=1.2,
        ge=1.0,
        le=3.0,
        description="Penalty factor applied to previously generated token logits."
    )


class GenerateResponse(BaseModel):
    phase: str
    prompt: str
    generated_text: str
    tokens_generated: int


class PhaseDetail(BaseModel):
    phase: str
    name: str
    description: str
    checkpoint_loaded: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_phases: list[str]