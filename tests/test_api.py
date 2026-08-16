import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest

# 1. Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import __main__
from config.model_config import ModelConfig

# Fix PyTorch unpickling for checkpoints saved from __main__
__main__.ModelConfig = ModelConfig

from api.main import app

# Create test client instance
client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_artifacts():
    """Mocks model loading and token generation so API tests do not require actual weights."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "Once upon a time..."

    mock_model = MagicMock()

    # Target the exact module path where load_artifacts and generate_tokens reside
    # e.g., if main.py does 'import utils.generation', patch 'utils.generation.load_artifacts'
    with patch("api.main.load_artifacts", return_value=(mock_model, mock_tokenizer), create=True), \
         patch("api.main.generate_tokens", return_value="Once upon a time...", create=True):
        yield


# 1. System Health & Metadata Endpoint Tests
def test_health_endpoint():
    """Verifies that /health returns HTTP 200 and expected schema fields."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert isinstance(data["loaded_phases"], list)


def test_list_phases_endpoint():
    """Verifies that /phases returns configuration details for supported model phases."""
    response = client.get("/phases")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2

    phase_keys = [phase["phase"] for phase in data]
    assert "1" in phase_keys
    assert "2" in phase_keys


# 2. Synchronous Generation Endpoint Tests (/generate)
def test_generate_endpoint_phase2_default_prompt():
    """Verifies POST /generate returns a complete JSON response using Phase 2 defaults."""
    payload = {
        "phase": "2",
        "max_tokens": 15,
        "temperature": 0.8,
        "top_k": 40
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["phase"] == "2"
    assert isinstance(data["prompt"], str) and len(data["prompt"]) > 0
    assert isinstance(data["generated_text"], str) and len(data["generated_text"]) > 0
    assert data["tokens_generated"] > 0


def test_generate_endpoint_custom_prompt():
    """Verifies POST /generate handles custom user prompts correctly."""
    custom_prompt = "Once upon a time, a small mouse found a big piece of cheese."
    payload = {
        "phase": "2",
        "prompt": custom_prompt,
        "max_tokens": 10,
        "temperature": 0.7
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prompt"] == custom_prompt
    assert isinstance(data["generated_text"], str)


def test_generate_endpoint_invalid_phase():
    """Verifies POST /generate returns HTTP 400 Bad Request for unsupported phase keys."""
    payload = {
        "phase": "99",
        "max_tokens": 10
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "Invalid phase" in response.json()["detail"]


# 3. Streaming Generation Endpoint Tests (/generate/stream)
def test_generate_stream_endpoint():
    """Verifies POST /generate/stream yields Server-Sent Events (SSE) chunks."""
    payload = {
        "phase": "2",
        "prompt": "Tim the bird went for a flight",
        "max_tokens": 15,
        "temperature": 0.8
    }

    response = client.post("/generate/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse streaming event payload text
    content = response.text
    assert "data: " in content
    assert "data: [DONE]" in content


def test_generate_stream_invalid_params():
    """Verifies validation errors for out-of-bounds generation request parameters."""
    payload = {
        "phase": "2",
        "max_tokens": -5  # Invalid: max_tokens must be >= 1
    }
    response = client.post("/generate/stream", json=payload)
    assert response.status_code == 422  # Unprocessable Entity