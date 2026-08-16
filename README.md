# Custom Decoder-Only Transformer: Training, FastAPI & Containerized Inference

[![CI Pipeline](https://github.com/Ana123-hub/decoder_transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/Ana123-hub/decoder_transformer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed.svg)](https://www.docker.com/)

A lightweight, custom PyTorch implementation of a **Decoder-Only Transformer** built from scratch (causal multi-head self-attention, positional encodings, layer normalization, and feed-forward networks). This project details a two-phase training lifecycle, progressing from character-level Shakespearean text generation to subword tokenized modern narrative generation and packaged into a production-ready FastAPI backend with Docker containerization and CI/CD pipelines.

---

## Key Features

* **Custom Transformer Architecture:** Pure PyTorch implementation of causal self-attention mechanisms without high-level wrapper libraries.
* **Two-Phase Progressive Training:**
* **Phase 1 (TinyShakespeare):** Baseline character-level language modeling capturing early structural patterns.
* **Phase 2 (TinyStories):** Subword-tokenized text synthesis generating coherent, multi-sentence narrative structures.
* **RESTful FastAPI Service:** Asynchronous inference server supporting dynamic checkpoint selection, health monitoring, and parameter tuning (temperature, max tokens, top-k sampling).
* **Automated CI/CD Pipeline:** GitHub Actions workflow running automated unit and integration tests on every pull request and push.
* **Containerized Deployment:** Optimized Docker build (CPU runtime optimized) with Docker Compose for seamless cross-platform execution.

---

## Directory Structure

```text
decoder_transformer/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI configuration
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI routes & server application
│   └── schemas.py                     # Pydantic request/response data contracts
├── artifacts/
│   ├── phase1_tinyshakespeare/
│   │   ├── checkpoints/               # Phase 1 epoch checkpoints
│   │   ├── tokenizer.json             # Character-level vocabulary mapping
│   │   └── best_model.pt              # Best Phase 1 model weights
│   └── phase2_tinystories/
│       ├── checkpoints/               # Phase 2 epoch checkpoints
│       ├── tokenizer.json             # Trained BPE tokenizer model
│       └── best_model2.pt             # Best Phase 2 model weights
├── config/
│   ├── model_config.py                # Transformer architectural hyperparameters
│   └── train_config.py                # Optimizer, learning rate & trainer settings
├── data/
│   ├── dataset.py                     # PyTorch Dataset & DataLoader implementation
│   ├── prepare_tinyshakespeare.py     # Raw text ingestion & split logic
│   └── pretokenize.py                 # Offline dataset pre-tokenization script
├── model/
│   ├── __init__.py
│   ├── attention.py                   # Multi-head causal self-attention layer
│   ├── block.py                       # Transformer decoder block wrapper
│   ├── embedding.py                   # Token & positional embedding modules
│   ├── mlp.py                         # Position-wise Feed-Forward Network
│   ├── normalization.py               # Custom LayerNorm / RMSNorm implementation
│   └── transformer.py                 # Full GPT-style decoder model architecture
├── notebooks/
│   └── phase2_tinystories.ipynb       # Google Colab training & evaluation notebook
├── scripts/
│   └── test_overfit.py                # Single-batch overfitting diagnostic test
├── tests/
│   ├── __init__.py
│   ├── test_api.py                    # FastAPI endpoint integration tests
│   └── test_generation.py             # Autoregressive generation & sampling unit tests
├── tokenizer/
│   ├── __init__.py
│   ├── bpe_tokenizer.py               # Custom Byte-Pair Encoding wrapper
│   └── train_tokenizer.py             # Tokenizer training script
├── training/
│   ├── checkpointing.py               # Model saving, loading & state serialization
│   ├── scheduler.py                   # Cosine decay learning rate scheduler with warmup
│   └── trainer.py                     # Training & evaluation step management loop
├── utils/
│   ├── __init__.py
│   ├── logger.py                      # Console & file logging helper
│   ├── metrics.py                     # Perplexity & loss calculation tools
│   └── sampling.py                    # Temperature & Top-K / Nucleus sampling routines
├── .dockerignore
├── .gitignore
├── Dockerfile                         # Optimized container image build
├── docker-compose.yml                 # Service orchestration configuration
├── generate.py                        # Root CLI generation entry point
├── pytest.ini                         # Test runner filter configurations
├── requirements.txt                   # Dependency manifest
└── train.py                           # Root training orchestration entry point
```

## Training Phases & Experimental Results

### Phase 1: TinyShakespeare (Baseline Character-Level)
* Dataset: Complete works of William Shakespeare (~1MB raw text).
* Objective: Verify autoregressive loss reduction, causal masking correctness and basic vocabulary dynamics.
* Configuration: 6 Layers, 6 Attention Heads, 384 Embedding Dimension, Context Length: 256 tokens. 
* Results: Rapid loss drop within 600 steps (2.45 - 1.62). Generated text captured Shakespearean rhythm, character names, and dialogue formatting.

### Phase 2: TinyStories (Subword Narrative Synthesis)
* Dataset: Synthetic dataset of stories generated by GPT-3.5/4 restricted to 3-year-old vocabulary.
* Objective: Evaluate long-range coherence, semantic consistency, and subword tokenization (BPE/Tokenizer).
* Configuration: 6 Layers, 6 Attention Heads, 384 Embedding Dimension, Context Length: 256 tokens.
* Results: Achieved stable validation loss convergence (~1.2262 at step 5000). Model produces grammatical, multi-sentence stories featuring character interactions and logical narrative resolution.

## Quickstart & Local Setup

### Prerequisites
* Python 3.11+
* Git
* Google Colab

### Installation
### Clone repository
```bash
git clone [https://github.com/Ana123-hub/decoder_transformer.git](https://github.com/Ana123-hub/decoder_transformer.git)
cd decoder_transformer
```
### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.
```

## Running Test Suite (pytest)
This project uses pytest for unit testing model tensor transformations and FastAPI HTTP endpoints.

### Run all tests with verbose output
```bash
python -m pytest -v
```
### Run API integration tests specifically
```bash
python -m pytest tests/test_api.py -v
```
### Run tests with coverage report
```bash
python -m pytest --cov=src --cov=api tests/
```

## REST API Documentation & Endpoints
Once the application is running (locally or via Docker), interactive OpenAPI documentation is available at http://localhost:8000/docs.

## Example API Request (POST /generate)
### cURL Command:
```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Once upon a time, a small puppy",
    "phase": "2",
    "max_tokens": 50,
    "temperature": 0.8,
    "top_k": 40
  }'
  ```
  ### JSON Response:
  ```json
  {
  "phase": "2",
  "prompt": "Once upon a time, a small puppy",
  "generated_text": "Once upon a time, a small puppy found a shiny red ball in the garden. He bounced with joy and ran toward his friend, the little bird.",
  "tokens_generated": 50
} 
```

# Continuous Integration (CI)
This repository enforces automated testing via GitHub Actions on every push or pull request to main. The workflow:
* Sets up a clean Ubuntu-latest runner with Python 3.11.
* Installs core dependencies and PyTorch CPU binaries.
* Executes the full pytest suite across model and API integration tests.

