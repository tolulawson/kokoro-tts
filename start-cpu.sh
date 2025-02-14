#!/bin/bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=false
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web

# Run FastAPI with CPU extras using uv run
# Note: espeak may still require manual installation,
uv pip install -e ".[cpu]"
uv run uvicorn api.src.main:app --reload --host 0.0.0.0 --port 8880