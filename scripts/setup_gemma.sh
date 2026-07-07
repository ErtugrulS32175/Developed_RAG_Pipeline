#!/bin/bash
# Gemma 4 E4B isolated service setup (separate venv, same rationale as
# setup_paddle.sh: keeps its Transformers version away from vLLM's pin
# in the main env).
# Run from the repo root, after ./scripts/setup.sh, on a fresh pod.

set -e

echo "[1/4] Creating isolated venv: gemma_env"
python3 -m venv gemma_env
gemma_env/bin/pip install --upgrade pip

echo "[2/4] Installing PyTorch + torchvision (CUDA 12.6 build)"
# torchvision is required even for text-only chat: transformers' Gemma4Processor
# unconditionally imports its image_processing_gemma4 module, which imports
# torchvision.transforms.v2 -- omitting it fails AutoProcessor.from_pretrained
# with a misleading "Could not import module 'Gemma4Processor'" error.
gemma_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "[3/4] Installing latest Transformers (Gemma4ForConditionalGeneration needs a recent release) + service deps"
gemma_env/bin/pip install -U transformers accelerate pillow bitsandbytes fastapi uvicorn python-multipart huggingface_hub hf_transfer

echo "[4/4] Verifying isolation (main env torch must stay intact)"
python3 -c "import torch; print('main env torch ok:', torch.cuda.is_available())"
gemma_env/bin/python -c "import torch; print('venv torch:', torch.__version__, torch.cuda.is_available())"
gemma_env/bin/python -c "from transformers import AutoModelForCausalLM; print('venv transformers ok')"

echo "Done."
echo "Model weights (~16GB) download on first request via from_pretrained."
echo "If google/gemma-4-E4B-it turns out to be gated, run first:"
echo "  export HF_TOKEN=your_token"
echo "Start the service with (from repo root):"
echo "  nohup gemma_env/bin/uvicorn gemma_service:app --app-dir services --host 127.0.0.1 --port 8101 > gemma_service.log 2>&1 &"
