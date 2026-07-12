#!/bin/bash
# HunyuanOCR isolated service setup (separate venv, same rationale as
# setup_gemma.sh: keeps its Transformers version away from the other services).
# HunyuanOCR-1.5 (1B, OCR-specialized, trained for faithfulness) is the second
# A/B candidate. transformers>=5.13 + trust_remote_code. The plain-transformers
# path is slow on small GPUs -- fine for an accuracy A/B (greedy output is the
# same as vLLM); for production serving speed use vLLM (+DFlash) on the H200.
# Run from the repo root, after ./scripts/setup.sh, on a fresh pod.

set -e

echo "[1/4] Creating isolated venv: hunyuan_env"
python3 -m venv hunyuan_env
hunyuan_env/bin/pip install --upgrade pip

echo "[2/4] Installing PyTorch + torchvision (CUDA 12.6 build)"
# torchvision is required: HunYuanVL's image processing imports it.
hunyuan_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "[3/4] Installing Transformers (>=5.13 for HunYuanVLForConditionalGeneration) + service deps"
hunyuan_env/bin/pip install -U "transformers>=5.13" accelerate pillow \
  fastapi uvicorn python-multipart huggingface_hub hf_transfer

echo "[4/4] Verifying isolation + import"
python3 -c "import torch; print('main env torch ok:', torch.cuda.is_available())" || true
hunyuan_env/bin/python -c "import torch; print('venv torch:', torch.__version__, torch.cuda.is_available())"
hunyuan_env/bin/python -c "from transformers import HunYuanVLForConditionalGeneration; print('HunYuanVL import ok')"

echo "Done."
echo "Model weights (tencent/HunyuanOCR) download on first request via from_pretrained."
echo "If it is gated, run first:  export HF_TOKEN=your_token"
echo "Start the service with (from repo root -- PYTHONPATH so pipeline.* imports):"
echo "  PYTHONPATH=\$(pwd) nohup hunyuan_env/bin/uvicorn hunyuan_service:app \\"
echo "    --app-dir services --host 127.0.0.1 --port 8105 > hunyuan_service.log 2>&1 &"
