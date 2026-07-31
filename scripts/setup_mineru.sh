#!/bin/bash
# MinerU2.5 isolated service setup (separate venv, same rationale as the other
# model setups: keeps its Transformers/vLLM versions away from the rest).
# MinerU2.5-Pro (1.2B) is a document-parsing VLM. The service reads the whole
# image as one table; the library's layout-then-recognize path does not work on
# table-only images (see services/mineru_service.py). Trial backend, run
# alongside the production consensus pair, not inside it.
# Run from the repo root, after ./scripts/setup.sh.

set -e

echo "[1/4] Creating isolated venv: mineru_env"
python3 -m venv mineru_env
mineru_env/bin/pip install --upgrade pip

echo "[2/4] Installing PyTorch + torchvision (CUDA 12.6 build)"
mineru_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "[3/4] Installing Transformers + mineru-vl-utils + service deps"
# mineru-vl-utils provides MinerUClient (content_extract / two_step_extract) and
# the OTSL->HTML converter the service uses.
# Swap the extra for [vllm] if serving through vLLM instead of transformers.
mineru_env/bin/pip install -U "transformers>=4.56" "mineru-vl-utils[transformers]" \
  accelerate pillow openpyxl fastapi uvicorn python-multipart huggingface_hub hf_transfer

echo "[4/4] Verifying import"
mineru_env/bin/python -c "import torch; print('venv torch:', torch.__version__, torch.cuda.is_available())"
mineru_env/bin/python -c "from mineru_vl_utils import MinerUClient; print('MinerUClient import ok')"

mkdir -p logs
echo "Done."
echo "Model weights download on first request via from_pretrained."
echo "Start the service with (from repo root -- PYTHONPATH so pipeline.* imports):"
echo "  PYTHONPATH=\$(pwd) nohup mineru_env/bin/python -m uvicorn mineru_service:app \\"
echo "    --app-dir services --host 127.0.0.1 --port 8106 > logs/mineru_service.log 2>&1 &"
echo ""
echo "Then score it against the labeled images:"
echo "  python -m eval.run_eval --backends mineru"
