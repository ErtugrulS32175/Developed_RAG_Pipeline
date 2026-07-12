#!/bin/bash
# PaddleOCR-VL isolated service setup (separate venv, same rationale as
# setup_gemma.sh: keeps paddlepaddle away from the other services' torch pins).
# PaddleOCR-VL is the OCRTurk Turkish-table winner (TEDS ~0.87) and frugal
# (~5-7GB), so it is the primary A/B candidate. On Windows its CUDA/CUDNN stack
# is broken; this is a Linux/RunPod service.
# Run from the repo root, after ./scripts/setup.sh, on a fresh pod.

set -e

echo "[1/4] Creating isolated venv: paddleocrvl_env"
python3 -m venv paddleocrvl_env
paddleocrvl_env/bin/pip install --upgrade pip

echo "[2/4] Installing paddlepaddle-gpu 3.2.1 (CUDA 12.6 build, Paddle's own index)"
paddleocrvl_env/bin/pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "[3/4] Installing PaddleOCR-VL + service deps"
# paddleocr>=3.x exposes the PaddleOCRVL pipeline; paddlex[ocr] pulls its runtime.
paddleocrvl_env/bin/pip install -U "paddleocr" "paddlex[ocr]" \
  fastapi uvicorn python-multipart huggingface_hub hf_transfer

echo "[4/4] Verifying isolation + import"
python3 -c "import torch; print('main env torch ok:', torch.cuda.is_available())" || true
paddleocrvl_env/bin/python -c "import paddle; print('paddle:', paddle.__version__, 'cuda:', paddle.device.is_compiled_with_cuda())"
paddleocrvl_env/bin/python -c "from paddleocr import PaddleOCRVL; print('PaddleOCRVL import ok')"

echo "Done."
echo "Model weights download on first request via PaddleOCRVL()."
echo "Start the service with (from repo root -- PYTHONPATH so pipeline.* imports):"
echo "  PYTHONPATH=\$(pwd) nohup paddleocrvl_env/bin/uvicorn paddleocrvl_service:app \\"
echo "    --app-dir services --host 127.0.0.1 --port 8104 > paddleocrvl_service.log 2>&1 &"
