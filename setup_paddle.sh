#!/bin/bash
# PaddleOCR isolated service setup (separate venv to avoid PaddlePaddle/PyTorch conflict)
# Run after ./setup.sh on a fresh pod.

set -e

echo "[1/4] Creating isolated venv: paddle_env"
python3 -m venv paddle_env
paddle_env/bin/pip install --upgrade pip

echo "[2/4] Installing PaddlePaddle 3.0.0 (GPU, CUDA 12.6 build)"
paddle_env/bin/pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "[3/4] Installing PaddleOCR 3.3.1 (compatible with PaddlePaddle 3.0.0) + service deps"
paddle_env/bin/pip install "paddleocr==3.3.1" fastapi uvicorn python-multipart pillow

echo "[4/4] Verifying isolation (main env torch must stay intact)"
python3 -c "import torch; print('main env torch ok:', torch.cuda.is_available())"
paddle_env/bin/python -c "import paddle; print('venv paddle:', paddle.__version__, paddle.device.is_compiled_with_cuda())"
paddle_env/bin/python -c "import paddleocr; print('venv paddleocr ok')"

echo "Done. Start the service with:"
echo "  nohup paddle_env/bin/uvicorn paddle_service:app --host 127.0.0.1 --port 8100 > paddle_service.log 2>&1 &"
