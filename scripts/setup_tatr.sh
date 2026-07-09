#!/bin/bash
# TATR table-extraction service setup (isolated venv). Kept separate from the
# main env because tatr_service needs transformers 5.x (TableTransformerConfig's
# strict dataclass), while the main env pins transformers <5.0.0 for docling/vLLM.
# Runs the deterministic table backend: TATR detection + structure, plus EasyOCR
# for Turkish text cells. Numbers come from paddle_service over HTTP, so this env
# does NOT need PaddlePaddle. Run from repo root, on a fresh pod, after setup.sh.

set -e

echo "[1/5] Creating isolated venv: tatr_env"
python3 -m venv tatr_env
tatr_env/bin/pip install --upgrade pip

echo "[2/5] Installing PyTorch (CUDA 12.4 build)"
tatr_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing TATR + text deps"
# transformers>=5 for TableTransformer config; timm for the DETR backbone;
# pymupdf (fitz.Rect) for the vendored TATR post-processing geometry; easyocr
# for Turkish text cells (reads ı/İ/ş/ğ/ç/ö/ü correctly, no master list needed).
tatr_env/bin/pip install "transformers>=5.0.0" timm pymupdf easyocr huggingface_hub pillow numpy

echo "[4/5] Installing service deps"
tatr_env/bin/pip install fastapi uvicorn python-multipart requests

echo "[5/5] Verifying"
tatr_env/bin/python -c "import torch, timm, fitz, easyocr, transformers; print('tatr_env ok | torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"

echo "Done. Start the service with (from repo root):"
echo "  # paddle_service must be up first (number cells); point PADDLE_OCR_URL at it."
echo "  PADDLE_OCR_URL=http://127.0.0.1:8100/ocr KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \\"
echo "    nohup tatr_env/bin/uvicorn services.tatr_service:app --host 127.0.0.1 --port 8102 > tatr_service.log 2>&1 &"
