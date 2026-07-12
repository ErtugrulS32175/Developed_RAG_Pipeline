"""Generic vLLM-backed table service (fast path for the A/B).

Thin adapter that keeps our pluggable-backend contract (POST image -> {tables})
but does inference on a local vLLM OpenAI-compatible server instead of the slow
in-process transformers/paddle path. One file, one instance per model -- point
each instance at its own `vllm serve` endpoint + prompt via env:

  PaddleOCR-VL:  VLLM_BASE_URL=http://127.0.0.1:8114/v1  VLLM_MODEL=PaddlePaddle/PaddleOCR-VL
  HunyuanOCR:    VLLM_BASE_URL=http://127.0.0.1:8115/v1  VLLM_MODEL=tencent/HunyuanOCR

vLLM does the heavy VLM inference (batched attention, paged KV) -> same greedy
output as transformers, far faster. This wrapper just base64s the image into an
OpenAI chat request, asks for HTML tables, and parses them with parse_html_tables.
Runs in any small env with fastapi+requests+openpyxl (e.g. gemma_env). Reaches
vLLM over localhost on the pod; the harness reaches THIS over the SSH tunnel.
"""
import base64
import os

import requests
from fastapi import FastAPI, File, UploadFile

from pipeline.table_export import parse_html_tables

app = FastAPI()

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.getenv("VLLM_MODEL", "")
MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "600"))
# Doc-parse prompt: markdown with tables as HTML (parse_html_tables consumes the
# <table> markup). Turkish preservation is spelled out -- both models can drop
# diacritics otherwise. Override per model with VLLM_PROMPT if needed.
PROMPT = os.getenv(
    "VLLM_PROMPT",
    "Extract every table in this image as HTML <table> markup. Preserve every row "
    "and column exactly, keep empty cells empty, and reproduce all text verbatim "
    "including Turkish characters (ğ Ğ ş Ş ı İ ç Ç ö Ö ü Ü). Output only the HTML.",
)


def _data_url(data: bytes, filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64," + base64.b64encode(data).decode()


@app.post("/table")
async def extract_table(file: UploadFile = File(...)):
    data = await file.read()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(data, file.filename)}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0.0,          # greedy: reproducible, best for OCR
        "top_k": 1,
        "repetition_penalty": 1.0,
        "max_tokens": MAX_TOKENS,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return {"tables": parse_html_tables(text), "raw": text}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "vllm": BASE_URL}
