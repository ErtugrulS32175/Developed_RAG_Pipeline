"""Deterministic table-extraction service (the no-VLM alternative to
gemma_service). Given a scanned page image it:
  1. detects + crops the table with TATR,
  2. asks paddle_service to OCR the crop (word boxes, best on digits),
  3. runs pipeline.table_tatr to rebuild the grid + read Turkish text cells
     with EasyOCR,
returning {headers, rows} in the same shape gemma_service's parsed output has,
so the router can swap backends via TABLE_BACKEND without downstream changes.

Runs in the TATR/EasyOCR torch env (not paddle_env); it reaches PaddleOCR over
HTTP so the two conflicting runtimes never share a process. Set PADDLE_OCR_URL
to point at the running paddle_service.
"""
import io
import os

import requests
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from pipeline import table_tatr as tt
from pipeline import image_preprocess as ip

app = FastAPI()

PADDLE_OCR_URL = os.getenv("PADDLE_OCR_URL", "http://127.0.0.1:8100/ocr")
PADDLE_TIMEOUT = float(os.getenv("SERVICE_TIMEOUT", "120"))
# OCR image-enhancement layer (deskew + denoise + CLAHE + 2x upscale). Helps
# hard scans (low-contrast/shaded/dense) without regressing clean ones, and is
# safe for TATR detection (hard binarization is NOT -- kept off). Default on.
PREPROCESS = os.getenv("PREPROCESS", "1").lower() not in ("0", "false", "no")
# Columns whose data cells are read by EasyOCR-tr (names / Turkish text) rather
# than PaddleOCR. Default: column 1 (customer name). Comma-separated indices.
TEXT_COLS = tuple(int(c) for c in os.getenv("TATR_TEXT_COLS", "1").split(",") if c.strip() != "")


def _paddle_boxes(crop):
    """OCR the crop via paddle_service and return [{text, box}] in crop pixels."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    r = requests.post(
        PADDLE_OCR_URL,
        files={"file": ("crop.png", buf.getvalue(), "image/png")},
        timeout=PADDLE_TIMEOUT,
    )
    r.raise_for_status()
    return [{"text": ln["text"], "box": ln["box"]}
            for ln in r.json().get("lines", []) if ln.get("box")]


@app.post("/table")
async def extract_table(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    if PREPROCESS:
        image = ip.enhance(image)
    crop, box = tt.detect_and_crop(image)
    if crop is None:
        return {"tables": [], "detected": False}
    words = _paddle_boxes(crop)
    table = tt.assemble(crop, words, text_cols=TEXT_COLS)
    # Wrap in "tables" to mirror gemma_service; router.tables_via_tatr reads it.
    return {
        "tables": [{"headers": table["headers"], "rows": table["rows"]}],
        "flags": table.get("flags", []),
        "detected": True,
        "crop_box": box,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
