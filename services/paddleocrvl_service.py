"""PaddleOCR-VL table backend (0.9B doc-parsing VLM, Turkish-proven on OCRTurk).

Runs in the PaddleOCR-VL env (paddlepaddle-gpu>=3.2.1 + `paddleocr` +
`paddlex[ocr]`). Returns {tables:[{headers,rows}]} per the pluggable-backend
contract, so router.TABLE_BACKEND=paddleocr_vl routes to it with no other change.
Frugal (~5-7GB); deploy on RunPod/H200 with a serving VRAM cap. On Windows the
native CUDA/CUDNN stack is finicky -- this is a Linux/RunPod service.
"""
import os
import tempfile

from fastapi import FastAPI, File, UploadFile

from pipeline.table_export import parse_html_tables

app = FastAPI()
_PIPE = {}


def _pipe():
    if "p" not in _PIPE:
        from paddleocr import PaddleOCRVL
        _PIPE["p"] = PaddleOCRVL()
    return _PIPE["p"]


def _markdown(res_item):
    md = getattr(res_item, "markdown", None)
    if isinstance(md, dict):
        md = md.get("markdown_texts") or md.get("text") or ""
    return md or ""


@app.post("/table")
async def extract_table(file: UploadFile = File(...)):
    data = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        out = _pipe().predict(path)
    finally:
        os.unlink(path)
    md = "\n".join(_markdown(r) for r in out)
    # PaddleOCR-VL emits tables as HTML inside the markdown; parse to records.
    return {"tables": parse_html_tables(md), "markdown": md}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_PIPE)}
