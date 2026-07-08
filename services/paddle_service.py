import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR

app = FastAPI()
ocr = PaddleOCR(use_textline_orientation=True, lang="tr", device="gpu")

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # Buffer the upload to a temp file and hand PaddleOCR a path: predict() is
    # happiest with a path and this avoids the RGB/BGR ambiguity of feeding it
    # a raw ndarray. Accepting the file (not a path) lets router run off-box.
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = ocr.predict(tmp_path)
    finally:
        os.unlink(tmp_path)
    lines = []
    for res in result:
        texts = res["rec_texts"]
        scores = res.get("rec_scores", [None] * len(texts))
        for text, score in zip(texts, scores):
            lines.append({"text": text, "confidence": score})
    # "text" kept as a single joined string for backward compatibility with
    # router.ocr_via_paddle; "lines" carries the per-line confidence
    # PaddleOCR already computes, previously discarded here.
    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "line_count": len(lines),
    }

@app.get("/health")
def health():
    return {"status": "ok"}
