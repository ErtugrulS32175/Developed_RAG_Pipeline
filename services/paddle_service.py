from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR

app = FastAPI()
ocr = PaddleOCR(use_textline_orientation=True, lang="tr", device="gpu")

class OCRRequest(BaseModel):
    image_path: str

@app.post("/ocr")
def run_ocr(req: OCRRequest):
    result = ocr.predict(req.image_path)
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
