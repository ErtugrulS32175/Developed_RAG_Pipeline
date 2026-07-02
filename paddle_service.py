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
        for text in res["rec_texts"]:
            lines.append(text)
    return {"text": "\n".join(lines), "line_count": len(lines)}

@app.get("/health")
def health():
    return {"status": "ok"}
