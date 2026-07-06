import io

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR, TableRecognitionPipelineV2

app = FastAPI()
ocr = PaddleOCR(use_textline_orientation=True, lang="tr", device="gpu")

# PPStructureV3 is not used here: in paddleocr==3.3.1 / paddlex==3.3.13 it eagerly
# initializes a VLM-based chart-recognition model even with use_chart_recognition=False
# (paddlex bug, layout_parsing/pipeline_v2.py has a bare "init the model at any time"
# TODO), which crashes because that model needs a paddle op not in paddlepaddle-gpu 3.0.0.
# TableRecognitionPipelineV2 gives the same table detection/structure output without
# pulling in layout-parsing's chart/seal/formula baggage.
table_pipeline = TableRecognitionPipelineV2(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
)

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

def _html_to_headers_rows(html: str):
    # PP-Structure emits every cell as <td> (no <th>), so pandas can't detect a
    # header row on its own. We assume the first row is the header, which holds
    # for typical invoice/product tables but is a heuristic, not a certainty --
    # the raw HTML is kept alongside so this assumption can always be checked.
    try:
        df = pd.read_html(io.StringIO(html), header=0)[0]
    except (ValueError, IndexError):
        return [], []
    headers = [str(c) for c in df.columns]
    rows = df.astype(object).where(df.notnull(), "").values.tolist()
    return headers, rows

@app.post("/table")
def run_table(req: OCRRequest):
    results = table_pipeline.predict(req.image_path)
    tables = []
    for res in results:
        for table_res in res["table_res_list"]:
            data = table_res.json["res"]
            html = data["pred_html"]
            scores = data.get("table_ocr_pred", {}).get("rec_scores", [])
            headers, rows = _html_to_headers_rows(html)
            tables.append({
                "headers": headers,
                "rows": rows,
                "confidence": sum(scores) / len(scores) if scores else None,
                "html": html,
                "cell_count": len(data.get("cell_box_list", [])),
            })
    return {"tables": tables}

@app.get("/health")
def health():
    return {"status": "ok"}
