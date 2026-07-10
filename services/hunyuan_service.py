"""HunyuanOCR-1.5 table backend (1B OCR-specialized VLM, trained for faithfulness).

Doc-parse prompt -> markdown with HTML tables -> {tables:[{headers,rows}]} per the
pluggable-backend contract (router.TABLE_BACKEND=hunyuan). transformers>=5.13 +
trust_remote_code. Plain transformers is slow on small GPUs; on RunPod/H200 serve
with vLLM (+DFlash) for practical speed -- this wrapper is the transformers path.
"""
import io
import os

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from pipeline.table_export import parse_html_tables

app = FastAPI()
MODEL_ID = os.getenv("HUNYUAN_MODEL", "tencent/HunyuanOCR")
# The model's own document-parsing prompt: markdown, tables as HTML, reading order.
PROMPT = os.getenv(
    "HUNYUAN_PROMPT",
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
)
MAX_NEW = int(os.getenv("HUNYUAN_MAX_NEW_TOKENS", "3000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE = {}


def _load():
    if not _STATE:
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        _STATE["proc"] = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _STATE["model"] = HunYuanVLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype="bfloat16", device_map="auto", trust_remote_code=True).eval()
    return _STATE["proc"], _STATE["model"]


@app.post("/table")
async def extract_table(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    proc, model = _load()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                             repetition_penalty=1.08)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {"tables": parse_html_tables(text), "raw": text}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_STATE)}
