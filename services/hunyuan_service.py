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
# Long/dense tables as HTML need many output tokens; a low cap truncates mid-table
# so </table> never arrives and the parser gets nothing. The model handles 128K ctx.
MAX_NEW = int(os.getenv("HUNYUAN_MAX_NEW_TOKENS", "8000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE = {}


def _load():
    if not _STATE:
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        # use_fast=False -> PIL-backed image processor, which supports the model's
        # native LANCZOS resample. The fast (torchvision) path silently downgrades
        # to BICUBIC on torchvision<0.27, degrading OCR fidelity vs the original.
        _STATE["proc"] = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=False)
        _STATE["model"] = HunYuanVLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype="bfloat16", device_map="auto", trust_remote_code=True).eval()
    return _STATE["proc"], _STATE["model"]


@app.post("/table")
async def extract_table(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    proc, model = _load()
    # The chat template expects a system turn; the official client sends an empty
    # one. Omitting it builds a malformed prompt -> scrambled output (Tencent's
    # 2025-11-28 "system prompt config" fix). Sampling stays greedy per the recipe.
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                             repetition_penalty=1.08)
    # clean_up_tokenization_spaces defaults True but is destructive for BPE (strips
    # spaces before punctuation) -> corrupts OCR text; the processor itself warns.
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0]
    dbg = os.getenv("HUNYUAN_DEBUG_DIR")
    if dbg:
        with open(os.path.join(dbg, "hy_last_raw.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return {"tables": parse_html_tables(text), "raw": text}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_STATE)}
