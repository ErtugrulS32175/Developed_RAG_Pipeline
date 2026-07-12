"""Gemma-4 E4B table backend (8B-param multimodal VLM, ~16GB BF16 weights).

General-purpose VLM prompted to emit a table as JSON directly from the image,
bypassing grid-cell regression. Registered as router.TABLE_BACKEND=gemma
(needs_raw=True): this service returns the model's raw text under `raw` and the
router parses it client-side with table_export.parse_table_json, so parser fixes
never require a service redeploy. Validated on a RunPod A40 (sample1: lower on the
full scan, much higher cropped to the table region -- cropping is the pipeline's job).

Isolated `gemma_env` venv (transformers kept off the main env's vLLM pin); see
scripts/setup_gemma.sh. Weights are heavy for an 8GB laptop -- an A40/H200 service.
"""
import io
import json
import os
import re

from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

# Configurable so the table-extraction model can be swapped independently of the
# chat/LLM role's model (services/llm_service.py) -- separate services on purpose,
# even when they both happen to be Gemma today. E4B = 8B params (~16GB BF16).
MODEL_ID = os.getenv("GEMMA_TABLE_MODEL", "google/gemma-4-E4B-it")
MAX_NEW = int(os.getenv("GEMMA_MAX_NEW_TOKENS", "4096"))

TABLE_PROMPT = os.getenv(
    "GEMMA_TABLE_PROMPT",
    "Extract every table in this image as a single JSON object of the form "
    '{"headers": [...], "rows": [[...], ...]}. '
    "Preserve every row and column exactly as shown, keep empty cells as \"\". "
    "Preserve Turkish characters exactly (ğ Ğ ş Ş ı İ ç Ç ö Ö ü Ü); never split "
    "a letter into a base character plus a separate accent mark. "
    "Return only the JSON, no other text.",
)

_STATE = {}


def _load():
    """Lazy singleton -- keeps /health cheap and startup GPU-free until first use."""
    if not _STATE:
        from transformers import AutoModelForCausalLM, AutoProcessor
        _STATE["proc"] = AutoProcessor.from_pretrained(MODEL_ID)
        _STATE["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype="auto", device_map="auto")
    return _STATE["proc"], _STATE["model"]


def _extract_table_json(text: str):
    """Best-effort strict parse for callers hitting this service directly; the
    router uses the tolerant table_export.parse_table_json on `raw` instead."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"headers": [], "rows": []}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"headers": [], "rows": []}
    return {"headers": data.get("headers", []), "rows": data.get("rows", [])}


@app.post("/table")
async def run_table(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    proc, model = _load()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": TABLE_PROMPT}]}]
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=text, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    # Greedy + a high token budget: table extraction is a structured task, not
    # creative chat -- sampling invites malformed JSON, and a 20-row table can
    # exceed a small budget and get truncated to an empty parse.
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    raw = proc.decode(outputs[0][input_len:], skip_special_tokens=True)
    return {"tables": [_extract_table_json(raw)], "raw": raw}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_STATE)}
