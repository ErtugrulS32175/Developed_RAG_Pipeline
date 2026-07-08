import io
import json
import os
import re

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

app = FastAPI()

# Configurable so the table-extraction model can be swapped independently of
# the chat/LLM role's model (services/llm_service.py) -- they're separate
# services on purpose, even when they happen to both be Gemma today.
MODEL_ID = os.getenv("GEMMA_TABLE_MODEL", "google/gemma-4-E4B-it")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# High visual token budget per the model card's guidance for OCR/document
# parsing (options are 70/140/280/560/1120) -- exact processor kwarg name
# is unconfirmed until we can run this against the real model, check the
# "Code for processing Images" snippet on the model page on first run.
TABLE_PROMPT = (
    "Extract every table in this image as a single JSON object of the form "
    '{"headers": [...], "rows": [[...], ...]}. '
    "Preserve every row and column exactly as shown, keep empty cells as \"\". "
    "Preserve Turkish characters exactly (ğ Ğ ş Ş ı İ ç Ç ö Ö ü Ü); never split "
    "a letter into a base character plus a separate accent mark. "
    "Return only the JSON, no other text."
)


def _extract_table_json(text: str):
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
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TABLE_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    # Greedy + higher token budget: table extraction is a structured task, not
    # creative chat -- sampling (temp 1.0) invites malformed JSON, and a 20-row
    # table can exceed 2048 tokens and get truncated (empty parse).
    outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    # TEMP DEBUG: surface the raw model text so we can diagnose empty parses.
    # Prints to the RunPod service log AND returns under "raw"; remove once the
    # extraction is confirmed working.
    print("[GEMMA RAW]", repr(response[:3000]), flush=True)
    return {"tables": [_extract_table_json(response)], "raw": response}


@app.get("/health")
def health():
    return {"status": "ok"}
