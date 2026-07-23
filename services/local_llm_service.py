import os
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

app = FastAPI()

# Kept independent from services/gemma_service.py (the table/image-extraction
# role) on purpose: LLM_SERVICE_MODEL can be swapped to a different model
# later without touching the table-extraction service, and vice versa.
MODEL_ID = os.getenv("LLM_SERVICE_MODEL", "google/gemma-4-E4B-it")

# device_map="auto" + 4-bit was tried first: accelerate's planner sizes the
# offload plan off the *unquantized* footprint, decided some modules (Gemma
# 4's large PLE per-layer-embedding tables) needed to live on CPU, and that
# mixed CPU/GPU placement crashed at generate() time with a bitsandbytes/
# accelerate hook incompatibility ("Cannot copy out of meta tensor; no
# data!") on the per-layer projection module. Forcing everything onto the
# single GPU sidesteps that -- the real 4-bit footprint comfortably fits.
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0})


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible text-only chat endpoint -- pipeline/query.py already
    talks to LLM_API_URL in this exact shape, so pointing that env var here
    is the only integration needed, no client-side code change."""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs, max_new_tokens=1024, do_sample=True,
        temperature=req.temperature, top_p=0.95, top_k=64,
    )
    answer = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
