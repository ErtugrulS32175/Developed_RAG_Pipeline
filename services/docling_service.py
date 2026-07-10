"""Standalone Granite-Docling (ibm-granite/granite-docling-258M) service, kept
independent of the main router/pipeline: POST an image, get back the raw DocTags,
the rendered markdown, and any tables as {headers, rows} -- purely for inspecting
what this model produces on your documents.

Why this exists / the KV-cache gotcha you hit: the model is only 258M params but
the SigLIP2 vision encoder turns a page into hundreds of vision tokens, and the
DocTags output (a full-page layout markup) is a long generated sequence. If the
model's config ships use_cache=false, transformers recomputes the whole sequence
every decode step -> it behaves like the "KV cache keeps filling up" slowdown you
saw. We force use_cache=True in generate() below, run on GPU in bf16, and cap
max_new_tokens; cropping to the table region first (as the TATR path does) cuts
the generated length further. For a bigger speedup still, a llama.cpp/GGUF build
is the community's 100x-faster route -- out of scope for this transformers svc.
"""
import io
import os
import time

import requests
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from pipeline import table_docling as td
from pipeline import image_preprocess as ip

app = FastAPI()

MODEL_ID = os.getenv("DOCLING_MODEL", "ibm-granite/granite-docling-258M")
MAX_NEW_TOKENS = int(os.getenv("DOCLING_MAX_NEW_TOKENS", "4096"))
PROMPT = os.getenv("DOCLING_PROMPT", "Convert this page to docling.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# /table_tr uses paddle_service for column geometry (Turkish-text correction).
PADDLE_OCR_URL = os.getenv("PADDLE_OCR_URL", "http://127.0.0.1:8100/ocr")
PADDLE_TIMEOUT = float(os.getenv("SERVICE_TIMEOUT", "120"))
TEXT_COLS = tuple(int(c) for c in os.getenv("DOCLING_TEXT_COLS", "1").split(",") if c.strip() != "")
# Optional master list (one name per line): snaps text columns to canonical
# spellings and flags unknowns for review. Off unless DOCLING_MASTER_FILE is set.
# OCR image-enhancement layer (deskew + denoise + CLAHE + 2x upscale). Default on.
PREPROCESS = os.getenv("PREPROCESS", "1").lower() not in ("0", "false", "no")
_MASTER_FILE = os.getenv("DOCLING_MASTER_FILE")
MASTER_NAMES = None
if _MASTER_FILE and os.path.exists(_MASTER_FILE):
    with open(_MASTER_FILE, encoding="utf-8") as f:
        MASTER_NAMES = [ln.strip() for ln in f if ln.strip()]

_STATE = {}  # lazy: {processor, model} -- loaded on first /docling, keeps startup fast


def _load():
    if not _STATE:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        _STATE["processor"] = AutoProcessor.from_pretrained(MODEL_ID)
        _STATE["model"] = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
    return _STATE["processor"], _STATE["model"]


def _doctags_to_doc(doctags, image):
    """DocTags string -> DoclingDocument (for markdown + structured tables).
    Returns None if the installed docling_core API differs; the raw DocTags are
    still returned to the caller regardless."""
    try:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        dt = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        return DoclingDocument.load_from_doctags(dt, document_name="Document")
    except Exception:
        return None


def _tables_from_doc(doc):
    """Pull tables out of a DoclingDocument as [{headers, rows}] so the output
    lines up with the rest of the pipeline's table shape."""
    tables = []
    for tbl in getattr(doc, "tables", []) or []:
        try:
            df = tbl.export_to_dataframe()
            headers = [str(c) for c in df.columns.tolist()]
            rows = [[("" if v is None else str(v)) for v in row] for row in df.values.tolist()]
            tables.append({"headers": headers, "rows": rows})
        except Exception:
            continue
    return tables


def _generate(image):
    """Run Granite-Docling on an image -> (doctags, gen_tokens, seconds)."""
    processor, model = _load()
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        # use_cache=True is the fix for the "KV cache" slowdown: it enables the
        # incremental key/value cache so each new token is O(1), not O(seq) --
        # overriding any use_cache=false left in the model config.
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, use_cache=True)
    elapsed = round(time.time() - t0, 2)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    doctags = processor.batch_decode(trimmed, skip_special_tokens=False)[0]
    return doctags.replace("<end_of_utterance>", "").strip(), int(trimmed.shape[1]), elapsed


def _paddle_boxes(crop):
    """OCR the crop via paddle_service -> word boxes [[x0,y0,x1,y1]] for column
    geometry (Granite emits no per-cell boxes in OTSL mode)."""
    buf = io.BytesIO(); crop.save(buf, format="PNG")
    r = requests.post(PADDLE_OCR_URL,
                      files={"file": ("crop.png", buf.getvalue(), "image/png")},
                      timeout=PADDLE_TIMEOUT)
    r.raise_for_status()
    return [ln["box"] for ln in r.json().get("lines", []) if ln.get("box")]


def _split_header(headers, rows):
    """Granite exports column-header cells as data row 0 with positional headers
    (0..N); promote that row to the real header."""
    if headers and all(str(h).isdigit() for h in headers) and rows:
        return rows[0], rows[1:]
    return headers, rows


@app.post("/docling")
async def run_docling(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    if PREPROCESS:
        image = ip.enhance(image)
    doctags, gen_tokens, elapsed = _generate(image)
    doc = _doctags_to_doc(doctags, image)
    markdown = doc.export_to_markdown() if doc is not None else None
    tables = _tables_from_doc(doc) if doc is not None else []
    return {
        "doctags": doctags,
        "markdown": markdown,
        "tables": tables,
        "n_tables": len(tables),
        "gen_tokens": gen_tokens,
        "seconds": elapsed,
        "device": DEVICE,
    }


@app.post("/table_tr")
async def table_tr(file: UploadFile = File(...)):
    """Granite grid + numbers, with Turkish text columns re-read by EasyOCR-tr
    (column geometry from paddle_service). Returns {headers, rows} like the other
    table backends. Independent of the router pipeline -- inspect it directly."""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    if PREPROCESS:
        image = ip.enhance(image)
    doctags, gen_tokens, elapsed = _generate(image)
    doc = _doctags_to_doc(doctags, image)
    tables = _tables_from_doc(doc) if doc is not None else []
    if not tables:
        return {"tables": [], "detected": False, "seconds": elapsed}

    headers, rows = _split_header(tables[0]["headers"], tables[0]["rows"])
    bbox = td.table_bbox_from_doctags(doctags, image.width, image.height)
    crop = image.crop(bbox) if bbox else image
    words = _paddle_boxes(crop)
    headers, rows, flags = td.correct(crop, headers, rows, words,
                                      text_cols=TEXT_COLS, master_names=MASTER_NAMES)

    return {
        "tables": [{"headers": headers, "rows": rows}],
        "flags": flags,
        "detected": True,
        "gen_tokens": gen_tokens,
        "seconds": elapsed,
    }


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_STATE), "model": MODEL_ID, "device": DEVICE}
