import base64
import os
import re
import time
import uuid
from pathlib import Path
from typing import Union

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import db
from pipeline import ingest_router
from pipeline.query import ask
from pipeline.table_pipeline import run_consensus
from pipeline.table_export import table_to_markdown

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Two OpenAI-style model ids OpenWebUI shows in its selector. The RAG one keeps
# the existing text pipeline; the table one runs the image->consensus->table flow.
RAG_MODEL_ID = "ragtest-rag"
TABLE_MODEL_ID = "ragtest-table"

app = FastAPI()

conn = db.get_conn()
db.init_schema(conn)


class ChatMessage(BaseModel):
    role: str
    # OpenWebUI sends a plain string for text turns, but an OpenAI-vision-style
    # list of content parts ({"type":"text",...}/{"type":"image_url",...}) once
    # an image is attached via the "+" button -- accept both.
    content: Union[str, list]


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}


def _message_text(content) -> str:
    """The user's typed text, whether the turn is a plain string or a
    multimodal content-part list."""
    if isinstance(content, str):
        return content
    return "\n".join(
        p.get("text", "") for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ).strip()


def _message_image_urls(content) -> list:
    """Every image_url attached to a multimodal turn, in order."""
    if isinstance(content, str):
        return []
    urls = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url:
                urls.append(url)
    return urls


def _save_data_url_image(data_url: str):
    """Decode a base64 data: URL (what OpenWebUI embeds) to a file on disk so the
    consensus pipeline, which works from a path, can read it. Returns None for a
    plain http(s) url, which we don't fetch."""
    m = _DATA_URL_RE.match(data_url)
    if not m:
        return None
    ext = _MIME_EXT.get(m.group("mime"), "png")
    dest = UPLOAD_DIR / f"owui-{uuid.uuid4().hex[:12]}.{ext}"
    dest.write_bytes(base64.b64decode(m.group("data")))
    return dest


def _extract_tables_reply(messages) -> str:
    """Run the two-VLM consensus on the image in the latest turn and render the
    result(s) as markdown OpenWebUI can display inline."""
    images = _message_image_urls(messages[-1].content)
    if not images:
        return ("Tablo çıkarımı için bir tablo görüntüsü yükleyin: sohbet "
                "kutusundaki + ile bir PNG/JPG ekleyip gönderin.")

    dest = _save_data_url_image(images[-1])
    if dest is None:
        return "Görüntü çözülemedi (yalnızca gömülü base64 görüntüler destekleniyor)."

    try:
        results = run_consensus(str(dest))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tablo cikarimi basarisiz: {e}")

    if not results:
        return "Görüntüde tablo bulunamadı."

    blocks = []
    for i, r in enumerate(results, 1):
        md = table_to_markdown(r["headers"], r["rows"])
        note = f"_Güven: {r['confidence']:.2f}_"
        if r.get("needs_review"):
            n = len(r.get("disagreements", []))
            note += (f" — ⚠️ {n} hücrede modeller uyuşmuyor, gözden geçirin"
                     if n else " — ⚠️ gözden geçirilmeli")
        blocks.append(f"**Tablo {i}**\n\n{md}\n\n{note}")
    return "\n\n---\n\n".join(blocks)


@app.get("/v1/models")
def list_models():
    """OpenWebUI (or any OpenAI-compatible client) calls this to discover which
    model ids to send chat completions requests for. Mark the table model as
    vision-capable in OpenWebUI's model settings so attached images are forwarded
    as base64 instead of being routed through document RAG."""
    return {
        "object": "list",
        "data": [
            {"id": RAG_MODEL_ID, "object": "model", "owned_by": "ragtest"},
            {"id": TABLE_MODEL_ID, "object": "model", "owned_by": "ragtest"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible wrapper. Routes by model id: the table model runs the
    image->consensus->table flow; anything else falls through to the existing
    retrieve/rerank/generate RAG pipeline. OpenWebUI does not know or care which
    happens underneath."""
    if req.model == TABLE_MODEL_ID:
        answer = _extract_tables_reply(req.messages)
    else:
        answer = ask(_message_text(req.messages[-1].content))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(await file.read())
    file_type = dest.suffix.lower().lstrip(".")
    document_id = db.upsert_document(conn, file.filename, file_type, status="pending")
    return {"document_id": document_id, "filename": file.filename, "status": "pending"}


@app.post("/documents/{document_id}/process")
def process_document(document_id: str):
    doc = db.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")

    path = UPLOAD_DIR / doc["filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"uploaded file missing on disk: {path}")

    try:
        ingest_router.main(str(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")

    return {"document_id": document_id, "status": "done"}


@app.get("/documents/{document_id}")
def read_document(document_id: str):
    doc = db.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@app.get("/health")
def health():
    return {"status": "ok"}
