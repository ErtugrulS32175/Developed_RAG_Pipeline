import os
import time
import uuid
from pathlib import Path
from typing import Union

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import db
from pipeline import ingest_router
from pipeline import owui_chat
from pipeline.query import ask

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
    # OpenWebUI streams by default; without this the flag would be silently
    # dropped and every reply would arrive as one lump after minutes of silence.
    stream: bool = False


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
    is_table = req.model == TABLE_MODEL_ID
    if req.stream:
        gen = (owui_chat.stream_tables(req.messages, req.model) if is_table
               else owui_chat.stream_text(
                   ask(owui_chat.message_text(req.messages[-1].content)), req.model))
        return StreamingResponse(gen, media_type="text/event-stream")

    if is_table:
        try:
            answer = owui_chat.tables_reply(req.messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"tablo cikarimi basarisiz: {e}")
    else:
        answer = ask(owui_chat.message_text(req.messages[-1].content))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
    }


@app.get("/files/{name}")
def download_export(name: str):
    """Serve a generated xlsx. Linked from the chat reply, so this is opened by
    the user's browser rather than by OpenWebUI itself."""
    if not owui_chat.EXPORT_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="gecersiz dosya adi")
    path = owui_chat.EXPORT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="dosya bulunamadi")
    return FileResponse(
        path,
        filename=name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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
