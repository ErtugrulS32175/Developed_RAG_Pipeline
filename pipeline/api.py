import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import db
from pipeline import ingest_router
from pipeline.query import ask

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

conn = db.get_conn()
db.init_schema(conn)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


@app.get("/v1/models")
def list_models():
    """OpenWebUI (or any OpenAI-compatible client) calls this to discover
    which model id to send chat completions requests for."""
    return {"object": "list", "data": [{"id": "ragtest-rag", "object": "model", "owned_by": "ragtest"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible wrapper around the existing retrieve/rerank/generate
    pipeline in pipeline.query -- this is the whole integration surface
    OpenWebUI needs; it does not know or care that RAG happens underneath."""
    question = req.messages[-1].content
    answer = ask(question)
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
