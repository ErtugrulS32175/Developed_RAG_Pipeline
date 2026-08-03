import json
import logging
import os
import secrets
import time
import traceback
import uuid
from pathlib import Path
from typing import Union

from fastapi import Depends, FastAPI, Header, HTTPException, Response, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline.index import db
from pipeline.index import ingest
from pipeline.api import owui_chat
from pipeline.retrieval import rag_backends
from pipeline.validation.rag.answer_guard import (
    ABSTAINED,
    ANSWERED,
    REVIEW_REQUIRED,
    GuardResult,
    is_abstention,
)

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Two OpenAI-style model ids OpenWebUI shows in its selector. The RAG one keeps
# the existing text pipeline; the table one runs the image->consensus->table flow.
RAG_MODEL_ID = "ragtest-rag"
TABLE_MODEL_ID = "ragtest-table"
# A third id so the alternative engine can be picked per conversation, the same
# way the table flow is picked today. Both answer the same questions from the
# same documents; which one is better is a measurement, not a default.
LLAMAINDEX_MODEL_ID = "ragtest-rag-llamaindex"
RAG_MODELS = {
    RAG_MODEL_ID: "native",
    LLAMAINDEX_MODEL_ID: "llamaindex",
}
REVIEW_MESSAGE = (
    "Yanıt kaynaklarla otomatik olarak doğrulanamadı; "
    "insan incelemesi gerekiyor."
)
RAG_UNAVAILABLE_MESSAGE = "Seçilen RAG motoru şu anda kullanılamıyor."
RAG_FAILURE_MESSAGE = "RAG yanıtı üretilemedi."
DOCUMENT_PROCESSING_FAILURE_MESSAGE = "Belge işlenemedi."
_WINDOWS_DEVICE_NAMES = frozenset({
    "con", "prn", "aux", "nul", "clock$", "conin$", "conout$",
    *(
        f"{prefix}{suffix}"
        for prefix in ("com", "lpt")
        for suffix in (*map(str, range(1, 10)), "¹", "²", "³")
    ),
})

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ragtest.api")

# Interactive docs stay on while the API is unauthenticated -- that is the local
# development case, where they are useful. Once a key is configured the service
# is reachable by someone else, and there is no reason to publish its surface to
# a caller who cannot use any of it.
_DOCS_OPEN = not os.getenv("API_KEY", "").strip()
app = FastAPI(
    docs_url="/docs" if _DOCS_OPEN else None,
    redoc_url="/redoc" if _DOCS_OPEN else None,
    openapi_url="/openapi.json" if _DOCS_OPEN else None,
)


@app.middleware("http")
async def log_requests(request, call_next):
    """One structured line per request.

    Method, path, status and duration only -- deliberately NOT the body or the
    query string. Requests here carry questions and answers drawn from private
    documents, and a log file is just another place that content can leak to.
    The request id makes a single call traceable without recording what it said.
    """
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        # An arbitrary exception message can contain a DSN, a local path or
        # document/model text. The shared helper keeps safe frame locations
        # without copying that untrusted detail into a second storage system.
        _log_safe_failure(
            error,
            "api_istek_hatasi",
            istek=request_id,
            yol=request.url.path,
            yontem=request.method,
            durum="exception",
        )
        raise
    log.info(json.dumps({
        "istek": request_id,
        "yontem": request.method,
        "yol": request.url.path,
        "durum": response.status_code,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }))
    return response


_conn = None


def get_conn():
    """Connect on first use rather than at import, so this module can be
    imported (by a test, by tooling) without a running database."""
    global _conn
    if _conn is None:
        _conn = db.get_conn()
        db.init_schema(_conn)
    return _conn


# Shared-secret auth. Enforced when API_KEY is set; when it is not, the API is
# open and says so loudly at startup rather than pretending to be protected.
# That mirrors how vLLM and the rest of this stack behave, and keeps a local
# run friction-free -- but anything reachable beyond localhost must set it.
API_KEY = os.getenv("API_KEY", "").strip()
if not API_KEY:
    log.warning("API_KEY tanimli degil: ucnoktalar kimlik dogrulamasiz. "
                "Yerel disinda calistiriyorsan API_KEY ayarla.")


def require_api_key(authorization: str = Header(default="")):
    """Bearer-token check.

    `compare_digest` rather than `==`: a plain comparison returns as soon as it
    finds a differing byte, which leaks the key one character at a time to
    anyone able to time the responses.
    """
    if not API_KEY:
        return
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="gecersiz veya eksik API anahtari")


AUTH = [Depends(require_api_key)]


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


@app.get("/v1/models", dependencies=AUTH)
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
            {"id": LLAMAINDEX_MODEL_ID, "object": "model", "owned_by": "ragtest"},
        ],
    }


@app.post("/v1/chat/completions", dependencies=AUTH)
def chat_completions(req: ChatRequest):
    """OpenAI-compatible wrapper with a checked publication boundary.

    Table extraction keeps its separate service path. A RAG model may publish
    only the text carried by an answered/abstained ``GuardResult``; review
    results become a fixed notice and never expose the unchecked model reply.
    """
    if req.model not in {*RAG_MODELS, TABLE_MODEL_ID}:
        raise HTTPException(status_code=404, detail="bilinmeyen model")
    if not req.messages:
        raise HTTPException(status_code=400, detail="en az bir mesaj gerekli")

    is_table = req.model == TABLE_MODEL_ID
    backend = RAG_MODELS.get(req.model)

    def ask_checked():
        question = owui_chat.message_text(req.messages[-1].content)
        if not question.strip():
            raise HTTPException(status_code=400, detail="soru bos olamaz")
        try:
            result = rag_backends.answer_checked(question, backend=backend)
        except RuntimeError as e:
            _log_rag_failure(e, backend)
            raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_MESSAGE)
        except Exception as e:
            _log_rag_failure(e, backend)
            raise HTTPException(status_code=502, detail=RAG_FAILURE_MESSAGE)
        return _publish_checked(result)

    if req.stream:
        if is_table:
            gen = owui_chat.stream_tables(req.messages, req.model)
        else:
            status, answer = ask_checked()
            gen = owui_chat.stream_text(answer, req.model, rag_status=status)
        return StreamingResponse(gen, media_type="text/event-stream")

    if is_table:
        try:
            answer = owui_chat.tables_reply(req.messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"tablo cikarimi basarisiz: {e}")
    else:
        status, answer = ask_checked()

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
    }
    if not is_table:
        response["rag_status"] = status
    return response


def _publish_checked(result):
    """Project a validated result to public status and text.

    Malformed internal values are programmer errors, not review decisions. They
    fail closed with a generic response rather than hiding the bug or coercing a
    raw string into something publishable.
    """
    if not isinstance(result, GuardResult):
        log.error("RAG backend checked contract returned %s",
                  type(result).__name__)
        raise HTTPException(status_code=500, detail="gecersiz RAG yanit sozlesmesi")
    has_text = isinstance(result.answer, str) and bool(result.answer.strip())
    clean = result.diagnostics == ()
    if (
        result.status == ANSWERED
        and has_text
        and clean
        and not is_abstention(result.answer)
    ):
        return result.status, result.answer
    if (
        result.status == ABSTAINED
        and has_text
        and clean
        and is_abstention(result.answer)
    ):
        return result.status, result.answer
    if result.status == REVIEW_REQUIRED and result.answer is None:
        return result.status, REVIEW_MESSAGE
    log.error("RAG backend returned inconsistent checked status")
    raise HTTPException(status_code=500, detail="gecersiz RAG yanit sozlesmesi")


def _log_safe_failure(error, event, **fields):
    """Log enough traceback structure to debug without logging its message.

    Exception messages from HTTP/model/database clients can contain endpoints,
    credentials or model text. File basename, line and function retain the code
    path while deliberately excluding those values and all request content.
    """
    frames = [
        {
            "dosya": Path(frame.filename).name,
            "satir": frame.lineno,
            "fonksiyon": frame.name,
        }
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    log.error(json.dumps({
        "olay": event,
        **fields,
        "hata": type(error).__name__,
        "iz": frames,
    }))


def _log_rag_failure(error, backend):
    _log_safe_failure(error, "rag_yanit_hatasi", backend=backend)


def _safe_upload_filename(filename):
    """Reject path syntax, trailing aliases and Windows device spellings."""
    if not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="gecersiz dosya adi")
    original = filename
    raw = original.strip()
    portable = raw.replace("\\", "/")
    stem = portable.split(".", 1)[0].rstrip(" .").casefold()
    if (
        not portable
        or raw != original
        or any(ord(char) < 32 for char in portable)
        or "\x00" in portable
        or ":" in portable
        or portable.rstrip(". ") != portable
        or stem in _WINDOWS_DEVICE_NAMES
        or portable in {".", ".."}
        or Path(portable).name != portable
    ):
        raise HTTPException(status_code=400, detail="gecersiz dosya adi")
    return portable


def _set_document_error(conn, document_id):
    """Best-effort terminal status update without masking the primary failure."""
    try:
        db.set_document_status(conn, document_id, "error")
    except Exception as status_error:
        _log_safe_failure(
            status_error,
            "belge_durumu_yazilamadi",
        )


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


@app.post("/documents/upload", dependencies=AUTH)
async def upload_document(file: UploadFile = File(...)):
    filename = _safe_upload_filename(file.filename)
    dest = UPLOAD_DIR / filename
    dest.write_bytes(await file.read())
    file_type = dest.suffix.lower().lstrip(".")
    document_id = db.upsert_document(
        get_conn(),
        filename,
        file_type,
        status="pending",
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "status": "pending",
    }


@app.post("/documents/{document_id}/process", dependencies=AUTH)
def process_document(document_id: str):
    conn = get_conn()
    doc = db.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")

    try:
        filename = _safe_upload_filename(doc["filename"])
    except HTTPException:
        _set_document_error(conn, document_id)
        log.error(json.dumps({
            "olay": "gecersiz_kayitli_dosya_adi",
            "hata": "InvalidFilename",
        }))
        raise HTTPException(
            status_code=500,
            detail=DOCUMENT_PROCESSING_FAILURE_MESSAGE,
        )

    path = UPLOAD_DIR / filename
    if not path.exists():
        _set_document_error(conn, document_id)
        raise HTTPException(status_code=404, detail="uploaded file missing")

    try:
        db.set_document_status(conn, document_id, "processing")
        ingest.main(str(path))
        processed = db.get_document(conn, document_id)
    except Exception as e:
        _set_document_error(conn, document_id)
        _log_safe_failure(e, "belge_isleme_hatasi")
        raise HTTPException(
            status_code=500,
            detail=DOCUMENT_PROCESSING_FAILURE_MESSAGE,
        )

    if processed is None or processed.get("status") != "done":
        _set_document_error(conn, document_id)
        log.error(json.dumps({
            "olay": "belge_tamamlanmadan_dondu",
            "hata": "IncompleteIngest",
        }))
        raise HTTPException(
            status_code=500,
            detail=DOCUMENT_PROCESSING_FAILURE_MESSAGE,
        )

    return {"document_id": document_id, "status": processed["status"]}


@app.get("/documents/{document_id}", dependencies=AUTH)
def read_document(document_id: str):
    doc = db.get_document(get_conn(), document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@app.get("/health")
def health():
    """Liveness: the process is up. Deliberately touches nothing else, so a
    restart loop is never triggered by a dependency being briefly unavailable."""
    return {"status": "ok"}


def _probe(name, fn):
    """Report whether a dependency answers -- and nothing more.

    The failure detail goes to the log, not the response. A connection error
    carries the host, port and user it was trying, and /ready has to stay
    reachable without a credential for a load balancer to use it.
    """
    try:
        fn()
        return name, True
    except Exception as e:
        log.warning(json.dumps({"kontrol": name, "hata": type(e).__name__}))
        log.debug("%s kontrolu basarisiz", name, exc_info=True)
        return name, False


@app.get("/ready")
def ready(response: Response):
    """Readiness: can this instance actually serve a request?

    Separate from /health on purpose. Liveness answers "should I be restarted",
    readiness answers "should traffic be sent to me" -- conflating them means a
    database blip restarts a healthy process. Returns 503 when a dependency is
    down so a load balancer or compose healthcheck can act on it.
    """
    import requests

    from pipeline.index import embeddings

    def check_db():
        with get_conn().cursor() as cur:
            cur.execute("SELECT 1")

    def check_embed():
        base = embeddings.EMBED_API_URL.rsplit("/v1/", 1)[0]
        requests.get(f"{base}/v1/models", timeout=3).raise_for_status()

    checks = dict([_probe("veritabani", check_db), _probe("embedding", check_embed)])
    healthy = all(checks.values())
    response.status_code = 200 if healthy else 503
    return {"status": "ready" if healthy else "degraded", "kontroller": checks}
