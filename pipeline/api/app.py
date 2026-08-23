import json
import logging
import os
import secrets
import time
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Union
from uuid import UUID

from fastapi import (
    Depends, FastAPI, Header, HTTPException, Query, Response, UploadFile, File,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import AwareDatetime, BaseModel, Field
from dotenv import load_dotenv

from pipeline.index import db
from pipeline.index import ingest
from pipeline.index import publication
from pipeline.index.attempt_contract import (
    AttemptAlreadyRunning,
    AttemptOutcome,
    CandidateConflict,
    CandidateNotPublished,
    CandidateSuperseded,
)
from pipeline.api import owui_chat
from pipeline.retrieval import rag_backends
from pipeline.validation.rag.answer_guard import (
    ABSTAINED,
    ANSWERED,
    REVIEW_REQUIRED,
    GuardResult,
    PageCitation,
    is_abstention,
)

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# Cap chosen from the data this system actually ingests: the largest source
# document seen so far is ~30MB, so 50MB passes everything legitimate while an
# unbounded read no longer lets one request hold the whole file in memory.
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(50 * 1024 * 1024)))

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


@asynccontextmanager
async def _lifespan(_app):
    # Nothing to do on startup: the pool is created lazily on first checkout.
    # Closing it explicitly makes reloads and test processes deterministic
    # instead of leaving idle connections to the OS.
    yield
    db.close_pool()


app = FastAPI(
    lifespan=_lifespan,
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


_schema_ready = False


@contextmanager
def db_conn():
    """One pooled connection per request, connected on first use rather than at
    import so this module stays importable without a running database.

    The pool's context manager commits on clean exit and rolls back on
    exception, and checkout revalidates the connection -- so one failed
    statement can no longer poison every request that follows, which is
    exactly what the previous single cached module-level connection did."""
    global _schema_ready
    with db.get_pool().connection() as conn:
        if not _schema_ready:
            db.init_schema(conn)
            _schema_ready = True
        yield conn


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


# How many documents one question may name. A scope is a NARROWING, so a
# bound is what keeps it one: without a cap, a caller could hand over an
# arbitrarily long array that every retrieval statement then carries. The
# number is a product decision, declared here once and enforced by the
# request model, so an oversized scope is refused before the endpoint body
# runs -- and therefore before a connection, an embedding or a backend.
DOCUMENT_SCOPE_MAX = 50

# The scope's SHAPE lives on the field, not in the body, for exactly the
# reason the inventory's page bounds do: a declaration is refused with 422
# before anything is borrowed or computed. `UUID` is the type because
# `documents.id` is `uuid` -- so "well-formed identifier" is answered by
# the column's own type rather than by a second rule invented here. A
# malformed element, a non-list value or a list outside 1..MAX is a
# validation error with a `body -> document_ids` location; it is never
# silently dropped, because a dropped scope is a question answered over the
# whole corpus while the caller believes it was narrowed.
DocumentScope = Annotated[
    list[UUID],
    Field(min_length=1, max_length=DOCUMENT_SCOPE_MAX),
]
TagName = Annotated[str, Field(min_length=1)]
TagScope = Annotated[
    list[TagName],
    Field(min_length=1, max_length=DOCUMENT_SCOPE_MAX),
]


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # OpenWebUI streams by default; without this the flag would be silently
    # dropped and every reply would arrive as one lump after minutes of silence.
    stream: bool = False
    # Absent means what it has always meant: the whole corpus, unchanged.
    document_ids: DocumentScope | None = None
    collection_ids: DocumentScope | None = None
    tags: TagScope | None = None


class CollectionRequest(BaseModel):
    name: str = Field(min_length=1)


class DocumentTagsRequest(BaseModel):
    tags: list[TagName] = Field(max_length=DOCUMENT_SCOPE_MAX)


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


def _document_scope(document_ids):
    """Collapse the requested identifiers into ONE canonical scope.

    Absent stays absent: `None` here is the whole corpus, and it must not
    acquire a scope by accident anywhere below.

    Supplied, the identifiers are collapsed with SET SEMANTICS and ordered,
    once, before the first backend call. A list naming one document three
    times therefore produces exactly the scope -- and exactly the retrieval
    filter -- that naming it once produces: a repetition can neither widen
    the scope nor travel to the database as a repeated filter value. The
    canonical spelling is `str(UUID)`, which the request model already
    produced by parsing, so two spellings of one identifier are one element
    here rather than two.
    """
    if document_ids is None:
        return None
    return tuple(sorted({str(document_id) for document_id in document_ids}))


def _chat_document_scope(req: ChatRequest, is_table: bool):
    """Resolve every RAG scope dimension through the document table once."""
    direct = _document_scope(req.document_ids)
    if is_table or (req.collection_ids is None and req.tags is None):
        return direct
    collections = _document_scope(req.collection_ids)
    try:
        with db_conn() as conn:
            return db.resolve_document_scope(
                conn, document_ids=direct, collection_ids=collections,
                tags=req.tags)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.post("/v1/chat/completions", dependencies=AUTH)
def chat_completions(req: ChatRequest):
    """OpenAI-compatible wrapper with a checked publication boundary.

    Table extraction keeps its separate service path. A RAG model may publish
    only the text carried by an answered/abstained ``GuardResult``; review
    results become a fixed notice and never expose the unchecked model reply.

    ``document_ids`` narrows a RAG question to a named set of documents.
    Its SHAPE is refused by the request model before this body runs; its
    MEANING is settled here, once, and then handed to the one checked call
    both response shapes go through -- which is why a streamed answer and a
    non-streamed one cannot be scoped differently.
    """
    if req.model not in {*RAG_MODELS, TABLE_MODEL_ID}:
        raise HTTPException(status_code=404, detail="bilinmeyen model")
    if not req.messages:
        raise HTTPException(status_code=400, detail="en az bir mesaj gerekli")

    is_table = req.model == TABLE_MODEL_ID
    backend = RAG_MODELS.get(req.model)
    # Settled BEFORE the first backend call, and outside the closure, so
    # both branches ask the same question of the same set. The table route
    # reads none of this: it keeps its separate service path unchanged.
    document_ids = _chat_document_scope(req, is_table)

    def ask_checked():
        question = owui_chat.message_text(req.messages[-1].content)
        if not question.strip():
            raise HTTPException(status_code=400, detail="soru bos olamaz")
        # Forwarded only when a scope was asked for: an unscoped request
        # must reach the backend as the call it has always been.
        scope = {} if document_ids is None else {"document_ids": document_ids}
        try:
            result = rag_backends.answer_checked(question, backend=backend,
                                                 **scope)
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
            status, answer, citations = ask_checked()
            gen = owui_chat.stream_text(
                answer,
                req.model,
                rag_status=status,
                rag_citations=_citation_payload(citations),
            )
        return StreamingResponse(gen, media_type="text/event-stream")

    if is_table:
        try:
            answer = owui_chat.tables_reply(req.messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"tablo cikarimi basarisiz: {e}")
    else:
        status, answer, citations = ask_checked()

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
        response["rag_citations"] = _citation_payload(citations)
    return response


def _citation_payload(citations):
    return [
        {"page": citation.page, "source": citation.source}
        for citation in citations
    ]


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
    valid_citations = (
        type(result.citations) is tuple
        and all(isinstance(citation, PageCitation)
                for citation in result.citations)
    )
    if not valid_citations:
        log.error("RAG backend returned invalid citation metadata")
        raise HTTPException(status_code=500, detail="gecersiz RAG yanit sozlesmesi")
    if (
        result.status == ANSWERED
        and has_text
        and clean
        and not is_abstention(result.answer)
    ):
        return result.status, result.answer, result.citations
    if (
        result.status == ABSTAINED
        and has_text
        and clean
        and is_abstention(result.answer)
        and result.citations == ()
    ):
        return result.status, result.answer, result.citations
    if result.status == REVIEW_REQUIRED and result.answer is None:
        return result.status, REVIEW_MESSAGE, ()
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


# THE API NO LONGER STAMPS A DOCUMENT `error` ANYWHERE. The helper that
# did it is gone rather than merely unused, so nothing can quietly start
# calling it again. Every remaining case it served has moved to the
# subject it was actually about: a run's own failure goes on that run's
# ATTEMPT, a request's failure is the HTTP status, and a source file
# that has gone missing is a storage problem that says nothing about the
# generation currently being served.


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


# The endpoint used to keep a per-filename lock of its own, IN THIS
# PROCESS, as a cheap first fence in front of its hand-written publish
# sequence. There is no sequence here any more: the publication service
# holds a database SESSION lock, which serialises across PROCESSES and
# therefore across every worker, not just this one. A second, weaker lock
# in front of it fenced nothing the first did not already fence.
@app.post("/documents/upload", dependencies=AUTH)
async def upload_document(file: UploadFile = File(...), replace: bool = False):
    """Publish a candidate -- through the SHARED SERVICE and nothing else.

    This endpoint used to carry its own copy of the whole sequence:
    resolve the canonical name, check the conflict, write a temp file,
    upsert the row, os.replace. The CLI had no publication step at all,
    so the two paths drifted until they carried different guarantees.
    Everything below the read is now one call, and the lock, the crash
    windows and the disk target live where they belong -- in
    ``publication.publish_candidate``.

    NO ``replaced`` FIELD any more. It used to be computed by hashing
    whatever was on disk before writing, which the service (rightly)
    does not report back; recomputing it here would mean a read OUTSIDE
    the publish lock, and a guess printed as a fact is exactly what this
    audit has been removing. ``candidate_id`` answers the same question
    truthfully: the same bytes keep it, different bytes mint a new one.
    """
    filename = _safe_upload_filename(file.filename)
    # Read in chunks against the cap BEFORE anything touches disk or the
    # database, so an oversized upload leaves no partial file and no row.
    pieces, size = [], 0
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        size += len(piece)
        if size > UPLOAD_MAX_BYTES:
            raise HTTPException(status_code=413, detail="dosya cok buyuk")
        pieces.append(piece)
    body = b"".join(pieces)
    file_type = Path(filename).suffix.lower().lstrip(".")

    with db_conn() as conn:
        try:
            document_id, candidate_id, canonical = (
                publication.publish_candidate(
                    conn, filename, file_type, body,
                    allow_replace=replace))
        except CandidateConflict:
            # same name, different bytes, no explicit authority. The
            # refusal is atomic in the database, so nothing was staged
            # and nothing reached the disk.
            raise HTTPException(
                status_code=409,
                detail="ayni adla farkli icerik zaten kayitli; degistirmek "
                       "bilincliyse replace=true ver")
        except CandidateSuperseded:
            # a NEWER candidate was staged while these bytes were being
            # written: the disk moved but this candidate was not
            # published. Answering 200 here would report a publication
            # the database declined.
            raise HTTPException(
                status_code=409,
                detail="yayin sirasinda daha yeni bir aday evrelendi; bu "
                       "yukleme yayimlanmadi")
        except publication.UnsafeCanonicalName:
            _log_safe_failure(ValueError("kanonik_ad"), "kanonik_ad_guvensiz")
            raise HTTPException(
                status_code=500,
                detail="kanonik ad tutarsizligi; dosya yazilmadi")
    return {
        "document_id": document_id,
        "filename": canonical,
        "candidate_id": candidate_id,
        "status": "pending",
    }


def _release_attempt(attempt, note):
    """Make sure the lease THIS REQUEST took does not outlive the request.

    The endpoint takes the attempt before anything is parsed, so a run
    that ends without recording its own verdict -- it raised on the way,
    or came back with nothing terminal -- leaves the lease held. The HTTP
    side was already fail-closed; the LIFECYCLE was not, and a retry then
    had to wait out the whole lease window for a run that was long over.

    Idempotent by construction, not by a flag: a run that did record its
    verdict cleared the lease in the same statement, so this second
    closure is refused as a lost lease and absorbed. Only an attempt that
    really ended still holding its lease is closed here.

    A closure that FAILS is not swallowed into silence -- it gets its own
    log event with the exception type. The request is answered 500
    either way, but "we could not close the attempt" is a second problem
    and it must be findable as one."""
    try:
        ingest.abandon_attempt(attempt, note)
    except Exception as closure_error:
        _log_safe_failure(closure_error, "deneme_kapatilamadi")


def _reported_outcome(returned):
    """The run's own terminal verdict, or None if it did not give one.

    Fail-closed by shape: only ``done`` and ``partial`` are outcomes a
    completed run may report. Anything else -- None, a bare string, an
    unexpected tuple -- means we do not know how the run ended, and not
    knowing is never reported to a client as success."""
    if (isinstance(returned, tuple) and len(returned) == 2
            and returned[0] in (AttemptOutcome.DONE, AttemptOutcome.PARTIAL)):
        return returned
    return None


@app.post("/documents/{document_id}/process", dependencies=AUTH)
def process_document(document_id: str):
    # Three SHORT borrows instead of one connection held across the whole
    # request: ingest can run for minutes on its own connection, and a pooled
    # connection parked here for that long would starve every other request.
    with db_conn() as conn:
        doc = db.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    if doc.get("archived_at") is not None:
        raise HTTPException(status_code=409, detail="document is archived")
    # A row with no recorded candidate has nothing for an ingest to bind
    # to: processing it would be exactly the unbound run the P0 exploited.
    if not doc.get("candidate_id") or not doc.get("content_sha256"):
        raise HTTPException(
            status_code=409,
            detail="belgenin kayitli adayi yok; once upload ile aday "
                   "kaydedilmeli")

    # NEITHER OF THE TWO CHECKS BELOW MARKS THE DOCUMENT. Both used to,
    # and both were saying the wrong thing about the wrong subject: a
    # source file that has gone missing, or a stored name that is not a
    # safe basename, tells you nothing about the generation currently
    # being SERVED. That generation's chunks are in the index and still
    # answering questions -- the file is only needed to build the NEXT
    # one. A probe made the mismatch plain: HTTP 404, active_generation
    # 4 -> 4, and `status` done -> error. A healthy index wearing a
    # failure label. These are failures of the REQUEST and of source
    # STORAGE; if they ever need to be visible on the row, they need a
    # column of their own, not this one.
    try:
        filename = _safe_upload_filename(doc["filename"])
    except HTTPException:
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
        raise HTTPException(status_code=404, detail="uploaded file missing")

    # THE LEASE IS TAKEN HERE, before anything is parsed. Two refusals
    # have to be answers rather than failures halfway through an ingest:
    # a candidate that is still STAGED (the upload committed its row but
    # has not finished writing the bytes) and a document another worker
    # is already indexing. Both are 409, both leave every document and
    # attempt column exactly as they were -- the audited version fell
    # into the publish gap, refused correctly deep inside the ingest, and
    # stamped the document `error` while the upload returned 200 pending.
    with db_conn() as conn:
        try:
            attempt = db.begin_attempt(conn, document_id)
        except CandidateNotPublished:
            raise HTTPException(
                status_code=409,
                detail="belgenin adayi henuz yayimlanmadi; yukleme bitince "
                       "tekrar deneyin")
        except AttemptAlreadyRunning:
            raise HTTPException(
                status_code=409,
                detail="bu belge icin calisan bir islem var; bitmesini "
                       "bekleyin")
        except db.DocumentLifecycleConflict:
            # Rechecked while taking the row lock: archive may have committed
            # after the read above and before begin_attempt started.
            raise HTTPException(
                status_code=409,
                detail="document is archived")

    # THE DOCUMENT ROW IS NOT THIS REQUEST'S SCRATCHPAD. It used to be
    # stamped `processing` here and `error` in the handler below, and the
    # result was then read back off it -- three mistakes with one root.
    # `documents.status` describes the SERVED version; a run's own
    # verdict belongs to its attempt (rule 5). A real PARTIAL run leaves
    # the row alone by design, so reading it back showed `processing`,
    # which this endpoint called "never finished": 500, and a healthy
    # served generation relabelled `error`. The run now REPORTS its
    # verdict, and nothing here writes the served status at all --
    # promotion is the only thing that moves it.
    try:
        # bound to the attempt, not to a tuple the endpoint read: the
        # candidate id, its bytes and the observed generation all travel
        # inside the one object the lease was minted with
        verdict = _reported_outcome(ingest.main(str(path), attempt=attempt))
    except Exception as e:
        _release_attempt(attempt, type(e).__name__)
        _log_safe_failure(e, "belge_isleme_hatasi")
        raise HTTPException(
            status_code=500,
            detail=DOCUMENT_PROCESSING_FAILURE_MESSAGE,
        )

    if verdict is None:
        # the run came back without a terminal verdict: not a partial and
        # not a failure, just an answer nobody can act on. It is reported
        # as a failure of THE REQUEST, and the served version -- which
        # this run never touched -- keeps saying what it said.
        _release_attempt(attempt, "IncompleteIngest")
        log.error(json.dumps({
            "olay": "belge_tamamlanmadan_dondu",
            "hata": "IncompleteIngest",
        }))
        raise HTTPException(
            status_code=500,
            detail=DOCUMENT_PROCESSING_FAILURE_MESSAGE,
        )

    # "partial" is a TRUE statement, not a failure to hide: some pages were
    # lost and the stored chunks are real. An earlier version rewrote it as
    # "error" and answered 500 -- destroying exactly the honesty the partial
    # status was built to carry.
    status, note = verdict
    response = {"document_id": document_id, "status": status}
    if note:
        response["status_note"] = note
    return response


# What an INVENTORY may say about a document. Deliberately not "the row
# minus a blocklist": a column added to `documents` later joins this list
# only when someone writes its name here, so the next candidate-shaped
# secret cannot arrive in a listing by default. `content_sha256` and
# `candidate_id` are absent for that reason -- they describe the recorded
# candidate's bytes and its immutable identity, which is single-document
# detail, not something to hand out one page at a time.
DOCUMENT_LIST_FIELDS = (
    "document_id",
    "filename",
    "file_type",
    "uploaded_at",
    "status",
    "status_note",
    "active_generation",
    "archived_at",
)

# A page nobody sized is a full table scan waiting for its first large
# corpus; a cap that only the database enforces is a scan that already
# started. Both bounds are decided here, in the signature, so an
# out-of-range page never reaches a connection at all.
DOCUMENT_PAGE_DEFAULT = 20
DOCUMENT_PAGE_MAX = 100

# The inventory filters are OPEN text by design: `documents.status` has no
# CHECK constraint (the closed done/error/partial/superseded set belongs to
# the ATTEMPTS table) and `file_type` is whatever suffix an upload carried.
# BOTH COLUMNS ARE UNBOUNDED `text`, so a length cap declared here would be
# a policy this layer invented: it would refuse a value the database itself
# stores and would diverge from the db seam, which enforces no such cap.
# The only shape asked for is therefore the one the filter needs to mean
# anything -- present or absent, and non-empty when present -- refused with
# 422 before a connection is ever borrowed. The value itself is a
# parameterized exact-equality filter, and an unknown one simply matches
# nothing.

# THE DATE WINDOW IS THE OPPOSITE CASE: `documents.uploaded_at` is
# `timestamptz NOT NULL DEFAULT now()` -- the database writes it, this code
# never does -- so an instant on that column is only comparable if the
# caller says WHICH instant they mean. A value with no offset and no `Z`
# does not; it is a wall-clock reading whose meaning depends on a timezone
# nobody sent. `AwareDatetime` is therefore declared on the parameters, in
# the same style as the page bounds above, and it refuses a naive value, a
# date-only value and malformed text with 422 before the body runs. A bare
# `datetime` annotation would NOT: it accepts a naive value and hands the
# body a `tzinfo` of None, which is how a wall-clock reading ends up being
# compared against absolute instants.
#
# ONE RULE CANNOT LIVE IN A PARAMETER DECLARATION: `after < before` is a
# statement about BOTH values, and a declaration only ever sees its own. It
# is checked as the first thing in the body -- above `db_conn()` -- so an
# empty or reversed window still costs no pooled connection and no scan.
# That refusal is an `HTTPException(422)` and therefore carries a text
# `detail`, where a parameter-declared refusal carries a list of
# `loc`/`type` error objects. The two 422 shapes differ on purpose: they
# come from different gates.


def _document_summary(row):
    """Project one listing row onto the published field set.

    The query already selects only these columns, so this is the SECOND
    guard rather than the only one -- and it is the guard that does not
    depend on remembering to keep a SELECT list narrow. Missing keys
    become None instead of raising: an inventory that fails outright
    because one legacy row lacks a note tells the caller nothing.
    """
    return {field: row.get(field) for field in DOCUMENT_LIST_FIELDS}


@app.get("/documents", dependencies=AUTH)
def list_documents(
    limit: int = Query(DOCUMENT_PAGE_DEFAULT, ge=1, le=DOCUMENT_PAGE_MAX),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, min_length=1),
    file_type: str | None = Query(None, min_length=1),
    uploaded_after: AwareDatetime | None = Query(None),
    uploaded_before: AwareDatetime | None = Query(None),
    q: str | None = Query(None, min_length=1),
    archived: bool = Query(False),
    collection_id: UUID | None = Query(None),
    tag: str | None = Query(None, min_length=1),
):
    """One page of the document inventory, newest first.

    The default is the active inventory. `archived=true` switches to the
    archived inventory; the two sets never mix in one page. Archive state is
    applied with every other filter before pagination and is published as
    `archived_at` so a lifecycle result remains observable.

    `status` and `file_type` narrow the inventory by exact equality --
    each may stand alone, together they AND -- and they narrow it BEFORE
    pagination, so `offset`, the page and `has_more` all describe the
    filtered sequence.

    `uploaded_after` and `uploaded_before` narrow the same sequence to a
    window on `uploaded_at`. Both bounds are EXCLUSIVE: a row sitting
    exactly on a bound is outside the window, so the two halves of a
    split never both claim it. Each may stand alone, together they AND,
    and they AND with `status` and `file_type` as well -- all four are
    applied before pagination like the two above.

    Each bound must carry an offset (`Z`, `+03:00`, `-05:00` are all
    fine), and two spellings of the same absolute instant are the same
    bound: the comparison here and in the database is between instants,
    never between the texts that were typed.

    `q` searches ONE column -- `filename` -- case-insensitively, for a
    LITERAL substring. It ANDs with the four filters above and is
    applied before pagination just as they are, so `offset`, the page
    and `has_more` describe the searched sequence. Literal means what it
    says: `%` and `_` are LIKE's metacharacters, not the caller's
    wildcards, so a search for `%` finds the names that really carry one
    rather than every document. The escaping that makes that true lives
    at the query seam, next to the clause that names the escape
    character; this layer forwards the value and invents nothing.

    ONLY THE SHAPE IS DECLARED HERE, and only the one shape the search
    needs to mean anything -- present and non-empty, or absent -- which
    FastAPI refuses with 422 before this body, and therefore before any
    checkout or statement, runs. No length cap: `filename` is unbounded
    `text`, so a limit declared here would refuse a name the database
    stores. And `_safe_upload_filename` is NOT reused as a validator: it
    is an UPLOAD gate, and it rejects slashes, colons, control
    characters and trailing spaces -- all of them legitimate things to
    search FOR, so reusing it would narrow the search silently instead
    of protecting anything.

    `has_more` comes from the query itself: the database is asked for
    ``limit + 1`` rows and the extra one, if it exists, is the evidence
    that another page follows. It is never published -- the page is
    truncated back to `limit` -- and no COUNT over the whole table is
    run, so the flag cannot disagree with the page it was computed with.

    The bounds are declared on the parameters, which means FastAPI
    refuses a bad page with 422 BEFORE this body runs: `db_conn()` is
    below the validation, so a limit of 0, 101, a negative offset, a
    malformed filter or a naive timestamp costs no pooled connection and
    no scan.
    """
    # The one rule no declaration can carry, checked where it still costs
    # nothing: an empty window (the bounds are exclusive, so equal bounds
    # can never match a row) and a reversed one are refused above the
    # checkout, not answered with an empty page the caller has to explain.
    if (uploaded_after is not None and uploaded_before is not None
            and uploaded_after >= uploaded_before):
        raise HTTPException(
            status_code=422,
            detail="uploaded_after, uploaded_before'dan kesin olarak once "
                   "olmali")
    try:
        with db_conn() as conn:
            filters = {"status": status, "file_type": file_type,
                       "uploaded_after": uploaded_after,
                       "uploaded_before": uploaded_before,
                       "q": q, "archived": archived}
            if collection_id is not None:
                filters["collection_id"] = str(collection_id)
            if tag is not None:
                filters["tag"] = tag
            rows = db.list_documents(conn, limit=limit, offset=offset, **filters)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    return {
        "documents": [_document_summary(row) for row in rows[:limit]],
        "limit": limit,
        "offset": offset,
        "has_more": len(rows) > limit,
    }


@app.post("/collections", dependencies=AUTH)
def create_collection(request: CollectionRequest):
    try:
        with db_conn() as conn:
            return db.create_collection(conn, request.name)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.get("/collections", dependencies=AUTH)
def list_collections():
    with db_conn() as conn:
        return {"collections": db.list_collections(conn)}


@app.get("/tags", dependencies=AUTH)
def list_tags():
    with db_conn() as conn:
        return {"tags": db.list_tags(conn)}


@app.delete("/tags/{tag_id}", dependencies=AUTH)
def delete_tag(tag_id: UUID):
    with db_conn() as conn:
        deleted = db.delete_tag(conn, str(tag_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="tag not found")
    return Response(status_code=204)


@app.delete("/collections/{collection_id}", dependencies=AUTH)
def delete_collection(collection_id: UUID):
    with db_conn() as conn:
        deleted = db.delete_collection(conn, str(collection_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="collection not found")
    return Response(status_code=204)


def _set_collection_membership(collection_id: UUID, document_id: UUID,
                               present: bool):
    with db_conn() as conn:
        result = db.set_collection_document(
            conn, str(collection_id), str(document_id), present)
    if result is None:
        raise HTTPException(status_code=404,
                            detail="collection or document not found")
    return {"collection_id": str(collection_id),
            "document_id": str(document_id), "present": result}


@app.put("/collections/{collection_id}/documents/{document_id}",
         dependencies=AUTH)
def add_collection_document(collection_id: UUID, document_id: UUID):
    return _set_collection_membership(collection_id, document_id, True)


@app.delete("/collections/{collection_id}/documents/{document_id}",
            dependencies=AUTH)
def remove_collection_document(collection_id: UUID, document_id: UUID):
    return _set_collection_membership(collection_id, document_id, False)


@app.put("/documents/{document_id}/tags", dependencies=AUTH)
def replace_document_tags(document_id: UUID, request: DocumentTagsRequest):
    try:
        with db_conn() as conn:
            result = db.replace_document_tags(
                conn, str(document_id), request.tags)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    if result is None:
        raise HTTPException(status_code=404, detail="document not found")
    return result


def _set_document_lifecycle(document_id: str, archived: bool):
    try:
        with db_conn() as conn:
            result = db.set_document_archived(conn, document_id, archived)
    except db.DocumentLifecycleConflict:
        raise HTTPException(
            status_code=409,
            detail="document has an active ingest attempt") from None
    if result is None:
        raise HTTPException(status_code=404, detail="document not found")
    return result


@app.post("/documents/{document_id}/archive", dependencies=AUTH)
def archive_document(document_id: str):
    return _set_document_lifecycle(document_id, True)


@app.post("/documents/{document_id}/restore", dependencies=AUTH)
def restore_document(document_id: str):
    return _set_document_lifecycle(document_id, False)


@app.get("/documents/{document_id}", dependencies=AUTH)
def read_document(document_id: str):
    with db_conn() as conn:
        doc = db.get_document(conn, document_id)
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
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")

    def check_embed():
        base = embeddings.EMBED_API_URL.rsplit("/v1/", 1)[0]
        requests.get(f"{base}/v1/models", timeout=3).raise_for_status()

    checks = dict([_probe("veritabani", check_db), _probe("embedding", check_embed)])
    healthy = all(checks.values())
    response.status_code = 200 if healthy else 503
    return {"status": "ready" if healthy else "degraded", "kontroller": checks}
