"""Local, closed-profile speech-to-text for OpenWebUI dictation.

    python -m services.speech_service

One process, one worker, one model: faster-whisper ``large-v3`` on CUDA in
float16, transcribing Turkish with VAD filtering, loaded once for the life
of the process. It speaks the OpenAI transcription shape on
``POST /v1/audio/transcriptions`` so OpenWebUI's microphone button can use
it unchanged, and answers ``{"text": ...}`` and nothing else.

WHAT THIS PATH IS NOT. It is the temporary dictation path only. Nothing is
stored: the upload lives in a task-specific temporary file for the length
of one request and the transcript exists only in the response. There is
no PostgreSQL, no RAG index, no tenant or hierarchy decision, no external
model API, and no log line ever carries a filename, the audio or the
text. Persistent recordings, transcription jobs, revisions and indexing
are later packages and are not scaffolded here.

THE PROFILE IS CLOSED ON PURPOSE. A request naming another ``model`` or
``language`` is refused, and there is no turbo, int8 or CPU fallback: a
GPU that cannot hold the model leaves ``/readyz`` at 503 rather than
quietly serving a smaller one. ``production_model_factory`` carries
exactly the three arguments the decision names, and the tests pin them.

THE KEY IS A SERVICE IDENTITY. ``SPEECH_API_KEY`` identifies OpenWebUI to
this process and nothing more -- not a user, not a tenant -- which is why
this path makes no audit or ownership claim at all.

ORDER OF REFUSALS. Identity, readiness and the byte ceiling are checked
in a small ASGI gate before the multipart body is parsed; model/language
and the format allowlist before the upload is staged; duration by
decoding, bounded by the limit, before the model sees a byte. One
transcription runs at a time; a bounded number of callers may wait a
bounded time, and everyone else gets ``speech_busy`` immediately.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hmac
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("speech_service")

# --- the closed profile ------------------------------------------------------

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "tr"
VAD_FILTER = True

DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_PORT = 8012
MIN_API_KEY_CHARS = 32

TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
UPLOAD_PREFIX = "speech-"
UPLOAD_SUFFIX = ".upload"

# By extension AND declared type, both from the client; neither is ever
# used as a path. mp4 is what Safari's MediaRecorder names an AAC
# dictation, so it belongs with m4a.
ALLOWED_EXTENSIONS = frozenset(
    {"wav", "webm", "mp3", "mp4", "m4a", "ogg", "flac"})
ALLOWED_CONTENT_TYPES = frozenset({
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/webm", "video/webm",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/x-m4a", "audio/m4a",
    "audio/ogg",
    "audio/flac", "audio/x-flac",
})
# OpenWebUI v0.11.0 relays the recording through aiohttp without a part
# type, which arrives as octet-stream; the extension is the signal then.
UNTYPED_CONTENT_TYPES = frozenset({"", "application/octet-stream"})
# What OpenWebUI itself accepts from the browser before relaying it here;
# docker-compose.speech.yml carries the same literal.
OPENWEBUI_SUPPORTED_CONTENT_TYPES = (
    "audio/wav,audio/x-wav,audio/webm,audio/ogg,audio/mp4,audio/mpeg,"
    "audio/x-m4a,audio/flac")

ERROR_CONTRACT_VERSION = 1
ERROR_STATUS_BY_CODE = MappingProxyType({
    "invalid_request": 400,
    "invalid_audio": 400,
    "authentication_required": 401,
    "resource_not_found": 404,
    "method_not_allowed": 405,
    "payload_too_large": 413,
    "audio_too_long": 413,
    "unsupported_media_type": 415,
    "validation_failed": 422,
    "internal_error": 500,
    "service_unavailable": 503,
    "speech_busy": 503,
    "speech_timeout": 504,
})
ErrorCode = Literal[
    "invalid_request", "invalid_audio", "authentication_required",
    "resource_not_found", "method_not_allowed", "payload_too_large",
    "audio_too_long", "unsupported_media_type", "validation_failed",
    "internal_error", "service_unavailable", "speech_busy",
    "speech_timeout",
]
_CODE_BY_HTTP_STATUS = MappingProxyType({
    400: "invalid_request",
    401: "authentication_required",
    404: "resource_not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_failed",
    503: "service_unavailable",
})


@dataclasses.dataclass(frozen=True)
class Limits:
    """The first local dictation profile. These are deliberately plain
    constants pinned by tests, not tunables: the E5 quota policy is where
    they become per-tenant decisions."""
    max_upload_bytes: int = 25 * 1024 * 1024
    multipart_overhead_bytes: int = 64 * 1024
    max_audio_seconds: float = 120.0
    transcribe_timeout_seconds: float = 90.0
    queue_wait_seconds: float = 20.0
    max_waiting: int = 2
    upload_chunk_bytes: int = 64 * 1024


DEFAULT_LIMITS = Limits()


# --- refusals ---------------------------------------------------------------

class SpeechConfigError(Exception):
    """The process must not start; the message is closed static text."""


class SpeechError(Exception):
    """A closed, content-free refusal: ``code`` is the whole message."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class InvalidAudio(SpeechError):
    def __init__(self):
        super().__init__("invalid_audio")


class AudioTooLong(SpeechError):
    def __init__(self):
        super().__init__("audio_too_long")


class UploadTooLarge(SpeechError):
    def __init__(self):
        super().__init__("payload_too_large")


class SpeechBusy(SpeechError):
    def __init__(self):
        super().__init__("speech_busy")


class SpeechTimeout(SpeechError):
    def __init__(self):
        super().__init__("speech_timeout")


class ModelUnavailable(SpeechError):
    def __init__(self):
        super().__init__("service_unavailable")


# --- configuration ----------------------------------------------------------

@dataclasses.dataclass(frozen=True, repr=False)
class Settings:
    api_key: str
    host: str = DEFAULT_BIND_HOST
    port: int = DEFAULT_PORT

    def __repr__(self):
        # the key is the one thing a repr must never carry into a log
        return f"Settings(host={self.host!r}, port={self.port})"


def load_settings(environ=os.environ) -> Settings:
    key = environ.get("SPEECH_API_KEY", "")
    if len(key) < MIN_API_KEY_CHARS:
        raise SpeechConfigError(
            f"SPEECH_API_KEY is required and must be at least "
            f"{MIN_API_KEY_CHARS} characters")
    host = environ.get("SPEECH_BIND_HOST", "").strip() or DEFAULT_BIND_HOST
    try:
        port = int(environ.get("SPEECH_PORT", "") or DEFAULT_PORT)
    except ValueError:
        raise SpeechConfigError("SPEECH_PORT must be an integer") from None
    if not 1 <= port <= 65535:
        raise SpeechConfigError("SPEECH_PORT must be between 1 and 65535")
    return Settings(api_key=key, host=host, port=port)


def key_matches(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode("utf-8"),
                               expected.encode("utf-8"))


def _presented_key(authorization: str) -> str:
    scheme, _, credential = authorization.partition(" ")
    if scheme != "Bearer":
        return ""
    return credential.strip()


# --- the model seam ----------------------------------------------------------

def production_model_factory():
    """Exactly the three arguments of the closed decision, nothing else.

    The import is deliberately here rather than at module level: the unit
    suite never has faster-whisper installed, and importing it must never
    be a side effect of collecting tests. A pre-downloaded model is served
    through the standard ``HF_HUB_CACHE`` / ``HF_HUB_OFFLINE=1`` pair, which
    keeps the identity the literal ``large-v3`` with no second code path.
    """
    from faster_whisper import WhisperModel
    # the vendor logger narrates durations and VAD counts; numbers only,
    # but nothing below WARNING is needed from it
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def probe_audio_seconds(path: str, limit: float) -> float:
    """Duration of the staged file, measured before the model sees it.

    A container that states its duration is answered from the header. A
    streamed recording (a browser's WebM has no duration element) is
    DECODED, and the decode stops the moment the limit is crossed -- so a
    tiny compressed file claiming hours costs at most ``limit`` seconds
    of audio to refuse. Anything that does not decode as audio is
    ``InvalidAudio``; vendor messages are dropped, never re-raised.
    """
    import av
    # ffmpeg's own diagnostics can carry the path they were about; they
    # are not this service's to publish
    av.logging.set_level(None)
    try:
        container = av.open(path, mode="r")
    except Exception:
        raise InvalidAudio() from None
    try:
        streams = [stream for stream in container.streams
                   if stream.type == "audio"]
        if not streams:
            raise InvalidAudio()
        duration = container.duration          # AV_TIME_BASE units, or None
        if duration is not None and duration > 0:
            seconds = duration / 1_000_000
            if seconds > limit:
                raise AudioTooLong()
            return seconds
        seconds = 0.0
        try:
            for frame in container.decode(streams[0]):
                seconds += frame.samples / frame.sample_rate
                if seconds > limit:
                    raise AudioTooLong()
        except SpeechError:
            raise
        except Exception:
            raise InvalidAudio() from None
        return seconds
    finally:
        container.close()


def _remove_exact(path: str) -> None:
    """The one path we minted, by that exact path. Never a glob."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("speech.tempfile_not_removed error_type=%s",
                    type(exc).__name__)


class SpeechEngine:
    """One model, one worker thread, one slot.

    The slot and the staged file are owned by whichever side is furthest
    along: the request until the job is submitted, the worker thread from
    then on. A request that times out or is cancelled after submission
    therefore changes nothing about the GPU -- the worker still finishes,
    removes the file and hands the slot back -- and a request that leaves
    before submission cleans up itself.
    """

    def __init__(self, model_factory, probe, limits: Limits):
        self._factory = model_factory
        self._probe = probe
        self._limits = limits
        self._model = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="speech-gpu")
        self._slot = asyncio.Semaphore(1)
        self._waiting = 0

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def waiting(self) -> int:
        return self._waiting

    def load(self) -> bool:
        started = time.monotonic()
        try:
            self._model = self._factory()
        except Exception as exc:
            self._model = None
            log.error("speech.model_load_failed error_type=%s",
                      type(exc).__name__)
            return False
        log.info("speech.model_loaded elapsed_ms=%d",
                 int((time.monotonic() - started) * 1000))
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._model = None

    async def transcribe(self, path: str) -> str:
        model = self._model
        submitted = False
        try:
            if model is None:
                raise ModelUnavailable()
            if self._waiting >= self._limits.max_waiting:
                raise SpeechBusy()
            self._waiting += 1
            try:
                await asyncio.wait_for(
                    self._slot.acquire(),
                    timeout=self._limits.queue_wait_seconds)
            except asyncio.TimeoutError:
                raise SpeechBusy() from None
            finally:
                self._waiting -= 1
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                self._executor, self._run, model, path, loop)
            submitted = True
        finally:
            if not submitted:
                _remove_exact(path)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._limits.transcribe_timeout_seconds)
        except asyncio.TimeoutError:
            raise SpeechTimeout() from None

    def _run(self, model, path: str, loop) -> str:
        try:
            seconds = self._probe(path, self._limits.max_audio_seconds)
            started = time.monotonic()
            segments, _info = model.transcribe(
                path, language=LANGUAGE, vad_filter=VAD_FILTER)
            # drained completely and in order; joined as the model spoke,
            # with nothing added, translated or normalised
            text = "".join(segment.text for segment in segments).strip()
            log.info("speech.transcribed audio_seconds=%.1f elapsed_ms=%d",
                     seconds, int((time.monotonic() - started) * 1000))
            return text
        finally:
            _remove_exact(path)
            try:
                loop.call_soon_threadsafe(self._slot.release)
            except RuntimeError:            # loop already closed
                log.warning("speech.slot_release_after_loop_closed")


# --- wire shapes --------------------------------------------------------------

class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Closed):
    status: Literal["ok"]


class ReadyResponse(_Closed):
    status: Literal["ready"]
    model: Literal["large-v3"]
    device: Literal["cuda"]
    compute_type: Literal["float16"]
    language: Literal["tr"]


class TranscriptionResponse(_Closed):
    text: str


class ErrorEnvelope(_Closed):
    version: Literal[1]
    code: ErrorCode
    request_id: str


class ErrorResponse(_Closed):
    error: ErrorEnvelope


def _error_body(request_id: str, code: str) -> dict:
    return ErrorResponse(error=ErrorEnvelope(
        version=ERROR_CONTRACT_VERSION, code=code, request_id=request_id,
    )).model_dump(mode="json")


def _error_response(request: Request, code: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unavailable")
    return JSONResponse(status_code=ERROR_STATUS_BY_CODE[code],
                        content=_error_body(request_id, code))


def _responses(*codes: str) -> dict:
    return {ERROR_STATUS_BY_CODE[code]: {"model": ErrorResponse,
                                        "description": f"Closed error ({code})"}
            for code in codes}


# --- the gate: before the body is parsed ---------------------------------------

class Gate:
    """Pure ASGI. Stamps a request id on every response and, for the
    transcription route, refuses unauthenticated, not-ready and oversized
    requests before a byte of multipart is parsed. The byte ceiling is
    enforced on Content-Length when there is one and on the stream when
    there is not -- OpenWebUI streams, so there usually is not."""

    def __init__(self, app, settings: Settings, limits: Limits):
        self.app = app
        self.settings = settings
        self.limits = limits

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:8]
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        if scope["method"] == "POST" and scope["path"] == TRANSCRIPTION_PATH:
            headers = {name.decode("latin-1").lower(): value.decode("latin-1")
                       for name, value in scope.get("headers", [])}
            presented = _presented_key(headers.get("authorization", ""))
            if not presented or not key_matches(presented,
                                                self.settings.api_key):
                await self._refuse(send_with_id, request_id,
                                   "authentication_required")
                return
            engine = scope["app"].state.engine
            if not engine.ready:
                await self._refuse(send_with_id, request_id,
                                   "service_unavailable")
                return
            ceiling = (self.limits.max_upload_bytes
                       + self.limits.multipart_overhead_bytes)
            declared = headers.get("content-length")
            if declared is not None and (
                    not declared.isdigit() or int(declared) > ceiling):
                await self._refuse(send_with_id, request_id,
                                   "payload_too_large")
                return
            receive = self._bounded(receive, ceiling)
        await self.app(scope, receive, send_with_id)

    @staticmethod
    def _bounded(receive, ceiling: int):
        total = 0

        async def bounded():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > ceiling:
                    raise UploadTooLarge()
            return message
        return bounded

    @staticmethod
    async def _refuse(send, request_id: str, code: str):
        payload = json.dumps(_error_body(request_id, code)).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": ERROR_STATUS_BY_CODE[code],
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": payload})


# --- request handling ------------------------------------------------------------

def _validate_fields(model, language) -> None:
    if model is not None and model != MODEL_NAME:
        raise SpeechError("invalid_request")
    if language is not None and language != LANGUAGE:
        raise SpeechError("invalid_request")


def _validate_format(filename, content_type) -> None:
    name = filename or ""
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise SpeechError("unsupported_media_type")
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if declared in UNTYPED_CONTENT_TYPES:
        return
    if declared not in ALLOWED_CONTENT_TYPES:
        raise SpeechError("unsupported_media_type")


async def _stage(upload: UploadFile, limits: Limits, upload_dir: Path) -> str:
    """Bounded chunks into a file this service named; the client's name is
    never part of the path. A partial file is removed on every exit."""
    handle, path = tempfile.mkstemp(prefix=UPLOAD_PREFIX, suffix=UPLOAD_SUFFIX,
                                    dir=str(upload_dir))
    total = 0
    try:
        with os.fdopen(handle, "wb") as sink:
            while True:
                chunk = await upload.read(limits.upload_chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_upload_bytes:
                    raise UploadTooLarge()
                sink.write(chunk)
        if total == 0:
            raise InvalidAudio()
    except BaseException:
        _remove_exact(path)
        raise
    return path


def create_app(settings: Settings, *, model_factory=production_model_factory,
               probe=probe_audio_seconds, limits: Limits = DEFAULT_LIMITS,
               upload_dir=None) -> FastAPI:
    engine = SpeechEngine(model_factory, probe, limits)
    staging_dir = Path(upload_dir) if upload_dir else Path(tempfile.gettempdir())

    @asynccontextmanager
    async def lifespan(app):
        # synchronous on purpose: the process accepts connections only
        # after the one load attempt has an answer, and a failed attempt
        # keeps the process alive with /readyz at 503
        engine.load()
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title="RAGTest local speech service", version="1",
                  lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.engine = engine
    app.add_middleware(Gate, settings=settings, limits=limits)

    @app.exception_handler(SpeechError)
    async def speech_error(request, exc):
        return _error_response(request, exc.code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return _error_response(request, "validation_failed")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        # FastAPI wraps anything raised while it parses the body into a
        # generic 400 "from" the original. The gate's byte ceiling fires
        # exactly there, inside receive(), so its own code is recovered
        # from the cause rather than reported as a malformed request.
        if isinstance(exc.__cause__, SpeechError):
            return _error_response(request, exc.__cause__.code)
        return _error_response(
            request, _CODE_BY_HTTP_STATUS.get(exc.status_code, "internal_error"))

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        log.error("speech.unhandled error_type=%s", type(exc).__name__)
        return _error_response(request, "internal_error")

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse,
             responses=_responses("service_unavailable"))
    async def readyz():
        if not engine.ready:
            raise ModelUnavailable()
        return ReadyResponse(status="ready", model=MODEL_NAME, device=DEVICE,
                             compute_type=COMPUTE_TYPE, language=LANGUAGE)

    @app.post(TRANSCRIPTION_PATH, response_model=TranscriptionResponse,
              responses=_responses(
                  "invalid_request", "authentication_required",
                  "payload_too_large", "unsupported_media_type",
                  "validation_failed", "internal_error",
                  "service_unavailable", "speech_timeout"))
    async def transcriptions(request: Request,
                             file: UploadFile = File(...),
                             model: str | None = Form(None),
                             language: str | None = Form(None)):
        # The two Form parameters document the request shape. Validation
        # reads the parsed form itself (cached, not parsed twice) because
        # FastAPI folds an EMPTY form value into the default, and in this
        # closed contract "present but empty" is a wrong value, not none.
        form = await request.form()
        _validate_fields(form.get("model"), form.get("language"))
        _validate_format(file.filename, file.content_type)
        path = await _stage(file, limits, staging_dir)
        try:
            text = await engine.transcribe(path)
        except SpeechError:
            raise
        except Exception as exc:
            # converted HERE so the vendor's message never reaches the
            # server's own exception logging
            log.error("speech.transcription_failed error_type=%s",
                      type(exc).__name__)
            raise SpeechError("internal_error") from None
        return TranscriptionResponse(text=text)

    return app


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = load_settings(os.environ)
    except SpeechConfigError as exc:
        print(f"speech_service: config_error: {exc}", file=sys.stderr)
        return 2
    app = create_app(settings)
    import uvicorn
    # one process, one worker: the model is loaded exactly once and the
    # GPU is never asked to hold two copies
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
