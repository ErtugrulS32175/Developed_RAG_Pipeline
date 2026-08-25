import base64
import json
import hashlib
import hmac
import logging
import os
import secrets
import time
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Literal, Union
from uuid import UUID

from fastapi import (
    Depends, FastAPI, Header, HTTPException, Query, Request, Response,
    UploadFile, File,
)
from fastapi.responses import StreamingResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from pipeline.index import db
from pipeline.index import ingest
from pipeline.index import publication
from pipeline.storage import handle_transport
from pipeline.evaluation import datasets as eval_datasets
from pipeline.index.attempt_contract import (
    AttemptAlreadyRunning,
    AttemptOutcome,
    CandidateConflict,
    CandidateNotPublished,
    CandidateSuperseded,
)
from pipeline.api import owui_chat
from pipeline.api import auth
from pipeline.api import contracts as api_contracts
from pipeline.api import identity
from pipeline.api import metrics
from pipeline.api import org_policy
from pipeline.retrieval import planner, rag_backends
from pipeline.retrieval.trace import RetrievalTrace
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
API_KEY = os.getenv("API_KEY", "").strip()
API_KEYS_JSON = os.getenv("API_KEYS_JSON", "").strip()
OPENWEBUI_GATEWAY_KEY = os.getenv("OPENWEBUI_GATEWAY_KEY", "").strip()
OPENWEBUI_USER_JWT_SECRET = os.getenv(
    "OPENWEBUI_USER_JWT_SECRET", "").strip()
ALLOW_INSECURE_LOCAL = os.getenv("ALLOW_INSECURE_LOCAL", "").strip() == "1"
API_BIND_HOST = os.getenv("API_BIND_HOST", "").strip()
AUTH_REGISTRY = auth.load_registry(
    API_KEY,
    API_KEYS_JSON,
    external_auth=bool(OPENWEBUI_GATEWAY_KEY),
    allow_insecure_local=ALLOW_INSECURE_LOCAL,
    bind_host=API_BIND_HOST,
)
EVIDENCE_HMAC_SECRET = os.getenv("EVIDENCE_HMAC_SECRET", "").strip()
if (EVIDENCE_HMAC_SECRET
        and len(EVIDENCE_HMAC_SECRET.encode("utf-8")) < 32):
    raise identity.IdentityConfigurationError(
        "evidence HMAC anahtari en az 32 bayt olmali")
if OPENWEBUI_GATEWAY_KEY and not EVIDENCE_HMAC_SECRET:
    raise identity.IdentityConfigurationError(
        "kimlik dogrulamali kurulumda evidence HMAC anahtari gerekli")
if bool(OPENWEBUI_GATEWAY_KEY) != bool(OPENWEBUI_USER_JWT_SECRET):
    raise identity.IdentityConfigurationError(
        "OpenWebUI gateway key ve JWT anahtari birlikte tanimlanmali")
if OPENWEBUI_GATEWAY_KEY and len(OPENWEBUI_GATEWAY_KEY.encode("utf-8")) < 32:
    raise identity.IdentityConfigurationError(
        "OpenWebUI gateway anahtari en az 32 bayt olmali")
OPENWEBUI_IDENTITY = (
    identity.Verifier.configured(
        OPENWEBUI_USER_JWT_SECRET,
        max_lifetime_seconds=int(os.getenv(
            "OPENWEBUI_USER_JWT_MAX_SECONDS", "60")),
        clock_skew_seconds=int(os.getenv(
            "OPENWEBUI_USER_JWT_CLOCK_SKEW", "5")),
    ) if OPENWEBUI_GATEWAY_KEY else None
)
_GATEWAY_DIGEST = (
    hashlib.sha256(OPENWEBUI_GATEWAY_KEY.encode("utf-8")).digest()
    if OPENWEBUI_GATEWAY_KEY else None
)
# A configured deployment derives this domain-specific MAC key from stable
# secret material.  The fixed fallback exists only in the deliberately open
# local-development mode, where there is no remote actor or authorization
# boundary.  It keeps local fixtures deterministic without pretending to be a
# production secret.
_EVIDENCE_KEY_MATERIAL = (
    EVIDENCE_HMAC_SECRET or "ragtest-open-local-development-only"
).encode("utf-8")
_EVIDENCE_HMAC_KEY = hashlib.sha256(
    b"ragtest-evidence-reference-v1\x00" + _EVIDENCE_KEY_MATERIAL).digest()
_REVIEW_HMAC_KEY = hashlib.sha256(
    b"ragtest-review-reference-v1\x00" + _EVIDENCE_KEY_MATERIAL).digest()
EVIDENCE_TICKET_SECONDS = 50
EVIDENCE_PASSAGE_MAX_CHARS = 4000
EXPORT_TICKET_SECONDS = 50
EXPORT_RECORD_SECONDS = 3600
EXPORT_MAX_BYTES = 32 << 20
_EXPORT_HMAC_KEY = hashlib.sha256(
    b"ragtest-table-export-v1\x00" + _EVIDENCE_KEY_MATERIAL).digest()
if _GATEWAY_DIGEST is not None:
    if any(hmac.compare_digest(_GATEWAY_DIGEST, item.digest)
           for item in AUTH_REGISTRY.credentials):
        raise identity.IdentityConfigurationError(
            "OpenWebUI gateway anahtari API anahtariyla ayni olamaz")
    # Configuring the gateway is a production auth boundary: absence of a
    # legacy key must not reactivate the local-development open principal.
    AUTH_REGISTRY = auth.Registry(AUTH_REGISTRY.credentials, None)
_DOCS_OPEN = not (AUTH_REGISTRY.configured or OPENWEBUI_IDENTITY is not None)


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


def _gateway_offered(authorization):
    if _GATEWAY_DIGEST is None or type(authorization) is not str:
        return False
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(
        _GATEWAY_DIGEST, hashlib.sha256(token.encode("utf-8")).digest())


def _resolve_forwarded_principal(assertion):
    """Resolve a verified subject; display claims never reach this seam."""
    conn = db.get_conn(service=True)
    try:
        _require_current_schema(conn)
        resolved = db.resolve_org_identity(
            conn, assertion.issuer, assertion.subject)
    finally:
        conn.close()
    if resolved is None:
        return None
    return auth.Principal(
        tenant_id=resolved["tenant_id"],
        role=resolved["role"],
        subject_id=resolved["identity_id"],
        source="openwebui",
        position_id=resolved["position_id"],
        org_architect=resolved["org_architect"],
    )


def _request_principal(request):
    authorizations = request.headers.getlist("authorization")
    if not authorizations and AUTH_REGISTRY.open_principal is not None:
        # Local development deliberately keeps the historical open principal.
        # Configuring either legacy credentials or the OpenWebUI gateway sets
        # this field to None, so the production boundary cannot enter here.
        return AUTH_REGISTRY.open_principal
    if len(authorizations) != 1:
        return None
    authorization = authorizations[0]
    principal = auth.authenticate(AUTH_REGISTRY, authorization)
    if principal is not None:
        return principal
    if OPENWEBUI_IDENTITY is None or not _gateway_offered(authorization):
        return None
    forwarded = request.headers.getlist(identity.HEADER_NAME)
    plain = {
        name.decode("latin-1").casefold()
        for name, _value in request.headers.raw
        if name.decode("latin-1").casefold().startswith("x-openwebui-user-")
        and name.decode("latin-1").casefold() != identity.HEADER_NAME
    }
    if len(forwarded) != 1 or plain:
        return None
    try:
        assertion = OPENWEBUI_IDENTITY.verify(forwarded[0])
        return _resolve_forwarded_principal(assertion)
    except (identity.IdentityRefused, ValueError):
        return None


@app.middleware("http")
async def bind_request_principal(request, call_next):
    principal = _request_principal(request)
    token = auth.bind(principal)
    try:
        return await call_next(request)
    finally:
        auth.reset(token)


@app.middleware("http")
async def log_requests(request, call_next):
    """One structured line per request.

    Method, path, status and duration only -- deliberately NOT the body or the
    query string. Requests here carry questions and answers drawn from private
    documents, and a log file is just another place that content can leak to.
    The request id makes a single call traceable without recording what it said.
    """
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    started = time.perf_counter()
    status = 500

    def route_template():
        route = request.scope.get("route")
        value = getattr(route, "path", "unmatched")
        return (value if isinstance(value, str) and value.startswith("/")
                and "?" not in value and len(value) <= 200 else "unmatched")

    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
    except Exception as error:
        # An arbitrary exception message can contain a DSN, a local path or
        # document/model text. The shared helper keeps safe frame locations
        # without copying that untrusted detail into a second storage system.
        _log_safe_failure(
            error,
            "api_istek_hatasi",
            istek=request_id,
            yol=route_template(),
            yontem=request.method,
            durum="exception",
        )
        raise
    finally:
        route_path = route_template()
        metrics.observe(
            request.method, route_path, status,
            (time.perf_counter() - started) * 1000)
    log.info(json.dumps({
        "istek": request_id,
        "yontem": request.method,
        "yol": route_path,
        "durum": response.status_code,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }))
    return response


_schema_ready = False


def _require_current_schema(conn):
    """Prove readiness once; request traffic never receives DDL authority."""
    global _schema_ready
    if _schema_ready:
        return
    db.require_runtime_ready(conn)
    _schema_ready = True


@contextmanager
def _principal_db_conn(principal):
    """One pooled connection per request, connected on first use rather than at
    import so this module stays importable without a running database.

    The pool's context manager commits on clean exit and rolls back on
    exception, and checkout revalidates the connection -- so one failed
    statement can no longer poison every request that follows, which is
    exactly what the previous single cached module-level connection did."""
    with db.get_pool().connection() as conn:
        _require_current_schema(conn)
        db.set_tenant_context(
            conn, principal.tenant_id, actor_id=principal.subject_id)
        try:
            yield conn
        finally:
            conn.rollback()
            db.clear_tenant_context(conn)


@contextmanager
def db_conn():
    with _principal_db_conn(auth.current_principal()) as conn:
        yield conn


# Shared-secret or forwarded identity auth.  The only unauthenticated mode is
# an explicit local-development opt-in bound to a literal loopback address.
if not (AUTH_REGISTRY.configured or OPENWEBUI_IDENTITY is not None):
    log.warning("Kimliksiz yerel gelistirme modu acik; dis aga baglanamaz.")


def _require_role(minimum_role, authorization):
    principal = auth.bound_principal()
    if principal is None:
        principal = auth.authenticate(AUTH_REGISTRY, authorization)
    if principal is None:
        raise HTTPException(status_code=401,
                            detail="gecersiz veya eksik API anahtari")
    if not auth.permits(principal, minimum_role):
        raise HTTPException(status_code=403, detail="bu islem icin rol yetersiz")
    return principal


def require_api_key(authorization: str = Header(default="")):
    """Backward-compatible reader dependency for every data-bearing route."""
    return _require_role("reader", authorization)


def require_editor(authorization: str = Header(default="")):
    return _require_role("editor", authorization)


def require_admin(authorization: str = Header(default="")):
    return _require_role("admin", authorization)


AUTH = [Depends(require_api_key)]
EDITOR_AUTH = [Depends(require_editor)]
ADMIN_AUTH = [Depends(require_admin)]


def require_org_identity():
    """Require a database-resolved OpenWebUI subject, not a legacy API key."""
    principal = auth.bound_principal()
    if (principal is None or principal.source != "openwebui"
            or principal.subject_id is None):
        raise HTTPException(status_code=403,
                            detail="organizasyon kimligi gerekli")
    return principal


def require_org_architect(
        request: Request,
        principal=Depends(require_org_identity)):
    """Topology authority grants no document role by itself."""
    if not principal.org_architect:
        path = request.url.path
        if path == "/v1/org/admin/audit-events":
            action = "events_view"
        elif path.startswith("/v1/org/admin/members/"):
            action = "membership_change"
        elif path == "/v1/org/admin/retention-policy":
            action = "retention_policy_change"
        elif path == "/v1/org/admin/retention-documents":
            action = "retention_inventory_view"
        elif "/legal-holds" in path:
            action = "legal_hold_change"
        elif "/purge-jobs" in path:
            action = "purge_schedule"
        else:
            action = ("topology_change" if request.method == "PUT"
                      else "topology_read")
        with db_conn() as conn:
            db.record_org_decision(
                conn,
                actor_id=principal.subject_id,
                subject_id=None,
                action=action,
                reason_code="system_operation",
                allowed=False,
                request_id=request.state.request_id,
            )
        raise HTTPException(status_code=403,
                            detail="organizasyon mimari yetkisi gerekli")
    return principal


def require_evidence_actor(
        principal=Depends(require_org_identity)):
    """Require a real content role in addition to OpenWebUI identity.

    An organization architect can design the hierarchy without acquiring any
    document visibility.  A person who separately holds an active reader,
    editor or admin membership is rechecked by the database at ticket mint and
    consumption; this dependency only closes the route shape early.
    """
    if not auth.permits(principal, "reader"):
        raise HTTPException(status_code=403,
                            detail="kanit goruntuleme rolu gerekli")
    return principal


def require_eval_writer(
        principal=Depends(require_org_identity)):
    """Require a real editor; hierarchy authority is rechecked by PostgreSQL."""
    if (principal is None or principal.source != "openwebui"
            or principal.subject_id is None
            or not auth.permits(principal, "editor")):
        raise HTTPException(status_code=403,
                            detail="degerlendirme yazma rolu gerekli")
    return principal


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
    include_trace: bool = False


class CollectionRequest(BaseModel):
    name: str = Field(min_length=1)


class DocumentTagsRequest(BaseModel):
    tags: list[TagName] = Field(max_length=DOCUMENT_SCOPE_MAX)


class DocumentVersionActivationRequest(BaseModel):
    """The one caller-owned value in an activation: its observed revision."""

    expected_revision: int = Field(ge=0)


class EvidenceTicketRequest(BaseModel):
    evidence_ref: str = Field(min_length=43, max_length=43,
                              pattern=r"^[A-Za-z0-9_-]+$")


class EvidencePreviewRequest(BaseModel):
    ticket: str = Field(min_length=43, max_length=43,
                        pattern=r"^[A-Za-z0-9_-]+$")


class ExportTicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    export_ref: str = Field(min_length=43, max_length=43,
                            pattern=r"^[A-Za-z0-9_-]+$")


class ExportDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ticket: str = Field(min_length=43, max_length=43,
                        pattern=r"^[A-Za-z0-9_-]+$")


class ReviewFeedbackRequest(BaseModel):
    feedback_ref: str = Field(min_length=43, max_length=43,
                              pattern=r"^[A-Za-z0-9_-]+$")
    verdict: Literal["helpful", "not_helpful"]
    reason_code: Literal[
        "incorrect", "missing_evidence", "outdated", "unsafe", "other",
    ] | None = None


class ReviewDecisionRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    expected_policy_epoch: int = Field(ge=1)
    decision: Literal["resolved", "dismissed"]
    resolution_code: Literal["corrected", "no_issue", "escalated"]
    reason_code: Literal["management_duty", "security_review"]


class EvalDatasetCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    slug: str = Field(min_length=1, max_length=80,
                      pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    label: str = Field(min_length=1, max_length=160)


class EvalDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)


class EvalPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    expected_policy_epoch: int = Field(ge=1)
    expected_draft_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class EvalRetireRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    expected_policy_epoch: int = Field(ge=1)


class OrgPositionRequest(BaseModel):
    id: UUID
    parent_id: UUID | None = None
    title: str = Field(min_length=1, max_length=200)
    kind: Literal["root", "manager", "member"]
    can_monitor_descendants: bool = False
    protected_from_monitoring: bool = False


class OrgMemberRequest(BaseModel):
    issuer: Literal["open-webui"] = "open-webui"
    subject: str = Field(min_length=1, max_length=200,
                         pattern=r"^[\x20-\x7e]+$")
    position_id: UUID
    display_label: str = Field(min_length=1, max_length=200)
    app_role: Literal["reader", "editor", "admin"] = "reader"
    state: Literal["active", "pending", "suspended"] = "active"


class OrgTopologyRequest(BaseModel):
    expected_version: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=200)
    positions: list[OrgPositionRequest] = Field(min_length=1, max_length=500)
    members: list[OrgMemberRequest] = Field(default_factory=list,
                                             max_length=5000)


class OrgMembershipUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_architecture_version: int = Field(ge=1, strict=True)
    expected_policy_epoch: int = Field(ge=1, strict=True)
    state: Literal["active", "pending", "suspended"] | None = None
    app_role: Literal["reader", "editor", "admin"] | None = None
    position_id: UUID | None = None


class RetentionPolicyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    expected_policy_epoch: int = Field(ge=1, strict=True)
    archive_retention_days: int = Field(ge=1, le=3650, strict=True)


class LegalHoldCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_document_revision: int = Field(ge=0, strict=True)
    expected_policy_epoch: int = Field(ge=1, strict=True)
    reason_code: Literal[
        "litigation", "regulatory", "security_investigation",
    ]


class LegalHoldReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    expected_policy_epoch: int = Field(ge=1, strict=True)


class PurgeScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_document_revision: int = Field(ge=0, strict=True)
    expected_policy_epoch: int = Field(ge=1, strict=True)


@app.get("/v1/org/me", response_model=api_contracts.OrganizationMeResponse)
def organization_me(principal=Depends(require_org_identity)):
    """Show only the caller's own level and content-free organization facts."""
    with db_conn() as conn:
        membership = db.org_context(conn, principal.subject_id)
    return {
        "tenant_id": principal.tenant_id,
        "identity_id": principal.subject_id,
        "architecture_admin": principal.org_architect,
        "membership": membership,
    }


@app.get("/v1/org/visible-members",
         response_model=api_contracts.VisibleOrgMembersResponse)
def organization_visible_members(
        request: Request,
        reason_code: Literal["management_duty", "security_review"] = Query(),
        principal=Depends(require_org_identity)):
    """Return strict descendants only; peers, ancestors and protected users stay out."""
    with db_conn() as conn:
        members = db.visible_org_members(conn, principal.subject_id)
        db.record_org_decision(
            conn, actor_id=principal.subject_id, subject_id=None,
            action="monitor_view", reason_code=reason_code, allowed=True,
            request_id=request.state.request_id)
    return {"members": members}


@app.get("/v1/org/admin/topology",
         response_model=api_contracts.OrgTopologyResponse)
def organization_topology(
        request: Request,
        principal=Depends(require_org_architect)):
    with db_conn() as conn:
        topology = db.org_topology(conn)
        if topology is None:
            raise HTTPException(status_code=404,
                                detail="organizasyon bulunamadi")
        db.record_org_decision(
            conn, actor_id=principal.subject_id, subject_id=None,
            action="topology_read", reason_code="system_operation",
            allowed=True, request_id=request.state.request_id)
    return topology


@app.get("/v1/org/admin/audit-events",
         response_model=api_contracts.OrgAuditEventListResponse)
def organization_audit_events(
        request: Request,
        limit: int = Query(20, ge=1, le=100),
        before_created_at: AwareDatetime | None = Query(default=None),
        before_id: UUID | None = Query(default=None),
        action: list[Literal[
            "monitor_view", "topology_read", "topology_change",
            "access_preview", "review_queue_view", "review_decision",
            "events_view", "membership_change", "retention_policy_change",
            "legal_hold_change", "retention_inventory_view",
            "purge_schedule", "purge_execute"
        ]] | None = Query(default=None),
        decision: list[Literal["allowed", "denied"]] | None = Query(
            default=None),
        reason_code: list[Literal["management_duty", "security_review",
                                 "system_operation", "policy_preview"]]
                      | None = Query(default=None),
        principal=Depends(require_org_architect)):
    """List immutable governance decisions with closed query filters."""
    if (before_created_at is None) != (before_id is None):
        raise HTTPException(status_code=422,
                            detail="governance imlec parametreleri eksik")
    before = (before_created_at, before_id) if before_id is not None else None
    try:
        with db_conn() as conn:
            rows = db.list_org_audit_events(
                conn, limit=limit, before=before, actions=action,
                decisions=decision, reasons=reason_code)
            db.record_org_decision(
                conn, actor_id=principal.subject_id, subject_id=None,
                action="events_view", reason_code="system_operation",
                allowed=True, request_id=request.state.request_id)
    except db.OrgAuditQueryRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    page = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page:
        next_cursor = {
            "before_created_at": page[-1]["created_at"],
            "before_id": str(page[-1]["id"]),
        }
    return {
        "events": [_org_audit_event_summary(row) for row in page],
        "limit": limit,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


@app.put("/v1/org/admin/topology",
         response_model=api_contracts.OrgVersionResponse)
def replace_organization_topology(
        body: OrgTopologyRequest, request: Request,
        principal=Depends(require_org_architect)):
    positions = [item.model_dump() for item in body.positions]
    members = [item.model_dump() for item in body.members]
    try:
        ordered, members = org_policy.ordered_topology(positions, members)
    except org_policy.TopologyRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    try:
        with db_conn() as conn:
            version = db.replace_org_topology(
                conn, expected_version=body.expected_version, name=body.name,
                positions=ordered, members=members,
                actor_id=principal.subject_id,
                request_id=request.state.request_id)
    except db.OrgVersionConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.OrgIdentityConflict as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    return version


@app.put("/v1/org/admin/members/{identity_id}",
         response_model=api_contracts.OrgMembershipResponse)
def update_organization_membership(
        identity_id: UUID, body: OrgMembershipUpdateRequest, request: Request,
        principal=Depends(require_org_architect)):
    if (body.state is None and body.app_role is None
            and body.position_id is None):
        raise HTTPException(status_code=422,
                            detail="uyelik degisikligi belirtilmedi")
    try:
        with db_conn() as conn:
            updated = db.update_org_member(
                conn, actor_id=principal.subject_id,
                target_identity_id=identity_id,
                expected_architecture_version=body.expected_architecture_version,
                expected_policy_epoch=body.expected_policy_epoch,
                state=body.state, app_role=body.app_role,
                position_id=body.position_id,
                request_id=request.state.request_id)
    except db.OrgVersionConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.OrgPolicyConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except (db.OrgIdentityConflict,
            db.OrgMembershipStateRefused) as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    return {
        "identity_id": str(updated["identity_id"]),
        "display_label": updated["display_label"],
        "app_role": updated["app_role"],
        "state": updated["state"],
        "position_id": (None if updated["position_id"] is None
                        else str(updated["position_id"])),
        "architecture_version": updated["architecture_version"],
        "policy_epoch": updated["policy_epoch"],
    }


@app.get("/v1/org/admin/retention-policy",
         response_model=api_contracts.RetentionPolicyResponse)
def organization_retention_policy(
        principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return db.get_tenant_retention_policy(
                conn, actor_id=principal.subject_id)
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=403, detail=str(error)) from None


@app.get("/v1/org/admin/retention-documents",
         response_model=api_contracts.RetentionDocumentListResponse)
def organization_retention_documents(
        request: Request,
        limit: int = Query(50, ge=1, le=100),
        before_uploaded_at: AwareDatetime | None = Query(default=None),
        before_id: UUID | None = Query(default=None),
        principal=Depends(require_org_architect)):
    """List content-free document lifecycle facts for the admin portal."""
    if (before_uploaded_at is None) != (before_id is None):
        raise HTTPException(status_code=422,
                            detail="retention belge imleci eksik")
    before = ((before_uploaded_at, before_id)
              if before_id is not None else None)
    try:
        with db_conn() as conn:
            rows = db.list_retention_documents(
                conn, actor_id=principal.subject_id, limit=limit,
                before=before)
            db.record_org_decision(
                conn, actor_id=principal.subject_id, subject_id=None,
                action="retention_inventory_view",
                reason_code="system_operation", allowed=True,
                request_id=request.state.request_id)
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    page = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = None
    if has_more and page:
        next_cursor = {
            "before_uploaded_at": page[-1]["uploaded_at"],
            "before_id": str(page[-1]["document_id"]),
        }
    documents = []
    for row in page:
        documents.append({
            "document_id": str(row["document_id"]),
            "status": row["status"],
            "revision": int(row["revision"]),
            "uploaded_at": row["uploaded_at"],
            "archived_at": row["archived_at"],
            "purged_at": row["purged_at"],
            "active_hold_count": int(row["active_hold_count"]),
            "latest_purge_job_id": (
                None if row["latest_purge_job_id"] is None
                else str(row["latest_purge_job_id"])),
            "latest_purge_state": row["latest_purge_state"],
        })
    return {
        "documents": documents,
        "limit": limit,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


@app.put("/v1/org/admin/retention-policy",
         response_model=api_contracts.RetentionPolicyResponse)
def update_organization_retention_policy(
        body: RetentionPolicyUpdateRequest, request: Request,
        principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return db.update_tenant_retention_policy(
                conn, actor_id=principal.subject_id,
                archive_retention_days=body.archive_retention_days,
                expected_revision=body.expected_revision,
                expected_policy_epoch=body.expected_policy_epoch,
                request_id=request.state.request_id)
    except (db.RetentionPolicyConflict,
            db.OrgPolicyConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.get("/documents/{document_id}/legal-holds",
         response_model=api_contracts.LegalHoldListResponse)
def document_legal_holds(
        document_id: UUID, principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return {"holds": db.list_document_legal_holds(
                conn, actor_id=principal.subject_id,
                document_id=document_id)}
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=403, detail=str(error)) from None


@app.post("/documents/{document_id}/legal-holds", status_code=201,
          response_model=api_contracts.LegalHoldResponse,
          response_model_exclude_unset=True)
def create_document_legal_hold(
        document_id: UUID, body: LegalHoldCreateRequest, request: Request,
        principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return db.create_document_legal_hold(
                conn, actor_id=principal.subject_id,
                document_id=document_id,
                reason_code=body.reason_code,
                expected_revision=body.expected_document_revision,
                expected_policy_epoch=body.expected_policy_epoch,
                request_id=request.state.request_id)
    except (db.DocumentVersionConflict,
            db.OrgPolicyConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.post("/documents/{document_id}/legal-holds/{hold_id}/release",
          response_model=api_contracts.LegalHoldResponse,
          response_model_exclude_unset=True)
def release_document_legal_hold(
        document_id: UUID, hold_id: UUID, body: LegalHoldReleaseRequest,
        request: Request, principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return db.release_document_legal_hold(
                conn, actor_id=principal.subject_id,
                document_id=document_id, hold_id=hold_id,
                expected_revision=body.expected_revision,
                expected_policy_epoch=body.expected_policy_epoch,
                request_id=request.state.request_id)
    except (db.DocumentVersionConflict,
            db.OrgPolicyConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.get("/documents/{document_id}/purge-jobs",
         response_model=api_contracts.PurgeJobListResponse,
         response_model_exclude_unset=True)
def document_purge_jobs(
        document_id: UUID, principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return {"jobs": db.list_document_purge_jobs(
                conn, actor_id=principal.subject_id,
                document_id=document_id)}
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=403, detail=str(error)) from None


@app.post("/documents/{document_id}/purge-jobs", status_code=202,
          response_model=api_contracts.PurgeJobResponse,
          response_model_exclude_unset=True)
def schedule_document_purge(
        document_id: UUID, body: PurgeScheduleRequest, request: Request,
        principal=Depends(require_org_architect)):
    try:
        with db_conn() as conn:
            return db.schedule_document_purge(
                conn, actor_id=principal.subject_id,
                document_id=document_id,
                expected_revision=body.expected_document_revision,
                expected_policy_epoch=body.expected_policy_epoch,
                request_id=request.state.request_id)
    except (db.DocumentVersionConflict,
            db.OrgPolicyConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.DocumentRetentionRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


def _eval_version_metadata(row, *, version_id=None):
    item = {
        "version_id": row.get("id", version_id),
        "version_number": row["version_number"],
        "state": row["state"],
        "revision": row["revision"],
        "case_count": row["case_count"],
    }
    digest = row.get("content_sha256")
    if digest is not None:
        item["content_sha256"] = digest
    sealed_at = row.get("sealed_at")
    if sealed_at is not None:
        item["sealed_at"] = sealed_at
    return item


def _eval_dataset_metadata(row):
    item = {
        "dataset_id": row["id"],
        "slug": row["slug"],
        "label": row["label"],
        "state": row["state"],
        "revision": row["revision"],
    }
    for name in ("owner_label", "policy_epoch", "current_version_id",
                 "current_version_number"):
        if name in row:
            item[name] = row[name]
    latest_id = row.get("latest_version_id")
    item["versions"] = ([] if latest_id is None else [{
        "version_id": latest_id,
        "version_number": row["latest_version_number"],
        "state": row["latest_version_state"],
        "revision": row["latest_version_revision"],
        "case_count": row["latest_case_count"],
    }])
    return item


def _eval_access_error():
    return HTTPException(status_code=404,
                         detail="degerlendirme seti bulunamadi")


def _eval_state_error():
    return HTTPException(status_code=409,
                         detail="degerlendirme gecisi reddedildi")


def _closed_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate")
        result[key] = value
    return result


def _reject_json_constant(_value):
    raise ValueError("constant")


async def _eval_import_body(request):
    """Bound and decode an import before FastAPI's ordinary JSON parser."""
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > eval_datasets.MAX_JSON_BYTES:
                raise HTTPException(status_code=413,
                                    detail="degerlendirme aktarimi cok buyuk")
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="gecersiz istek uzunlugu") from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > eval_datasets.MAX_JSON_BYTES:
            raise HTTPException(status_code=413,
                                detail="degerlendirme aktarimi cok buyuk")
    try:
        text = bytes(raw).decode("utf-8", errors="strict")
        if not text or text.startswith("\ufeff"):
            raise ValueError("encoding")
        value = json.loads(
            text, object_pairs_hook=_closed_json_pairs,
            parse_constant=_reject_json_constant)
        if (type(value) is not dict
                or set(value) != {"expected_revision", "cases"}
                or type(value["expected_revision"]) is not int
                or value["expected_revision"] < 1):
            raise ValueError("shape")
        cases = eval_datasets.normalize_cases(value["cases"])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
            ValueError, eval_datasets.EvalDatasetError):
        raise HTTPException(status_code=422,
                            detail="gecersiz degerlendirme aktarimi") from None
    return value["expected_revision"], cases


@app.get("/v1/eval/datasets",
         response_model=api_contracts.EvalDatasetListResponse,
         response_model_exclude_unset=True)
def eval_dataset_list(
        limit: int = Query(100, ge=1, le=100),
        principal=Depends(require_org_identity)):
    try:
        with db_conn() as conn:
            rows = db.list_eval_datasets(
                conn, actor_id=principal.subject_id, limit=limit)
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    return {"datasets": [_eval_dataset_metadata(row) for row in rows]}


@app.post("/v1/eval/datasets", status_code=201,
          response_model=api_contracts.EvalDatasetResponse,
          response_model_exclude_unset=True)
def eval_dataset_create(
        body: EvalDatasetCreateRequest,
        principal=Depends(require_eval_writer)):
    try:
        with db_conn() as conn:
            row = db.create_eval_dataset(
                conn, actor_id=principal.subject_id,
                slug=body.slug, label=body.label)
    except db.EvalDatasetConflict:
        raise HTTPException(status_code=409,
                            detail="degerlendirme seti zaten var") from None
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    return {"dataset": _eval_dataset_metadata(row)}


@app.get("/v1/eval/datasets/{dataset_id}/versions",
         response_model=api_contracts.EvalVersionListResponse,
         response_model_exclude_unset=True)
def eval_version_list(
        dataset_id: UUID, principal=Depends(require_org_identity)):
    try:
        with db_conn() as conn:
            rows = db.list_eval_versions(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id)
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    if not rows:
        raise _eval_access_error()
    return {"versions": [_eval_version_metadata(row) for row in rows]}


@app.post("/v1/eval/datasets/{dataset_id}/drafts", status_code=201,
          response_model=api_contracts.EvalVersionResponse,
          response_model_exclude_unset=True)
def eval_draft_create(
        dataset_id: UUID, body: EvalDraftRequest,
        principal=Depends(require_eval_writer)):
    try:
        with db_conn() as conn:
            row = db.create_eval_draft(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id,
                expected_revision=body.expected_revision)
    except db.EvalDatasetConflict:
        raise HTTPException(status_code=409,
                            detail="degerlendirme revision degisti") from None
    except db.EvalDatasetStateRefused:
        raise _eval_state_error() from None
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    return {"version": _eval_version_metadata(row)}


@app.post("/v1/eval/datasets/{dataset_id}/versions/"
          "{version_id}/cases/import",
          response_model=api_contracts.EvalVersionResponse,
          response_model_exclude_unset=True)
async def eval_cases_import(
        dataset_id: UUID, version_id: UUID, request: Request,
        principal=Depends(require_eval_writer)):
    expected_revision, cases = await _eval_import_body(request)
    try:
        with db_conn() as conn:
            row = db.replace_eval_cases(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id, version_id=version_id,
                expected_revision=expected_revision, cases=cases)
    except db.EvalDatasetConflict:
        raise HTTPException(status_code=409,
                            detail="degerlendirme revision degisti") from None
    except db.EvalDatasetStateRefused:
        raise _eval_state_error() from None
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    return {"version": _eval_version_metadata(row, version_id=version_id)}


@app.get("/v1/eval/datasets/{dataset_id}/versions/{version_id}/cases",
         response_model=api_contracts.EvalCaseListResponse)
def eval_cases_read(
        dataset_id: UUID, version_id: UUID, response: Response,
        principal=Depends(require_org_identity)):
    try:
        with db_conn() as conn:
            versions = db.list_eval_versions(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id)
            if not any(row["id"] == version_id for row in versions):
                raise db.EvalDatasetAccessRefused("eval version bulunamadi")
            cases = db.read_eval_cases(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id, version_id=version_id)
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    response.headers["Cache-Control"] = "no-store"
    return {"dataset_id": dataset_id, "version_id": version_id,
            "cases": cases}


@app.post("/v1/eval/datasets/{dataset_id}/versions/{version_id}/publish",
          response_model=api_contracts.EvalVersionResponse,
          response_model_exclude_unset=True)
def eval_version_publish(
        dataset_id: UUID, version_id: UUID, body: EvalPublishRequest,
        principal=Depends(require_eval_writer)):
    try:
        with db_conn() as conn:
            row = db.publish_eval_version(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id, version_id=version_id,
                expected_revision=body.expected_revision,
                expected_policy_epoch=body.expected_policy_epoch,
                expected_draft_sha256=body.expected_draft_sha256)
    except db.EvalDatasetConflict:
        raise HTTPException(status_code=409,
                            detail="degerlendirme yayin kapisi degisti") from None
    except db.EvalDatasetStateRefused:
        raise _eval_state_error() from None
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    return {"version": _eval_version_metadata(row, version_id=version_id)}


@app.post("/v1/eval/datasets/{dataset_id}/retire",
          response_model=api_contracts.EvalDatasetResponse,
          response_model_exclude_unset=True)
def eval_dataset_retire(
        dataset_id: UUID, body: EvalRetireRequest,
        principal=Depends(require_eval_writer)):
    try:
        with db_conn() as conn:
            row = db.retire_eval_dataset(
                conn, actor_id=principal.subject_id,
                dataset_id=dataset_id,
                expected_revision=body.expected_revision,
                expected_policy_epoch=body.expected_policy_epoch)
    except db.EvalDatasetConflict:
        raise HTTPException(status_code=409,
                            detail="degerlendirme emeklilik kapisi degisti") from None
    except db.EvalDatasetStateRefused:
        raise _eval_state_error() from None
    except db.EvalDatasetAccessRefused:
        raise _eval_access_error() from None
    return {"dataset": _eval_dataset_metadata(row)}


@app.get("/v1/models", dependencies=AUTH,
         response_model=api_contracts.ModelListResponse)
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


def _chat_retrieval_scope(req: ChatRequest):
    """Return only canonical caller scope dimensions, never authority ids.

    Meaning is resolved later by the checked backend on the same connection
    and repeatable-read snapshot that performs retrieval.  Resolving here and
    then opening another connection left a policy-change window in which a
    collection/tag scope could be stale before either ranking began.
    """
    direct = _document_scope(req.document_ids)
    collections = _document_scope(req.collection_ids)
    tags = None if req.tags is None else tuple(req.tags)
    return {
        name: value for name, value in (
            ("document_ids", direct),
            ("collection_ids", collections),
            ("tags", tags),
        ) if value is not None
    }


@app.post(
    "/v1/chat/completions",
    dependencies=AUTH,
    response_model=api_contracts.ChatCompletionResponse,
    response_model_exclude_unset=True,
    responses={
        200: {
            "description": "JSON completion or an SSE event stream",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "Each data event validates as "
                            "ChatCompletionChunkResponse; the terminal event "
                            "is the literal [DONE]."
                        ),
                    },
                },
            },
        },
    },
)
def chat_completions(req: ChatRequest):
    """OpenAI-compatible wrapper with a checked publication boundary.

    Table extraction keeps its separate service path. A RAG model may publish
    only the text carried by an answered/abstained ``GuardResult``; review
    results become a fixed notice and never expose the unchecked model reply.

    Retrieval scope SHAPE is refused by the request model before this body
    runs.  Its canonical dimensions are handed to the one checked call both
    response shapes use; the backend resolves their MEANING inside the same
    authority snapshot as retrieval.  A streamed answer and a non-streamed
    one therefore cannot be scoped differently.
    """
    if req.model not in {*RAG_MODELS, TABLE_MODEL_ID}:
        raise HTTPException(status_code=404, detail="bilinmeyen model")
    if not req.messages:
        raise HTTPException(status_code=400, detail="en az bir mesaj gerekli")

    is_table = req.model == TABLE_MODEL_ID
    backend = RAG_MODELS.get(req.model)
    table_principal = auth.current_principal() if is_table else None
    export_ref_for = (
        (lambda name: _register_export_reference(table_principal, name))
        if table_principal is not None else None)
    # Canonicalised BEFORE the first backend call and outside the closure, so
    # both branches offer the same dimensions.  Their meaning is settled in
    # the backend transaction.  The table route reads none of this.
    retrieval_scope = {} if is_table else _chat_retrieval_scope(req)

    def ask_checked():
        question = owui_chat.message_text(req.messages[-1].content)
        if not question.strip():
            raise HTTPException(status_code=400, detail="soru bos olamaz")
        # Forwarded only when a scope was asked for: an unscoped request
        # must reach the backend as the call it has always been.
        principal = auth.current_principal()
        tenant_token = db.bind_execution_tenant(
            principal.tenant_id, actor_id=principal.subject_id)
        try:
            result = rag_backends.answer_checked(question, backend=backend,
                                                 **retrieval_scope)
        except planner.PlannerError:
            raise HTTPException(
                status_code=422, detail="gecersiz retrieval istegi") from None
        except RuntimeError as e:
            _log_rag_failure(e, backend)
            raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_MESSAGE)
        except Exception as e:
            _log_rag_failure(e, backend)
            raise HTTPException(status_code=502, detail=RAG_FAILURE_MESSAGE)
        finally:
            db.reset_execution_tenant(tenant_token)
        published = _publish_checked(result)
        if req.include_trace and published[3] is None:
            log.error("RAG backend omitted requested retrieval trace")
            raise HTTPException(status_code=500,
                                detail="gecersiz RAG izleme sozlesmesi")
        feedback_ref = _persist_review_interaction(result)
        return (*published, feedback_ref)

    if req.stream:
        if is_table:
            gen = owui_chat.stream_tables(
                req.messages, req.model,
                namespace=table_principal.tenant_id.hex,
                export_ref_for=export_ref_for)
        else:
            status, answer, citations, trace, feedback_ref = ask_checked()
            gen = owui_chat.stream_text(
                answer,
                req.model,
                rag_status=status,
                rag_citations=_citation_payload(
                    citations, persist=_browser_evidence_enabled(),
                    feedback_ref=feedback_ref),
                rag_trace=(_trace_payload(trace)
                           if req.include_trace else None),
            )
        return StreamingResponse(gen, media_type="text/event-stream")

    if is_table:
        try:
            answer = owui_chat.tables_reply(
                req.messages,
                namespace=table_principal.tenant_id.hex,
                export_ref_for=export_ref_for)
        except Exception as error:
            _log_safe_failure(error, "tablo_cikarimi_hatasi")
            raise HTTPException(
                status_code=500, detail="tablo cikarimi basarisiz") from None
    else:
        status, answer, citations, trace, feedback_ref = ask_checked()

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
        response["rag_citations"] = _citation_payload(
            citations, persist=_browser_evidence_enabled(),
            feedback_ref=feedback_ref)
        if req.include_trace:
            response["rag_trace"] = _trace_payload(trace)
    return response


def _browser_evidence_enabled():
    principal = auth.bound_principal()
    return (principal is not None and principal.source == "openwebui"
            and principal.subject_id is not None
            and auth.permits(principal, "reader"))


def _citation_payload(citations, *, persist=False, feedback_ref=None):
    payload = []
    references = {}
    for citation in citations:
        item = {"page": citation.page, "source": citation.source}
        # Legacy backends may not yet carry trusted chunk identity.  Never
        # synthesize a reference from filename/page -- neither is unique.  A
        # fully bound citation gets the closed browser contract; the internal
        # chunk UUID itself is never published.
        if (persist and citation.chunk_id is not None
                and citation.document_name is not None):
            digest = _evidence_digest_for_chunk(citation.chunk_id)
            item.update({
                "evidence_ref": _b64url(digest),
                "document_name": citation.document_name,
            })
            if feedback_ref is not None:
                item["feedback_ref"] = feedback_ref
            references[digest] = citation.chunk_id
        payload.append(item)
    if persist and references:
        try:
            with db_conn() as conn:
                db.register_evidence_references(
                    conn, tuple(sorted(references.items())))
        except db.EvidenceAccessRefused:
            raise HTTPException(status_code=500,
                                detail="kanit referansi kaydedilemedi") from None
    return payload


def _persist_review_interaction(result):
    """Record only content-free metadata for verified browser publications."""
    if not _browser_evidence_enabled():
        return None
    if result.status == ANSWERED:
        if not any(citation.chunk_id is not None
                   and citation.document_name is not None
                   for citation in result.citations):
            return None
        interaction_id = uuid.uuid4()
        digest = hmac.new(
            _REVIEW_HMAC_KEY, b"interaction\x00" + interaction_id.bytes,
            hashlib.sha256).digest()
        with db_conn() as conn:
            db.create_review_interaction(
                conn, interaction_id=interaction_id,
                actor_id=auth.current_principal().subject_id,
                ref_digest=digest, outcome=ANSWERED,
                citation_count=len(result.citations))
        return _b64url(digest)
    if result.status == REVIEW_REQUIRED:
        with db_conn() as conn:
            db.create_review_interaction(
                conn, interaction_id=uuid.uuid4(),
                actor_id=auth.current_principal().subject_id,
                ref_digest=None, outcome=REVIEW_REQUIRED,
                citation_count=len(result.citations))
    return None


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _evidence_reference(chunk_id):
    """Return a persistent opaque locator; it grants no read authority."""
    return _b64url(_evidence_digest_for_chunk(chunk_id))


def _evidence_digest_for_chunk(chunk_id):
    try:
        raw = UUID(str(chunk_id)).bytes
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500,
                            detail="gecersiz RAG kanit sozlesmesi") from exc
    return hmac.new(
        _EVIDENCE_HMAC_KEY, b"chunk\x00" + raw,
        hashlib.sha256).digest()


def _evidence_digest(reference):
    """Decode one opaque digest; only a DB mapping can resolve its target."""
    if type(reference) is not str or len(reference) != 43:
        return None
    try:
        raw = base64.b64decode(
            reference + "=", altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) != 32 or _b64url(raw) != reference:
        return None
    return raw


def _export_reference_digest(export_id):
    return hmac.new(
        _EXPORT_HMAC_KEY, b"export\x00" + UUID(str(export_id)).bytes,
        hashlib.sha256).digest()


def _export_id_for_storage(storage_name):
    """Give one random code-name a stable, non-reversible database identity."""
    if (type(storage_name) is not str
            or owui_chat.EXPORT_NAME_RE.fullmatch(storage_name) is None):
        raise db.ExportAccessRefused("disari aktarim dosyasi gecersiz")
    material = hmac.new(
        _EXPORT_HMAC_KEY, b"storage\x00" + storage_name.encode("ascii"),
        hashlib.sha256).digest()
    return uuid.UUID(bytes=material[:16], version=4)


def _decode_export_reference(reference):
    """Decode one opaque export locator; it grants no download authority."""
    return _evidence_digest(reference)


def _read_export_bytes(storage_name):
    """Read one code-named export through no-follow handles under a ceiling."""
    if (type(storage_name) is not str
            or owui_chat.EXPORT_NAME_RE.fullmatch(storage_name) is None):
        raise db.ExportAccessRefused("disari aktarim dosyasi gecersiz")
    root = file_handle = None
    file_closed = root_closed = True
    try:
        root = handle_transport.open_root(owui_chat.EXPORT_DIR)
        file_handle = handle_transport.open_child_file(root, storage_name)
        body = handle_transport.read_all(file_handle, EXPORT_MAX_BYTES)
    except handle_transport.TransportError as exc:
        raise db.ExportAccessRefused(
            "disari aktarim dosyasi kullanilamiyor") from exc
    finally:
        if file_handle is not None:
            file_closed = handle_transport.close_handle_quietly(file_handle)
        if root is not None:
            root_closed = handle_transport.close_directory_quietly(root)
    if not file_closed or not root_closed or not body:
        raise db.ExportAccessRefused("disari aktarim dosyasi kullanilamiyor")
    return body


def _register_export_reference(principal, storage_name):
    """Measure and bind one generated file without persisting its contents."""
    if principal.subject_id is None or principal.source != "openwebui":
        return None
    try:
        body = _read_export_bytes(storage_name)
        export_id = _export_id_for_storage(storage_name)
        ref_digest = _export_reference_digest(export_id)
        with _principal_db_conn(principal) as conn:
            db.register_table_export(
                conn,
                export_id=export_id,
                actor_id=principal.subject_id,
                ref_digest=ref_digest,
                storage_name=storage_name,
                file_sha256=hashlib.sha256(body).digest(),
                file_size=len(body),
                ttl_seconds=EXPORT_RECORD_SECONDS,
            )
    except db.ExportAccessRefused:
        return None
    return _b64url(ref_digest)


@app.post("/v1/evidence/tickets",
          response_model=api_contracts.ShortLivedTicketResponse)
def create_evidence_ticket(
        body: EvidenceTicketRequest, response: Response,
        principal=Depends(require_evidence_actor)):
    """Exchange an opaque citation locator for one 50-second preview ticket."""
    ref_digest = _evidence_digest(body.evidence_ref)
    if ref_digest is None:
        raise HTTPException(status_code=404, detail="kanit bulunamadi")
    ticket = secrets.token_urlsafe(32)
    digest = hashlib.sha256(ticket.encode("ascii")).digest()
    try:
        with db_conn() as conn:
            db.mint_evidence_preview_ticket(
                conn, actor_id=principal.subject_id, ref_digest=ref_digest,
                token_digest=digest, ttl_seconds=EVIDENCE_TICKET_SECONDS)
    except db.EvidenceAccessRefused:
        raise HTTPException(status_code=404,
                            detail="kanit bulunamadi") from None
    response.headers["Cache-Control"] = "no-store"
    return {"ticket": ticket, "expires_in": EVIDENCE_TICKET_SECONDS}


@app.post("/v1/evidence/preview",
          response_model=api_contracts.EvidencePreviewResponse)
def preview_evidence(
        body: EvidencePreviewRequest, response: Response,
        principal=Depends(require_evidence_actor)):
    """Consume one actor-bound ticket and return one bounded passage only."""
    digest = hashlib.sha256(body.ticket.encode("ascii")).digest()
    try:
        with db_conn() as conn:
            preview = db.consume_evidence_preview_ticket(
                conn, actor_id=principal.subject_id, token_digest=digest,
                passage_max_chars=EVIDENCE_PASSAGE_MAX_CHARS)
    except db.EvidenceAccessRefused:
        raise HTTPException(status_code=404,
                            detail="kanit bileti gecersiz") from None
    if (set(preview) != {"document_name", "page", "passage"}
            or type(preview["document_name"]) is not str
            or type(preview["page"]) is not int
            or type(preview["passage"]) is not str):
        raise HTTPException(status_code=500,
                            detail="gecersiz kanit onizleme sozlesmesi")
    response.headers["Cache-Control"] = "no-store"
    return {
        "document_name": preview["document_name"],
        "page": preview["page"],
        "content_type": "passage",
        "passage": preview["passage"],
    }


@app.post("/v1/exports/tickets",
          response_model=api_contracts.ShortLivedTicketResponse)
def create_export_ticket(
        body: ExportTicketRequest, response: Response,
        principal=Depends(require_evidence_actor)):
    """Exchange an opaque export locator for one 50-second actor ticket."""
    ref_digest = _decode_export_reference(body.export_ref)
    if ref_digest is None:
        raise HTTPException(status_code=404,
                            detail="disari aktarim bulunamadi")
    ticket = secrets.token_urlsafe(32)
    token_digest = hashlib.sha256(ticket.encode("ascii")).digest()
    try:
        with db_conn() as conn:
            db.mint_table_export_ticket(
                conn,
                actor_id=principal.subject_id,
                ref_digest=ref_digest,
                token_digest=token_digest,
                ttl_seconds=EXPORT_TICKET_SECONDS,
            )
    except db.ExportAccessRefused:
        raise HTTPException(status_code=404,
                            detail="disari aktarim bulunamadi") from None
    response.headers["Cache-Control"] = "no-store"
    return {"ticket": ticket, "expires_in": EXPORT_TICKET_SECONDS}


@app.post(
    "/v1/exports/download",
    response_class=Response,
    responses={
        200: {"content": {
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet": {}}},
    },
)
def download_export(
        body: ExportDownloadRequest,
        principal=Depends(require_evidence_actor)):
    """Atomically consume one actor ticket, then serve its measured bytes."""
    token_digest = hashlib.sha256(body.ticket.encode("ascii")).digest()
    try:
        with db_conn() as conn:
            measured = db.consume_table_export_ticket(
                conn, actor_id=principal.subject_id,
                token_digest=token_digest)
        if set(measured) != {"storage_name", "file_sha256", "file_size"}:
            raise db.ExportAccessRefused("disari aktarim sozlesmesi gecersiz")
        digest_value = measured["file_sha256"]
        if (type(measured["storage_name"]) is not str
                or type(measured["file_size"]) is not int
                or isinstance(measured["file_size"], bool)
                or not isinstance(digest_value, (bytes, bytearray, memoryview))):
            raise db.ExportAccessRefused("disari aktarim sozlesmesi gecersiz")
        expected_digest = bytes(digest_value)
        if len(expected_digest) != 32:
            raise db.ExportAccessRefused("disari aktarim sozlesmesi gecersiz")
        body_bytes = _read_export_bytes(measured["storage_name"])
        if (len(body_bytes) != measured["file_size"]
                or not hmac.compare_digest(
                    hashlib.sha256(body_bytes).digest(), expected_digest)):
            raise db.ExportAccessRefused("disari aktarim olcumu degisti")
    except db.ExportAccessRefused:
        raise HTTPException(status_code=404,
                            detail="disari aktarim bileti gecersiz") from None
    return Response(
        content=body_bytes,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"),
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="table-export.xlsx"',
        },
    )


@app.post("/v1/reviews/feedback",
          response_model=api_contracts.ReviewFeedbackResponse)
def submit_review_feedback(
        body: ReviewFeedbackRequest,
        principal=Depends(require_evidence_actor)):
    """Record one closed verdict; the opaque reference grants no authority."""
    if ((body.verdict == "helpful" and body.reason_code is not None)
            or (body.verdict == "not_helpful" and body.reason_code is None)):
        raise HTTPException(status_code=422,
                            detail="geri bildirim nedeni kararla uyusmuyor")
    digest = _evidence_digest(body.feedback_ref)
    if digest is None:
        raise HTTPException(status_code=404,
                            detail="geri bildirim hedefi bulunamadi")
    try:
        with db_conn() as conn:
            result = db.submit_review_feedback(
                conn, actor_id=principal.subject_id, ref_digest=digest,
                verdict=body.verdict, reason_code=body.reason_code)
    except db.ReviewAccessRefused:
        raise HTTPException(status_code=404,
                            detail="geri bildirim hedefi bulunamadi") from None
    return {"status": "recorded", **result}


@app.get("/v1/reviews/queue",
         response_model=api_contracts.ReviewQueueResponse)
def review_queue(
        request: Request,
        limit: int = Query(20, ge=1, le=100),
        before_created_at: AwareDatetime | None = Query(default=None),
        before_id: UUID | None = Query(default=None),
        reason_code: Literal["management_duty", "security_review"] = Query(),
        principal=Depends(require_org_identity)):
    """List only fresh, strict-descendant cases; never return model content."""
    if (before_created_at is None) != (before_id is None):
        raise HTTPException(status_code=422,
                            detail="inceleme imleci eksik")
    before = (before_created_at, before_id) if before_id is not None else None
    try:
        with db_conn() as conn:
            rows = db.list_review_cases(
                conn, reviewer_id=principal.subject_id,
                limit=limit, before=before)
            db.record_org_decision(
                conn, actor_id=principal.subject_id, subject_id=None,
                action="review_queue_view", reason_code=reason_code,
                allowed=True, request_id=request.state.request_id)
    except db.ReviewAccessRefused:
        raise HTTPException(status_code=403,
                            detail="inceleme kuyrugu yetkisi yok") from None
    has_more = len(rows) > limit
    page = rows[:limit]
    items = [{
        "case_id": row["id"],
        "trigger_code": row["trigger_code"],
        "state": row["state"],
        "revision": row["revision"],
        "created_at": row["created_at"],
        "outcome": row["outcome"],
        "citation_count": row["citation_count"],
        "subject_label": row["display_label"],
        "position_title": row["position_title"],
        "policy_epoch": row["policy_epoch"],
    } for row in page]
    next_cursor = None
    if has_more and page:
        next_cursor = {
            "before_created_at": page[-1]["created_at"],
            "before_id": page[-1]["id"],
        }
    return {"cases": items, "has_more": has_more,
            "next_cursor": next_cursor}


@app.post("/v1/reviews/{case_id}/decision",
          response_model=api_contracts.ReviewDecisionResponse)
def decide_review_case(
        case_id: UUID, body: ReviewDecisionRequest, request: Request,
        principal=Depends(require_org_identity)):
    try:
        with db_conn() as conn:
            result = db.decide_review_case(
                conn, reviewer_id=principal.subject_id, case_id=case_id,
                expected_revision=body.expected_revision,
                expected_policy_epoch=body.expected_policy_epoch,
                decision=body.decision,
                resolution_code=body.resolution_code,
                reason_code=body.reason_code,
                request_id=request.state.request_id)
    except db.ReviewConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except db.ReviewAccessRefused:
        raise HTTPException(status_code=404,
                            detail="inceleme vakasi bulunamadi") from None
    return result


def _trace_payload(trace):
    if not isinstance(trace, RetrievalTrace):
        raise HTTPException(status_code=500,
                            detail="gecersiz RAG izleme sozlesmesi")
    return trace.public()


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
    if result.trace is not None and not isinstance(result.trace, RetrievalTrace):
        log.error("RAG backend returned invalid retrieval trace")
        raise HTTPException(status_code=500,
                            detail="gecersiz RAG yanit sozlesmesi")
    if not valid_citations:
        log.error("RAG backend returned invalid citation metadata")
        raise HTTPException(status_code=500, detail="gecersiz RAG yanit sozlesmesi")
    if (
        result.status == ANSWERED
        and has_text
        and clean
        and not is_abstention(result.answer)
    ):
        return result.status, result.answer, result.citations, result.trace
    if (
        result.status == ABSTAINED
        and has_text
        and clean
        and is_abstention(result.answer)
        and result.citations == ()
    ):
        return result.status, result.answer, result.citations, result.trace
    if result.status == REVIEW_REQUIRED and result.answer is None:
        return result.status, REVIEW_MESSAGE, (), result.trace
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


# The endpoint used to keep a per-filename lock of its own, IN THIS
# PROCESS, as a cheap first fence in front of its hand-written publish
# sequence. There is no sequence here any more: the publication service
# holds a database SESSION lock, which serialises across PROCESSES and
# therefore across every worker, not just this one. A second, weaker lock
# in front of it fenced nothing the first did not already fence.
@app.post("/documents/upload", dependencies=EDITOR_AUTH,
          response_model=api_contracts.DocumentUploadResponse)
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
                    allow_replace=replace,
                    tenant_id=auth.current_principal().tenant_id))
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
        except publication.VersionSourceRefused as error:
            _log_safe_failure(error, "surum_kaynagi_yayimlanamadi")
            raise HTTPException(
                status_code=500,
                detail="document version source could not be published") \
                from None
    return {
        "document_id": document_id,
        "filename": canonical,
        "candidate_id": candidate_id,
        # During the compatibility window candidate identity is the immutable
        # source-version identity.  Publishing both names lets new clients move
        # to the durable vocabulary without breaking existing callers.
        "version_id": candidate_id,
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


@app.post("/documents/{document_id}/process", dependencies=EDITOR_AUTH,
          response_model=api_contracts.DocumentProcessResponse,
          response_model_exclude_unset=True)
def process_document(document_id: str):
    # Three SHORT borrows instead of one connection held across the whole
    # request: ingest can run for minutes on its own connection, and a pooled
    # connection parked here for that long would starve every other request.
    with db_conn() as conn:
        doc = db.get_document(conn, document_id)
        queued_job = db.active_ingest_job(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    if queued_job is not None:
        raise HTTPException(status_code=409,
                            detail="document already has an active ingest job")
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
        except db.IngestJobConflict:
            # The earlier read is only a fast refusal. begin_attempt repeats
            # it while holding the document row lock, closing enqueue races.
            raise HTTPException(
                status_code=409,
                detail="document already has an active ingest job")
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
        with db_conn() as conn:
            publication.ensure_bound_version_source(
                conn,
                UPLOAD_DIR,
                auth.current_principal().tenant_id,
                document_id,
                attempt.candidate_id,
                filename,
                expected_sha256=attempt.candidate_sha,
                max_bytes=UPLOAD_MAX_BYTES,
            )
        # bound to the attempt, not to a tuple the endpoint read: the
        # candidate id, its bytes and the observed generation all travel
        # inside the one object the lease was minted with
        verdict = _reported_outcome(ingest.ingest_version_source(
            UPLOAD_DIR,
            auth.current_principal().tenant_id,
            document_id,
            attempt.candidate_id,
            filename,
            attempt,
            expected_sha256=attempt.candidate_sha,
            max_bytes=UPLOAD_MAX_BYTES,
        ))
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


@app.post("/documents/{document_id}/ingest-jobs", dependencies=EDITOR_AUTH,
          status_code=202, response_model=api_contracts.IngestJobResponse,
          response_model_exclude_unset=True)
def enqueue_ingest_job(
        document_id: UUID,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
):
    try:
        with db_conn() as conn:
            job = db.enqueue_ingest_job(
                conn, str(document_id), idempotency_key)
    except (db.IngestJobConflict, db.DocumentLifecycleConflict,
            CandidateNotPublished) as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    if job is None:
        raise HTTPException(status_code=404, detail="document not found")
    return job


@app.get("/ingest-jobs/{job_id}", dependencies=AUTH,
         response_model=api_contracts.IngestJobResponse,
         response_model_exclude_unset=True)
def read_ingest_job(job_id: UUID):
    with db_conn() as conn:
        job = db.get_ingest_job(conn, str(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="ingest job not found")
    return job


@app.delete("/ingest-jobs/{job_id}", dependencies=EDITOR_AUTH,
            response_model=api_contracts.IngestJobResponse,
            response_model_exclude_unset=True)
def cancel_ingest_job(job_id: UUID):
    try:
        with db_conn() as conn:
            job = db.cancel_ingest_job(conn, str(job_id))
    except db.IngestJobConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    if job is None:
        raise HTTPException(status_code=404, detail="ingest job not found")
    return job


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

# Version history is metadata, not an internal catalogue dump. Keep a second
# explicit projection at the HTTP boundary even though the DB seam is already
# narrow, so a future internal column cannot become public by accident.
DOCUMENT_VERSION_LIST_FIELDS = (
    "version_id",
    "version_number",
    "created_at",
    "is_active",
    "index_ready",
    "document_revision",
)

DOCUMENT_DETAIL_FIELDS = (
    "id",
    "filename",
    "file_type",
    "uploaded_at",
    "status",
    "status_note",
    "active_generation",
    "content_sha256",
    "candidate_id",
    "active_version_id",
    "revision",
    "archived_at",
    "purged_at",
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


def _document_version_summary(row):
    return {field: row.get(field) for field in DOCUMENT_VERSION_LIST_FIELDS}


def _document_detail(row):
    return {field: row[field] for field in DOCUMENT_DETAIL_FIELDS if field in row}


def _org_audit_event_summary(row):
    return {
        "event_id": str(row["id"]),
        "action": row["action"],
        "reason_code": row["reason_code"],
        "decision": row["decision"],
        "request_id": row["request_id"],
        "actor_id": str(row["actor_id"]),
        "subject_id": (None if row["subject_id"] is None else str(
            row["subject_id"])),
        "created_at": row["created_at"],
    }


@app.get("/documents", dependencies=AUTH,
         response_model=api_contracts.DocumentListResponse)
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
    before_uploaded_at: AwareDatetime | None = Query(None),
    before_id: UUID | None = Query(None),
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

    Deep pages may use the returned ``next_cursor`` as
    ``before_uploaded_at`` plus ``before_id``. The pair is exact and cannot be
    mixed with a non-zero offset; omitting both keeps the legacy offset path.
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
    if (before_uploaded_at is None) != (before_id is None):
        raise HTTPException(
            status_code=422,
            detail="before_uploaded_at ve before_id birlikte verilmeli")
    if before_uploaded_at is not None and offset != 0:
        raise HTTPException(
            status_code=422,
            detail="cursor ve offset birlikte kullanilamaz")
    try:
        with db_conn() as conn:
            filters = {"status": status, "file_type": file_type,
                       "uploaded_after": uploaded_after,
                       "uploaded_before": uploaded_before,
                       "q": q, "archived": archived,
                       "tenant_id": str(
                           auth.current_principal().tenant_id)}
            if collection_id is not None:
                filters["collection_id"] = str(collection_id)
            if tag is not None:
                filters["tag"] = tag
            if before_uploaded_at is not None:
                filters["before"] = (before_uploaded_at, str(before_id))
            rows = db.list_documents(conn, limit=limit, offset=offset, **filters)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit and page:
        next_cursor = {
            "before_uploaded_at": page[-1]["uploaded_at"],
            "before_id": page[-1]["document_id"],
        }
    return {
        "documents": [_document_summary(row) for row in page],
        "limit": limit,
        "offset": offset,
        "has_more": len(rows) > limit,
        "next_cursor": next_cursor,
    }


@app.post("/collections", dependencies=EDITOR_AUTH)
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


@app.delete("/tags/{tag_id}", dependencies=ADMIN_AUTH)
def delete_tag(tag_id: UUID):
    with db_conn() as conn:
        deleted = db.delete_tag(conn, str(tag_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="tag not found")
    return Response(status_code=204)


@app.delete("/collections/{collection_id}", dependencies=ADMIN_AUTH)
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
         dependencies=EDITOR_AUTH)
def add_collection_document(collection_id: UUID, document_id: UUID):
    return _set_collection_membership(collection_id, document_id, True)


@app.delete("/collections/{collection_id}/documents/{document_id}",
            dependencies=EDITOR_AUTH)
def remove_collection_document(collection_id: UUID, document_id: UUID):
    return _set_collection_membership(collection_id, document_id, False)


@app.put("/documents/{document_id}/tags", dependencies=EDITOR_AUTH)
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


@app.post("/documents/{document_id}/archive", dependencies=ADMIN_AUTH,
          response_model=api_contracts.DocumentLifecycleResponse)
def archive_document(document_id: str):
    return _set_document_lifecycle(document_id, True)


@app.post("/documents/{document_id}/restore", dependencies=ADMIN_AUTH,
          response_model=api_contracts.DocumentLifecycleResponse)
def restore_document(document_id: str):
    return _set_document_lifecycle(document_id, False)


@app.get("/documents/{document_id}/versions", dependencies=AUTH,
         response_model=api_contracts.DocumentVersionListResponse)
def list_document_versions(
    document_id: UUID,
    limit: int = Query(DOCUMENT_PAGE_DEFAULT, ge=1, le=DOCUMENT_PAGE_MAX),
    before_version_number: int | None = Query(None, ge=1),
):
    """List content-safe immutable version metadata, newest first."""
    try:
        with db_conn() as conn:
            rows = db.list_document_versions(
                conn, str(document_id), limit=limit,
                before_version_number=before_version_number)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    has_more = len(rows) > limit
    page = [_document_version_summary(row) for row in rows[:limit]]
    return {
        "versions": page,
        "limit": limit,
        "before_version_number": before_version_number,
        "has_more": has_more,
        "next_before_version_number": (
            page[-1]["version_number"] if has_more and page else None),
    }


@app.post(
    "/documents/{document_id}/versions/{version_id}/activate",
    dependencies=ADMIN_AUTH,
    response_model=api_contracts.DocumentVersionActivationResponse,
)
def activate_document_version(
    document_id: UUID,
    version_id: UUID,
    request: DocumentVersionActivationRequest,
):
    """Activate one retained ready version after a fresh source proof."""
    document_text = str(document_id)
    version_text = str(version_id)
    with db_conn() as conn:
        digest = db.document_version_source_digest(
            conn, document_text, version_text)
        if digest is None:
            raise HTTPException(status_code=404,
                                detail="document version not found")
        try:
            proof = publication.verify_version_source(
                UPLOAD_DIR,
                auth.current_principal().tenant_id,
                document_text,
                version_text,
                expected_sha256=digest,
                max_bytes=UPLOAD_MAX_BYTES,
            )
        except publication.VersionSourceMissing:
            raise HTTPException(status_code=409,
                                detail="document version source unavailable") \
                from None
        except publication.VersionSourceRefused:
            raise HTTPException(status_code=409,
                                detail="document version source invalid") \
                from None
        try:
            activated = db.activate_document_version(
                conn,
                document_text,
                version_text,
                request.expected_revision,
                verified_source_sha256=proof.sha256,
            )
        except db.DocumentVersionConflict:
            raise HTTPException(
                status_code=409,
                detail="document version revision conflict") from None
        except db.DocumentLifecycleConflict:
            raise HTTPException(
                status_code=409,
                detail="document lifecycle conflict") from None
        except db.IngestJobConflict:
            raise HTTPException(
                status_code=409,
                detail="document ingest job conflict") from None
    if activated is None:
        raise HTTPException(status_code=409,
                            detail="document version is not activation-ready")
    return activated


@app.get("/documents/{document_id}", dependencies=AUTH,
         response_model=api_contracts.DocumentDetailResponse,
         response_model_exclude_unset=True)
def read_document(document_id: str):
    with db_conn() as conn:
        doc = db.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _document_detail(doc)


@app.get("/health", response_model=api_contracts.HealthResponse)
def health():
    """Liveness: the process is up. Deliberately touches nothing else, so a
    restart loop is never triggered by a dependency being briefly unavailable."""
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics():
    """Public monitoring surface containing route templates and counts only."""
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


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


@app.get("/ready", response_model=api_contracts.ReadinessResponse)
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

    def check_schema():
        with db_conn() as conn:
            if not db.schema_is_current(conn):
                raise RuntimeError("schema drift")

    def check_embed():
        base = embeddings.EMBED_API_URL.rsplit("/v1/", 1)[0]
        requests.get(f"{base}/v1/models", timeout=3).raise_for_status()

    checks = dict([
        _probe("veritabani", check_db),
        _probe("sema", check_schema),
        _probe("embedding", check_embed),
    ])
    healthy = all(checks.values())
    response.status_code = 200 if healthy else 503
    return {"status": "ready" if healthy else "degraded", "kontroller": checks}
