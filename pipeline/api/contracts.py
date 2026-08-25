"""Closed response contracts shared by the HTTP API and generated clients.

Request validation already protects most write paths.  Responses historically
relied on tests alone, however, so a misspelled or newly leaked field could
reach a client without FastAPI noticing.  Migrate the surface incrementally:
each model here forbids additions and must preserve the route's existing JSON
bytes when it is first attached.
"""
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


class _ClosedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _wire_timestamp(value):
    """Keep legacy timestamp spelling while accepting real DB datetimes."""
    return value.isoformat() if isinstance(value, datetime) else value


WireTimestamp = Annotated[
    str | datetime,
    Field(union_mode="left_to_right"),
    PlainSerializer(_wire_timestamp, return_type=str, when_used="json"),
]
WireId = str | UUID

ErrorCode = Literal[
    "invalid_request", "authentication_required", "permission_denied",
    "resource_not_found", "method_not_allowed", "conflict",
    "payload_too_large", "validation_failed", "rate_limited",
    "internal_error", "upstream_failed", "service_unavailable",
    "request_failed",
]


class ErrorEnvelopeResponse(_ClosedResponse):
    version: Literal[1]
    code: ErrorCode
    request_id: str = Field(pattern=r"^(?:[a-f0-9]{8}|unavailable)$")


class LegacyValidationIssueResponse(_ClosedResponse):
    type: str
    loc: list[str | int]
    msg: str


class ErrorResponse(_ClosedResponse):
    error: ErrorEnvelopeResponse
    detail: str | list[LegacyValidationIssueResponse]


class HealthResponse(_ClosedResponse):
    status: Literal["ok"]


class ReadinessChecks(_ClosedResponse):
    veritabani: bool
    sema: bool
    embedding: bool


class ReadinessResponse(_ClosedResponse):
    status: Literal["ready", "degraded"]
    kontroller: ReadinessChecks


class ModelDescriptor(_ClosedResponse):
    id: str = Field(min_length=1, max_length=200,
                    pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
    object: Literal["model"]
    owned_by: Literal["ragtest"]


class ModelListResponse(_ClosedResponse):
    object: Literal["list"]
    data: tuple[ModelDescriptor, ...] = Field(min_length=1, max_length=20)


class BrowserSessionResponse(_ClosedResponse):
    authenticated: Literal[True]
    tenant_id: WireId
    role: Literal["reader", "editor", "admin", "org_architect"]
    source: Literal["oidc"]
    position_id: WireId | None
    org_architect: bool
    csrf_token: str = Field(min_length=32, max_length=128)
    expires_at: int = Field(gt=0)


class OrgSelfMembership(_ClosedResponse):
    identity_id: WireId
    position_id: WireId
    title: str
    kind: Literal["root", "manager", "member"]
    level: int = Field(ge=1)
    can_monitor_descendants: bool
    protected_from_monitoring: bool
    architecture_version: int = Field(ge=1)
    policy_epoch: int = Field(ge=1)
    display_label: str
    app_role: Literal["reader", "editor", "admin"]


class OrganizationMeResponse(_ClosedResponse):
    tenant_id: WireId
    identity_id: WireId
    architecture_admin: bool
    membership: OrgSelfMembership | None


class VisibleOrgMember(_ClosedResponse):
    identity_id: WireId
    display_label: str
    app_role: Literal["reader", "editor", "admin"]
    position_id: WireId
    title: str
    kind: Literal["root", "manager", "member"]
    level: int = Field(ge=1)


class VisibleOrgMembersResponse(_ClosedResponse):
    members: list[VisibleOrgMember]


class OrgPositionResponse(_ClosedResponse):
    id: WireId
    parent_id: WireId | None
    title: str
    kind: Literal["root", "manager", "member"]
    can_monitor_descendants: bool
    protected_from_monitoring: bool


class OrgTopologyMember(_ClosedResponse):
    identity_id: WireId
    issuer: str
    subject: str
    position_id: WireId
    display_label: str
    app_role: Literal["reader", "editor", "admin"]
    state: Literal["active", "pending", "suspended"]


class OrgTopologyResponse(_ClosedResponse):
    id: WireId
    name: str
    architecture_version: int = Field(ge=1)
    policy_epoch: int = Field(ge=1)
    positions: list[OrgPositionResponse]
    members: list[OrgTopologyMember]


class OrgVersionResponse(_ClosedResponse):
    architecture_version: int = Field(ge=1)
    policy_epoch: int = Field(ge=1)


class OrgMembershipResponse(_ClosedResponse):
    identity_id: WireId
    display_label: str
    app_role: Literal["reader", "editor", "admin"]
    state: Literal["active", "pending", "suspended"]
    position_id: WireId | None
    architecture_version: int = Field(ge=1)
    policy_epoch: int = Field(ge=1)


class OrgAuditEvent(_ClosedResponse):
    event_id: WireId
    action: str
    reason_code: str
    decision: Literal["allowed", "denied"]
    request_id: str
    actor_id: WireId
    subject_id: WireId | None
    created_at: WireTimestamp


class OrgAuditCursor(_ClosedResponse):
    before_created_at: WireTimestamp
    before_id: WireId


class OrgAuditEventListResponse(_ClosedResponse):
    events: list[OrgAuditEvent]
    limit: int = Field(ge=1, le=100)
    has_more: bool
    next_cursor: OrgAuditCursor | None


class RetentionPolicyResponse(_ClosedResponse):
    tenant_id: WireId
    archive_retention_days: int = Field(ge=1, le=3650)
    revision: int = Field(ge=1)
    policy_epoch: int = Field(ge=1)
    updated_at: WireTimestamp | None


class RetentionDocument(_ClosedResponse):
    document_id: WireId
    status: str
    revision: int = Field(ge=0)
    uploaded_at: WireTimestamp
    archived_at: WireTimestamp | None
    purged_at: WireTimestamp | None
    active_hold_count: int = Field(ge=0)
    latest_purge_job_id: WireId | None
    latest_purge_state: Literal[
        "pending", "running", "completed", "failed", "cancelled"
    ] | None


class RetentionDocumentCursor(_ClosedResponse):
    before_uploaded_at: WireTimestamp
    before_id: WireId


class RetentionDocumentListResponse(_ClosedResponse):
    documents: list[RetentionDocument]
    limit: int = Field(ge=1, le=100)
    has_more: bool
    next_cursor: RetentionDocumentCursor | None


class LegalHoldResponse(_ClosedResponse):
    id: WireId
    document_id: WireId
    reason_code: Literal[
        "litigation", "regulatory", "security_investigation"
    ]
    state: Literal["active", "released"]
    revision: int = Field(ge=1)
    created_at: WireTimestamp | None
    released_at: WireTimestamp | None
    policy_epoch: int | None = Field(default=None, ge=1)


class LegalHoldListResponse(_ClosedResponse):
    holds: list[LegalHoldResponse]


class PurgeJobResponse(_ClosedResponse):
    id: WireId
    document_id: WireId
    state: Literal["pending", "running", "completed", "failed", "cancelled"]
    eligible_at: WireTimestamp | None
    attempt_count: int = Field(ge=0, le=20)
    policy_revision: int | None = Field(default=None, ge=1)
    policy_epoch: int | None = Field(default=None, ge=1)
    document_revision: int | None = Field(default=None, ge=0)
    failure_code: Literal[
        "storage_refused", "storage_unavailable"
    ] | None = None
    created_at: WireTimestamp | None
    started_at: WireTimestamp | None = None
    completed_at: WireTimestamp | None = None


class PurgeJobListResponse(_ClosedResponse):
    jobs: list[PurgeJobResponse]


class DocumentSummary(_ClosedResponse):
    document_id: WireId
    filename: str
    file_type: str
    uploaded_at: WireTimestamp | None
    status: str
    status_note: str | None
    active_generation: int = Field(ge=0)
    archived_at: WireTimestamp | None


class DocumentCursor(_ClosedResponse):
    before_uploaded_at: WireTimestamp
    before_id: WireId


class DocumentListResponse(_ClosedResponse):
    documents: list[DocumentSummary]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    has_more: bool
    next_cursor: DocumentCursor | None


class DocumentDetailResponse(_ClosedResponse):
    id: WireId
    filename: str
    file_type: str
    uploaded_at: WireTimestamp | None = None
    status: str
    status_note: str | None = None
    active_generation: int = Field(default=0, ge=0)
    content_sha256: str | None = None
    candidate_id: WireId | None = None
    active_version_id: WireId | None = None
    revision: int | None = Field(default=None, ge=0)
    archived_at: WireTimestamp | None = None
    purged_at: WireTimestamp | None = None


class DocumentLifecycleResponse(_ClosedResponse):
    document_id: WireId
    archived: bool
    archived_at: WireTimestamp | None


class DocumentVersion(_ClosedResponse):
    version_id: WireId
    version_number: int = Field(ge=1)
    created_at: WireTimestamp
    is_active: bool
    index_ready: bool
    document_revision: int = Field(ge=0)


class DocumentVersionListResponse(_ClosedResponse):
    versions: list[DocumentVersion]
    limit: int = Field(ge=1, le=100)
    before_version_number: int | None = Field(default=None, ge=1)
    has_more: bool
    next_before_version_number: int | None = Field(default=None, ge=1)


class DocumentVersionActivationResponse(_ClosedResponse):
    document_id: WireId
    active_version_id: WireId
    active_generation: int = Field(ge=0)
    revision: int = Field(ge=0)
    changed: bool




class DocumentUploadResponse(_ClosedResponse):
    document_id: WireId
    filename: str
    candidate_id: WireId
    version_id: WireId
    status: Literal["pending"]


class DocumentProcessResponse(_ClosedResponse):
    document_id: WireId
    status: Literal["done", "partial"]
    status_note: str | None = None


class IngestJobResponse(_ClosedResponse):
    job_id: WireId
    document_id: WireId
    candidate_id: WireId
    version_id: WireId | None = None
    status: Literal[
        "queued", "running", "succeeded", "partial", "failed", "cancelled"
    ]
    attempt_count: int = Field(ge=0)
    created_at: WireTimestamp | None
    started_at: WireTimestamp | None
    finished_at: WireTimestamp | None
    outcome_note: str | None




class ShortLivedTicketResponse(_ClosedResponse):
    ticket: str
    expires_in: int = Field(ge=1)


class EvidencePreviewResponse(_ClosedResponse):
    document_name: str
    page: int = Field(ge=0)
    content_type: Literal["passage"]
    passage: str


class ReviewFeedbackResponse(_ClosedResponse):
    status: Literal["recorded"]
    revision: int = Field(ge=1)
    review_open: bool


class ReviewCase(_ClosedResponse):
    case_id: WireId
    trigger_code: Literal["user_feedback", "guard_review"]
    state: Literal["open"]
    revision: int = Field(ge=1)
    created_at: WireTimestamp
    outcome: Literal["answered", "review_required"]
    citation_count: int = Field(ge=0)
    subject_label: str
    position_title: str
    policy_epoch: int = Field(ge=1)


class ReviewCursor(_ClosedResponse):
    before_created_at: WireTimestamp
    before_id: WireId


class ReviewQueueResponse(_ClosedResponse):
    cases: list[ReviewCase]
    has_more: bool
    next_cursor: ReviewCursor | None


class ReviewDecisionResponse(_ClosedResponse):
    state: Literal["resolved", "dismissed"]
    revision: int = Field(ge=2)
    decided_at: WireTimestamp




class EvalVersionMetadata(_ClosedResponse):
    version_id: WireId
    version_number: int = Field(ge=1)
    state: Literal["draft", "published"]
    revision: int = Field(ge=1)
    case_count: int = Field(ge=0)
    content_sha256: str | None = None
    sealed_at: WireTimestamp | None = None


class EvalDatasetMetadata(_ClosedResponse):
    dataset_id: WireId
    slug: str
    label: str
    state: Literal["active", "retired"]
    revision: int = Field(ge=1)
    owner_label: str | None = None
    policy_epoch: int | None = Field(default=None, ge=1)
    current_version_id: WireId | None = None
    current_version_number: int | None = Field(default=None, ge=1)
    versions: list[EvalVersionMetadata]


class EvalDatasetListResponse(_ClosedResponse):
    datasets: list[EvalDatasetMetadata]


class EvalDatasetResponse(_ClosedResponse):
    dataset: EvalDatasetMetadata


class EvalVersionListResponse(_ClosedResponse):
    versions: list[EvalVersionMetadata]


class EvalVersionResponse(_ClosedResponse):
    version: EvalVersionMetadata


class EvalCase(_ClosedResponse):
    case_key: WireId
    q: str
    key: str
    answer: str
    pages: list[int]
    type: Literal["metin", "sayisal", "tablo"]


class EvalCaseListResponse(_ClosedResponse):
    dataset_id: WireId
    version_id: WireId
    cases: list[EvalCase]


class RagCitationResponse(_ClosedResponse):
    page: int = Field(ge=1)
    source: Literal["model", "derived"]
    evidence_ref: str | None = None
    document_name: str | None = None
    feedback_ref: str | None = None


class RetrievalTraceStagesResponse(_ClosedResponse):
    plan: int = Field(ge=0)
    retrieve: int = Field(ge=0)
    rerank: int | None = Field(default=None, ge=0)
    context: int = Field(ge=0)
    generate: int = Field(ge=0)
    validate_ms: int = Field(alias="validate", serialization_alias="validate",
                             ge=0)


class RetrievalTraceResponse(_ClosedResponse):
    trace_version: Literal[2]
    trace_id: str
    backend: Literal["native", "llamaindex"]
    planner_policy_version: Literal[1]
    query_class: Literal["factual"]
    retrieval_mode: Literal["hybrid_balanced"]
    fallback: Literal["none"]
    scope_kind: Literal[
        "all_visible", "explicit_documents", "metadata_filters",
        "intersection", "empty",
    ]
    policy_epoch: int = Field(ge=1)
    top_k: int = Field(ge=1)
    candidate_limit: int = Field(ge=1)
    scope_document_count: int | None = Field(default=None, ge=0)
    retrieved_count: int = Field(ge=0)
    reranked_count: int | None = Field(default=None, ge=0)
    context_passage_count: int = Field(ge=0)
    context_utf8_bytes: int = Field(ge=0)
    stages_ms: RetrievalTraceStagesResponse


class ChatMessageResponse(_ClosedResponse):
    role: Literal["assistant"]
    content: str


class ChatChoiceResponse(_ClosedResponse):
    index: Literal[0]
    message: ChatMessageResponse
    finish_reason: Literal["stop"]


class ChatCompletionResponse(_ClosedResponse):
    id: str
    object: Literal["chat.completion"]
    created: int = Field(ge=0)
    model: str
    choices: list[ChatChoiceResponse] = Field(min_length=1, max_length=1)
    rag_status: Literal["answered", "abstained", "review_required"] | None = None
    rag_citations: list[RagCitationResponse] | None = None
    rag_trace: RetrievalTraceResponse | None = None


class ChatDeltaResponse(_ClosedResponse):
    content: str | None = None


class ChatChunkChoiceResponse(_ClosedResponse):
    index: Literal[0]
    delta: ChatDeltaResponse
    finish_reason: Literal["stop"] | None


class ChatCompletionChunkResponse(_ClosedResponse):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int = Field(ge=0)
    model: str
    choices: list[ChatChunkChoiceResponse] = Field(min_length=1, max_length=1)
    rag_status: Literal["answered", "abstained", "review_required"] | None = None
    rag_citations: list[RagCitationResponse] | None = None
    rag_trace: RetrievalTraceResponse | None = None


class CollectionCreatedResponse(_ClosedResponse):
    collection_id: WireId
    name: str
    created_at: WireTimestamp | None


class CollectionSummaryResponse(CollectionCreatedResponse):
    document_count: int = Field(ge=0)


class CollectionListResponse(_ClosedResponse):
    collections: list[CollectionSummaryResponse]


class TagSummaryResponse(_ClosedResponse):
    tag_id: WireId
    name: str
    created_at: WireTimestamp | None
    document_count: int = Field(ge=0)


class TagListResponse(_ClosedResponse):
    tags: list[TagSummaryResponse]


class CollectionMembershipResponse(_ClosedResponse):
    collection_id: WireId
    document_id: WireId
    present: bool


class DocumentTagsResponse(_ClosedResponse):
    document_id: WireId
    tags: list[str]
