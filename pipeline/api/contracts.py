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
