"""Closed citation references and actor-bound evidence preview tickets."""
from contextlib import contextmanager
from datetime import datetime, timezone
import base64
import hashlib
from uuid import UUID

import pytest
from fastapi import HTTPException, Response

from pipeline.api import auth
from pipeline.api import app as api
from pipeline.index import db
from pipeline.validation.rag.answer_guard import MODEL_CITATION, PageCitation


TENANT = UUID("71000000-0000-0000-0000-000000000001")
ACTOR = UUID("71000000-0000-0000-0000-000000000002")
CHUNK = UUID("71000000-0000-0000-0000-000000000003")


def _principal(role="reader", *, architect=False):
    return auth.Principal(
        TENANT, role, subject_id=ACTOR, source="openwebui",
        org_architect=architect)


def test_evidence_reference_is_stable_opaque_and_integrity_protected():
    first = api._evidence_reference(str(CHUNK))
    assert first == api._evidence_reference(str(CHUNK))
    assert len(first) == 43
    assert str(CHUNK) not in first
    raw = base64.urlsafe_b64decode(first + "=")
    assert len(raw) == 32
    assert CHUNK.bytes not in raw
    assert api._evidence_digest(first) == raw
    replacement = "A" if first[0] != "A" else "B"
    # A syntactically valid different digest has no reversible identity; only
    # the tenant-scoped persistent DB mapping can resolve it.
    assert api._evidence_digest(replacement + first[1:]) != raw


def test_chat_citation_projects_no_internal_id_path_or_excerpt():
    payload = api._citation_payload((PageCitation(
        7, MODEL_CITATION, chunk_id=str(CHUNK),
        document_name="yonetim-raporu.pdf"),))
    assert payload == [{"page": 7, "source": "model"}]
    rendered = repr(payload)
    assert str(CHUNK) not in rendered
    assert "passage" not in rendered
    assert "source_path" not in rendered


def test_legacy_citation_without_trusted_chunk_does_not_invent_a_reference():
    assert api._citation_payload((PageCitation(7, MODEL_CITATION),)) == [
        {"page": 7, "source": "model"}]


def test_browser_citation_is_persisted_before_its_reference_is_returned(
        monkeypatch):
    digest = api._evidence_digest_for_chunk(str(CHUNK))
    conn = _Conn({"ref_digest": digest})

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(api, "db_conn", connection)
    payload = api._citation_payload((PageCitation(
        3, MODEL_CITATION, chunk_id=str(CHUNK),
        document_name="rapor.pdf"),), persist=True)
    assert payload[0]["evidence_ref"] == api._b64url(digest)
    assert conn.cur.params == (digest, CHUNK)
    assert conn.commits == 1


def test_architecture_authority_alone_never_becomes_content_authority():
    with pytest.raises(HTTPException) as caught:
        api.require_evidence_actor(_principal("org_architect", architect=True))
    assert caught.value.status_code == 403


def test_evidence_browser_surface_is_post_only_and_body_carried():
    routes = {
        route.path: set(route.methods or ())
        for route in api.app.routes
        if route.path.startswith("/v1/evidence/")
    }
    assert routes == {
        "/v1/evidence/tickets": {"POST"},
        "/v1/evidence/preview": {"POST"},
    }
    schema = api.app.openapi()
    ticket = schema["paths"]["/v1/evidence/tickets"]["post"]
    preview = schema["paths"]["/v1/evidence/preview"]["post"]
    assert "requestBody" in ticket and "parameters" not in ticket
    assert "requestBody" in preview and "parameters" not in preview


def test_an_altered_reference_has_no_mapping_and_is_rejected_closed(
        monkeypatch):
    reference = api._evidence_reference(str(CHUNK))
    altered = ("A" if reference[0] != "A" else "B") + reference[1:]

    @contextmanager
    def connection():
        yield object()

    seen = {}

    def refuse(_conn, **kwargs):
        seen.update(kwargs)
        raise db.EvidenceAccessRefused("no mapping")

    monkeypatch.setattr(api, "db_conn", connection)
    monkeypatch.setattr(api.db, "mint_evidence_preview_ticket", refuse)
    with pytest.raises(HTTPException) as caught:
        api.create_evidence_ticket(
            api.EvidenceTicketRequest(evidence_ref=altered),
            Response(), _principal())
    assert caught.value.status_code == 404
    assert seen["ref_digest"] == api._evidence_digest(altered)


def test_ticket_endpoint_binds_actor_chunk_digest_and_closed_lifetime(
        monkeypatch):
    conn = _Conn({"expires_at": datetime.now(timezone.utc)})

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(api, "db_conn", connection)
    response = Response()
    result = api.create_evidence_ticket(
        api.EvidenceTicketRequest(
            evidence_ref=api._evidence_reference(str(CHUNK))),
        response, _principal())
    assert result["expires_in"] == 50
    assert len(result["ticket"]) == 43
    assert conn.cur.params == (
        hashlib.sha256(result["ticket"].encode("ascii")).digest(),
        ACTOR, 50, ACTOR,
        api._evidence_digest(api._evidence_reference(str(CHUNK))))
    assert conn.commits == 1
    assert response.headers["cache-control"] == "no-store"


def test_preview_endpoint_returns_only_the_bounded_passage_contract(monkeypatch):
    conn = _Conn({"document_name": "rapor.pdf", "page": 9,
                  "passage": "yalniz ilgili pasaj"})

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(api, "db_conn", connection)
    response = Response()
    ticket = "A" * 43
    result = api.preview_evidence(
        api.EvidencePreviewRequest(ticket=ticket), response, _principal())
    assert result == {
        "document_name": "rapor.pdf",
        "page": 9,
        "content_type": "passage",
        "passage": "yalniz ilgili pasaj",
    }
    assert conn.cur.params == (
        ACTOR, hashlib.sha256(b"A" * 43).digest(), ACTOR, 4000)
    assert conn.commits == 1
    assert response.headers["cache-control"] == "no-store"


class _Cursor:
    def __init__(self, row):
        self.row = row
        self.sql = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        self.sql = sql
        self.params = params

    def fetchone(self):
        return self.row


class _Conn:
    def __init__(self, row):
        self.cur = _Cursor(row)
        self.commits = 0

    def cursor(self, **_kwargs):
        return self.cur

    def commit(self):
        self.commits += 1


def test_reference_mapping_is_active_resource_bound_and_committed():
    conn = _Conn({"ref_digest": b"r" * 32})
    db.register_evidence_references(conn, ((b"r" * 32, CHUNK),))
    assert "INSERT INTO evidence_references" in conn.cur.sql
    assert "d.archived_at IS NULL" in conn.cur.sql
    assert "c.generation = d.active_generation" in conn.cur.sql
    assert "c.version_id = d.active_version_id" in conn.cur.sql
    assert "ON CONFLICT (tenant_id, chunk_id)" in conn.cur.sql
    assert "ref_digest = EXCLUDED.ref_digest" in conn.cur.sql
    assert conn.cur.params == (b"r" * 32, CHUNK)
    assert conn.commits == 1


def test_db_mint_is_a_fresh_rls_membership_and_active_resource_check():
    conn = _Conn({"expires_at": datetime.now(timezone.utc)})
    digest = b"x" * 32
    db.mint_evidence_preview_ticket(
        conn, actor_id=ACTOR, ref_digest=b"r" * 32, token_digest=digest,
        ttl_seconds=50)
    sql = conn.cur.sql
    assert "JOIN org_memberships" in sql
    assert "m.state = 'active'" in sql
    assert "m.app_role IN ('reader', 'editor', 'admin')" in sql
    assert "d.archived_at IS NULL" in sql
    assert "c.generation = d.active_generation" in sql
    assert "c.version_id = d.active_version_id" in sql
    assert "ON CONFLICT (tenant_id, actor_id, purpose)" in sql
    assert "org_architects" not in sql
    assert conn.cur.params == (digest, ACTOR, 50, ACTOR, b"r" * 32)
    assert conn.commits == 1


def test_db_consume_is_single_statement_single_use_and_bounded():
    conn = _Conn({"document_name": "rapor.pdf", "page": 2,
                  "passage": "kanit"})
    result = db.consume_evidence_preview_ticket(
        conn, actor_id=ACTOR, token_digest=b"y" * 32,
        passage_max_chars=4000)
    assert result == {"document_name": "rapor.pdf", "page": 2,
                      "passage": "kanit"}
    sql = conn.cur.sql
    assert "SET consumed_at = now()" in sql
    assert "t.consumed_at IS NULL" in sql
    assert "t.expires_at > now()" in sql
    assert "t.purpose = 'preview'" in sql
    assert "t.actor_id = %s" in sql
    assert "t.tenant_id = c.tenant_id" in sql
    assert "d.archived_at IS NULL" in sql
    assert "c.generation = d.active_generation" in sql
    assert "c.version_id = d.active_version_id" in sql
    assert "left(c.text, %s)" in sql
    assert "source_path" not in sql
    assert conn.cur.params == (ACTOR, b"y" * 32, ACTOR, 4000)
    assert conn.commits == 1


@pytest.mark.parametrize(
    "closed_reason",
    ["cross_tenant", "inactive_membership", "archived_document",
     "stale_non_active_version"],
)
def test_mint_refuses_every_fresh_acl_or_resource_miss(closed_reason):
    # PostgreSQL RLS and the one INSERT..SELECT predicate collapse all of these
    # misses to the same closed absence.  The label keeps the threat matrix
    # explicit without inventing a different observable error for each secret.
    assert closed_reason
    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            _Conn(None), actor_id=ACTOR, ref_digest=b"r" * 32,
            token_digest=b"z" * 32, ttl_seconds=50)


@pytest.mark.parametrize(
    "closed_reason", ["other_actor", "expired", "replay"],
)
def test_consume_refuses_actor_lifetime_and_single_use_misses(closed_reason):
    assert closed_reason
    with pytest.raises(db.EvidenceAccessRefused):
        db.consume_evidence_preview_ticket(
            _Conn(None), actor_id=ACTOR, token_digest=b"w" * 32)


def test_ticket_schema_is_closed_tenant_isolated_and_content_free():
    sql = (api.Path(db.__file__).with_name("schema.sql")
           .read_text(encoding="utf-8"))
    table = sql.split(
        "CREATE TABLE IF NOT EXISTS evidence_preview_tickets", 1)[1]
    table = table.split(");", 1)[0]
    assert "token_digest bytea PRIMARY KEY" in table
    assert "actor_id" in table and "tenant_id" in table
    assert "purpose" in table and "expires_at" in table
    assert "UNIQUE (tenant_id, actor_id, purpose)" in table
    assert "FOREIGN KEY (tenant_id, chunk_id)" in table
    assert "REFERENCES chunks(tenant_id, id) ON DELETE CASCADE" in table
    for forbidden in ("passage", "source_path", "question", "answer"):
        assert forbidden not in table
    assert "ALTER TABLE evidence_preview_tickets FORCE ROW LEVEL SECURITY" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS chunks_tenant_id_key" in sql
    assert "CREATE POLICY tenant_isolation ON evidence_preview_tickets" in sql
