"""Real PostgreSQL proof for actor-bound, single-use evidence tickets."""
import hashlib
import os
from pathlib import Path
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_EVIDENCE_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_EVIDENCE_PG_DSN is absent")
TENANT = uuid.UUID("73000000-0000-0000-0000-000000000001")
OTHER_TENANT = uuid.UUID("73000000-0000-0000-0000-000000000002")
ACTOR = uuid.UUID("73000000-0000-0000-0000-000000000010")
OTHER_ACTOR = uuid.UUID("73000000-0000-0000-0000-000000000011")
ARCHITECT = uuid.UUID("73000000-0000-0000-0000-000000000012")
POSITION = uuid.UUID("73000000-0000-0000-0000-000000000020")
OTHER_POSITION = uuid.UUID("73000000-0000-0000-0000-000000000021")
DOCUMENT = uuid.UUID("73000000-0000-0000-0000-000000000030")
CHUNK = uuid.UUID("73000000-0000-0000-0000-000000000040")
REF_DIGEST = hashlib.sha256(b"persistent-opaque-reference").digest()


@pytest.fixture(scope="module")
def evidence_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_evidence_" + uuid.uuid4().hex[:12]
    role = "ragtest_evidence_role_" + uuid.uuid4().hex[:12]
    password = "evidence-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cur.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        conn = psycopg.connect(DSN, user=role, password=password)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
            cur.execute(Path(db.__file__).with_name("schema.sql").read_text(
                encoding="utf-8"))
        conn.commit()
        db.set_tenant_context(conn, TENANT, service=True)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO org_tenants (id, name) VALUES (%s, %s)",
                [(TENANT, "Tenant"), (OTHER_TENANT, "Other")])
            cur.execute(
                "INSERT INTO org_positions "
                "(id, tenant_id, title, kind, can_monitor_descendants, "
                "protected_from_monitoring) "
                "VALUES (%s, %s, 'Root', 'root', true, true)",
                (POSITION, TENANT))
            cur.execute(
                "INSERT INTO org_positions "
                "(id, tenant_id, parent_id, title, kind) "
                "VALUES (%s, %s, %s, 'Member', 'member')",
                (OTHER_POSITION, TENANT, POSITION))
            for actor, subject in ((ACTOR, "actor"),
                                   (OTHER_ACTOR, "other"),
                                   (ARCHITECT, "architect")):
                cur.execute(
                    "INSERT INTO org_identities (id, issuer, subject) "
                    "VALUES (%s, 'open-webui', %s)", (actor, subject))
            for actor, position in ((ACTOR, POSITION),
                                    (OTHER_ACTOR, OTHER_POSITION)):
                cur.execute(
                    "INSERT INTO org_memberships "
                    "(tenant_id, identity_id, position_id, display_label, "
                    "app_role, state) VALUES (%s, %s, %s, 'member', "
                    "'reader', 'active')", (TENANT, actor, position))
            cur.execute(
                "INSERT INTO org_architects (tenant_id, identity_id) "
                "VALUES (%s, %s)", (TENANT, ARCHITECT))
            cur.execute(
                "INSERT INTO documents "
                "(id, tenant_id, filename, file_type, status, active_generation) "
                "VALUES (%s, %s, 'evidence.pdf', 'pdf', 'ready', 1)",
                (DOCUMENT, TENANT))
            cur.execute(
                "INSERT INTO chunks "
                "(id, tenant_id, document_id, type, text, source_tag, page, "
                "headings, dense, sparse, generation) VALUES "
                "(%s, %s, %s, 'text', 'bounded evidence passage', 'page:4', "
                "4, '[]'::jsonb, %s::vector, %s::sparsevec, 1)",
                (CHUNK, TENANT, DOCUMENT,
                 "[" + ",".join(["0"] * 1024) + "]",
                 "{1:1}/999999937"))
        conn.commit()
        db.set_tenant_context(conn, TENANT)
        db.register_evidence_references(conn, ((REF_DIGEST, CHUNK),))
        yield conn
    finally:
        if conn is not None:
            conn.close()
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cur.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        finally:
            admin.close()


def _digest(label):
    return hashlib.sha256(label.encode("ascii")).digest()


def test_ticket_is_actor_bound_short_lived_and_single_use(evidence_database):
    conn = evidence_database
    db.mint_evidence_preview_ticket(
        conn, actor_id=ACTOR, ref_digest=REF_DIGEST,
        token_digest=_digest("ticket-one"), ttl_seconds=50)
    # A second mint atomically revokes the first and replaces the actor's one
    # bounded row; repeated clicks cannot grow the table without bound.
    db.mint_evidence_preview_ticket(
        conn, actor_id=ACTOR, ref_digest=REF_DIGEST,
        token_digest=_digest("ticket-two"), ttl_seconds=50)
    with pytest.raises(db.EvidenceAccessRefused):
        db.consume_evidence_preview_ticket(
            conn, actor_id=ACTOR, token_digest=_digest("ticket-one"))
    with pytest.raises(db.EvidenceAccessRefused):
        db.consume_evidence_preview_ticket(
            conn, actor_id=OTHER_ACTOR,
            token_digest=_digest("ticket-two"))
    preview = db.consume_evidence_preview_ticket(
        conn, actor_id=ACTOR, token_digest=_digest("ticket-two"))
    assert preview == {"document_name": "evidence.pdf", "page": 4,
                       "passage": "bounded evidence passage"}
    with pytest.raises(db.EvidenceAccessRefused):
        db.consume_evidence_preview_ticket(
            conn, actor_id=ACTOR, token_digest=_digest("ticket-two"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM evidence_preview_tickets "
            "WHERE tenant_id = %s AND actor_id = %s", (TENANT, ACTOR))
        assert cur.fetchone()[0] == 1


def test_cross_tenant_and_architect_only_cannot_mint(evidence_database):
    conn = evidence_database
    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            conn, actor_id=ARCHITECT, ref_digest=REF_DIGEST,
            token_digest=_digest("architect"))
    conn.rollback()
    db.set_tenant_context(conn, OTHER_TENANT)
    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            conn, actor_id=ACTOR, ref_digest=REF_DIGEST,
            token_digest=_digest("cross-tenant"))
    conn.rollback()
    db.set_tenant_context(conn, TENANT)


def test_archived_or_stale_active_generation_refuses_mint(evidence_database):
    conn = evidence_database
    with conn.cursor() as cur:
        cur.execute("UPDATE chunks SET generation = 2 WHERE id = %s", (CHUNK,))
    conn.commit()
    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            conn, actor_id=ACTOR, ref_digest=REF_DIGEST,
            token_digest=_digest("stale"))
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("UPDATE chunks SET generation = 1 WHERE id = %s", (CHUNK,))
        cur.execute("UPDATE documents SET archived_at = now() WHERE id = %s",
                    (DOCUMENT,))
    conn.commit()
    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            conn, actor_id=ACTOR, ref_digest=REF_DIGEST,
            token_digest=_digest("archived"))


def test_inactive_membership_and_expired_ticket_are_refused(evidence_database):
    conn = evidence_database
    # The preceding test leaves the document archived; restore it through the
    # product seam before exercising identity/lifetime gates.
    db.set_document_archived(conn, str(DOCUMENT), False)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'suspended' "
            "WHERE tenant_id = %s AND identity_id = %s", (TENANT, ACTOR))
    conn.commit()
    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            conn, actor_id=ACTOR, ref_digest=REF_DIGEST,
            token_digest=_digest("inactive"))
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'active' "
            "WHERE tenant_id = %s AND identity_id = %s", (TENANT, ACTOR))
    conn.commit()
    digest = _digest("expired")
    db.mint_evidence_preview_ticket(
        conn, actor_id=ACTOR, ref_digest=REF_DIGEST, token_digest=digest)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE evidence_preview_tickets "
            "SET created_at = now() - interval '100 seconds', "
            "expires_at = now() - interval '1 second' "
            "WHERE token_digest = %s", (digest,))
    conn.commit()
    with pytest.raises(db.EvidenceAccessRefused):
        db.consume_evidence_preview_ticket(
            conn, actor_id=ACTOR, token_digest=digest)


def test_registering_a_rotated_digest_replaces_only_that_chunks_old_reference(
        evidence_database):
    conn = evidence_database
    old_digest = _digest("rotation-old-reference")
    new_digest = _digest("rotation-new-reference")

    db.register_evidence_references(conn, ((old_digest, CHUNK),))
    db.mint_evidence_preview_ticket(
        conn, actor_id=ACTOR, ref_digest=old_digest,
        token_digest=_digest("rotation-old-ticket"))

    db.register_evidence_references(conn, ((new_digest, CHUNK),))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ref_digest FROM evidence_references "
            "WHERE tenant_id = %s AND chunk_id = %s", (TENANT, CHUNK))
        assert cur.fetchall() == [(new_digest,)]

    with pytest.raises(db.EvidenceAccessRefused):
        db.mint_evidence_preview_ticket(
            conn, actor_id=ACTOR, ref_digest=old_digest,
            token_digest=_digest("rotation-refused-ticket"))
    conn.rollback()
    db.mint_evidence_preview_ticket(
        conn, actor_id=ACTOR, ref_digest=new_digest,
        token_digest=_digest("rotation-new-ticket"))
