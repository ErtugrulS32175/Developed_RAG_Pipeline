"""Real PostgreSQL proof for actor-bound, single-use table exports."""
import hashlib
import os
from pathlib import Path
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_OPERATIONS_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_OPERATIONS_PG_DSN is absent")
TENANT = uuid.UUID("76000000-0000-0000-0000-000000000001")
OTHER_TENANT = uuid.UUID("76000000-0000-0000-0000-000000000002")
ACTOR = uuid.UUID("76000000-0000-0000-0000-000000000010")
OTHER_ACTOR = uuid.UUID("76000000-0000-0000-0000-000000000011")
CROSS_ACTOR = uuid.UUID("76000000-0000-0000-0000-000000000012")
POSITION = uuid.UUID("76000000-0000-0000-0000-000000000020")
OTHER_POSITION = uuid.UUID("76000000-0000-0000-0000-000000000021")
PEER_POSITION = uuid.UUID("76000000-0000-0000-0000-000000000022")


@pytest.fixture(scope="module")
def export_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_export_" + uuid.uuid4().hex[:12]
    role = "ragtest_export_role_" + uuid.uuid4().hex[:12]
    password = "export-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
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
            cur.execute(
                "INSERT INTO rag_context_secrets (singleton, secret) "
                "VALUES (true, %s)", (db._context_secret(),))
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
                "(id, tenant_id, title, kind, can_monitor_descendants, "
                "protected_from_monitoring) "
                "VALUES (%s, %s, 'Other Root', 'root', true, true)",
                (OTHER_POSITION, OTHER_TENANT))
            cur.execute(
                "INSERT INTO org_positions "
                "(id, tenant_id, parent_id, title, kind) "
                "VALUES (%s, %s, %s, 'Peer', 'member')",
                (PEER_POSITION, TENANT, POSITION))
            for actor, subject in (
                (ACTOR, "actor"), (OTHER_ACTOR, "other"),
                (CROSS_ACTOR, "cross"),
            ):
                cur.execute(
                    "INSERT INTO org_identities (id, issuer, subject) "
                    "VALUES (%s, 'open-webui', %s)", (actor, subject))
            for tenant, actor, position in (
                (TENANT, ACTOR, POSITION),
                (TENANT, OTHER_ACTOR, PEER_POSITION),
                (OTHER_TENANT, CROSS_ACTOR, OTHER_POSITION),
            ):
                cur.execute(
                    "INSERT INTO org_memberships "
                    "(tenant_id, identity_id, position_id, display_label, "
                    "app_role, state) VALUES (%s, %s, %s, 'member', "
                    "'reader', 'active')", (tenant, actor, position))
        conn.commit()
        db.set_tenant_context(conn, TENANT)
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


def _register(conn, label):
    export_id = uuid.uuid4()
    ref_digest = _digest("ref-" + label)
    storage_name = hashlib.sha256(label.encode("ascii")).hexdigest()[:32] + ".xlsx"
    db.register_table_export(
        conn, export_id=export_id, actor_id=ACTOR,
        ref_digest=ref_digest, storage_name=storage_name,
        file_sha256=_digest("file-" + label), file_size=1234)
    return ref_digest


def test_ticket_is_exact_actor_bound_single_use_and_measured(export_database):
    conn = export_database
    ref_digest = _register(conn, "single-use")
    ticket = _digest("single-use-ticket")
    db.mint_table_export_ticket(
        conn, actor_id=ACTOR, ref_digest=ref_digest,
        token_digest=ticket, ttl_seconds=50)
    with pytest.raises(db.ExportAccessRefused):
        db.consume_table_export_ticket(
            conn, actor_id=OTHER_ACTOR, token_digest=ticket)
    measured = db.consume_table_export_ticket(
        conn, actor_id=ACTOR, token_digest=ticket)
    assert measured == {
        "storage_name": hashlib.sha256(b"single-use").hexdigest()[:32] + ".xlsx",
        "file_sha256": _digest("file-single-use"),
        "file_size": 1234,
    }
    with pytest.raises(db.ExportAccessRefused):
        db.consume_table_export_ticket(
            conn, actor_id=ACTOR, token_digest=ticket)


def test_another_actor_and_tenant_cannot_mint_from_the_reference(
        export_database):
    conn = export_database
    ref_digest = _register(conn, "actor-and-tenant")
    with pytest.raises(db.ExportAccessRefused):
        db.mint_table_export_ticket(
            conn, actor_id=OTHER_ACTOR, ref_digest=ref_digest,
            token_digest=_digest("other-actor"))
    db.set_tenant_context(conn, OTHER_TENANT)
    try:
        with pytest.raises(db.ExportAccessRefused):
            db.mint_table_export_ticket(
                conn, actor_id=CROSS_ACTOR, ref_digest=ref_digest,
                token_digest=_digest("other-tenant"))
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM table_exports")
            assert cur.fetchone()[0] == 0
    finally:
        db.set_tenant_context(conn, TENANT)


def test_inactive_membership_and_expired_export_are_closed(export_database):
    conn = export_database
    ref_digest = _register(conn, "lifetime")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'suspended' "
            "WHERE tenant_id = %s AND identity_id = %s", (TENANT, ACTOR))
    conn.commit()
    with pytest.raises(db.ExportAccessRefused):
        db.mint_table_export_ticket(
            conn, actor_id=ACTOR, ref_digest=ref_digest,
            token_digest=_digest("inactive"))
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'active' "
            "WHERE tenant_id = %s AND identity_id = %s", (TENANT, ACTOR))
        cur.execute(
            "UPDATE table_exports "
            "SET created_at = now() - interval '2 seconds', "
            "expires_at = now() - interval '1 second' "
            "WHERE ref_digest = %s", (ref_digest,))
    conn.commit()
    with pytest.raises(db.ExportAccessRefused):
        db.mint_table_export_ticket(
            conn, actor_id=ACTOR, ref_digest=ref_digest,
            token_digest=_digest("expired-export"))


def test_expired_ticket_cannot_be_consumed(export_database):
    conn = export_database
    ref_digest = _register(conn, "expired-ticket")
    ticket = _digest("expired-ticket-token")
    db.mint_table_export_ticket(
        conn, actor_id=ACTOR, ref_digest=ref_digest,
        token_digest=ticket, ttl_seconds=50)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE table_export_tickets "
            "SET created_at = now() - interval '2 seconds', "
            "expires_at = now() - interval '1 second' "
            "WHERE token_digest = %s", (ticket,))
    conn.commit()
    with pytest.raises(db.ExportAccessRefused):
        db.consume_table_export_ticket(
            conn, actor_id=ACTOR, token_digest=ticket)
