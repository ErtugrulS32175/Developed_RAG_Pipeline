"""Real PostgreSQL proof for retention, hold and terminal purge semantics."""
import hashlib
import os
import uuid

import pytest

from pipeline.index import db, publication


DSN = os.getenv("RAGTEST_TENANT_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_TENANT_PG_DSN is absent")

TENANT = uuid.UUID("44000000-0000-4000-8000-000000000001")
ARCHITECT = uuid.UUID("44000000-0000-4000-8000-000000000002")
POSITION = uuid.UUID("44000000-0000-4000-8000-000000000003")
OTHER_TENANT = uuid.UUID("45000000-0000-4000-8000-000000000001")
OTHER_DOCUMENT = uuid.UUID("45000000-0000-4000-8000-000000000002")


@pytest.fixture(scope="module")
def retention_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_retention_" + uuid.uuid4().hex[:12]
    role = "ragtest_retention_role_" + uuid.uuid4().hex[:12]
    password = "retention-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cursor.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        connection = psycopg.connect(DSN, user=role, password=password)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
        connection.commit()
        db.init_schema(connection)
        db.set_tenant_context(connection, TENANT, service=True)
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO org_tenants (id, name) VALUES (%s, 'Retention')",
                (TENANT,))
            cursor.execute(
                "INSERT INTO org_positions "
                "(id, tenant_id, parent_id, title, kind, "
                "can_monitor_descendants, protected_from_monitoring) "
                "VALUES (%s, %s, NULL, 'Root', 'root', true, true)",
                (POSITION, TENANT))
            cursor.execute(
                "INSERT INTO org_identities (id, issuer, subject) "
                "VALUES (%s, 'open-webui', 'retention-architect')",
                (ARCHITECT,))
            cursor.execute(
                "INSERT INTO org_memberships "
                "(tenant_id, identity_id, position_id, display_label, "
                "app_role, state) VALUES (%s, %s, %s, 'Architect', "
                "'admin', 'active')", (TENANT, ARCHITECT, POSITION))
            cursor.execute(
                "INSERT INTO org_architects (tenant_id, identity_id) "
                "VALUES (%s, %s)", (TENANT, ARCHITECT))
        connection.commit()
        yield connection
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cursor.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        finally:
            admin.close()


def _actor_context(connection):
    db.set_tenant_context(connection, TENANT, actor_id=ARCHITECT)
    connection.commit()


def _document_fixture(connection):
    document = uuid.uuid4()
    version = uuid.uuid4()
    chunk = uuid.uuid4()
    digest = "a" * 64
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO documents "
            "(id, tenant_id, filename, file_type, status, revision) "
            "VALUES (%s, %s, %s, 'pdf', 'ready', 0)",
            (document, TENANT, f"retention-{document.hex}.pdf"))
        cursor.execute(
            "INSERT INTO document_versions "
            "(id, tenant_id, document_id, version_number, content_sha256) "
            "VALUES (%s, %s, %s, 1, %s)",
            (version, TENANT, document, digest))
        cursor.execute(
            "INSERT INTO chunks "
            "(id, tenant_id, document_id, version_id, type, text, "
            "source_tag, page, headings, dense, sparse, generation) VALUES "
            "(%s, %s, %s, %s, 'text', 'fixture passage', 'page:1', 1, "
            "'[]'::jsonb, %s::vector, %s::sparsevec, 1)",
            (chunk, TENANT, document, version,
             "[" + ",".join(["0"] * 1024) + "]",
             "{1:1}/999999937"))
    connection.commit()
    return document, version, chunk, digest


def _policy(connection, days):
    _actor_context(connection)
    current = db.get_tenant_retention_policy(
        connection, actor_id=ARCHITECT)
    return db.update_tenant_retention_policy(
        connection, actor_id=ARCHITECT,
        archive_retention_days=days,
        expected_revision=current["revision"],
        expected_policy_epoch=current["policy_epoch"],
        request_id="policy-" + uuid.uuid4().hex[:20])


def test_policy_is_cas_guarded_and_appends_immutable_evidence(
        retention_database):
    updated = _policy(retention_database, 30)
    assert updated["archive_retention_days"] == 30
    assert updated["revision"] >= 2

    with pytest.raises(db.RetentionPolicyConflict):
        db.update_tenant_retention_policy(
            retention_database, actor_id=ARCHITECT,
            archive_retention_days=31,
            expected_revision=updated["revision"] - 1,
            expected_policy_epoch=updated["policy_epoch"],
            request_id="stale-" + uuid.uuid4().hex[:20])
    retention_database.rollback()

    with retention_database.cursor() as cursor:
        cursor.execute(
            "SELECT event_type FROM document_retention_events "
            "WHERE event_type = 'policy_changed'")
        assert cursor.fetchall() == [("policy_changed",)]
        with pytest.raises(Exception, match="immutable"):
            cursor.execute(
                "UPDATE document_retention_events SET request_id = "
                "'replacement-request' WHERE event_type = 'policy_changed'")
    retention_database.rollback()


def test_hold_retention_and_worker_purge_form_one_terminal_chain(
        retention_database, tmp_path):
    policy = _policy(retention_database, 1)
    _actor_context(retention_database)
    document, version, chunk, _digest = _document_fixture(retention_database)
    filename = f"retention-{document.hex}.pdf"

    archived = db.set_document_archived(
        retention_database, str(document), True)
    assert archived["archived"] is True
    with pytest.raises(db.DocumentRetentionRefused, match="retention"):
        db.schedule_document_purge(
            retention_database, actor_id=ARCHITECT,
            document_id=document, expected_revision=0,
            expected_policy_epoch=policy["policy_epoch"],
            request_id="early-" + uuid.uuid4().hex[:20])
    retention_database.rollback()
    with retention_database.cursor() as cursor:
        cursor.execute(
            "UPDATE documents SET archived_at = now() - interval '2 days' "
            "WHERE id = %s", (document,))
    retention_database.commit()

    hold = db.create_document_legal_hold(
        retention_database, actor_id=ARCHITECT, document_id=document,
        reason_code="litigation", expected_revision=0,
        expected_policy_epoch=policy["policy_epoch"],
        request_id="hold-" + uuid.uuid4().hex[:20])
    with pytest.raises(db.DocumentRetentionRefused, match="legal hold"):
        db.schedule_document_purge(
            retention_database, actor_id=ARCHITECT,
            document_id=document, expected_revision=0,
            expected_policy_epoch=hold["policy_epoch"],
            request_id="held-" + uuid.uuid4().hex[:20])
    retention_database.rollback()

    released = db.release_document_legal_hold(
        retention_database, actor_id=ARCHITECT, document_id=document,
        hold_id=hold["id"], expected_revision=hold["revision"],
        expected_policy_epoch=hold["policy_epoch"],
        request_id="release-" + uuid.uuid4().hex[:20])
    cancelled_job = db.schedule_document_purge(
        retention_database, actor_id=ARCHITECT, document_id=document,
        expected_revision=0,
        expected_policy_epoch=released["policy_epoch"],
        request_id="purge-" + uuid.uuid4().hex[:20])
    assert cancelled_job["state"] == "pending"
    second_hold = db.create_document_legal_hold(
        retention_database, actor_id=ARCHITECT, document_id=document,
        reason_code="regulatory", expected_revision=0,
        expected_policy_epoch=released["policy_epoch"],
        request_id="hold-two-" + uuid.uuid4().hex[:16])
    with retention_database.cursor() as cursor:
        cursor.execute(
            "SELECT state FROM document_purge_jobs WHERE id = %s",
            (cancelled_job["id"],))
        assert cursor.fetchone()[0] == "cancelled"
    second_release = db.release_document_legal_hold(
        retention_database, actor_id=ARCHITECT, document_id=document,
        hold_id=second_hold["id"],
        expected_revision=second_hold["revision"],
        expected_policy_epoch=second_hold["policy_epoch"],
        request_id="release-two-" + uuid.uuid4().hex[:13])
    job = db.schedule_document_purge(
        retention_database, actor_id=ARCHITECT, document_id=document,
        expected_revision=0,
        expected_policy_epoch=second_release["policy_epoch"],
        request_id="purge-two-" + uuid.uuid4().hex[:15])
    assert job["state"] == "pending"

    object_root = tmp_path / "uploads"
    object_root.mkdir()
    publication.publish_version_source(
        object_root, TENANT, document, version, b"fixture",
        expected_sha256=hashlib.sha256(b"fixture").hexdigest())
    legacy = publication.tenant_upload_root(object_root, TENANT) / filename
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"fixture")

    db.set_tenant_context(retention_database, TENANT, service=True)
    retention_database.commit()
    claimed = db.claim_document_purge(
        retention_database, "integration-worker")
    assert claimed["id"] == job["id"]
    assert claimed["version_ids"] == (str(version),)
    removed = publication.purge_document_sources(
        object_root, TENANT, document, claimed["version_ids"], filename)
    assert removed.removed_version_sources == 1
    assert removed.removed_legacy_source is True
    completed = db.complete_document_purge(
        retention_database, job_id=job["id"],
        worker_id="integration-worker")
    assert completed["state"] == "completed"
    assert completed["removed_chunks"] == 1

    _actor_context(retention_database)
    with retention_database.cursor() as cursor:
        cursor.execute(
            "SELECT status, purged_at, active_version_id, content_sha256, "
            "filename FROM documents WHERE id = %s", (document,))
        tombstone = cursor.fetchone()
        cursor.execute("SELECT count(*) FROM chunks WHERE id = %s", (chunk,))
        assert cursor.fetchone()[0] == 0
        cursor.execute(
            "SELECT count(*) FROM document_versions WHERE id = %s", (version,))
        assert cursor.fetchone()[0] == 1
        cursor.execute(
            "SELECT event_type FROM document_retention_events "
            "WHERE document_id = %s ORDER BY created_at, id", (document,))
        assert {row[0] for row in cursor.fetchall()} == {
            "hold_created", "hold_released", "purge_scheduled",
            "purge_completed",
        }
        cursor.execute(
            "SELECT action, decision FROM org_audit_events "
            "WHERE request_id = %s", (claimed["request_id"],))
        assert ("purge_execute", "allowed") in cursor.fetchall()
    assert tombstone[0] == "purged"
    assert tombstone[1] is not None
    assert tombstone[2] is None and tombstone[3] is None
    assert tombstone[4] == "purged-" + str(document)
    assert not publication.version_source_path(
        object_root, TENANT, document, version).exists()

    with pytest.raises(db.DocumentLifecycleConflict, match="purged"):
        db.set_document_archived(retention_database, str(document), False)
    retention_database.rollback()
    with pytest.raises(Exception, match="immutable"):
        with retention_database.cursor() as cursor:
            cursor.execute(
                "UPDATE documents SET status = 'ready' WHERE id = %s",
                (document,))
    retention_database.rollback()


def test_cross_tenant_document_identity_cannot_create_or_reveal_a_hold(
        retention_database):
    db.set_tenant_context(retention_database, TENANT, service=True)
    with retention_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO org_tenants (id, name) VALUES (%s, 'Other')",
            (OTHER_TENANT,))
        cursor.execute(
            "INSERT INTO documents "
            "(id, tenant_id, filename, file_type, status, revision) "
            "VALUES (%s, %s, 'other-retention.pdf', 'pdf', 'ready', 0)",
            (OTHER_DOCUMENT, OTHER_TENANT))
    retention_database.commit()
    _actor_context(retention_database)
    policy_epoch = db.get_tenant_retention_policy(
        retention_database, actor_id=ARCHITECT)["policy_epoch"]

    assert db.list_document_legal_holds(
        retention_database, actor_id=ARCHITECT,
        document_id=OTHER_DOCUMENT) == []
    with pytest.raises(db.DocumentRetentionRefused, match="bulunamadi"):
        db.create_document_legal_hold(
            retention_database, actor_id=ARCHITECT,
            document_id=OTHER_DOCUMENT, reason_code="litigation",
            expected_revision=0, expected_policy_epoch=policy_epoch,
            request_id="foreign-" + uuid.uuid4().hex[:20])
    retention_database.rollback()
