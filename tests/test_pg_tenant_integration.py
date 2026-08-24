"""Real PostgreSQL proof for RLS tenant separation and service claiming."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_TENANT_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_TENANT_PG_DSN is absent")

TENANT_A = uuid.UUID("10000000-0000-0000-0000-000000000001")
TENANT_B = uuid.UUID("20000000-0000-0000-0000-000000000002")
DOCUMENT_A = uuid.UUID("10000000-0000-0000-0000-000000000010")
DOCUMENT_B = uuid.UUID("20000000-0000-0000-0000-000000000020")


@pytest.fixture(scope="module")
def tenant_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_tenant_" + uuid.uuid4().hex[:12]
    role = "ragtest_role_" + uuid.uuid4().hex[:12]
    password = "tenant-integration-only"
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
        connection = psycopg.connect(
            DSN, user=role, password=password)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
        connection.commit()
        db.init_schema(connection)
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


def _tenant(connection, tenant, *, service=False):
    db.set_tenant_context(connection, tenant, service=service)
    connection.commit()


def _insert_document(connection, document_id, filename):
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO documents (id, filename, file_type, status) "
            "VALUES (%s, %s, 'pdf', 'ready')", (document_id, filename))
    connection.commit()


def test_two_tenants_can_own_the_same_name_but_never_read_each_other(
        tenant_database):
    _tenant(tenant_database, TENANT_A)
    _insert_document(tenant_database, DOCUMENT_A, "shared.pdf")
    _tenant(tenant_database, TENANT_B)
    _insert_document(tenant_database, DOCUMENT_B, "shared.pdf")

    with tenant_database.cursor() as cursor:
        cursor.execute("SELECT id, filename FROM documents ORDER BY id")
        assert cursor.fetchall() == [(DOCUMENT_B, "shared.pdf")]
    assert db.get_document(tenant_database, str(DOCUMENT_A)) is None

    _tenant(tenant_database, TENANT_A)
    with tenant_database.cursor() as cursor:
        cursor.execute("SELECT id, filename FROM documents ORDER BY id")
        assert cursor.fetchall() == [(DOCUMENT_A, "shared.pdf")]
    assert db.get_document(tenant_database, str(DOCUMENT_B)) is None


def test_service_context_sees_both_tenants_without_changing_row_ownership(
        tenant_database):
    _tenant(tenant_database, TENANT_A, service=True)
    with tenant_database.cursor() as cursor:
        cursor.execute("SELECT tenant_id, id FROM documents ORDER BY tenant_id")
        assert cursor.fetchall() == [
            (TENANT_A, DOCUMENT_A), (TENANT_B, DOCUMENT_B)]


def test_snapshot_scope_keys_do_not_collapse_same_named_tenant_documents(
        tenant_database):
    _tenant(tenant_database, TENANT_A)
    assert db.lock_retrieval_scope_keys(tenant_database) == [
        f"{TENANT_A}:{DOCUMENT_A}:legacy:0"]
    tenant_database.rollback()

    _tenant(tenant_database, TENANT_B)
    assert db.lock_retrieval_scope_keys(tenant_database) == [
        f"{TENANT_B}:{DOCUMENT_B}:legacy:0"]
    tenant_database.rollback()


def test_version_catalogue_and_activation_are_tenant_closed(tenant_database):
    _tenant(tenant_database, TENANT_A)
    document, version, _name = db.stage_candidate(
        tenant_database, "shared.pdf", "pdf", content_sha256="a" * 64,
        allow_replace=True)
    assert document == str(DOCUMENT_A)
    assert db.finalize_candidate_publication(
        tenant_database, document, version)
    assert [row["version_id"] for row in db.list_document_versions(
        tenant_database, document)] == [version]

    _tenant(tenant_database, TENANT_B)
    assert db.list_document_versions(tenant_database, document) == []
    assert db.document_version_source_digest(
        tenant_database, document, version) is None
    assert db.activate_document_version(
        tenant_database, document, version, 0,
        verified_source_sha256="a" * 64) is None
    tenant_database.rollback()


def test_composite_foreign_keys_refuse_cross_tenant_membership(tenant_database):
    collection = uuid.UUID("10000000-0000-0000-0000-000000000030")
    _tenant(tenant_database, TENANT_A, service=True)
    with tenant_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO collections (id, tenant_id, name, name_key) "
            "VALUES (%s, %s, 'A', 'a')", (collection, TENANT_A))
    tenant_database.commit()
    with pytest.raises(Exception) as caught:
        with tenant_database.cursor() as cursor:
            cursor.execute(
                "INSERT INTO collection_documents "
                "(tenant_id, collection_id, document_id) VALUES (%s, %s, %s)",
                (TENANT_A, collection, DOCUMENT_B))
    tenant_database.rollback()
    assert type(caught.value).__name__ == "ForeignKeyViolation"


def test_request_context_cannot_enable_service_access_with_a_tenant_value(
        tenant_database):
    _tenant(tenant_database, TENANT_B, service=False)
    with tenant_database.cursor() as cursor:
        cursor.execute("SELECT rag_service_access(), rag_effective_tenant()")
        assert cursor.fetchone() == (False, TENANT_B)
