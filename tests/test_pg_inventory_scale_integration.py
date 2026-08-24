"""Real PostgreSQL proof that deep inventory pages use bounded access paths.

The ordinary local suite skips this module when no disposable server is
supplied. CI already supplies ``RAGTEST_OPERATIONS_PG_DSN``; this test turns
the scale claims into measured plans instead of comments beside indexes.
"""
import os
from pathlib import Path
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_OPERATIONS_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_OPERATIONS_PG_DSN is absent")

TENANT = uuid.UUID("a7000000-0000-4000-8000-000000000001")
ROW_COUNT = 25000


def _plan_nodes(plan):
    yield plan
    for child in plan.get("Plans", ()):
        yield from _plan_nodes(child)


def _index_names(explain_rows):
    plan = explain_rows[0][0][0]["Plan"]
    return {
        node["Index Name"] for node in _plan_nodes(plan)
        if node.get("Index Name")
    }


def _node_types(explain_rows):
    plan = explain_rows[0][0][0]["Plan"]
    return {node["Node Type"] for node in _plan_nodes(plan)}


@pytest.fixture(scope="module")
def inventory_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_inventory_scale_" + uuid.uuid4().hex[:10]
    role = "ragtest_inventory_role_" + uuid.uuid4().hex[:10]
    password = "inventory-scale-only"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cursor.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        connection = psycopg.connect(DSN, user=role, password=password)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
            cursor.execute(Path(db.__file__).with_name("schema.sql").read_text(
                encoding="utf-8"))
        connection.commit()
        db.set_tenant_context(connection, TENANT)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents
                    (id, filename, file_type, uploaded_at, status, tenant_id,
                     archived_at)
                SELECT
                    md5('inventory-scale-' || item::text)::uuid,
                    'inventory-document-' || item::text || '.pdf',
                    CASE WHEN item %% 1000 = 0 THEN 'rare-type'
                         ELSE 'pdf' END,
                    timestamptz '2026-08-24 12:00:00+00'
                        - item * interval '1 second',
                    CASE WHEN item %% 1000 = 0 THEN 'rare'
                         ELSE 'done' END,
                    %s,
                    CASE WHEN item %% 777 = 0 THEN now() ELSE NULL END
                FROM generate_series(1, %s) AS item
                """,
                (TENANT, ROW_COUNT),
            )
            cursor.execute("ANALYZE documents")
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


def _boundary(connection, item):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT uploaded_at, id FROM documents "
            "WHERE filename = %s",
            (f"inventory-document-{item}.pdf",),
        )
        return cursor.fetchone()


def test_a_deep_cursor_page_is_contiguous_and_uses_the_active_order_index(
        inventory_database):
    connection = inventory_database
    boundary_time, boundary_id = _boundary(connection, 15000)

    rows = db.list_documents(
        connection, limit=5, offset=0,
        before=(boundary_time, str(boundary_id)), tenant_id=str(TENANT))

    assert [row["filename"] for row in rows] == [
        f"inventory-document-{item}.pdf" for item in range(15001, 15007)]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            EXPLAIN (FORMAT JSON)
            SELECT id FROM documents
            WHERE archived_at IS NULL
              AND tenant_id = %s::uuid
              AND (uploaded_at, id) < (%s, %s::uuid)
            ORDER BY uploaded_at DESC, id DESC
            LIMIT 6
            """,
            (TENANT, boundary_time, boundary_id),
        )
        plan = cursor.fetchall()
    assert "documents_tenant_active_inventory_idx" in _index_names(plan), plan
    assert "Seq Scan" not in _node_types(plan), plan


def test_a_selective_status_cursor_uses_its_partial_index(inventory_database):
    connection = inventory_database
    boundary_time, boundary_id = _boundary(connection, 9000)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            EXPLAIN (FORMAT JSON)
            SELECT id FROM documents
            WHERE archived_at IS NULL AND status = 'rare'
              AND tenant_id = %s::uuid
              AND (uploaded_at, id) < (%s, %s::uuid)
            ORDER BY uploaded_at DESC, id DESC
            LIMIT 6
            """,
            (TENANT, boundary_time, boundary_id),
        )
        plan = cursor.fetchall()
    assert "documents_tenant_active_status_inventory_idx" in _index_names(plan), plan
    assert "Seq Scan" not in _node_types(plan), plan


def test_a_selective_file_type_cursor_uses_its_partial_index(
        inventory_database):
    connection = inventory_database
    boundary_time, boundary_id = _boundary(connection, 9000)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            EXPLAIN (FORMAT JSON)
            SELECT id FROM documents
            WHERE archived_at IS NULL AND file_type = 'rare-type'
              AND tenant_id = %s::uuid
              AND (uploaded_at, id) < (%s, %s::uuid)
            ORDER BY uploaded_at DESC, id DESC
            LIMIT 6
            """,
            (TENANT, boundary_time, boundary_id),
        )
        plan = cursor.fetchall()
    assert "documents_tenant_active_type_inventory_idx" in _index_names(plan), plan
    assert "Seq Scan" not in _node_types(plan), plan


def test_archived_pages_use_their_partial_order_index(inventory_database):
    connection = inventory_database
    boundary_time, boundary_id = _boundary(connection, 9000)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            EXPLAIN (FORMAT JSON)
            SELECT id FROM documents
            WHERE archived_at IS NOT NULL
              AND tenant_id = %s::uuid
              AND (uploaded_at, id) < (%s, %s::uuid)
            ORDER BY uploaded_at DESC, id DESC
            LIMIT 6
            """,
            (TENANT, boundary_time, boundary_id),
        )
        plan = cursor.fetchall()
    assert "documents_tenant_archived_inventory_idx" in _index_names(plan), plan
    assert "Seq Scan" not in _node_types(plan), plan
