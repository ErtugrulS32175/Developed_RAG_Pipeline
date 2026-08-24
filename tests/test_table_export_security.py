"""Actor-bound, single-use access to generated table exports."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import uuid

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError

from pipeline.api import app as api
from pipeline.api import auth, owui_chat
from pipeline.index import db


TENANT = uuid.UUID("75000000-0000-0000-0000-000000000001")
ACTOR = uuid.UUID("75000000-0000-0000-0000-000000000002")
EXPORT = uuid.UUID("75000000-0000-0000-0000-000000000003")
STORAGE_NAME = "a" * 32 + ".xlsx"
FILE_BYTES = b"bounded-xlsx-placeholder"
FILE_SHA = hashlib.sha256(FILE_BYTES).digest()


def _principal(*, source="openwebui"):
    return auth.Principal(
        TENANT, "reader", subject_id=ACTOR, source=source,
        org_architect=False)


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


def test_export_surface_is_post_only_and_the_filename_route_is_gone():
    routes = {
        route.path: set(route.methods or ())
        for route in api.app.routes
        if route.path.startswith("/v1/exports/")
    }
    assert routes == {
        "/v1/exports/tickets": {"POST"},
        "/v1/exports/download": {"POST"},
    }
    assert not any(route.path == "/files/{name}" for route in api.app.routes)
    schema = api.app.openapi()
    for path in routes:
        operation = schema["paths"][path]["post"]
        assert "requestBody" in operation
        assert "parameters" not in operation


@pytest.mark.parametrize("model, field", [
    (api.ExportTicketRequest, "export_ref"),
    (api.ExportDownloadRequest, "ticket"),
])
def test_export_request_bodies_are_strict_closed_and_canonical(model, field):
    value = "A" * 43
    assert getattr(model(**{field: value}), field) == value
    with pytest.raises(ValidationError):
        model(**{field: value, "extra": "not accepted"})
    with pytest.raises(ValidationError):
        model(**{field: value[:-1] + "="})


def test_export_reference_is_opaque_and_has_no_reversible_file_name():
    export_id = api._export_id_for_storage(STORAGE_NAME)
    assert export_id == api._export_id_for_storage(STORAGE_NAME)
    assert export_id != api._export_id_for_storage("b" * 32 + ".xlsx")
    digest = api._export_reference_digest(export_id)
    reference = api._b64url(digest)
    assert len(digest) == 32 and len(reference) == 43
    assert str(EXPORT) not in reference
    assert STORAGE_NAME not in reference
    assert api._decode_export_reference(reference) == digest
    assert api._decode_export_reference(reference + "=") is None


def test_register_reference_measures_file_and_binds_the_openwebui_actor(
        tmp_path, monkeypatch):
    monkeypatch.setattr(owui_chat, "EXPORT_DIR", tmp_path)
    (tmp_path / STORAGE_NAME).write_bytes(FILE_BYTES)
    conn = _Conn({"ref_digest": b"r" * 32,
                  "expires_at": datetime.now(timezone.utc)})

    @contextmanager
    def connection(_principal_value):
        assert _principal_value == _principal()
        yield conn

    monkeypatch.setattr(api, "_principal_db_conn", connection)
    reference = api._register_export_reference(_principal(), STORAGE_NAME)
    assert len(reference) == 43
    assert conn.cur.params[1] == ACTOR
    assert conn.cur.params[3] == STORAGE_NAME
    assert conn.cur.params[4:] == (FILE_SHA, len(FILE_BYTES), 3600, ACTOR)
    assert conn.commits == 1
    assert api._register_export_reference(
        _principal(source="api_key"), STORAGE_NAME) is None


@pytest.mark.parametrize("name", [
    "../" + STORAGE_NAME, "B" * 32 + ".xlsx", "a.xlsx", "a" * 32,
])
def test_file_reader_refuses_every_name_it_did_not_issue(name):
    with pytest.raises(db.ExportAccessRefused):
        api._read_export_bytes(name)


def test_reader_refuses_to_publish_after_a_handle_close_failure(monkeypatch):
    monkeypatch.setattr(api.handle_transport, "open_root", lambda _path: object())
    monkeypatch.setattr(
        api.handle_transport, "open_child_file", lambda _root, _name: object())
    monkeypatch.setattr(
        api.handle_transport, "read_all", lambda _handle, _ceiling: FILE_BYTES)
    monkeypatch.setattr(
        api.handle_transport, "close_handle_quietly", lambda _handle: False)
    monkeypatch.setattr(
        api.handle_transport, "close_directory_quietly", lambda _root: True)
    with pytest.raises(db.ExportAccessRefused):
        api._read_export_bytes(STORAGE_NAME)


def test_ticket_endpoint_binds_actor_reference_and_short_lifetime(monkeypatch):
    conn = _Conn({"expires_at": datetime.now(timezone.utc)})

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(api, "db_conn", connection)
    response = Response()
    reference = api._b64url(api._export_reference_digest(EXPORT))
    result = api.create_export_ticket(
        api.ExportTicketRequest(export_ref=reference), response, _principal())
    assert result["expires_in"] == 50
    assert len(result["ticket"]) == 43
    assert conn.cur.params == (
        hashlib.sha256(result["ticket"].encode("ascii")).digest(),
        ACTOR, 50, ACTOR, api._decode_export_reference(reference), ACTOR)
    assert conn.commits == 1
    assert response.headers["cache-control"] == "no-store"


def test_download_consumes_before_read_and_checks_the_measurement(
        tmp_path, monkeypatch):
    monkeypatch.setattr(owui_chat, "EXPORT_DIR", tmp_path)
    (tmp_path / STORAGE_NAME).write_bytes(FILE_BYTES)
    seen = []

    def consume(_conn, **kwargs):
        seen.append(kwargs)
        return {"storage_name": STORAGE_NAME,
                "file_sha256": FILE_SHA, "file_size": len(FILE_BYTES)}

    @contextmanager
    def connection():
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    monkeypatch.setattr(api.db, "consume_table_export_ticket", consume)
    response = api.download_export(
        api.ExportDownloadRequest(ticket="A" * 43), _principal())
    assert seen == [{"actor_id": ACTOR,
                     "token_digest": hashlib.sha256(b"A" * 43).digest()}]
    assert response.body == FILE_BYTES
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == (
        'attachment; filename="table-export.xlsx"')
    assert STORAGE_NAME not in repr(response.headers)


@pytest.mark.parametrize("changed", ["file_size", "file_sha256"])
def test_download_refuses_a_file_changed_after_registration(
        changed, tmp_path, monkeypatch):
    monkeypatch.setattr(owui_chat, "EXPORT_DIR", tmp_path)
    (tmp_path / STORAGE_NAME).write_bytes(FILE_BYTES)
    measured = {"storage_name": STORAGE_NAME,
                "file_sha256": FILE_SHA, "file_size": len(FILE_BYTES)}
    measured[changed] = (len(FILE_BYTES) + 1 if changed == "file_size"
                         else hashlib.sha256(b"other").digest())

    @contextmanager
    def connection():
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    monkeypatch.setattr(
        api.db, "consume_table_export_ticket",
        lambda _conn, **_kwargs: measured)
    with pytest.raises(HTTPException) as caught:
        api.download_export(
            api.ExportDownloadRequest(ticket="A" * 43), _principal())
    assert caught.value.status_code == 404


@pytest.mark.parametrize("malformed", [
    {"storage_name": STORAGE_NAME, "file_sha256": None,
     "file_size": len(FILE_BYTES)},
    {"storage_name": STORAGE_NAME, "file_sha256": b"short",
     "file_size": len(FILE_BYTES)},
    {"storage_name": STORAGE_NAME, "file_sha256": FILE_SHA,
     "file_size": True},
])
def test_download_refuses_a_malformed_database_measurement(
        malformed, monkeypatch):
    @contextmanager
    def connection():
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    monkeypatch.setattr(
        api.db, "consume_table_export_ticket",
        lambda _conn, **_kwargs: malformed)
    with pytest.raises(HTTPException) as caught:
        api.download_export(
            api.ExportDownloadRequest(ticket="A" * 43), _principal())
    assert caught.value.status_code == 404


def test_db_register_is_current_tenant_actor_and_membership_bound():
    conn = _Conn({"ref_digest": b"r" * 32,
                  "expires_at": datetime.now(timezone.utc)})
    db.register_table_export(
        conn, export_id=EXPORT, actor_id=ACTOR, ref_digest=b"r" * 32,
        storage_name=STORAGE_NAME, file_sha256=FILE_SHA,
        file_size=len(FILE_BYTES))
    sql = conn.cur.sql
    assert "INSERT INTO table_exports" in sql
    assert "FROM org_memberships" in sql
    assert "m.identity_id = %s" in sql and "m.state = 'active'" in sql
    assert "m.app_role IN ('reader', 'editor', 'admin')" in sql
    assert "org_architects" not in sql
    assert conn.commits == 1


def test_db_mint_rechecks_actor_membership_export_owner_and_expiry():
    conn = _Conn({"expires_at": datetime.now(timezone.utc)})
    db.mint_table_export_ticket(
        conn, actor_id=ACTOR, ref_digest=b"r" * 32,
        token_digest=b"t" * 32, ttl_seconds=50)
    sql = conn.cur.sql
    for claim in (
        "JOIN org_memberships", "m.state = 'active'",
        "e.ref_digest = %s", "e.actor_id = %s", "e.expires_at > now()",
    ):
        assert claim in sql
    assert conn.cur.params == (
        b"t" * 32, ACTOR, 50, ACTOR, b"r" * 32, ACTOR)
    assert conn.commits == 1


def test_db_consume_is_one_statement_actor_bound_and_single_use():
    row = {"storage_name": STORAGE_NAME, "file_sha256": FILE_SHA,
           "file_size": len(FILE_BYTES)}
    conn = _Conn(row)
    assert db.consume_table_export_ticket(
        conn, actor_id=ACTOR, token_digest=b"t" * 32) == row
    sql = conn.cur.sql
    for claim in (
        "UPDATE table_export_tickets", "SET consumed_at = now()",
        "t.actor_id = %s", "e.actor_id = %s", "t.consumed_at IS NULL",
        "t.expires_at > now()", "e.expires_at > now()",
        "t.purpose = 'download'",
    ):
        assert claim in sql
    assert conn.commits == 1


@pytest.mark.parametrize("operation", ["register", "mint", "consume"])
def test_every_db_authority_miss_is_closed(operation):
    with pytest.raises(db.ExportAccessRefused):
        if operation == "register":
            db.register_table_export(
                _Conn(None), export_id=EXPORT, actor_id=ACTOR,
                ref_digest=b"r" * 32, storage_name=STORAGE_NAME,
                file_sha256=FILE_SHA, file_size=len(FILE_BYTES))
        elif operation == "mint":
            db.mint_table_export_ticket(
                _Conn(None), actor_id=ACTOR, ref_digest=b"r" * 32,
                token_digest=b"t" * 32)
        else:
            db.consume_table_export_ticket(
                _Conn(None), actor_id=ACTOR, token_digest=b"t" * 32)


def test_export_schema_is_content_free_closed_and_forced_through_rls():
    sql = Path(db.__file__).with_name("schema.sql").read_text(encoding="utf-8")
    export_table = sql.split(
        "CREATE TABLE IF NOT EXISTS table_exports", 1)[1].split(");", 1)[0]
    ticket_table = sql.split(
        "CREATE TABLE IF NOT EXISTS table_export_tickets", 1)[1].split(");", 1)[0]
    assert "actor_id" in export_table and "tenant_id" in export_table
    assert "ref_digest" in export_table and "file_sha256" in export_table
    assert "storage_name" in export_table and "file_size" in export_table
    assert "consumed_at" in ticket_table and "purpose" in ticket_table
    for forbidden in (
        "content", "cells", "rows", "question", "answer", "source_path",
    ):
        assert forbidden not in export_table.lower()
        assert forbidden not in ticket_table.lower()
    for table in ("table_exports", "table_export_tickets"):
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in sql
        assert f"CREATE POLICY tenant_isolation ON {table}" in sql
