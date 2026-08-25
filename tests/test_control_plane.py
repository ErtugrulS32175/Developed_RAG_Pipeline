"""Content-free control-plane repository and schema contract."""
import hashlib
import uuid
from pathlib import Path

import pytest

from pipeline.control import db


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "pipeline" / "control" / "schema.sql"


def test_control_connections_never_fall_back_to_the_data_plane(monkeypatch):
    monkeypatch.delenv("PG_CONTROL_DSN", raising=False)
    monkeypatch.setenv("PG_DSN", "must-not-be-used")
    with pytest.raises(db.ControlPlaneRefused, match="PG_CONTROL_DSN"):
        db.get_conn()

    monkeypatch.delenv("PG_CONTROL_MIGRATION_DSN", raising=False)
    monkeypatch.setenv("PG_MIGRATION_DSN", "must-not-be-used")
    with pytest.raises(
            db.ControlPlaneRefused, match="PG_CONTROL_MIGRATION_DSN"):
        db.get_migration_conn()


def test_external_identity_coordinates_are_delimiter_safe_hmacs(monkeypatch):
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "s" * 32)
    first = db.identity_digest("issuer:a", "subject")
    second = db.identity_digest("issuer", "a:subject")
    assert first != second
    assert len(first) == 32
    assert type(first) is bytes
    assert b"issuer" not in first
    assert b"subject" not in first


@pytest.mark.parametrize("issuer,subject", [
    ("", "subject"),
    ("issuer", ""),
    (None, "subject"),
    ("issuer", 7),
    ("x" * 513, "subject"),
    ("issuer", "\ud800"),
])
def test_invalid_identity_coordinates_fail_closed(monkeypatch, issuer, subject):
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "s" * 32)
    with pytest.raises(db.ControlPlaneRefused):
        db.identity_digest(issuer, subject)


def test_identity_hmac_key_is_independent_and_long(monkeypatch):
    monkeypatch.delenv("CONTROL_IDENTITY_HMAC_SECRET", raising=False)
    monkeypatch.setenv("OPENWEBUI_USER_JWT_SECRET", "w" * 64)
    with pytest.raises(db.ControlPlaneRefused):
        db.identity_digest("issuer", "subject")
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "short")
    with pytest.raises(db.ControlPlaneRefused):
        db.identity_digest("issuer", "subject")


def test_schema_is_fixed_and_closed_to_operational_facts_only():
    text = SCHEMA.read_text(encoding="utf-8")
    for table in (
            "control_regions", "control_tenants", "control_tenant_routes",
            "control_feature_catalog", "control_tenant_features",
            "control_tenant_quotas", "control_identity_routes",
            "control_identity_route_digests", "control_platform_operators",
            "control_platform_operator_digests", "control_admin_events"):
        assert f"rag_control.{table}" in text

    lowered = text.lower()
    for forbidden in (
            "prompt", "message_content", "document_content", "chunk_text",
            "embedding", "filename", "email", "display_name", " dsn",
            "bucket", "object_key", "jsonb default"):
        assert forbidden not in lowered
    assert text.count("octet_length(digest) = 32") == 2
    assert "CHECK (quota_enforcement = 'declared')" in text
    assert "CHECK (position('://' in connection_ref) = 0)" in text
    assert "CHECK (position('@' in connection_ref) = 0)" in text
    assert "ON DELETE CASCADE" not in text
    assert "SET search_path FROM CURRENT" not in text
    assert text.count("SET search_path = pg_catalog, rag_control") == 3


def test_runtime_role_has_only_lookup_function_authority():
    script = (
        ROOT / "scripts" / "init_control_runtime_role.sh"
    ).read_text(encoding="utf-8")
    assert "NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT" in script
    assert "REVOKE ALL ON ALL TABLES IN SCHEMA rag_control" in script
    assert "REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control" in script
    assert "rag_control.control_tenant_facts(uuid)" in script
    assert "rag_control.control_resolve_identity(integer, bytea)" in script
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES" not in script


def test_events_are_owner_immutable_and_public_exec_is_revoked():
    text = SCHEMA.read_text(encoding="utf-8")
    assert "BEFORE UPDATE OR DELETE ON rag_control.control_admin_events" in text
    assert "ERRCODE = '55000'" in text
    assert "MESSAGE = 'control_event_immutable'" in text
    assert text.count("STABLE SECURITY DEFINER") == 2
    assert "control_tenant_facts(uuid) FROM PUBLIC" in text
    assert "control_resolve_identity(integer, bytea)" in text


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.cursor_instance = _Cursor(rows)

    def cursor(self, **_kwargs):
        return self.cursor_instance


def _row(*, route=True):
    row = {
        "tenant_id": uuid.UUID("10000000-0000-0000-0000-000000000001"),
        "deployment_profile": "enterprise",
        "region_code": "eu-central",
        "route_kind": "dedicated_postgres",
        "configuration_revision": 4,
        "policy_revision": 9,
        "features": {"review": True},
        "quotas": {"document_count": 100},
        "quota_enforcement": "declared",
    }
    if route:
        row["connection_ref"] = "vault:tenant-a/postgres"
    return row


def test_internal_resolution_and_public_facts_are_separate():
    digest = b"a" * 32
    connection = _Connection([_row()])
    route = db.resolve_identity(connection, 3, digest)
    assert route.connection_ref == "vault:tenant-a/postgres"
    assert not hasattr(route.facts, "connection_ref")
    assert route.facts.deployment_profile == "enterprise"
    assert route.facts.quota_enforcement == "declared"
    with pytest.raises(TypeError):
        route.facts.features["new"] = True
    assert connection.cursor_instance.params == (3, digest)

    public = db.tenant_facts(_Connection([_row(route=False)]),
                             route.facts.tenant_id)
    assert public == route.facts
    assert not hasattr(public, "connection_ref")


@pytest.mark.parametrize("rows", [
    [{**_row(), "raw_subject": "forbidden"}],
    [{**_row(), "features": {"x": 1}}],
    [{**_row(), "quotas": {"x": True}}],
    [{**_row(), "policy_revision": -1}],
    [_row(), _row()],
])
def test_malformed_or_ambiguous_results_fail_closed(rows):
    with pytest.raises(db.ControlPlaneRefused):
        db.resolve_identity(_Connection(rows), 1, b"a" * 32)


@pytest.mark.parametrize("version,digest", [
    (0, b"a" * 32),
    (True, b"a" * 32),
    (1, b"short"),
    (1, bytearray(b"a" * 32)),
])
def test_invalid_digest_versions_and_shapes_fail_before_sql(version, digest):
    connection = _Connection([])
    with pytest.raises(db.ControlPlaneRefused):
        db.resolve_identity(connection, version, digest)
    assert connection.cursor_instance.query is None


def test_control_schema_version_binds_the_shipped_bytes():
    assert db.CONTROL_SCHEMA_VERSION == 1
    digest = hashlib.sha256(SCHEMA.read_bytes()).hexdigest()
    assert len(digest) == 64
    assert db._SCHEMA_LOCK_NAME == "ragtest-control-schema-migration"
