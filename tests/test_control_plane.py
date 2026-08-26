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
            "control_platform_operator_digests", "control_service_accounts",
            "control_service_account_scopes",
            "control_service_account_credentials",
            "control_service_account_events",
            "control_service_account_approvals",
            "control_service_account_approval_events",
            "control_service_account_assertion_keys",
            "control_service_account_assertion_nonces",
            "control_admin_events"):
        assert f"rag_control.{table}" in text

    lowered = text.lower()
    for forbidden in (
            "prompt", "message_content", "document_content", "chunk_text",
            "embedding", "filename", "email", "display_name", " dsn",
            "bucket", "object_key", "jsonb default"):
        assert forbidden not in lowered
    assert text.count("octet_length(digest) = 32") == 3
    assert "CHECK (quota_enforcement = 'declared')" in text
    assert "CHECK (position('://' in connection_ref) = 0)" in text
    assert "CHECK (position('@' in connection_ref) = 0)" in text
    assert "ON DELETE CASCADE" not in text
    assert "SET search_path FROM CURRENT" not in text
    assert text.count("SET search_path = pg_catalog, rag_control") == 21


def test_runtime_role_has_only_lookup_function_authority():
    script = (
        ROOT / "scripts" / "init_control_runtime_role.sh"
    ).read_text(encoding="utf-8")
    assert ("NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT "
            "NOREPLICATION") in script
    assert "ALTER ROLE rag_control_runtime LOGIN PASSWORD :'runtime_password'" in script
    assert "\\getenv runtime_password CONTROL_RUNTIME_PASSWORD" in script
    assert "--set=runtime_password" not in script
    assert "REVOKE ALL ON ALL TABLES IN SCHEMA rag_control" in script
    assert "REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control" in script
    assert "REVOKE ALL ON ALL FUNCTIONS IN SCHEMA rag_control" in script
    assert "REVOKE CREATE ON DATABASE" in script
    assert "rag_control.control_tenant_facts(uuid)" in script
    assert "rag_control.control_resolve_identity(integer, bytea)" in script
    assert "rag_control.control_resolve_platform_operator(" in script
    assert "rag_control.control_resolve_service_account(" in script
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES" not in script


def test_redeemer_role_is_a_separate_exact_function_capability():
    script = (
        ROOT / "scripts" / "init_control_redeemer_role.sh"
    ).read_text(encoding="utf-8")
    assert "rag_control_redeemer" in script
    assert script.count("GRANT EXECUTE ON FUNCTION") == 4
    assert "control_asserted_list_redeemable" in script
    assert "control_asserted_get_redeemable" in script
    assert "control_asserted_redeem_service_account_issue" in script
    assert "control_asserted_redeem_service_account_rotation" in script
    for forbidden in (
            "control_tenant_facts", "control_resolve_identity",
            "control_approve_service_account_issue",
            "control_revoke_service_account"):
        assert forbidden not in script


def test_events_are_owner_immutable_and_public_exec_is_revoked():
    text = SCHEMA.read_text(encoding="utf-8")
    assert "BEFORE UPDATE OR DELETE ON rag_control.control_admin_events" in text
    assert ("BEFORE UPDATE OR DELETE ON "
            "rag_control.control_service_account_events") in text
    assert ("BEFORE UPDATE OR DELETE ON rag_control."
            "control_service_account_approval_events") in text
    assert "control_service_account_approval_events_seal" in text
    assert "ERRCODE = '55000'" in text
    assert "MESSAGE = 'control_event_immutable'" in text
    assert text.count("STABLE SECURITY DEFINER") == 5
    assert text.count("VOLATILE SECURITY DEFINER") >= 9
    assert "control_tenant_facts(uuid) FROM PUBLIC" in text
    assert "control_resolve_identity(integer, bytea)" in text
    assert "control_resolve_platform_operator(" in text
    assert "control_seal_service_account_approval_event() FROM PUBLIC" in text
    assert "control_resolve_service_account(" in text


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
    assert db.CONTROL_SCHEMA_VERSION == 6
    digest = hashlib.sha256(SCHEMA.read_bytes()).hexdigest()
    assert len(digest) == 64
    assert db._SCHEMA_LOCK_NAME == "ragtest-control-schema-migration"
    assert "control_schema_state_monotonic" in db._SCHEMA_MONOTONIC_GUARD_DDL


class _InitCursor:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchone(self):
        return next(self.results)


class _InitConnection:
    def __init__(self, results):
        self.cursor_instance = _InitCursor(results)
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True


def test_conflicting_schema_history_receipt_is_refused():
    connection = _InitConnection([None, ("0" * 64,)])
    with pytest.raises(db.ControlPlaneRefused, match="history mismatch"):
        db.init_schema(connection)
    assert not connection.committed
    assert any(
        "SELECT schema_sha256 FROM rag_control.control_schema_history" in query
        for query, _params in connection.cursor_instance.calls)
    schema_sql = SCHEMA.read_text(encoding="utf-8")
    assert not any(
        query == schema_sql for query, _params in connection.cursor_instance.calls)
