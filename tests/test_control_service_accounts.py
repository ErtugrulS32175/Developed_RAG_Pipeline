"""Closed service-account token and runtime-route contracts."""
import uuid
from pathlib import Path

import pytest

from pipeline.control import db, service_accounts


ACCOUNT_ID = uuid.UUID("30000000-0000-0000-0000-000000000003")
TENANT_ID = uuid.UUID("10000000-0000-0000-0000-000000000001")


def test_credential_is_opaque_canonical_and_round_trips(monkeypatch):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    token, issued = service_accounts.issue_credential(ACCOUNT_ID, 7)
    parsed = service_accounts.parse_credential(token)

    assert token.startswith("ragsa.v1.")
    assert str(TENANT_ID) not in token
    assert issued == parsed
    assert issued.service_account_id == ACCOUNT_ID
    assert issued.credential_version == 7
    assert type(issued.digest) is bytes and len(issued.digest) == 32
    assert token.encode() not in issued.digest
    assert "digest" not in repr(issued)


def test_service_account_key_is_independent(monkeypatch):
    monkeypatch.delenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", raising=False)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 64)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 64)
    with pytest.raises(service_accounts.ServiceAccountRefused):
        service_accounts.issue_credential(ACCOUNT_ID, 1)

    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "a" * 32)
    token, first = service_accounts.issue_credential(ACCOUNT_ID, 1)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "b" * 32)
    second = service_accounts.parse_credential(token)
    assert first.digest != second.digest


@pytest.mark.parametrize("secret", [" " * 32, "x" * 31, "x" * 31 + "\n"])
def test_service_account_key_refuses_weak_shapes(monkeypatch, secret):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", secret)
    with pytest.raises(service_accounts.ServiceAccountRefused):
        service_accounts.issue_credential(ACCOUNT_ID, 1)


@pytest.mark.parametrize("peer", [
    "CONTROL_IDENTITY_HMAC_SECRET", "OIDC_SESSION_SECRET",
])
def test_service_account_key_cannot_reuse_another_security_domain(
        monkeypatch, peer):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "x" * 32)
    monkeypatch.setenv(peer, "x" * 32)
    with pytest.raises(service_accounts.ServiceAccountRefused):
        service_accounts.issue_credential(ACCOUNT_ID, 1)


@pytest.mark.parametrize("token", [
    None,
    7,
    "",
    " ragsa.v1." + ACCOUNT_ID.hex + ".1." + "a" * 43,
    "ragsa.v2." + ACCOUNT_ID.hex + ".1." + "a" * 43,
    "ragsa.v1." + str(ACCOUNT_ID) + ".1." + "a" * 43,
    "ragsa.v1.A0000000000000000000000000000003.1." + "a" * 43,
    "ragsa.v1." + ACCOUNT_ID.hex + ".01." + "a" * 43,
    "ragsa.v1." + ACCOUNT_ID.hex + ".0." + "a" * 43,
    "ragsa.v1." + ACCOUNT_ID.hex + ".1." + "a" * 42,
    "ragsa.v1." + ACCOUNT_ID.hex + ".1." + "a" * 42 + "!",
    "ragsa.v1." + ACCOUNT_ID.hex + ".1." + "a" * 43 + ".extra",
    "ragsa.v1." + ACCOUNT_ID.hex + ".1." + "a" * 42 + "\u011f",
])
def test_malformed_service_tokens_fail_closed(monkeypatch, token):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    with pytest.raises(service_accounts.ServiceAccountRefused):
        service_accounts.parse_credential(token)


@pytest.mark.parametrize("account,version", [
    ("bad", 1),
    (ACCOUNT_ID, True),
    (ACCOUNT_ID, 0),
    (ACCOUNT_ID, 2147483648),
])
def test_invalid_issue_coordinates_fail_closed(monkeypatch, account, version):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    with pytest.raises(service_accounts.ServiceAccountRefused):
        service_accounts.issue_credential(account, version)


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


def _route_row(**changes):
    row = {
        "service_account_id": ACCOUNT_ID,
        "tenant_id": TENANT_ID,
        "scopes": ["documents.read", "rag.query"],
        "deployment_profile": "enterprise",
        "region_code": "eu-central",
        "route_kind": "dedicated_postgres",
        "connection_ref": "vault:tenant-a/postgres",
        "configuration_revision": 4,
        "policy_revision": 9,
        "features": {"review": True},
        "quotas": {"document_count": 100},
        "quota_enforcement": "declared",
    }
    row.update(changes)
    return row


def test_runtime_route_carries_machine_scopes_without_a_human_role():
    connection = _Connection([_route_row()])
    route = db.resolve_service_account(
        connection, ACCOUNT_ID, 3, b"d" * 32)
    assert route.service_account_id == ACCOUNT_ID
    assert route.facts.tenant_id == TENANT_ID
    assert route.scopes == ("documents.read", "rag.query")
    assert not hasattr(route, "role")
    assert connection.cursor_instance.params == (ACCOUNT_ID, 3, b"d" * 32)


@pytest.mark.parametrize("rows", [
    [],
    [_route_row(), _route_row()],
    [_route_row(scopes=[])],
    [_route_row(scopes=["rag.query", "documents.read"])],
    [_route_row(scopes=["rag.query", "rag.query"])],
    [_route_row(scopes=["tenant.admin"])],
    [_route_row(role="admin")],
])
def test_runtime_route_refuses_absence_ambiguity_and_scope_drift(rows):
    connection = _Connection(rows)
    if not rows:
        assert db.resolve_service_account(
            connection, ACCOUNT_ID, 1, b"d" * 32) is None
    else:
        with pytest.raises(db.ControlPlaneRefused):
            db.resolve_service_account(
                connection, ACCOUNT_ID, 1, b"d" * 32)


@pytest.mark.parametrize("account,version,digest", [
    ("bad", 1, b"d" * 32),
    (ACCOUNT_ID, True, b"d" * 32),
    (ACCOUNT_ID, 1, b"short"),
    (ACCOUNT_ID, 1, bytearray(b"d" * 32)),
])
def test_invalid_runtime_coordinates_fail_before_sql(
        account, version, digest):
    connection = _Connection([])
    with pytest.raises(db.ControlPlaneRefused):
        db.resolve_service_account(
            connection, account, version, digest)
    assert connection.cursor_instance.query is None


def test_scope_vocabulary_is_closed_and_machine_only():
    assert db.SERVICE_ACCOUNT_SCOPES == tuple(sorted({
        "rag.query", "documents.read", "documents.write",
        "documents.lifecycle", "collections.manage", "tables.extract",
    }))
    assert all("admin" not in scope for scope in db.SERVICE_ACCOUNT_SCOPES)
    assert all("org" not in scope for scope in db.SERVICE_ACCOUNT_SCOPES)


def test_token_parse_refusal_does_not_chain_attacker_text(monkeypatch):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    hostile = "ragsa.v1." + ACCOUNT_ID.hex + ".not-a-version." + "a" * 43
    with pytest.raises(service_accounts.ServiceAccountRefused) as caught:
        service_accounts.parse_credential(hostile)
    assert caught.value.__cause__ is None
    assert "not-a-version" not in str(caught.value)


def test_database_not_the_caller_owns_credential_time():
    schema = (Path(__file__).resolve().parent.parent / "pipeline" /
              "control" / "schema.sql").read_text(encoding="utf-8")
    resolver = schema.split(
        "CREATE OR REPLACE FUNCTION rag_control.control_resolve_service_account",
        1)[1].split(
            "CREATE OR REPLACE FUNCTION "
            "rag_control.control_require_platform_security", 1)[0]
    assert "requested_at" not in resolver
    assert resolver.count("statement_timestamp()") == 3


def test_schema_enforces_one_live_credential_and_normalized_scopes():
    schema = (Path(__file__).resolve().parent.parent / "pipeline" /
              "control" / "schema.sql").read_text(encoding="utf-8")
    assert "control_one_active_service_credential" in schema
    assert "WHERE state = 'active'" in schema
    assert "PRIMARY KEY (service_account_id, scope_code)" in schema
    for scope in db.SERVICE_ACCOUNT_SCOPES:
        assert f"'{scope}'" in schema
