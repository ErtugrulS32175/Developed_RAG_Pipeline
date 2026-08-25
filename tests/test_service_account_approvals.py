"""Content-free approval queue for tenant-delivered service credentials."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import pytest

from pipeline.control import db
from pipeline.index import db as index_db


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = (ROOT / "pipeline" / "control" / "schema.sql").read_text(
    encoding="utf-8")
TENANT = uuid.UUID("10000000-0000-0000-0000-000000000001")
ACCOUNT = uuid.UUID("30000000-0000-0000-0000-000000000003")
APPROVAL = uuid.UUID("50000000-0000-0000-0000-000000000005")


class _Cursor:
    def __init__(self, rows, error=None):
        self.rows = rows
        self.error = error
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.query = query
        self.params = params
        if self.error is not None:
            raise self.error

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _Connection:
    def __init__(self, rows, error=None):
        self.cursor_instance = _Cursor(rows, error)

    def cursor(self, **_kwargs):
        return self.cursor_instance


def _created(policy=7):
    now = datetime.now(timezone.utc)
    return {
        "approval_revision": 1,
        "control_policy_revision": policy,
        "created_at": now,
        "expires_at": now + timedelta(minutes=15),
    }


def _listed(*, action="issue"):
    now = datetime.now(timezone.utc)
    issue = action == "issue"
    return {
        "approval_id": APPROVAL,
        "tenant_id": TENANT,
        "service_account_id": ACCOUNT,
        "action": action,
        "state": "approved",
        "approval_revision": 1,
        "reason_code": (
            "security_provisioning" if issue else "scheduled_rotation"),
        "scopes": ["documents.read", "rag.query"] if issue else None,
        "account_expires_at": now + timedelta(days=30) if issue else None,
        "credential_expires_at": now + timedelta(days=7),
        "expected_account_revision": None if issue else 4,
        "control_policy_revision": 7,
        "expires_at": now + timedelta(minutes=15),
        "created_at": now,
    }


def _audit(monkeypatch):
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "a" * 32)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 32)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 32)


def test_approval_schema_is_closed_content_free_and_short_lived():
    table = SCHEMA.split(
        "CREATE TABLE IF NOT EXISTS "
        "rag_control.control_service_account_approvals", 1)[1].split(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "control_one_pending_account_approval", 1)[0]
    assert "interval '15 minutes'" in table
    assert "action IN ('issue', 'rotate')" in table
    assert "'approved', 'redeemed', 'cancelled'" in table
    assert "control_one_pending_account_approval" in SCHEMA
    assert "WHERE state = 'approved'" in SCHEMA
    for forbidden in (
            "token", "secret", "credential_digest", "subject", "email",
            "display_name", "filename", "prompt", "content"):
        assert forbidden not in table.lower()


def test_approval_events_bind_actor_kind_and_cannot_be_rewritten():
    event = SCHEMA.split(
        "CREATE TABLE IF NOT EXISTS "
        "rag_control.control_service_account_approval_events", 1)[1].split(
            "CREATE TABLE IF NOT EXISTS rag_control.control_admin_events", 1)[0]
    assert "platform_security', 'tenant_org_admin', 'system" in event
    assert "FOREIGN KEY (approval_id, target_tenant_id, service_account_id)" in event
    assert ("BEFORE UPDATE OR DELETE ON "
            "rag_control.control_service_account_approval_events") in SCHEMA
    assert "control_service_account_approval_events_seal" in SCHEMA
    assert "approval_event_request_v1" in SCHEMA
    assert "approval_event_result_v1" in SCHEMA
    assert "tenant_actor_digest" in event
    for field in (
            "prior_state", "resulting_state", "prior_revision",
            "approval_created_at", "approval_expires_at", "reason_code"):
        assert field in event
    for forbidden in ("token", "secret", "subject", "email", "content"):
        assert forbidden not in event.lower()


def test_issue_approval_carries_only_closed_metadata(monkeypatch):
    _audit(monkeypatch)
    now = datetime.now(timezone.utc)
    connection = _Connection([_created()])
    result = db.approve_service_account_issue(
        connection, operator_key_version=1, operator_digest=b"p" * 32,
        approval_id=APPROVAL, tenant_id=TENANT, account_id=ACCOUNT,
        scopes=("documents.read", "rag.query"),
        account_expires_at=now + timedelta(days=30),
        credential_expires_at=now + timedelta(days=7),
        expected_policy_revision=7,
        reason_code="security_provisioning")
    assert result.action == "issue"
    assert result.scopes == ("documents.read", "rag.query")
    assert result.expected_account_revision is None
    assert result.control_policy_revision == 7
    assert "control_approve_service_account_issue" in (
        connection.cursor_instance.query)
    assert connection.cursor_instance.params[8:10] == (
        7, "security_provisioning")
    assert all(type(value) is bytes and len(value) == 32
               for value in connection.cursor_instance.params[-2:])


def test_rotation_approval_is_revision_and_policy_bound(monkeypatch):
    _audit(monkeypatch)
    now = datetime.now(timezone.utc)
    connection = _Connection([_created()])
    result = db.approve_service_account_rotation(
        connection, operator_key_version=1, operator_digest=b"p" * 32,
        approval_id=APPROVAL, tenant_id=TENANT, account_id=ACCOUNT,
        expected_account_revision=4,
        credential_expires_at=now + timedelta(days=7),
        expected_policy_revision=7, reason_code="scheduled_rotation")
    assert result.action == "rotate"
    assert result.scopes is None
    assert result.account_expires_at is None
    assert result.expected_account_revision == 4
    assert connection.cursor_instance.params[5:9] == (
        4, result.credential_expires_at, 7, "scheduled_rotation")


@pytest.mark.parametrize("field,value", [
    ("scopes", ("rag.query", "rag.query")),
    ("expected_policy_revision", True),
    ("reason_code", "arbitrary"),
])
def test_invalid_issue_approval_is_refused_before_sql(monkeypatch, field, value):
    _audit(monkeypatch)
    now = datetime.now(timezone.utc)
    arguments = {
        "operator_key_version": 1, "operator_digest": b"p" * 32,
        "approval_id": APPROVAL, "tenant_id": TENANT, "account_id": ACCOUNT,
        "scopes": ("rag.query",),
        "account_expires_at": now + timedelta(days=30),
        "credential_expires_at": now + timedelta(days=7),
        "expected_policy_revision": 7,
        "reason_code": "security_provisioning",
    }
    arguments[field] = value
    connection = _Connection([])
    with pytest.raises(db.ControlPlaneRefused):
        db.approve_service_account_issue(connection, **arguments)
    assert connection.cursor_instance.query is None


@pytest.mark.parametrize("error,expected,message", [
    (db.psycopg.errors.SerializationFailure("SENTINEL serialization"),
     db.ControlPlaneConflict, "service account approval conflict"),
    (db.psycopg.errors.InsufficientPrivilege("SENTINEL privilege"),
     db.ControlPlaneDenied, "service account approval denied"),
    (db.psycopg.errors.CheckViolation("SENTINEL check"),
     db.ControlPlaneRefused, "service account approval refused"),
])
def test_approval_creation_hides_database_error_details(
        monkeypatch, error, expected, message):
    _audit(monkeypatch)
    now = datetime.now(timezone.utc)
    with pytest.raises(expected) as refused:
        db.approve_service_account_issue(
            _Connection([], error), operator_key_version=1,
            operator_digest=b"p" * 32, approval_id=APPROVAL,
            tenant_id=TENANT, account_id=ACCOUNT,
            scopes=("rag.query",),
            account_expires_at=now + timedelta(days=30),
            credential_expires_at=now + timedelta(days=7),
            expected_policy_revision=7,
            reason_code="security_provisioning")
    assert str(refused.value) == message
    assert "SENTINEL" not in str(refused.value)
    assert refused.value.__cause__ is None


def test_listing_returns_only_tenant_safe_approval_metadata():
    first = _listed()
    second = {
        **_listed(action="rotate"), "approval_id": uuid.uuid4(),
        "created_at": first["created_at"] + timedelta(seconds=1),
        "expires_at": first["expires_at"] + timedelta(seconds=1),
    }
    connection = _Connection([first, second])
    rows = db.list_redeemable_service_account_approvals(
        connection, TENANT, limit=2)
    assert [row.action for row in rows] == ["issue", "rotate"]
    assert all(row.tenant_id == TENANT and row.state == "approved"
               for row in rows)
    assert connection.cursor_instance.params == (TENANT, 2)


def test_listing_rejects_open_or_impossible_rows():
    for change in (
            {"access_token": "forbidden"},
            {"state": "redeemed"},
            {"approval_revision": 2},
            {"tenant_id": uuid.uuid4()},
            {"expires_at": datetime.now(timezone.utc) + timedelta(minutes=16)},
            {"action": "rotate", "scopes": ["rag.query"],
             "account_expires_at": None, "expected_account_revision": 1}):
        row = {**_listed(), **change}
        with pytest.raises(db.ControlPlaneRefused):
            db.list_redeemable_service_account_approvals(
                _Connection([row]), TENANT)


@pytest.mark.parametrize("created,expires", [
    (timedelta(minutes=-20), timedelta(minutes=-5)),
    (timedelta(minutes=1), timedelta(minutes=10)),
])
def test_listing_rejects_expired_or_future_created_rows(created, expires):
    now = datetime.now(timezone.utc)
    row = {
        **_listed(), "created_at": now + created,
        "expires_at": now + expires,
    }
    with pytest.raises(db.ControlPlaneRefused):
        db.list_redeemable_service_account_approvals(
            _Connection([row]), TENANT)


@pytest.mark.parametrize("shape,limit", [
    ("duplicate", 2),
    ("excess", 1),
    ("out_of_order", 2),
])
def test_listing_rejects_duplicate_excess_or_out_of_order_rows(shape, limit):
    first = _listed()
    second = _listed()
    if shape == "duplicate":
        rows = [first, {**first}]
    elif shape == "excess":
        rows = [first, {**second, "approval_id": uuid.uuid4()}]
    else:
        rows = [
            {**first, "approval_id": uuid.uuid4(),
             "created_at": first["created_at"] + timedelta(seconds=2),
             "expires_at": first["expires_at"]},
            second,
        ]
    with pytest.raises(db.ControlPlaneRefused) as refused:
        db.list_redeemable_service_account_approvals(
            _Connection(rows), TENANT, limit=limit)
    assert str(refused.value) == "service account approvals result is invalid"


def test_listing_hides_database_error_details():
    connection = _Connection(
        [], db.psycopg.OperationalError("SENTINEL database detail"))
    with pytest.raises(db.ControlPlaneRefused) as refused:
        db.list_redeemable_service_account_approvals(connection, TENANT)
    assert str(refused.value) == "service account approvals refused"
    assert "SENTINEL" not in str(refused.value)
    assert refused.value.__cause__ is None


def test_platform_revoke_cancels_pending_approvals_in_the_same_function():
    revoke = SCHEMA.split(
        "CREATE OR REPLACE FUNCTION "
        "rag_control.control_revoke_service_account", 1)[1].split(
            "REVOKE ALL ON FUNCTION", 1)[0]
    cancel = revoke.index("UPDATE rag_control.control_service_account_approvals")
    lifecycle_event = revoke.index(
        "INSERT INTO rag_control.control_service_account_events")
    assert cancel < lifecycle_event
    assert "AND state = 'approved'" in revoke
    assert ("'approval_cancelled', 'service_account_revoked',\n"
            "           'platform_security'" in revoke)
    assert "requested_request_digest,\n           requested_resulting_fact_digest\n    FROM cancelled" not in revoke


def test_redeem_wrapper_selects_only_the_closed_action(monkeypatch):
    _audit(monkeypatch)
    now = datetime.now(timezone.utc)
    row = {
        "account_revision": 1,
        "credential_version": 1,
        "account_expires_at": now + timedelta(days=30),
        "credential_expires_at": now + timedelta(days=7),
    }
    connection = _Connection([row])
    approval = db.ServiceAccountApproval(
        APPROVAL, TENANT, ACCOUNT, "issue", "approved", 1,
        "security_provisioning", ("rag.query",),
        row["account_expires_at"], row["credential_expires_at"], None, 7,
        now + timedelta(minutes=15), now)
    result = db.redeem_service_account_approval(
        connection, approval, tenant_actor_digest=b"t" * 32,
        org_policy_epoch=4, credential_digest=b"c" * 32)
    assert result.service_account_id == ACCOUNT
    assert result.credential_version == 1
    assert "control_redeem_service_account_issue" in (
        connection.cursor_instance.query)
    assert b"c" * 32 == connection.cursor_instance.params[6]
    assert all(type(value) is bytes and len(value) == 32
               for value in connection.cursor_instance.params[7:])


@pytest.mark.parametrize("error,expected,message", [
    (db.psycopg.errors.SerializationFailure("SENTINEL serialization"),
     db.ControlPlaneConflict, "service account redemption conflict"),
    (db.psycopg.errors.InsufficientPrivilege("SENTINEL privilege"),
     db.ControlPlaneDenied, "service account redemption denied"),
    (db.psycopg.errors.CheckViolation("SENTINEL check"),
     db.ControlPlaneRefused, "service account redemption refused"),
])
def test_redemption_hides_database_error_details(
        monkeypatch, error, expected, message):
    _audit(monkeypatch)
    now = datetime.now(timezone.utc)
    approval = db.ServiceAccountApproval(
        APPROVAL, TENANT, ACCOUNT, "issue", "approved", 1,
        "security_provisioning", ("rag.query",),
        now + timedelta(days=30), now + timedelta(days=7), None, 7,
        now + timedelta(minutes=15), now)
    with pytest.raises(expected) as refused:
        db.redeem_service_account_approval(
            _Connection([], error), approval,
            tenant_actor_digest=b"t" * 32, org_policy_epoch=4,
            credential_digest=b"c" * 32)
    assert str(refused.value) == message
    assert "SENTINEL" not in str(refused.value)
    assert refused.value.__cause__ is None


def test_cancel_wrapper_is_platform_proof_and_revision_bound(monkeypatch):
    _audit(monkeypatch)
    connection = _Connection([(2,)])
    assert db.cancel_service_account_approval(
        connection, operator_key_version=1, operator_digest=b"p" * 32,
        approval_id=APPROVAL, tenant_id=TENANT, account_id=ACCOUNT,
        expected_approval_revision=1,
        reason_code="approval_cancelled") == 2
    assert connection.cursor_instance.params[2:] == (
        APPROVAL, TENANT, ACCOUNT, 1, "approval_cancelled")


def test_control_schema_version_advances_for_the_approval_authority():
    assert db.CONTROL_SCHEMA_VERSION == 5


def test_tenant_redemption_gate_locks_both_authority_rows():
    connection = _Connection([{"tenant_id": TENANT, "policy_epoch": 7}])
    assert index_db.lock_service_account_redeemer(
        connection, actor_id=ACCOUNT,
        expected_policy_epoch=7) == {
            "tenant_id": TENANT, "policy_epoch": 7}
    query = connection.cursor_instance.query
    assert "architect.active = true" in query
    assert "membership.state = 'active'" in query
    assert "membership.app_role = 'admin'" in query
    assert "FOR UPDATE OF tenant, architect, membership" in query


@pytest.mark.parametrize("rows,expected", [
    ([], index_db.ServiceAccountRedemptionRefused),
    ([{"tenant_id": TENANT, "policy_epoch": 8}],
     index_db.OrgPolicyConflict),
    ([{"tenant_id": TENANT, "policy_epoch": 7, "extra": True}],
     index_db.ServiceAccountRedemptionRefused),
])
def test_tenant_redemption_gate_fails_closed(rows, expected):
    with pytest.raises(expected):
        index_db.lock_service_account_redeemer(
            _Connection(rows), actor_id=ACCOUNT,
            expected_policy_epoch=7)
