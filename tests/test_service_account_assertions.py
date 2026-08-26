"""Closed Python boundary for cross-database service-account assertions."""
from dataclasses import FrozenInstanceError
import uuid

import pytest

from pipeline.service_account_assertions import (
    ServiceAccountAssertion,
    ServiceAccountAssertionRefused,
)


TENANT = uuid.UUID("10000000-0000-0000-0000-000000000001")
APPROVAL = uuid.UUID("50000000-0000-0000-0000-000000000005")
ACCOUNT = uuid.UUID("30000000-0000-0000-0000-000000000003")


def _assertion(purpose="approval_list", **changes):
    shaped = purpose != "approval_list"
    redeem = purpose.startswith("approval_redeem_")
    values = {
        "assertion_version": 1,
        "purpose": purpose,
        "key_version": 7,
        "tenant_id": TENANT,
        "tenant_actor_digest": b"a" * 32,
        "org_policy_epoch": 9,
        "approval_id": APPROVAL if shaped else None,
        "approval_revision": 1 if shaped else None,
        "service_account_id": ACCOUNT if shaped else None,
        "credential_digest": b"c" * 32 if redeem else None,
        "assertion_limit": 10 if purpose == "approval_list" else None,
        "issued_at": 2_000_000_000,
        "expires_at": 2_000_000_030,
        "nonce": b"n" * 16,
        "mac": b"m" * 32,
    }
    values.update(changes)
    return ServiceAccountAssertion(**values)


@pytest.mark.parametrize("purpose", [
    "approval_list", "approval_get",
    "approval_redeem_issue", "approval_redeem_rotate",
])
def test_each_closed_purpose_has_one_exact_in_memory_shape(purpose):
    assertion = _assertion(purpose)
    assert assertion.purpose == purpose
    assert len(assertion.sql_values()) == 14
    assert assertion.sql_values()[0] == 1


@pytest.mark.parametrize(("purpose", "changes"), [
    ("unknown", {}),
    ("approval_list", {"approval_id": APPROVAL}),
    ("approval_list", {"assertion_limit": 0}),
    ("approval_get", {"approval_revision": None}),
    ("approval_get", {"credential_digest": b"c" * 32}),
    ("approval_redeem_issue", {"credential_digest": None}),
    ("approval_redeem_rotate", {"assertion_limit": 1}),
    ("approval_list", {"expires_at": 2_000_000_031}),
    ("approval_list", {"nonce": b"n" * 15}),
    ("approval_list", {"mac": b"m" * 31}),
])
def test_malformed_or_cross_purpose_shapes_are_refused(purpose, changes):
    with pytest.raises(ServiceAccountAssertionRefused) as refused:
        _assertion(purpose, **changes)
    assert str(refused.value) == "service account assertion is invalid"


def test_proof_material_is_not_repr_or_mutable_state():
    assertion = _assertion("approval_redeem_issue")
    rendered = repr(assertion)
    for secret in (repr(b"a" * 32), repr(b"c" * 32),
                   repr(b"n" * 16), repr(b"m" * 32)):
        assert secret not in rendered
    assert not hasattr(assertion, "__dict__")
    with pytest.raises(FrozenInstanceError):
        assertion.purpose = "approval_list"


def test_exact_type_checks_reject_bool_and_mutable_bytes():
    for changes in (
            {"key_version": True}, {"org_policy_epoch": True},
            {"tenant_actor_digest": bytearray(b"a" * 32)},
            {"nonce": bytearray(b"n" * 16)},
            {"mac": memoryview(b"m" * 32)}):
        with pytest.raises(ServiceAccountAssertionRefused):
            _assertion(**changes)
