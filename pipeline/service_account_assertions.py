"""Closed in-memory shape for one cross-database tenant assertion."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


PURPOSES = (
    "approval_get",
    "approval_list",
    "approval_redeem_issue",
    "approval_redeem_rotate",
)


class ServiceAccountAssertionRefused(ValueError):
    """An assertion row escaped the closed cross-database contract."""


@dataclass(frozen=True, slots=True)
class ServiceAccountAssertion:
    assertion_version: int
    purpose: str
    key_version: int
    tenant_id: uuid.UUID
    tenant_actor_digest: bytes = field(repr=False)
    org_policy_epoch: int
    approval_id: uuid.UUID | None
    approval_revision: int | None
    service_account_id: uuid.UUID | None
    credential_digest: bytes | None = field(repr=False)
    assertion_limit: int | None
    issued_at: int
    expires_at: int
    nonce: bytes = field(repr=False)
    mac: bytes = field(repr=False)

    def __post_init__(self) -> None:
        exact_ints = (
            self.assertion_version, self.key_version,
            self.org_policy_epoch, self.issued_at, self.expires_at,
        )
        if (any(type(value) is not int for value in exact_ints)
                or self.assertion_version != 1
                or self.key_version < 1 or self.org_policy_epoch < 1
                or self.expires_at - self.issued_at != 30
                or type(self.purpose) is not str
                or self.purpose not in PURPOSES
                or type(self.tenant_id) is not uuid.UUID
                or type(self.tenant_actor_digest) is not bytes
                or len(self.tenant_actor_digest) != 32
                or type(self.nonce) is not bytes or len(self.nonce) != 16
                or type(self.mac) is not bytes or len(self.mac) != 32):
            raise ServiceAccountAssertionRefused(
                "service account assertion is invalid")
        optional_uuids = (self.approval_id, self.service_account_id)
        if any(value is not None and type(value) is not uuid.UUID
               for value in optional_uuids):
            raise ServiceAccountAssertionRefused(
                "service account assertion is invalid")
        if (self.approval_revision is not None
                and (type(self.approval_revision) is not int
                     or self.approval_revision < 1)):
            raise ServiceAccountAssertionRefused(
                "service account assertion is invalid")
        if (self.assertion_limit is not None
                and (type(self.assertion_limit) is not int
                     or not 1 <= self.assertion_limit <= 100)):
            raise ServiceAccountAssertionRefused(
                "service account assertion is invalid")
        if (self.credential_digest is not None
                and (type(self.credential_digest) is not bytes
                     or len(self.credential_digest) != 32)):
            raise ServiceAccountAssertionRefused(
                "service account assertion is invalid")
        list_shape = (
            self.approval_id is None
            and self.approval_revision is None
            and self.service_account_id is None
            and self.credential_digest is None
            and self.assertion_limit is not None
        )
        get_shape = (
            self.approval_id is not None
            and self.approval_revision is not None
            and self.service_account_id is not None
            and self.credential_digest is None
            and self.assertion_limit is None
        )
        redeem_shape = (
            self.approval_id is not None
            and self.approval_revision is not None
            and self.service_account_id is not None
            and self.credential_digest is not None
            and self.assertion_limit is None
        )
        expected = {
            "approval_list": list_shape,
            "approval_get": get_shape,
            "approval_redeem_issue": redeem_shape,
            "approval_redeem_rotate": redeem_shape,
        }
        if expected[self.purpose] is not True:
            raise ServiceAccountAssertionRefused(
                "service account assertion is invalid")

    def sql_values(self) -> tuple:
        """Return the exact wire-order values without logging-friendly text."""
        return (
            self.assertion_version, self.key_version, self.tenant_id,
            self.tenant_actor_digest, self.org_policy_epoch,
            self.approval_id, self.approval_revision,
            self.service_account_id, self.credential_digest,
            self.assertion_limit, self.issued_at, self.expires_at,
            self.nonce, self.mac,
        )
