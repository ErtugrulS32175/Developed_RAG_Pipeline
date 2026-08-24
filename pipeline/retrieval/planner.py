"""Deterministic, content-free retrieval planning authority.

V1 deliberately preserves the production retrieval breadth.  Its sole query
class and mode are recorded now so a later policy can be evaluated before it
changes behaviour.  Query and scope digests bind execution internally; neither
digest nor any content or identifier crosses the public decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re


POLICY_VERSION = 1
QUERY_UTF8_MAX = 8000

BACKENDS = frozenset({"native", "llamaindex"})
QUERY_CLASSES = frozenset({"factual"})
SCOPE_KINDS = frozenset({
    "all_visible", "explicit_documents", "metadata_filters",
    "intersection", "empty",
})
MODES = frozenset({"hybrid_balanced"})
FALLBACKS = frozenset({"none"})

TOP_K = 15
CANDIDATE_LIMIT = 60
RERANK_LIMIT = 15
CONTEXT_LIMIT = 15
CANDIDATE_UTF8_MAX = 262_144
CONTEXT_UTF8_MAX = 131_072

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PlannerError(ValueError):
    """An offered planner value failed a closed contract gate."""


class PlannerQueryMismatch(PlannerError):
    """Execution offered a query other than the one this plan binds."""


def _refuse(code: str):
    raise PlannerError(code)


def _query(value) -> str:
    if type(value) is not str:
        _refuse("planner_query_invalid")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        _refuse("planner_query_invalid")
    if (not encoded or len(encoded) > QUERY_UTF8_MAX
            or value != value.strip()
            or any(ord(char) < 0x20 or ord(char) == 0x7f for char in value)):
        _refuse("planner_query_invalid")
    return value


def _scope(value) -> str:
    if type(value) is not str or value not in SCOPE_KINDS:
        _refuse("planner_scope_invalid")
    return value


def _canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def query_class(query: str) -> str:
    """Return V1's sole class after validating the exact query bytes."""
    _query(query)
    return "factual"


@dataclass(frozen=True, slots=True)
class PlannerBudget:
    top_k: int = TOP_K
    candidate_limit: int = CANDIDATE_LIMIT
    rerank_limit: int = RERANK_LIMIT
    context_limit: int = CONTEXT_LIMIT
    candidate_utf8_max: int = CANDIDATE_UTF8_MAX
    context_utf8_max: int = CONTEXT_UTF8_MAX

    def __post_init__(self):
        if (
            type(self.top_k) is not int or self.top_k != TOP_K
            or type(self.candidate_limit) is not int
            or self.candidate_limit != CANDIDATE_LIMIT
            or type(self.rerank_limit) is not int
            or self.rerank_limit != RERANK_LIMIT
            or type(self.context_limit) is not int
            or self.context_limit != CONTEXT_LIMIT
            or type(self.candidate_utf8_max) is not int
            or self.candidate_utf8_max != CANDIDATE_UTF8_MAX
            or type(self.context_utf8_max) is not int
            or self.context_utf8_max != CONTEXT_UTF8_MAX
        ):
            _refuse("planner_budget_invalid")

    def public(self) -> dict:
        return {
            "top_k": self.top_k,
            "candidate_limit": self.candidate_limit,
            "rerank_limit": self.rerank_limit,
            "context_limit": self.context_limit,
            "candidate_utf8_max": self.candidate_utf8_max,
            "context_utf8_max": self.context_utf8_max,
        }


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    policy_version: int
    policy_epoch: int
    backend: str
    query_class: str
    mode: str
    fallback: str
    scope_kind: str
    budget: PlannerBudget
    query_sha256: str = field(repr=False)
    scope_sha256: str = field(repr=False)
    plan_sha256: str = field(repr=False)

    def __post_init__(self):
        if type(self.policy_version) is not int or self.policy_version != 1:
            _refuse("planner_plan_invalid")
        if type(self.policy_epoch) is not int or self.policy_epoch < 1:
            _refuse("planner_plan_invalid")
        if type(self.backend) is not str or self.backend not in BACKENDS:
            _refuse("planner_plan_invalid")
        if (type(self.query_class) is not str
                or self.query_class not in QUERY_CLASSES):
            _refuse("planner_plan_invalid")
        if type(self.mode) is not str or self.mode not in MODES:
            _refuse("planner_plan_invalid")
        if type(self.fallback) is not str or self.fallback not in FALLBACKS:
            _refuse("planner_plan_invalid")
        _scope(self.scope_kind)
        if type(self.budget) is not PlannerBudget:
            _refuse("planner_plan_invalid")
        if any(type(value) is not str or not _HEX64.fullmatch(value) for value in (
                self.query_sha256, self.scope_sha256, self.plan_sha256)):
            _refuse("planner_plan_invalid")
        expected = _plan_digest(
            self.policy_version, self.policy_epoch, self.backend,
            self.query_class, self.mode, self.fallback, self.scope_kind,
            self.budget, self.query_sha256, self.scope_sha256)
        if self.plan_sha256 != expected:
            _refuse("planner_plan_invalid")

    def public(self) -> dict:
        """Return the exact content-free decision allowed at the API edge."""
        return {
            "policy_version": self.policy_version,
            "policy_epoch": self.policy_epoch,
            "backend": self.backend,
            "query_class": self.query_class,
            "mode": self.mode,
            "fallback": self.fallback,
            "scope_kind": self.scope_kind,
            "budget": self.budget.public(),
        }


def _plan_digest(policy_version, policy_epoch, backend, classification, mode,
                 fallback, scope_kind, budget, query_sha256,
                 scope_sha256) -> str:
    return _digest({
        "backend": backend,
        "budget": budget.public(),
        "fallback": fallback,
        "mode": mode,
        "policy_epoch": policy_epoch,
        "policy_version": policy_version,
        "query_class": classification,
        "query_sha256": query_sha256,
        "scope_kind": scope_kind,
        "scope_sha256": scope_sha256,
    })


def build_plan(query, *, backend, scope_kind, policy_epoch,
               scope_digest=None):
    """Build one V1 plan bound to the exact query, policy and scope."""
    checked_query = _query(query)
    if type(backend) is not str or backend not in BACKENDS:
        _refuse("planner_backend_invalid")
    checked_scope = _scope(scope_kind)
    if type(policy_epoch) is not int or policy_epoch < 1:
        _refuse("planner_policy_invalid")
    if scope_digest is None:
        scope_sha256 = _digest({"scope_kind": checked_scope})
    elif type(scope_digest) is str and _HEX64.fullmatch(scope_digest):
        scope_sha256 = scope_digest
    else:
        _refuse("planner_scope_invalid")

    budget = PlannerBudget()
    classification = query_class(checked_query)
    query_sha256 = hashlib.sha256(checked_query.encode("utf-8")).hexdigest()
    plan_sha256 = _plan_digest(
        POLICY_VERSION, policy_epoch, backend, classification,
        "hybrid_balanced", "none", checked_scope, budget, query_sha256,
        scope_sha256)
    return RetrievalPlan(
        policy_version=POLICY_VERSION,
        policy_epoch=policy_epoch,
        backend=backend,
        query_class=classification,
        mode="hybrid_balanced",
        fallback="none",
        scope_kind=checked_scope,
        budget=budget,
        query_sha256=query_sha256,
        scope_sha256=scope_sha256,
        plan_sha256=plan_sha256,
    )


def verify_query(plan, query) -> None:
    """Prove execution uses the exact UTF-8 query bound by ``plan``."""
    if type(plan) is not RetrievalPlan:
        _refuse("planner_plan_invalid")
    # Direct dataclass construction is not authority: re-run every invariant.
    plan.__post_init__()
    checked = _query(query)
    offered = hashlib.sha256(checked.encode("utf-8")).hexdigest()
    if offered != plan.query_sha256:
        raise PlannerQueryMismatch("planner_query_mismatch")


# Kept for callers built against the initial isolated slice.
assert_query = verify_query
