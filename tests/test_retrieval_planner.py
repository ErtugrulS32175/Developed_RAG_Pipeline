from dataclasses import FrozenInstanceError, replace
import json

import pytest

from pipeline.retrieval import planner


QUERY_SENTINEL = "PRIVATE_QUERY_SENTINEL"
SCOPE_SENTINEL = "3" * 64


def _plan(query="kurgu soru", **overrides):
    values = {
        "backend": "native",
        "scope_kind": "all_visible",
        "policy_epoch": 1,
    }
    values.update(overrides)
    return planner.build_plan(query, **values)


def test_v1_preserves_the_measured_balanced_policy():
    plan = _plan()
    assert plan.policy_version == 1
    assert plan.policy_epoch == 1
    assert plan.query_class == "factual"
    assert plan.mode == "hybrid_balanced"
    assert plan.fallback == "none"
    assert plan.budget == planner.PlannerBudget(
        top_k=15, candidate_limit=60, rerank_limit=15, context_limit=15,
        candidate_utf8_max=262_144, context_utf8_max=131_072)


def test_the_two_backends_are_closed_and_share_v1_policy():
    native = _plan(backend="native")
    llama = _plan(backend="llamaindex")
    assert native.backend == "native" and llama.backend == "llamaindex"
    assert native.budget == llama.budget
    with pytest.raises(planner.PlannerError, match="^planner_backend_invalid$"):
        _plan(backend="unknown")


@pytest.mark.parametrize("query", [
    None, b"bytes", True, 7, "", " ", " leading", "trailing ",
    "line\nbreak", "x\x7fy", "x" * 8001, "\ud800",
])
def test_queries_are_exact_bounded_clean_utf8_strings(query):
    with pytest.raises(planner.PlannerError, match="^planner_query_invalid$"):
        _plan(query)


def test_query_limit_counts_utf8_bytes():
    accepted = "\u011f" * 4000
    assert _plan(accepted).mode == "hybrid_balanced"
    with pytest.raises(planner.PlannerError):
        _plan(accepted + "a")


@pytest.mark.parametrize("query", [
    "kurgu soru", '"tam ifade" nerede', "kurgu 47 degeri",
    "bir iki uc dort bes alti yedi sekiz dokuz on onbir oniki",
])
def test_v1_classification_is_deliberately_factual_only(query):
    first = _plan(query)
    second = _plan(query)
    assert first.query_class == "factual"
    assert first == second
    assert first.plan_sha256 == second.plan_sha256


def test_all_visible_and_empty_are_distinct_scope_semantics():
    visible = _plan(scope_kind="all_visible")
    empty = _plan(scope_kind="empty")
    assert visible.scope_kind == "all_visible"
    assert empty.scope_kind == "empty"
    assert visible.scope_sha256 != empty.scope_sha256
    assert visible.plan_sha256 != empty.plan_sha256


@pytest.mark.parametrize("scope_kind", sorted(planner.SCOPE_KINDS))
def test_every_closed_scope_kind_is_accepted(scope_kind):
    assert _plan(scope_kind=scope_kind).scope_kind == scope_kind


@pytest.mark.parametrize("scope_kind", [
    None, True, 7, "unscoped", "documents", "other",
])
def test_open_or_ambiguous_scope_kinds_are_refused(scope_kind):
    with pytest.raises(planner.PlannerError, match="^planner_scope_invalid$"):
        _plan(scope_kind=scope_kind)


@pytest.mark.parametrize("epoch", [None, True, 0, -1, 1.0, "1"])
def test_policy_epoch_is_an_exact_positive_integer(epoch):
    with pytest.raises(planner.PlannerError, match="^planner_policy_invalid$"):
        _plan(policy_epoch=epoch)
    assert _plan(policy_epoch=2).policy_epoch == 2


def test_optional_scope_digest_is_exact_lowercase_sha256():
    bound = _plan(scope_digest=SCOPE_SENTINEL)
    assert bound.scope_sha256 == SCOPE_SENTINEL
    assert SCOPE_SENTINEL not in json.dumps(bound.public())
    for attack in (True, b"x", "3" * 63, "G" * 64):
        with pytest.raises(planner.PlannerError):
            _plan(scope_digest=attack)


@pytest.mark.parametrize("field, value", [
    ("top_k", True), ("top_k", 14),
    ("candidate_limit", 59), ("rerank_limit", 14),
    ("context_limit", 16), ("candidate_utf8_max", 262_143),
    ("context_utf8_max", 131_071),
])
def test_budget_constructor_cannot_widen_or_narrow_v1(field, value):
    values = {"top_k": 15, "candidate_limit": 60,
              "rerank_limit": 15, "context_limit": 15,
              "candidate_utf8_max": 262_144,
              "context_utf8_max": 131_072}
    values[field] = value
    with pytest.raises(planner.PlannerError, match="^planner_budget_invalid$"):
        planner.PlannerBudget(**values)


def test_plan_and_budget_are_frozen_and_slotted():
    plan = _plan()
    with pytest.raises(FrozenInstanceError):
        plan.mode = "other"
    with pytest.raises(FrozenInstanceError):
        plan.budget.top_k = 100
    assert not hasattr(plan, "__dict__")
    assert not hasattr(plan.budget, "__dict__")


def test_execution_query_must_match_the_bound_query_exactly():
    plan = _plan("kurgu soru")
    assert planner.verify_query(plan, "kurgu soru") is None
    with pytest.raises(
            planner.PlannerQueryMismatch, match="^planner_query_mismatch$"):
        planner.verify_query(plan, "baska soru")


def test_plan_digest_binds_policy_scope_backend_and_query():
    base = _plan()
    alternatives = (
        _plan(backend="llamaindex"),
        _plan(scope_kind="empty"),
        _plan(policy_epoch=2),
        _plan("baska soru"),
        _plan(scope_digest=SCOPE_SENTINEL),
    )
    assert all(base.plan_sha256 != item.plan_sha256 for item in alternatives)


def test_direct_dataclass_replacement_cannot_forge_a_valid_plan():
    plan = _plan()
    with pytest.raises(planner.PlannerError, match="^planner_plan_invalid$"):
        replace(plan, policy_epoch=2)


def test_public_decision_is_exact_and_content_free():
    plan = _plan(QUERY_SENTINEL, scope_kind="explicit_documents",
                 policy_epoch=7, scope_digest=SCOPE_SENTINEL)
    public = plan.public()
    assert public == {
        "policy_version": 1,
        "policy_epoch": 7,
        "backend": "native",
        "query_class": "factual",
        "mode": "hybrid_balanced",
        "fallback": "none",
        "scope_kind": "explicit_documents",
        "budget": {
            "top_k": 15, "candidate_limit": 60,
            "rerank_limit": 15, "context_limit": 15,
            "candidate_utf8_max": 262_144,
            "context_utf8_max": 131_072,
        },
    }
    rendered = json.dumps(public, sort_keys=True) + repr(plan)
    assert QUERY_SENTINEL not in rendered
    assert SCOPE_SENTINEL not in rendered
    assert plan.query_sha256 not in rendered
    assert plan.plan_sha256 not in rendered


def test_errors_and_reprs_never_echo_private_values():
    rendered = []
    with pytest.raises(planner.PlannerError) as caught:
        _plan(QUERY_SENTINEL + "\n")
    rendered.append(repr(caught.value))
    with pytest.raises(planner.PlannerError) as caught:
        _plan(scope_digest=SCOPE_SENTINEL.upper().replace("3", "G"))
    rendered.append(repr(caught.value))
    joined = " ".join(rendered)
    assert QUERY_SENTINEL not in joined
    assert SCOPE_SENTINEL not in joined
