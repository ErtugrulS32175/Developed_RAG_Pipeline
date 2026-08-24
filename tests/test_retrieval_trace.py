"""V2 retrieval traces expose process evidence, never retrieved content."""
from dataclasses import FrozenInstanceError

import pytest

from pipeline.retrieval import trace
from pipeline.retrieval.trace import RetrievalTrace, TraceStage


def _native_stages():
    return (
        TraceStage("plan", 1), TraceStage("retrieve", 2),
        TraceStage("rerank", 1), TraceStage("context", 1),
        TraceStage("generate", 4), TraceStage("validate", 1),
    )


def _valid(**overrides):
    values = {
        "trace_version": 2,
        "trace_id": "b" * 32,
        "backend": "native",
        "planner_policy_version": 1,
        "query_class": "factual",
        "retrieval_mode": "hybrid_balanced",
        "fallback": "none",
        "scope_kind": "all_visible",
        "policy_epoch": 3,
        "top_k": 15,
        "candidate_limit": 60,
        "scope_document_count": None,
        "retrieved_count": 15,
        "reranked_count": 15,
        "context_passage_count": 15,
        "context_utf8_bytes": 4096,
        "stages": _native_stages(),
    }
    values.update(overrides)
    return RetrievalTrace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trace_version", 1),
        ("trace_version", True),
        ("trace_id", "not-an-id"),
        ("backend", "private-backend"),
        ("planner_policy_version", 2),
        ("planner_policy_version", True),
        ("query_class", "summary"),
        ("retrieval_mode", "dense_only"),
        ("fallback", "retry"),
        ("scope_kind", "private"),
        ("policy_epoch", 0),
        ("policy_epoch", True),
        ("top_k", 14),
        ("top_k", True),
        ("candidate_limit", 59),
        ("candidate_limit", True),
        ("scope_document_count", -1),
        ("retrieved_count", True),
        ("reranked_count", -1),
        ("context_passage_count", "15"),
        ("context_utf8_bytes", -1),
        ("context_utf8_bytes", True),
        ("stages", []),
        ("stages", (TraceStage("plan", 1),
                    TraceStage("retrieve", 2))),
        ("reranked_count", 16),
        ("retrieved_count", 16),
        ("context_passage_count", 16),
    ],
)
def test_trace_contract_refuses_open_or_ambiguous_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        _valid(**{field: value})


@pytest.mark.parametrize("scope_kind", [
    "explicit_documents", "metadata_filters", "intersection",
])
def test_named_scopes_require_a_positive_resolved_document_count(scope_kind):
    with pytest.raises(ValueError, match="named scope"):
        _valid(scope_kind=scope_kind, scope_document_count=None)
    with pytest.raises(ValueError, match="named scope"):
        _valid(scope_kind=scope_kind, scope_document_count=0)
    assert _valid(scope_kind=scope_kind, scope_document_count=1)


def test_empty_and_all_visible_scope_counts_are_unambiguous():
    assert _valid(scope_kind="empty", scope_document_count=0,
                  retrieved_count=0, reranked_count=0,
                  context_passage_count=0, context_utf8_bytes=0)
    with pytest.raises(ValueError, match="empty scope"):
        _valid(scope_kind="empty", scope_document_count=None)
    with pytest.raises(ValueError, match="cannot produce"):
        _valid(scope_kind="empty", scope_document_count=0)
    assert _valid(scope_kind="all_visible", scope_document_count=None)
    assert _valid(scope_kind="all_visible", scope_document_count=3)
    with pytest.raises(ValueError, match="visible scope"):
        _valid(scope_kind="all_visible", scope_document_count=0)


@pytest.mark.parametrize(("passages", "size"), [(0, 1), (1, 0)])
def test_context_passage_count_and_utf8_bytes_agree_in_both_directions(
        passages, size):
    with pytest.raises(ValueError, match="context count and bytes"):
        _valid(context_passage_count=passages, context_utf8_bytes=size)


def test_backend_specific_stage_and_rerank_contracts_start_with_plan():
    llama = _valid(
        backend="llamaindex",
        reranked_count=None,
        stages=(
            TraceStage("plan", 1), TraceStage("retrieve", 2),
            TraceStage("context", 1), TraceStage("generate", 4),
            TraceStage("validate", 1),
        ),
    )
    assert llama.stages[0].name == "plan"
    with pytest.raises(ValueError, match="incomplete or out of order"):
        _valid(stages=_native_stages()[1:])
    with pytest.raises(ValueError, match="cannot claim"):
        _valid(
            backend="llamaindex",
            stages=(
                TraceStage("plan", 1), TraceStage("retrieve", 2),
                TraceStage("context", 1), TraceStage("generate", 4),
                TraceStage("validate", 1),
            ),
        )
    with pytest.raises(ValueError, match="requires a reranked"):
        _valid(reranked_count=None)
    with pytest.raises(ValueError, match="context count exceeds"):
        _valid(reranked_count=14)
    with pytest.raises(ValueError, match="context count exceeds"):
        _valid(
            backend="llamaindex", retrieved_count=14, reranked_count=None,
            stages=(
                TraceStage("plan", 1), TraceStage("retrieve", 2),
                TraceStage("context", 1), TraceStage("generate", 4),
                TraceStage("validate", 1),
            ),
        )


def test_closed_words_cannot_be_impersonated_by_string_subclasses():
    class LyingText(str):
        pass

    for field, value in (
            ("trace_id", LyingText("b" * 32)),
            ("backend", LyingText("native")),
            ("query_class", LyingText("factual")),
            ("retrieval_mode", LyingText("hybrid_balanced")),
            ("fallback", LyingText("none")),
            ("scope_kind", LyingText("all_visible"))):
        with pytest.raises(ValueError):
            _valid(**{field: value})


def test_public_trace_has_one_closed_v2_shape_and_no_content_slots():
    payload = _valid().public()
    assert payload == {
        "trace_version": 2,
        "trace_id": "b" * 32,
        "backend": "native",
        "planner_policy_version": 1,
        "query_class": "factual",
        "retrieval_mode": "hybrid_balanced",
        "fallback": "none",
        "scope_kind": "all_visible",
        "policy_epoch": 3,
        "top_k": 15,
        "candidate_limit": 60,
        "scope_document_count": None,
        "retrieved_count": 15,
        "reranked_count": 15,
        "context_passage_count": 15,
        "context_utf8_bytes": 4096,
        "stages_ms": {
            "plan": 1, "retrieve": 2, "rerank": 1, "context": 1,
            "generate": 4, "validate": 1,
        },
    }
    forbidden = {
        "query", "query_hash", "text", "filename", "tenant_id",
        "actor_id", "document_id", "chunk_id", "version_id",
        "citation_ref", "score", "content", "passages", "answer",
    }
    assert not (forbidden & set(payload))
    assert set(payload["stages_ms"]) == {
        "plan", "retrieve", "rerank", "context", "generate", "validate"}


def test_trace_is_frozen_slotted_and_new_trace_issues_only_a_correlation_id():
    created = trace.new_trace(
        backend="native", planner_policy_version=1, query_class="factual",
        retrieval_mode="hybrid_balanced", fallback="none",
        scope_kind="explicit_documents", policy_epoch=7, top_k=15,
        candidate_limit=60, scope_document_count=2, retrieved_count=2,
        reranked_count=2, context_passage_count=2,
        context_utf8_bytes=128, stages=_native_stages())
    assert len(created.trace_id) == 32
    assert created.trace_version == 2
    assert not hasattr(created, "__dict__")
    with pytest.raises(FrozenInstanceError):
        created.policy_epoch = 8
