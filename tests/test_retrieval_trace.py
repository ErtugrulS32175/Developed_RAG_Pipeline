"""Closed retrieval traces expose process evidence, never retrieved content."""
import pytest

from pipeline.retrieval.trace import RetrievalTrace, TraceStage


def _valid(**overrides):
    values = {
        "trace_id": "b" * 32,
        "backend": "native",
        "scope_document_count": None,
        "retrieved_count": 15,
        "reranked_count": 15,
        "context_passage_count": 15,
        "stages": (
            TraceStage("retrieve", 2), TraceStage("rerank", 1),
            TraceStage("context", 1), TraceStage("generate", 4),
            TraceStage("validate", 1),
        ),
    }
    values.update(overrides)
    return RetrievalTrace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trace_id", "not-an-id"),
        ("backend", "private-backend"),
        ("scope_document_count", -1),
        ("retrieved_count", True),
        ("reranked_count", -1),
        ("context_passage_count", "15"),
        ("stages", []),
        ("stages", (TraceStage("retrieve", 1),
                    TraceStage("retrieve", 2))),
        ("reranked_count", 16),
        ("context_passage_count", 16),
    ],
)
def test_trace_contract_refuses_open_or_ambiguous_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        _valid(**{field: value})


def test_public_trace_has_one_closed_shape_and_no_content_slots():
    payload = _valid().public()
    assert payload == {
        "trace_id": "b" * 32,
        "backend": "native",
        "scope_document_count": None,
        "retrieved_count": 15,
        "reranked_count": 15,
        "context_passage_count": 15,
        "stages_ms": {"retrieve": 2, "rerank": 1, "context": 1,
                      "generate": 4, "validate": 1},
    }
    assert not ({"query", "text", "filename", "document_id", "score"}
                & set(payload))
