"""Context assembly: every retrieved passage must reach the model with a usable
source citation, since the answer prompt requires the answer to cite its page.
"""
import pytest

from pipeline.retrieval.context import RagContext
from pipeline.retrieval.query import build_context, build_rag_context, citation


def _chunk(**kw):
    base = {"type": "text", "text": "govde", "filename": "belge.pdf",
            "page": 26, "headings": [], "table_data": None}
    base.update(kw)
    return base


def test_text_chunk_cites_document_and_page():
    assert citation(_chunk()) == "[belge.pdf | Sayfa 26]"


def test_table_from_a_native_pdf_is_still_cited():
    """The regression this exists for: table chunks used to be passed through
    verbatim on the assumption their text carried its own header. Tables parsed
    out of a native PDF have no such header, so they reached the model with no
    source at all."""
    c = citation(_chunk(type="table", table_data=None))
    assert "belge.pdf" in c and "Sayfa 26" in c and "tablo" in c


def test_verified_table_carries_its_confidence():
    c = citation(_chunk(type="table", table_data={"confidence": 0.87}))
    assert "guven 0.87" in c


def test_section_path_is_included():
    c = citation(_chunk(headings=["Bolum A", "Alt baslik"]))
    assert "Bolum A > Alt baslik" in c


def test_missing_metadata_degrades_gracefully():
    assert citation({"text": "x"}) == "[? | Sayfa 0]"


def test_every_passage_in_the_context_gets_a_citation():
    chunks = [_chunk(page=3), _chunk(type="table", page=7), _chunk(page=11)]
    ctx = build_context(chunks)
    assert ctx.count("belge.pdf") == 3
    for page in (3, 7, 11):
        assert f"Sayfa {page}" in ctx


def test_model_text_and_provenance_are_built_from_the_same_chunks():
    context = build_rag_context(
        [_chunk(page=3, text="zeta"), _chunk(page=7, text="gamma")],
        numbered=True,
    )

    assert "[P1]" in context.model_text and "[P2]" in context.model_text
    assert [(p.handle, p.page, p.text) for p in context.passages] == [
        (1, 3, "zeta"),
        (2, 7, "gamma"),
    ]


def test_database_identity_stays_out_of_model_text_but_in_provenance():
    chunk_id = "00000000-0000-0000-0000-000000000271"
    context = build_rag_context(
        [_chunk(id=chunk_id, filename="kurgu.pdf")], numbered=True)

    passage = context.passages[0]
    assert passage.chunk_id == chunk_id
    assert passage.document_name == "kurgu.pdf"
    assert chunk_id not in context.model_text


def test_untrusted_chunk_identity_cannot_become_evidence_authority():
    context = build_rag_context([_chunk(id="not-a-chunk")], numbered=True)
    assert context.passages[0].chunk_id is None


def test_numbered_text_cannot_discard_its_paired_provenance():
    with pytest.raises(ValueError, match="build_rag_context"):
        build_context([_chunk()], numbered=True)


def test_provenance_collection_cannot_be_mutable():
    with pytest.raises(TypeError, match="immutable tuple"):
        RagContext(passages=[], numbered=False)


def test_passages_stay_separated():
    ctx = build_context([_chunk(text="bir"), _chunk(text="iki")])
    assert ctx.count("\n\n---\n\n") == 1
    assert "bir" in ctx and "iki" in ctx


def test_empty_context_is_empty_not_an_error():
    assert build_context([]) == ""


def test_reranking_never_shrinks_the_context():
    """The reranker orders passages; discarding some of them silently removes
    content that retrieval had already found. Measured: cutting the list made a
    question unanswerable while every page-level metric still read 1.0."""
    from pipeline.retrieval import query
    assert query.TOP_RERANK >= query.TOP_K


def test_candidate_byte_budget_refuses_before_rerank(monkeypatch):
    from pipeline.retrieval import planner, query

    chunks = [_chunk(text="x" * 20_000) for _ in range(15)]
    monkeypatch.setattr(query, "retrieve", lambda _question: chunks)
    monkeypatch.setattr(
        query, "rerank",
        lambda *_args, **_kwargs: pytest.fail("reranker was called"))
    with pytest.raises(planner.PlannerError,
                       match="^planner_candidate_limit$"):
        query.ask_checked("bounded question")


def test_context_byte_budget_refuses_before_generation(monkeypatch):
    from pipeline.retrieval import planner, query

    chunks = [_chunk(text="x" * planner.CONTEXT_UTF8_MAX)]
    monkeypatch.setattr(query, "retrieve", lambda _question: chunks)
    monkeypatch.setattr(
        query.gen, "generate_structured",
        lambda *_args, **_kwargs: pytest.fail("generator was called"))
    with pytest.raises(planner.PlannerError,
                       match="^planner_context_limit$"):
        query.ask_checked("bounded question")


@pytest.mark.parametrize("name", ["TOP_K", "TOP_RERANK"])
def test_runtime_retrieval_knobs_cannot_disagree_with_the_plan(
        monkeypatch, name):
    from pipeline.retrieval import planner, query

    current = getattr(query, name)
    monkeypatch.setattr(query, name, current + 1)
    monkeypatch.setattr(
        query, "retrieve",
        lambda *_args, **_kwargs: pytest.fail("retrieval was called"))

    with pytest.raises(planner.PlannerError,
                       match="^planner_runtime_policy_mismatch$"):
        query.ask_checked("bounded question")


# --- answering-endpoint auth (lives with generation, not retrieval) ---

def test_no_header_when_no_key_is_set(monkeypatch):
    """The local and tunnelled cases: an unauthenticated endpoint must not be
    sent an empty Bearer token, which some servers reject outright."""
    import pipeline.generation.answer as q
    monkeypatch.setattr(q, "LLM_API_KEY", "")
    assert q.llm_headers() == {}


def test_key_is_sent_as_bearer(monkeypatch):
    """Needed once the endpoint is a rented GPU behind a public proxy URL."""
    import pipeline.generation.answer as q
    monkeypatch.setattr(q, "LLM_API_KEY", "test-api-key")
    assert q.llm_headers() == {"Authorization": "Bearer test-api-key"}
