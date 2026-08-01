"""Checks that an answer stayed inside the passages it was given.

Every figure below is invented -- test fixtures must never be copied out of a
real document.

These are one-sided by construction: passing means nothing was invented, not
that the right value was chosen. The tests that matter most are the ones
pinning down what the checks CANNOT see, because that is what decides whether
they may ever be allowed to block an answer.
"""
from pipeline.retrieval.query import build_rag_context
from pipeline.validation.rag.answer_guard import (
    check, context_pages, unsupported_figures, unsupported_pages)

CHUNKS = [
    {
        "filename": "belge.pdf",
        "page": 42,
        "headings": ["Bolum"],
        "text": "47 000 zeta uretildi.",
    },
    {
        "filename": "belge.pdf",
        "page": 43,
        "text": "milyonda 47 oraninda ucret alinir.",
    },
]
RAG_CONTEXT = build_rag_context(CHUNKS)
CONTEXT = RAG_CONTEXT.model_text


# --- which pages were actually supplied -------------------------------------

def test_pages_come_from_trusted_retrieval_metadata():
    assert context_pages(RAG_CONTEXT) == {42, 43}


def test_a_page_named_inside_a_passage_is_not_a_supplied_page():
    """A document refers to its own pages; only retrieval metadata is trusted."""
    ctx = build_rag_context([{
        "filename": "belge.pdf",
        "page": 42,
        "text": "ayrintilar Sayfa 99 uzerinde yer alir.",
    }])
    assert context_pages(ctx) == {42}


def test_a_cited_page_that_was_never_supplied_is_flagged():
    assert unsupported_pages("Sayfa 90'a gore boyledir.", RAG_CONTEXT) == [90]
    assert unsupported_pages("Sayfa 42'ye gore boyledir.", RAG_CONTEXT) == []


# --- figures ----------------------------------------------------------------

def test_a_figure_taken_from_a_passage_is_supported():
    assert unsupported_figures("Sayfa 42'ye gore 47 000 zeta uretilmistir",
                               CONTEXT) == []


def test_a_figure_no_passage_contains_is_flagged():
    assert unsupported_figures("Sayfa 42'ye gore 8 000 zeta uretilmistir",
                               CONTEXT) == [[8000.0]]


def test_the_page_citation_is_not_treated_as_a_claim():
    """'Sayfa 42' is the model pointing at its source, not an assertion about
    the world. Counting it as one would flag every correctly cited answer,
    since a page number is rarely also a figure in the text."""
    assert unsupported_figures("Sayfa 90 icin veri yok", "[belge.pdf | Sayfa 90]\nx") == []


def test_the_same_figure_written_as_words_still_counts():
    assert unsupported_figures("kirk yedi bin zeta uretilmistir", CONTEXT) == []


def test_small_numbers_can_be_ignored():
    """Clause and item markers are numbers but not data."""
    answer = "Sayfa 42'ye gore, madde 3 uyarinca 47 000 zeta uretilmistir"
    assert unsupported_figures(answer, CONTEXT) == [[3.0]]
    assert unsupported_figures(answer, CONTEXT, minimum=10) == []


# --- rates ------------------------------------------------------------------

def test_a_rate_restated_as_a_percentage_is_supported():
    """Turkish states a small rate as a fraction phrase and a model restates it
    as a percentage. The restated value is nowhere in the source, so without
    this every such answer is flagged."""
    assert unsupported_figures("oran yuzde 0,0047 seviyesindedir", CONTEXT) == []


def test_without_derivation_the_same_answer_is_flagged():
    assert unsupported_figures("oran yuzde 0,0047 seviyesindedir", CONTEXT,
                               derive=False) == [[0.0047]]


def test_deriving_over_the_whole_context_can_absorb_a_real_error():
    """Measured on real answers, and the reason derivation is optional: a wrong
    restatement was covered because an UNRELATED passage implied the same
    value. Nothing here can tell the two apart -- only an answer that says
    WHICH passage it used can, which is why the check has to become passage
    scoped before it is ever allowed to block anything."""
    ctx = CONTEXT + "\n\n---\n\n[belge.pdf | Sayfa 44]\nyuzde 47 pay ayrilir."
    assert unsupported_figures("oran 0,47 seviyesindedir", ctx) == []
    assert unsupported_figures("oran 0,47 seviyesindedir", ctx,
                               derive=False) == [[0.47]]


# --- the combined check -----------------------------------------------------

def test_a_clean_answer_raises_nothing():
    assert check("Sayfa 42'ye gore 47 000 zeta uretilmistir", RAG_CONTEXT) == []


def test_both_kinds_of_flag_are_reported_together():
    names = {name for name, _ in
             check("Sayfa 90'a gore 8 000 zeta uretilmistir", RAG_CONTEXT)}
    assert names == {"kaynaksiz_sayi", "kaynaksiz_sayfa"}


def test_empty_answer_and_context_do_not_raise():
    assert check("", build_rag_context([])) == []
    assert check(None, build_rag_context([])) == []
