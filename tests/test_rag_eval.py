"""Scoring logic of the retrieval eval. The live half (embedding, pgvector,
reranker) is exercised by running the harness; these cover the arithmetic that
every conclusion about retrieval quality will rest on.
"""
from collections import Counter

from eval.rag_eval import first_hit_rank, rarest_terms, summarize


# --- first_hit_rank ---

def test_rank_is_one_based():
    assert first_hit_rank([9, 4, 7], [4]) == 2
    assert first_hit_rank([4, 9, 7], [4]) == 1


def test_miss_is_none_not_zero():
    """0 would silently average into MRR as a perfect-ish score."""
    assert first_hit_rank([1, 2, 3], [99]) is None


def test_any_expected_page_counts():
    assert first_hit_rank([5, 8], [8, 12]) == 2


def test_empty_result_is_a_miss():
    assert first_hit_rank([], [3]) is None


# --- summarize ---

def test_mrr_counts_a_miss_as_zero():
    """So runs with different k stay comparable -- otherwise widening the net
    would look worse simply by surfacing more answerable questions."""
    m = summarize([1, 2, None, None])
    assert m["mrr"] == round((1 + 0.5) / 4, 4)
    assert m["miss"] == 2
    assert m["n"] == 4


def test_hit_at_k_is_cumulative():
    m = summarize([1, 3, 11, None])
    assert m["hit@1"] == 0.25
    assert m["hit@3"] == 0.5
    assert m["hit@10"] == 0.5
    assert m["hit@20"] == 0.75


def test_perfect_run():
    m = summarize([1, 1, 1])
    assert m["mrr"] == 1.0 and m["miss"] == 0 and m["hit@1"] == 1.0


def test_empty_question_set_does_not_divide_by_zero():
    assert summarize([]) == {"n": 0}


def test_median_rank_ignores_misses():
    assert summarize([2, 4, None])["median_rank"] == 3


# --- rarest_terms ---

def test_picks_the_least_common_words():
    freq = Counter({"sirket": 500, "rapor": 400, "tesvik": 3, "kalem": 7})
    assert rarest_terms("sirket rapor tesvik kalem", freq, 2) == ["tesvik", "kalem"]


def test_short_words_and_numbers_are_not_terms():
    freq = Counter({"kalem": 1})
    assert rarest_terms("bir iki 12345 kalem", freq, 5) == ["kalem"]


def test_returns_fewer_than_asked_when_text_is_thin():
    assert len(rarest_terms("kalem", Counter({"kalem": 1}), 6)) == 1
