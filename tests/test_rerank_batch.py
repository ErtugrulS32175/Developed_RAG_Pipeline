"""The reranker sends ONE request and must fail loudly on a bad response.

The service seam is faked; live-service equivalence (same ordering as the
per-chunk loop, scores within batched-kernel numerics) was verified
separately against the running reranker.
"""
import pytest

from pipeline.retrieval import query


class FakeResponse:
    def __init__(self, scores, shuffle=False):
        items = [{"index": i, "score": s} for i, s in enumerate(scores)]
        if shuffle:
            items = items[::-1]  # deterministic disorder
        self._items = items

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self._items}


def _capture(monkeypatch, scores, shuffle=False):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(json)
        return FakeResponse(scores, shuffle=shuffle)

    monkeypatch.setattr(query.requests, "post", post)
    return calls


def test_all_candidates_go_in_one_request(monkeypatch):
    calls = _capture(monkeypatch, [0.1, 0.9, 0.5])
    chunks = [{"text": "bir"}, {"text": "iki"}, {"text": "uc"}]

    ordered = query.rerank("kurgu soru", chunks, top_n=3)

    assert len(calls) == 1
    assert calls[0]["text_2"] == ["bir", "iki", "uc"]
    assert [c["text"] for c in ordered] == ["iki", "uc", "bir"]


def test_top_n_still_cuts_after_ordering(monkeypatch):
    _capture(monkeypatch, [0.1, 0.9, 0.5])
    chunks = [{"text": "bir"}, {"text": "iki"}, {"text": "uc"}]
    assert [c["text"] for c in query.rerank("s", chunks, top_n=2)] == ["iki", "uc"]


def test_scores_bind_through_the_index_field_not_response_order(monkeypatch):
    """The auditor's attack: a service answering out of order must not attach
    scores to the wrong chunks. Binding goes through each item's index."""
    _capture(monkeypatch, [0.1, 0.9, 0.5], shuffle=True)
    chunks = [{"text": "bir"}, {"text": "iki"}, {"text": "uc"}]

    ordered = query.rerank("kurgu soru", chunks, top_n=3)

    assert [c["text"] for c in ordered] == ["iki", "uc", "bir"]


def test_a_non_permutation_response_is_refused(monkeypatch):
    def post(url, json=None, timeout=None):
        class Bad:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"index": 0, "score": 0.1},
                                 {"index": 0, "score": 0.2},
                                 {"index": 2, "score": 0.3}]}
        return Bad()

    monkeypatch.setattr(query.requests, "post", post)
    with pytest.raises(ValueError):
        query.rerank("kurgu soru", [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                     top_n=3)


def test_a_mismatched_score_count_is_an_error_not_a_silent_drop(monkeypatch):
    _capture(monkeypatch, [0.1, 0.9])
    chunks = [{"text": "bir"}, {"text": "iki"}, {"text": "uc"}]
    with pytest.raises(ValueError):
        query.rerank("kurgu soru", chunks, top_n=3)


def test_empty_candidates_never_touch_the_service(monkeypatch):
    calls = _capture(monkeypatch, [])
    assert query.rerank("kurgu soru", [], top_n=5) == []
    assert calls == []


def test_one_candidate_is_already_ranked_and_never_touches_the_service(monkeypatch):
    calls = _capture(monkeypatch, [0.5])
    chunk = {"text": "tek pasaj"}
    assert query.rerank("kurgu soru", [chunk], top_n=1) == [chunk]
    assert calls == []
