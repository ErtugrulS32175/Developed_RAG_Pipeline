import pytest

from pipeline.index import db


def _row(identity, text=None):
    return {"id": identity, "text": text or f"passage-{identity}"}


def test_candidate_breadth_is_bounded_but_never_shrinks_the_request():
    assert db.rrf_candidate_limit(1) == 4
    assert db.rrf_candidate_limit(15) == 60
    assert db.rrf_candidate_limit(100) == 200
    assert db.rrf_candidate_limit(400) == 400


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "15", None])
def test_candidate_breadth_refuses_ambiguous_sizes(value):
    with pytest.raises(ValueError):
        db.rrf_candidate_limit(value)


def test_agreement_below_each_final_cut_can_win_the_fusion():
    dense = [_row("dense-only"), _row("agreed"), _row("tail-a")]
    sparse = [_row("sparse-only"), _row("agreed"), _row("tail-b")]

    fused = db.reciprocal_rank_fusion(dense, sparse, top_k=1, rrf_k=1)

    assert [row["id"] for row in fused] == ["agreed"]


def test_equal_scores_have_a_stable_chunk_identity_tie_break():
    dense = [_row("zeta")]
    sparse = [_row("alpha")]
    assert [row["id"] for row in db.reciprocal_rank_fusion(
        dense, sparse, top_k=2, rrf_k=1)] == ["alpha", "zeta"]


def test_a_repeated_identity_inside_one_ranking_is_refused():
    with pytest.raises(ValueError, match="tekrarladi"):
        db.reciprocal_rank_fusion(
            [_row("same"), _row("same")], [], top_k=1, rrf_k=1)


def test_two_rankings_cannot_disagree_about_one_chunk_payload():
    with pytest.raises(ValueError, match="farkli veri"):
        db.reciprocal_rank_fusion(
            [_row("same", "first")], [_row("same", "second")],
            top_k=1, rrf_k=1,
        )


@pytest.mark.parametrize("rrf_k", [-1, True, 1.5, "1", None])
def test_rrf_policy_is_a_closed_non_negative_integer(rrf_k):
    with pytest.raises(ValueError):
        db.reciprocal_rank_fusion([_row("a")], [], 1, rrf_k)
