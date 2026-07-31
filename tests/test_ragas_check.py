"""Judge-agreement arithmetic for the Ragas cross-check.

Ragas scores with an LLM. Whether an LLM judges Turkish reliably is unknown, so
before depending on it we compare its verdict to the deterministic one we
already trust. These cover the comparison itself; the scoring needs a live LLM.
"""
from eval.answer.ragas_check import agreement


def _row(fc, ours):
    return {"factual_correctness": fc, "bizim_dogru": ours}


def test_full_agreement():
    rows = [_row(0.9, True), _row(0.1, False), _row(0.8, True)]
    a = agreement(rows)
    assert a["uyum"] == 1.0
    assert a["hakem_daha_katı"] == 0 and a["hakem_daha_gevşek"] == 0


def test_a_judge_that_fails_answers_we_scored_right_is_flagged_as_stricter():
    """The direction matters: a judge that rejects correct answers wastes
    investigation time, one that passes wrong answers hides real defects."""
    a = agreement([_row(0.2, True), _row(0.1, True)])
    assert a["uyum"] == 0.0
    assert a["hakem_daha_katı"] == 2


def test_a_judge_that_passes_answers_we_scored_wrong_is_flagged_as_looser():
    a = agreement([_row(0.9, False)])
    assert a["hakem_daha_gevşek"] == 1


def test_rows_the_judge_could_not_score_are_excluded_not_counted_as_agreement():
    """A metric that errored must not quietly inflate the agreement rate."""
    a = agreement([_row(0.9, True), _row(None, True), _row(None, False)])
    assert a["karsilastirilabilir"] == 1
    assert a["uyum"] == 1.0


def test_nothing_comparable_does_not_divide_by_zero():
    assert agreement([_row(None, True)]) == {"karsilastirilabilir": 0}
    assert agreement([]) == {"karsilastirilabilir": 0}


def test_the_threshold_sits_at_half():
    """Ragas returns a fraction; the deterministic verdict is binary, so the
    comparison needs a stated cut rather than an implicit one."""
    assert agreement([_row(0.5, True)])["uyum"] == 1.0
    assert agreement([_row(0.49, True)])["uyum"] == 0.0
