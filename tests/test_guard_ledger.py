"""The saved-run ledger replays one policy instead of trusting mixed flags."""
from collections import Counter

from eval.answer.guard_ledger import (
    STRUCTURED_DERIVED,
    STRUCTURED_EXPLICIT,
    _new_counter,
    _record,
    replay_flags,
)
from eval.answer.judge import DOGRU, YANLIS
from pipeline.retrieval.query import build_rag_context


def _context():
    return build_rag_context(
        [{
            "filename": "kurgu.pdf",
            "page": 953761,
            "text": "KURGU_OMEGA_NESNESI 731641 birimdir.",
        }],
        numbered=True,
    )


def test_replay_can_compare_derived_and_explicit_page_policies():
    row = {
        "cevap": "KURGU_OMEGA_NESNESI 731641 birimdir.",
        "dayanak": [{
            "pasaj": 1,
            "alinti": "KURGU_OMEGA_NESNESI 731641 birimdir.",
        }],
    }

    derived = replay_flags(row, _context(), STRUCTURED_DERIVED)
    explicit = replay_flags(row, _context(), STRUCTURED_EXPLICIT)

    assert derived == []
    assert explicit == [("eksik_sayfa", [])]


def test_ledger_separates_false_review_from_wrong_answer_benefit():
    summary = _new_counter()
    flags = Counter()

    _record(
        summary,
        flags,
        {"durum": DOGRU, "sayfa_dogru": False},
        "KURGU_OMEGA_NESNESI desteklenir.",
        [("kurgu_tani", [])],
    )
    _record(
        summary,
        flags,
        {"durum": YANLIS, "sayfa_dogru": False},
        "KURGU_OMEGA_NESNESI desteklenmez.",
        [("kurgu_tani", [])],
    )

    assert summary["correct_withheld"] == 1
    assert summary["noncorrect_caught"] == 1
    assert summary["settled_wrong_caught"] == 1
