"""The offline citation-policy counter-metric uses aggregate saved rows only."""
import json

from eval.answer.page_policy_eval import (
    FIGURE_PAGE,
    ONE_PAGE,
    ONE_PASSAGE,
    measure_rows,
    saved_rows,
    saved_sets,
)
from eval.answer.judge import DOGRU, YANLIS
from pipeline.retrieval.query import build_rag_context


def _row(chunks, dayanak, answer, correct=True):
    context = build_rag_context(chunks, numbered=True)
    return {
        "baglam": context.model_text,
        "dayanak": dayanak,
        "cevap": answer,
        "cevap_dogru": correct,
        "durum": DOGRU if correct else YANLIS,
        "bayraklar": [["eksik_sayfa", []]],
    }


def test_measurement_charges_each_relaxation_for_wrong_publication():
    rows = [
        _row(
            [{
                "filename": "kurgu.pdf",
                "page": 953761,
                "text": "KURGU_OMEGA_NESNESI 731641 birimdir.",
            }],
            [{
                "pasaj": 1,
                "alinti": "KURGU_OMEGA_NESNESI 731641 birimdir.",
            }],
            "KURGU_OMEGA_NESNESI 731641 birimdir.",
        ),
        _row(
            [
                {
                    "filename": "kurgu.pdf",
                    "page": 953762,
                    "text": "KURGU_OMEGA_NESNESI ilk satirdadir.",
                },
                {
                    "filename": "kurgu.pdf",
                    "page": 953762,
                    "text": "KURGU_OMEGA_NESNESI ikinci satirdadir.",
                },
            ],
            [
                {
                    "pasaj": 1,
                    "alinti": "KURGU_OMEGA_NESNESI ilk satirdadir.",
                },
                {
                    "pasaj": 2,
                    "alinti": "KURGU_OMEGA_NESNESI ikinci satirdadir.",
                },
            ],
            "KURGU_OMEGA_NESNESI desteklenir.",
        ),
        _row(
            [
                {
                    "filename": "kurgu.pdf",
                    "page": 953763,
                    "text": "KURGU_OMEGA_NESNESI 842753 birimdir.",
                },
                {
                    "filename": "kurgu.pdf",
                    "page": 953764,
                    "text": "KURGU_OMEGA_NESNESI aciklamasi buradadir.",
                },
            ],
            [
                {
                    "pasaj": 1,
                    "alinti": "KURGU_OMEGA_NESNESI 842753 birimdir.",
                },
                {
                    "pasaj": 2,
                    "alinti": "KURGU_OMEGA_NESNESI aciklamasi buradadir.",
                },
            ],
            "KURGU_OMEGA_NESNESI 842753 birimdir.",
            correct=False,
        ),
    ]

    report = measure_rows(rows)["policies"]

    assert report[ONE_PASSAGE]["correct_answers_rescued"] == 1
    assert report[ONE_PASSAGE]["settled_wrong_answers_newly_publishable"] == 0
    assert report[ONE_PAGE]["correct_answers_rescued"] == 2
    assert report[ONE_PAGE]["settled_wrong_answers_newly_publishable"] == 0
    assert report[FIGURE_PAGE]["correct_answers_rescued"] == 2
    assert report[FIGURE_PAGE]["settled_wrong_answers_newly_publishable"] == 1


def test_multi_page_evidence_without_a_unique_figure_page_stays_ambiguous():
    row = _row(
        [
            {
                "filename": "kurgu.pdf",
                "page": 953763,
                "text": "KURGU_OMEGA_NESNESI ilk aciklamadir.",
            },
            {
                "filename": "kurgu.pdf",
                "page": 953764,
                "text": "KURGU_OMEGA_NESNESI ikinci aciklamadir.",
            },
        ],
        [
            {
                "pasaj": 1,
                "alinti": "KURGU_OMEGA_NESNESI ilk aciklamadir.",
            },
            {
                "pasaj": 2,
                "alinti": "KURGU_OMEGA_NESNESI ikinci aciklamadir.",
            },
        ],
        "KURGU_OMEGA_NESNESI desteklenir.",
    )

    report = measure_rows([row])

    assert report["policies"][FIGURE_PAGE]["unresolved_missing_page"] == 1
    assert report["genuinely_ambiguous_after_all_candidates"] == 1


def test_saved_rows_excludes_only_a_numeric_sweep_suffix(tmp_path):
    payload = {"sorular": [{"cevap_dogru": True}]}
    (tmp_path / "rag_answers_kurgu.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (tmp_path / "rag_answers_kurgu_k15.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    files, rows = saved_rows([tmp_path])
    set_files, sets = saved_sets([tmp_path])

    assert files == 1
    assert rows == payload["sorular"]
    assert set_files == 1
    assert sets == {"kurgu": payload["sorular"]}
