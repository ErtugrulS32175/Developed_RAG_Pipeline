"""Row binding over structured tables: the fact-based check and its limits.

Fixtures use the invented, absence-verified vocabulary of the binding tests
(Alfa/Beta Sirketi, Zeta Endeksi, 47 000 / 88 000). What is locked: the
verdict table -- a figure occurring only outside the bound row while the
bound row offers a different figure in the same column is flagged; every
incomplete fact pattern is silence; and the module stays out of the
publication import graph exactly like the flat-text signal.
"""
import pytest

from pipeline.validation.rag.row_binding import (
    WRONG_ROW_BINDING, check_row_binding)

QUESTION = "Alfa Sirketi'nin Zeta Endeksi nedir?"


def _table_chunk(headers, rows, **extra):
    data = {"table_id": "kurgu_t1", "page": 42, "headers": headers,
            "rows": rows, "confidence": 1.0, "needs_review": False,
            "issues": []}
    data.update(extra)
    return {"type": "table", "text": "duzlestirilmis metin",
            "page": 42, "table_data": data}


SIRA_TABLE = _table_chunk(
    ["Kurulus", "Zeta Endeksi"],
    [["Alfa Sirketi", "47 000"],
     ["Beta Sirketi", "88 000"]])


def test_a_sibling_row_value_is_flagged_as_a_fact():
    flags = check_row_binding(QUESTION, "Cevap 88 000 birimdir.", [SIRA_TABLE])
    assert [code for code, _ in flags] == [WRONG_ROW_BINDING]
    assert "88" in flags[0][1][0]


def test_the_bound_rows_own_value_is_clean():
    assert check_row_binding(QUESTION, "Cevap 47 000 birimdir.",
                             [SIRA_TABLE]) == []


def test_number_formatting_differences_do_not_unbind():
    # the answer writes dotted thousands; the cell carries spaced thousands
    assert check_row_binding(QUESTION, "Cevap 47.000 birimdir.",
                             [SIRA_TABLE]) == []


def test_a_figure_absent_from_every_table_is_not_this_checks_job():
    assert check_row_binding(QUESTION, "Cevap 12 345.", [SIRA_TABLE]) == []


def test_no_term_signal_means_silence():
    assert check_row_binding("Gama Kurulusu hakkinda ne denir?",
                             "Cevap 88 000.", [SIRA_TABLE]) == []


def test_an_equal_best_tie_means_silence():
    assert check_row_binding("Zeta Endeksi kactir?", "Cevap 88 000.",
                             [SIRA_TABLE]) == []


def test_the_bound_row_must_offer_a_different_figure_to_accuse():
    # the bound row's cell in the figure's column is EMPTY: incomplete
    # pattern, silence -- the table may simply not carry the asked value
    sparse = _table_chunk(
        ["Kurulus", "Zeta Endeksi"],
        [["Alfa Sirketi", ""],
         ["Beta Sirketi", "88 000"]])
    assert check_row_binding(QUESTION, "Cevap 88 000.", [sparse]) == []


def test_any_table_binding_the_figure_correctly_clears_it():
    # a second retrieved table carries the figure in the bound row: the
    # figure has a legitimate source, so the first table may not accuse
    other = _table_chunk(
        ["Kurulus", "Zeta Endeksi"],
        [["Alfa Sirketi", "88 000"]])
    flags = check_row_binding(QUESTION, "Cevap 88 000.", [SIRA_TABLE, other])
    assert flags == []


def test_inflected_question_terms_still_bind_through_stems():
    flags = check_row_binding("Alfa Sirketinin Zeta Endeksi kactir?",
                              "Cevap 88 000.", [SIRA_TABLE])
    assert [code for code, _ in flags] == [WRONG_ROW_BINDING]


def test_header_words_planted_in_a_cell_are_a_known_false_annotation_limit():
    """Auditor finding, round 14, PINNED not fixed: row scoring compares
    every question term against every cell, so a wrong row whose spare cell
    carries the COLUMN NAME collects the attribute terms and outranks the
    true row -- the correct answer gets annotated. Any lexical row scorer
    is spoofable this way; separating entity terms from attribute terms is
    the next measured step, not a quiet patch. The signal stays
    annotation-only and publication-isolated, so the cost is reviewer
    minutes and measurement noise, never a withheld answer. If this test
    ever fails, the limit moved -- that is an announcement, not a bug."""
    poisoned = _table_chunk(
        ["Kurulus", "Zeta Endeksi", "Not"],
        [["Alfa Sirketi", "47 000", ""],
         ["Beta Sirketi", "88 000", "Zeta Endeksi kaydi"]])
    flags = check_row_binding(QUESTION, "Cevap 47 000 birimdir.", [poisoned])
    assert [code for code, _ in flags] == [WRONG_ROW_BINDING]  # the limit


def test_wrong_column_in_the_bound_row_is_a_declared_limit():
    """v0 binds ROWS only. The answer takes the bound row's OTHER column
    (wrong year, right entity) and passes clean. Pinned so the limit is
    visible; closing it is the next measured step, not a silent hope."""
    two_years = _table_chunk(
        ["Kurulus", "Endeks 1903", "Endeks 1907"],
        [["Alfa Sirketi", "41 500", "47 000"],
         ["Beta Sirketi", "77 000", "88 000"]])
    assert check_row_binding("Alfa Sirketi 1907 Zeta Endeksi nedir?",
                             "Cevap 41 500.", [two_years]) == []


def test_malformed_table_data_is_ignored_not_fatal():
    broken = [
        {"table_data": None},
        {"table_data": {"headers": "duz", "rows": [["Alfa", "1"]]}},
        {"table_data": {"headers": ["a"], "rows": "duz"}},
        {"table_data": {"headers": ["a"], "rows": [None, 3]}},
        "duz metin",
        None,
    ]
    assert check_row_binding(QUESTION, "Cevap 88 000.", broken) == []
    # and a malformed neighbour does not blind the check to a real table
    flags = check_row_binding(QUESTION, "Cevap 88 000.",
                              broken + [SIRA_TABLE])
    assert [code for code, _ in flags] == [WRONG_ROW_BINDING]


def test_ragged_rows_do_not_crash_the_column_comparison():
    ragged = _table_chunk(
        ["Kurulus", "Zeta Endeksi", "Not"],
        [["Alfa Sirketi"],
         ["Beta Sirketi", "88 000", "kurgu not"]])
    assert check_row_binding(QUESTION, "Cevap 88 000.", [ragged]) == []


def test_row_binding_stays_out_of_the_publication_import_graph():
    """Same architecture rule as binding_guard, enforced the same two ways:
    a fresh interpreter's eager import graph, and below, the static token
    ban. Annotation-only means annotation-only."""
    import os
    import subprocess
    import sys
    code = (
        "import sys\n"
        "import pipeline.validation.rag.answer_guard\n"
        "import pipeline.api.app\n"
        "print([m for m in sys.modules if 'row_binding' in m])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_no_pipeline_module_references_row_binding_even_lazily():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "pipeline"
    offenders = [
        str(path.relative_to(root.parent))
        for path in sorted(root.rglob("*.py"))
        if path.name != "row_binding.py"
        and "row_binding" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []
