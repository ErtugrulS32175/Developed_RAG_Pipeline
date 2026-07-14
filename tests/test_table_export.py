from openpyxl import load_workbook

from pipeline.table_export import (
    estimate_table_confidence,
    export_result_xlsx,
    table_to_markdown,
)


def test_confidence_all_rows_match_header_width():
    headers = ["Product", "Quantity"]
    rows = [["Pen", 5], ["Pencil", 3]]
    assert estimate_table_confidence(headers, rows) == 1.0


def test_confidence_penalizes_mismatched_rows():
    headers = ["Product", "Quantity", "Total"]
    rows = [["Pen", 5, 100], ["Pencil", 3]]  # second row missing a column
    assert estimate_table_confidence(headers, rows) == 0.5


def test_confidence_empty_table_is_zero():
    assert estimate_table_confidence([], []) == 0.0
    assert estimate_table_confidence(["Product"], []) == 0.0


def test_markdown_includes_citation_header_when_provided():
    md = table_to_markdown(
        ["Product", "Quantity"], [["Pen", 5]],
        filename="invoice_001.pdf", page=1, table_id="table_001", confidence=0.94,
    )
    assert "Belge: invoice_001.pdf" in md
    assert "Tablo: table_001" in md
    assert "Güven: 0.94" in md
    assert "| Product | Quantity |" in md


def test_markdown_without_citation_args_stays_plain():
    md = table_to_markdown(["Product"], [["Pen"]])
    assert "Belge:" not in md
    assert md.startswith("| Product |")


def _consensus_result():
    return {
        "mode": "consensus",
        "backends": ["vl", "hy"],
        "headers": ["Product", "Qty"],
        "rows": [["Pen", "5"], ["Book", "3"]],
        "confidence": 0.83,
        "structural_confidence": 1.0,
        "number_fidelity": 1.0,
        "agreement": 0.83,
        "needs_review": True,
        "issues": ["1 hucrede modeller ayristi"],
        "disagreements": [{"kind": "cell", "pos": (1, 1), "vl": "3", "hy": "8"}],
    }


def test_export_writes_data_and_report_sheets(tmp_path):
    out = tmp_path / "t.xlsx"
    export_result_xlsx(_consensus_result(), str(out))
    wb = load_workbook(out)
    assert wb.sheetnames == ["Tablo", "Rapor"]
    ws = wb["Tablo"]
    assert [c.value for c in ws[1]] == ["Product", "Qty"]
    assert ws["A1"].font.bold is True
    assert ws["B3"].value == "3"                       # row (1,1) -> excel B3


def test_export_highlights_disagreement_cell(tmp_path):
    out = tmp_path / "t.xlsx"
    export_result_xlsx(_consensus_result(), str(out))
    ws = load_workbook(out)["Tablo"]
    assert ws["B3"].fill.patternType == "solid"             # disagreed cell filled
    assert ws["B3"].fill.fgColor.rgb.endswith("FFE699")
    assert ws["A2"].fill.patternType is None                # agreeing cell not filled


def test_export_report_lists_both_candidates(tmp_path):
    out = tmp_path / "t.xlsx"
    export_result_xlsx(_consensus_result(), str(out))
    rows = list(load_workbook(out)["Rapor"].iter_rows(values_only=True))
    flat = [str(c) for r in rows for c in r if c is not None]
    assert "vl + hy" in flat
    assert "3" in flat and "8" in flat                 # both candidates present


def test_export_single_backend_result_has_no_highlights(tmp_path):
    out = tmp_path / "t.xlsx"
    result = {
        "backend": "vl",
        "headers": ["Product", "Qty"],
        "rows": [["Pen", "5"]],
        "confidence": 1.0,
        "structural_confidence": 1.0,
        "number_fidelity": 1.0,
        "needs_review": False,
        "issues": [],
    }
    export_result_xlsx(result, str(out))
    ws = load_workbook(out)["Tablo"]
    assert ws["B2"].fill.patternType is None
