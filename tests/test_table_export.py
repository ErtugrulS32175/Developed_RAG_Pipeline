from pipeline.table_export import estimate_table_confidence, table_to_markdown


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
