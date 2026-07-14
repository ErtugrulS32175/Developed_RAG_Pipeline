from pipeline.consensus import reconcile


def test_full_agreement_no_disagreements():
    a = {"headers": ["Product", "Qty"], "rows": [["Pen", "5"], ["Book", "3"]]}
    rec = reconcile(a, a, "vl", "hy")
    assert rec["shape_match"] is True
    assert rec["agreement"] == 1.0
    assert rec["disagreements"] == []


def test_single_cell_disagreement_is_flagged_with_both_candidates():
    a = {"headers": ["Product", "Qty"], "rows": [["Pen", "5"], ["Book", "3"]]}
    b = {"headers": ["Product", "Qty"], "rows": [["Pen", "5"], ["Book", "8"]]}
    rec = reconcile(a, b, "vl", "hy")
    assert rec["shape_match"] is True
    assert len(rec["disagreements"]) == 1
    d = rec["disagreements"][0]
    assert d["kind"] == "cell" and d["pos"] == (1, 1)
    assert d["vl"] == "3" and d["hy"] == "8"          # both candidates preserved
    assert rec["rows"][1][1] == "3"                    # primary value displayed
    assert rec["review_mask_rows"][1][1] is True


def test_turkish_fold_and_number_format_do_not_count_as_disagreement():
    # ş/ı folded, thousands/decimal separators stripped -> same value, no flag
    a = {"headers": ["Kod"], "rows": [["Subat"], ["1000.50"]]}
    b = {"headers": ["Kod"], "rows": [["Şubat"], ["1.000,50"]]}
    rec = reconcile(a, b, "vl", "hy")
    assert rec["agreement"] == 1.0
    assert rec["disagreements"] == []


def test_shape_mismatch_forces_whole_table_review():
    a = {"headers": ["Product", "Qty"], "rows": [["Pen", "5"]]}
    b = {"headers": ["Product", "Qty"], "rows": [["Pen", "5"], ["Book", "3"]]}
    rec = reconcile(a, b, "vl", "hy")
    assert rec["shape_match"] is False
    assert rec["agreement"] == 0.0
    assert rec["disagreements"][0]["kind"] == "shape"
    assert all(all(m) for m in rec["review_mask_rows"])   # everything to review


def test_header_disagreement_is_flagged():
    a = {"headers": ["Product", "Qty"], "rows": [["Pen", "5"]]}
    b = {"headers": ["Product", "Count"], "rows": [["Pen", "5"]]}
    rec = reconcile(a, b, "vl", "hy")
    assert any(d["kind"] == "header" for d in rec["disagreements"])
    assert rec["review_mask_headers"][1] is True


def test_agreement_ratio_counts_headers_and_cells():
    a = {"headers": ["A", "B"], "rows": [["1", "2"], ["3", "4"]]}
    b = {"headers": ["A", "B"], "rows": [["1", "9"], ["3", "4"]]}   # 1 of 6 differs
    rec = reconcile(a, b, "vl", "hy")
    assert rec["agreement"] == round(5 / 6, 4)
