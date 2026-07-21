from pipeline.table_pipeline import _finalize, _finalize_consensus


def _grouped_table():
    # non-numeric data cells -> number-verify stays clean, so the only signal
    # under test is the header/template handling
    return {
        "headers": ["g1", "g2", "g3", "g4"],
        "header_rows": [["Zzz", "Qqq", "Www", "Rrr"]],
        "rows": [["aa", "bb", "cc", "dd"]],
    }


def _matching_template():
    return {
        "name": "form_x",
        "header_rows": [["ColA", "GroupB", "", "ColC"], ["", "Sub1", "Sub2", ""]],
        "header_merges": [[0, 0, 2, 1], [0, 1, 1, 2], [0, 3, 2, 1]],
    }


def test_finalize_flags_undefined_grouped_form():
    r = _finalize(_grouped_table(), ocr_text="", backend="b1",
                  review_threshold=0.9, templates=[])
    assert r["needs_review"] is True
    assert r.get("review_all_headers") is True
    assert any("tanimlanmamis form" in i for i in r["issues"])
    # the two-level structure is preserved for the exporter
    assert r["header_rows"] == [["Zzz", "Qqq", "Www", "Rrr"]]


def test_finalize_stamps_recognized_form():
    table = {
        "header_rows": [["C0l4", "Grup8", "", "ColC"], ["", "Sub1", "Sub2", ""]],
        "header_merges": [[0, 0, 2, 1]],
        "rows": [["aa", "bb", "cc", "dd"]],
    }
    r = _finalize(table, ocr_text="", backend="b1",
                  review_threshold=0.9, templates=[_matching_template()])
    assert r.get("template") == "form_x"
    assert r.get("review_all_headers") is None          # recognized -> not flagged
    assert r["headers"] == ["ColA", "GroupB - Sub1", "GroupB - Sub2", "ColC"]


def test_finalize_flat_table_untouched_by_templates():
    table = {"headers": ["A", "B"], "rows": [["aa", "bb"]]}
    r = _finalize(table, ocr_text="", backend="b1",
                  review_threshold=0.9, templates=[_matching_template()])
    assert "header_rows" not in r
    assert r.get("review_all_headers") is None
    assert r["headers"] == ["A", "B"]


def _rec(headers, rows, **extra):
    rec = {
        "headers": headers, "rows": rows,
        "shape_match": True, "shape_primary": (len(rows), len(headers)),
        "shape_secondary": (len(rows), len(headers)),
        "disagreements": [], "agreement": 1.0,
    }
    rec.update(extra)
    return rec


def test_finalize_consensus_flags_undefined_grouped_form():
    rec = _rec(["g1", "g2", "g3", "g4"], [["aa", "bb", "cc", "dd"]],
               header_rows=[["Zzz", "Qqq", "Www", "Rrr"]], header_merges=[])
    r = _finalize_consensus(rec, "", ["b1", "b2"], 0.9, templates=[])
    assert r.get("review_all_headers") is True
    assert r["needs_review"] is True
    assert r["header_rows"] == [["Zzz", "Qqq", "Www", "Rrr"]]


def test_finalize_consensus_stamps_and_drops_header_disagreements():
    rec = _rec(
        ["C0l4", "Grup8 - Sub1", "Grup8 - Sub2", "ColC"],
        [["aa", "bb", "cc", "dd"]],
        header_rows=[["C0l4", "Grup8", "", "ColC"], ["", "Sub1", "Sub2", ""]],
        header_merges=[[0, 0, 2, 1]],
        disagreements=[{"kind": "header", "pos": 0, "b1": "C0l4", "b2": "ColA"},
                       {"kind": "cell", "pos": (0, 1), "b1": "bb", "b2": "xx"}],
        agreement=0.9,
    )
    r = _finalize_consensus(rec, "", ["b1", "b2"], 0.9, templates=[_matching_template()])
    assert r.get("template") == "form_x"
    assert r.get("review_all_headers") is None
    assert r["headers"] == ["ColA", "GroupB - Sub1", "GroupB - Sub2", "ColC"]
    # header disagreement dropped (header trusted after stamp), cell one kept
    kinds = [d["kind"] for d in r["disagreements"]]
    assert "header" not in kinds and "cell" in kinds


def test_finalize_consensus_flat_passes_through():
    rec = _rec(["A", "B"], [["aa", "bb"]])
    r = _finalize_consensus(rec, "", ["b1", "b2"], 0.9, templates=[_matching_template()])
    assert "header_rows" not in r
    assert r.get("review_all_headers") is None
    assert r["headers"] == ["A", "B"]
