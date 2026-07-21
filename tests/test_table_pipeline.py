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


def test_finalize_consensus_shows_both_on_unresolved_shape_mismatch():
    rec = _rec(["A", "B", "C"], [["1", "2", "3"]],
               shape_match=False, shape_primary=(1, 3), shape_secondary=(1, 4),
               disagreements=[{"kind": "shape"}], agreement=0.0)
    cands = [{"backend": "b1", "headers": ["A", "B", "C"], "rows": [["1", "2", "3"]]},
             {"backend": "b2", "headers": ["A", "B", "C", "D"], "rows": [["1", "2", "3", "4"]]}]
    r = _finalize_consensus(rec, "", ["b1", "b2"], 0.9, templates=[], candidates=cands)
    assert r["needs_review"] is True
    assert len(r["candidates"]) == 2
    assert [c["backend"] for c in r["candidates"]] == ["b1", "b2"]


def test_finalize_consensus_scores_candidate_quality_from_ocr():
    rec = _rec(["A", "B"], [["10", "20"]],
               shape_match=False, shape_primary=(1, 2), shape_secondary=(1, 3),
               disagreements=[{"kind": "shape"}], agreement=0.0)
    # ocr_text has 10 and 20 but NOT 99 -> b2's "99" is a suspect number
    cands = [{"backend": "b1", "headers": ["A", "B"], "rows": [["10", "20"]]},
             {"backend": "b2", "headers": ["A", "B", "C"], "rows": [["10", "20", "99"]]}]
    r = _finalize_consensus(rec, "10 20", ["b1", "b2"], 0.9, templates=[], candidates=cands)
    a, b = r["candidates"]
    assert a["suspect_count"] == 0
    assert b["suspect_count"] == 1 and (0, 2) in b["review_cells"]


def test_finalize_consensus_no_candidates_when_template_arbitrates():
    # shape mismatch, but a template recognizes the form and stamps it (data
    # width matches) -> we have the answer, so DON'T dump both candidates
    rec = _rec(["x", "x", "x", "x"], [["aa", "bb", "cc", "dd"]],
               shape_match=False, shape_primary=(1, 4), shape_secondary=(1, 5),
               disagreements=[{"kind": "shape"}], agreement=0.0,
               header_rows=[["C0l4", "Grup8", "", "ColC"], ["", "Sub1", "Sub2", ""]],
               header_merges=[[0, 0, 2, 1]])
    cands = [{"backend": "b1", "headers": ["x"] * 4, "rows": [["aa", "bb", "cc", "dd"]]},
             {"backend": "b2", "headers": ["y"] * 5, "rows": [["a", "b", "c", "d", "e"]]}]
    r = _finalize_consensus(rec, "", ["b1", "b2"], 0.9,
                            templates=[_matching_template()], candidates=cands)
    assert r.get("template") == "form_x"
    assert "candidates" not in r          # arbitrated -> no need to show both
    assert r["headers"] == ["ColA", "GroupB - Sub1", "GroupB - Sub2", "ColC"]


def test_consensus_metrics_agreement_primary_total():
    from eval.table_eval import consensus_metrics
    vl = {"headers": ["A", "B"], "rows": [["1", "2"], ["3", "4"]]}
    hy = {"headers": ["A", "B"], "rows": [["1", "2"], ["9", "4"]]}   # differs at (1,0)
    gt = {"headers": ["A", "B"], "rows": [["1", "2"], ["9", "4"]]}   # hy right, vl wrong there
    m = consensus_metrics(vl, hy, gt)
    assert m["agreement"] == round(5 / 6, 4)      # 1 of 6 cells disagree
    assert m["primary_acc"] == round(5 / 6, 4)    # VL wrong on (1,0)
    assert m["total_acc"] == 1.0                  # HY recovers it -> at least one right
