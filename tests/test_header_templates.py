import json

from pipeline.header_templates import (
    apply_template,
    load_templates,
    match_template,
    resolve_header,
)


def _template():
    # generic two-level form: ColA | GroupB(Sub1,Sub2) | ColC(rowspan)
    return {
        "name": "form_x",
        "header_rows": [
            ["ColA", "GroupB", "", "ColC"],
            ["", "Sub1", "Sub2", ""],
        ],
        "header_merges": [[0, 0, 2, 1], [0, 1, 1, 2], [0, 3, 2, 1]],
    }


def test_load_templates_reads_json_dir(tmp_path):
    (tmp_path / "form_x.json").write_text(json.dumps(_template()), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    templates = load_templates(tmp_path)
    assert [t["name"] for t in templates] == ["form_x"]


def test_load_templates_missing_dir_returns_empty(tmp_path):
    assert load_templates(tmp_path / "nope") == []


def test_load_templates_defaults_name_to_filename(tmp_path):
    t = _template()
    del t["name"]
    (tmp_path / "my_form.json").write_text(json.dumps(t), encoding="utf-8")
    assert load_templates(tmp_path)[0]["name"] == "my_form"


def test_match_identifies_form_despite_garbled_header():
    # incoming header text is OCR-garbled but still closest to form_x
    garbled = {
        "header_rows": [
            ["Col4", "Grup8", "", "Col C"],   # noisy versions of the template text
            ["", "Svb1", "Suh2", ""],
        ],
    }
    tpl, score = match_template(garbled["header_rows"], [_template()])
    assert tpl is not None and tpl["name"] == "form_x"
    assert score >= 0.5


def test_match_returns_none_for_unrelated_header():
    other = {"header_rows": [["Zzz", "Qqq", "Www"]]}
    tpl, score = match_template(other["header_rows"], [_template()])
    assert tpl is None


def test_match_empty_incoming_header_returns_none():
    tpl, score = match_template([["", ""]], [_template()])
    assert tpl is None and score == 0.0


def test_apply_stamps_canonical_header_and_keeps_data():
    parsed = {
        "headers": ["garbage"] * 4,
        "header_rows": [["C0l4", "Grup8", "", "ColC"], ["", "Sub1", "Sub2", ""]],
        "header_merges": [[0, 0, 2, 1]],           # model's messy spans
        "rows": [["1", "2", "3", "4"], ["5", "6", "7", "8"]],
    }
    out = apply_template(parsed, _template())
    assert out is not None
    assert out["template"] == "form_x"
    # canonical text + spans win over the model's garbled ones
    assert out["header_rows"] == _template()["header_rows"]
    assert out["header_merges"] == [(0, 0, 2, 1), (0, 1, 1, 2), (0, 3, 2, 1)]
    assert out["headers"] == ["ColA", "GroupB - Sub1", "GroupB - Sub2", "ColC"]
    # data is untouched
    assert out["rows"] == [["1", "2", "3", "4"], ["5", "6", "7", "8"]]


def test_apply_returns_none_on_column_count_mismatch():
    parsed = {"rows": [["1", "2", "3", "4", "5"]]}   # 5 data cols, template is 4
    assert apply_template(parsed, _template()) is None


def test_resolve_stamps_when_form_recognized():
    parsed = {
        "header_rows": [["C0l4", "Grup8", "", "ColC"], ["", "Sub1", "Sub2", ""]],
        "header_merges": [[0, 0, 2, 1]],
        "rows": [["a", "b", "c", "d"]],
    }
    out, info = resolve_header(parsed, [_template()])
    assert info["template"] == "form_x" and info["undefined_form"] is False
    assert out["headers"] == ["ColA", "GroupB - Sub1", "GroupB - Sub2", "ColC"]


def test_resolve_flags_unrecognized_grouped_header():
    parsed = {"header_rows": [["Zzz", "Qqq", "Www", "Rrr"]], "rows": [["a", "b", "c", "d"]]}
    out, info = resolve_header(parsed, [_template()])
    assert info["undefined_form"] is True and info["template"] is None
    assert out is parsed          # unchanged, nothing stamped


def test_resolve_flags_recognized_but_column_mismatch():
    parsed = {
        "header_rows": [["C0l4", "Grup8", "", "ColC"], ["", "Sub1", "Sub2", ""]],
        "rows": [["a", "b", "c", "d", "e"]],       # 5 data cols, template is 4
    }
    out, info = resolve_header(parsed, [_template()])
    assert info["template"] == "form_x" and info["undefined_form"] is True


def test_resolve_passes_flat_header_through():
    parsed = {"headers": ["A", "B"], "rows": [["1", "2"]]}
    out, info = resolve_header(parsed, [_template()])
    assert out is parsed and info["undefined_form"] is False
