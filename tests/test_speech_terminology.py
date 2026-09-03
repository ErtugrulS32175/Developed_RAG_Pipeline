"""Closed registry and bounded-selection tests; no model or network needed."""
import json
from pathlib import Path
import pytest

from services import speech_terminology as terminology


def _document():
    return {
        "schema_version": 1,
        "profile_id": "kurumsal_tr",
        "revision": "rev_1",
        "language": "tr",
        "terms": [
            {"canonical": "Ortak Terim", "aliases": ["OT"],
             "contexts": ["default", "equities"], "priority": 10},
            {"canonical": "Dusuk Oncelik", "aliases": [],
             "contexts": ["default"], "priority": 1},
        ],
    }


def _write(tmp_path, document=None):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(
        _document() if document is None else document,
        ensure_ascii=False), encoding="utf-8")
    return path


def _tokens(text):
    return len(text.split())


def test_the_built_in_registry_is_closed_and_preserves_the_measured_pack():
    path = (Path(__file__).resolve().parent.parent / "services" /
            "terminology" / "capital_markets_tr.json")
    registry = terminology.load_registry(path)
    pack = registry.select("default", _tokens)
    assert registry.profile_id == "capital_markets_tr"
    assert len(registry.terms) == 8
    assert pack.phrase_count == 11
    assert pack.text == (
        "Borsa İstanbul, yatırım hesabı, hisse senedi, Türk Hava Yolları, "
        "THYAO, Garanti BBVA, Garanti Bankası, GARAN, lot, satış emri, "
        "alış emri")


@pytest.mark.parametrize("edit", [
    lambda value: value.update(extra=True),
    lambda value: value.update(schema_version=True),
    lambda value: value.update(language="en"),
    lambda value: value.update(terms=[]),
])
def test_invalid_root_shapes_are_refused(tmp_path, edit):
    document = _document()
    edit(document)
    with pytest.raises(terminology.TerminologyConfigError):
        terminology.load_registry(_write(tmp_path, document))


def test_unknown_term_fields_are_refused(tmp_path):
    document = _document()
    document["terms"][0]["definition"] = "must not enter this registry"
    with pytest.raises(terminology.TerminologyConfigError):
        terminology.load_registry(_write(tmp_path, document))


def test_duplicate_json_keys_are_refused(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(terminology.TerminologyConfigError):
        terminology.load_registry(path)


@pytest.mark.parametrize("duplicate", ["ortak terim", "ＯＴ"])
def test_normalized_canonical_and_alias_collisions_are_refused(
        tmp_path, duplicate):
    document = _document()
    document["terms"].append({
        "canonical": duplicate, "aliases": [],
        "contexts": ["default"], "priority": 2,
    })
    with pytest.raises(terminology.TerminologyConfigError):
        terminology.load_registry(_write(tmp_path, document))


@pytest.mark.parametrize("bad", [" satir", "satir\nsonu", "virgul,terim"])
def test_terms_cannot_smuggle_layout_or_delimiters(tmp_path, bad):
    document = _document()
    document["terms"][0]["canonical"] = bad
    with pytest.raises(terminology.TerminologyConfigError):
        terminology.load_registry(_write(tmp_path, document))


def test_selection_is_deterministic_and_context_is_closed(tmp_path):
    registry = terminology.load_registry(_write(tmp_path))
    assert registry.select("default", _tokens).text == (
        "Ortak Terim, OT, Dusuk Oncelik")
    assert registry.select("equities", _tokens).text == "Ortak Terim, OT"
    with pytest.raises(terminology.TerminologyConfigError):
        registry.select("caller_supplied", _tokens)


def test_selection_obeys_the_real_token_counter_and_phrase_ceiling(tmp_path):
    registry = terminology.load_registry(_write(tmp_path))
    calls = []

    def measured(text):
        calls.append(text)
        return len(text)

    pack = registry.select("default", measured, max_tokens=15,
                           max_phrases=2)
    assert pack.text == "Ortak Terim, OT"
    assert pack.phrase_count == 2
    assert pack.token_count == 15
    assert calls


@pytest.mark.parametrize("answer", [0, -1, 1.5, None])
def test_invalid_tokenizer_answers_fail_closed(tmp_path, answer):
    registry = terminology.load_registry(_write(tmp_path))
    with pytest.raises(terminology.TerminologyConfigError):
        registry.select("default", lambda _text: answer)


def test_repr_exposes_metadata_but_not_terms(tmp_path):
    registry = terminology.load_registry(_write(tmp_path))
    shown = repr(registry)
    assert "kurumsal_tr" in shown
    assert "Ortak Terim" not in shown
    assert str(tmp_path) not in shown


def test_registry_must_be_a_regular_file(tmp_path):
    with pytest.raises(terminology.TerminologyConfigError):
        terminology.load_registry(tmp_path)


def test_validation_cli_prints_only_closed_metadata(tmp_path, capsys):
    path = _write(tmp_path)
    assert terminology.main([str(path)]) == 0
    output = capsys.readouterr()
    body = json.loads(output.out)
    assert set(body) == {
        "schema_version", "profile_id", "revision", "language", "sha256",
        "term_count", "context_count",
    }
    assert body["term_count"] == 2
    assert "Ortak Terim" not in output.out
    assert str(path) not in output.out
    assert output.err == ""
