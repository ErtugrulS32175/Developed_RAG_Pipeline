"""Ingest side of the table flow: how a pipeline result becomes a RAG chunk,
and that ingest extracts through the verified pipeline rather than a raw backend.
"""
import json

from pipeline import ingest_router, router


def _consensus_table():
    """Shaped like table_pipeline.run_consensus output."""
    return {
        "mode": "consensus",
        "backends": ["b1", "b2"],
        "headers": ["Kod", "Tutar"],
        "rows": [["1", "10,00"], ["2", "20,00"]],
        "confidence": 0.95,
        "agreement": 0.75,
        "structural_confidence": 1.0,
        "number_fidelity": 0.95,
        "shape_match": True,
        "disagreements": [{"kind": "cell", "pos": (1, 1), "b1": "20,00", "b2": "26,00"}],
        "issues": ["1 hucrede modeller ayristi (insan gozden gecirmeli)"],
        "needs_review": True,
    }


def _chunks(table, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_router, "OUTPUT_DIR", tmp_path)
    return ingest_router.chunks_from_tables([table], "image:tables", "doc", "doc.png")


# --- trust signals reach the chunk ---

def test_chunk_carries_consensus_metadata(tmp_path, monkeypatch):
    chunks = _chunks(_consensus_table(), tmp_path, monkeypatch)
    data = chunks[0]["table_data"]

    assert data["mode"] == "consensus"
    assert data["backends"] == ["b1", "b2"]
    assert data["agreement"] == 0.75
    assert data["disagreements"][0]["pos"] == (1, 1)
    assert data["number_fidelity"] == 0.95


def test_chunk_table_data_is_json_serializable(tmp_path, monkeypatch):
    """It is stored in a jsonb column, so a stray set/tuple would fail at insert."""
    chunks = _chunks(_consensus_table(), tmp_path, monkeypatch)
    assert json.dumps(chunks[0]["table_data"])


def test_pipeline_review_decision_is_respected_not_recomputed(tmp_path, monkeypatch):
    """The pipeline weighs agreement and number fidelity; ingest must not
    second-guess it with a confidence threshold it knows less about."""
    table = _consensus_table()
    table["confidence"] = 1.0          # would pass ingest's own threshold
    table["issues"] = []               # and carry no issues
    table["needs_review"] = True       # but the pipeline still says review

    chunks = _chunks(table, tmp_path, monkeypatch)
    assert chunks[0]["table_data"]["needs_review"] is True


def test_falls_back_to_shape_confidence_for_a_plain_table(tmp_path, monkeypatch):
    plain = {"headers": ["A", "B"], "rows": [["1", "2"]]}
    chunks = _chunks(plain, tmp_path, monkeypatch)
    data = chunks[0]["table_data"]

    assert data["confidence"] == 1.0
    assert data["needs_review"] is False
    assert "mode" not in data


# --- exports ---

def test_flagged_table_is_copied_to_review_with_a_report(tmp_path, monkeypatch):
    _chunks(_consensus_table(), tmp_path, monkeypatch)
    report = tmp_path / "tables" / "_review" / "doc_image_tables_0.issues.txt"

    assert (tmp_path / "tables" / "_review" / "doc_image_tables_0.xlsx").exists()
    text = report.read_text(encoding="utf-8")
    assert "b1+b2" in text
    assert "1 hucrede ayrisma" in text


def test_pipeline_result_gets_the_rich_export(tmp_path, monkeypatch):
    """A consensus result has disagreements to highlight, so it must go through
    export_result_xlsx (Tablo + Rapor sheets), not the plain writer."""
    from openpyxl import load_workbook

    _chunks(_consensus_table(), tmp_path, monkeypatch)
    wb = load_workbook(tmp_path / "tables" / "doc_image_tables_0.xlsx")
    assert "Rapor" in wb.sheetnames


def test_plain_table_gets_the_plain_export(tmp_path, monkeypatch):
    from openpyxl import load_workbook

    _chunks({"headers": ["A"], "rows": [["1"]]}, tmp_path, monkeypatch)
    wb = load_workbook(tmp_path / "tables" / "doc_image_tables_0.xlsx")
    assert "Rapor" not in wb.sheetnames


# --- ingest extracts through the pipeline, not a raw backend ---

def test_ingest_uses_consensus_by_default(monkeypatch):
    seen = {}

    def fake_run_consensus(image_path, ocr_text=None):
        seen["mode"] = "consensus"
        seen["ocr_text"] = ocr_text
        return [{"headers": ["A"], "rows": [["1"]]}]

    import pipeline.table_pipeline as tp
    monkeypatch.setattr(tp, "run_consensus", fake_run_consensus)
    monkeypatch.setattr(router, "INGEST_TABLE_MODE", "consensus")

    out = router.tables_from_image_verified("x.png", ocr_text="onceden okundu")
    assert seen["mode"] == "consensus"
    # the caller's OCR reading is reused instead of re-running the service
    assert seen["ocr_text"] == "onceden okundu"
    assert out[0]["headers"] == ["A"]


def test_ingest_single_mode_runs_one_backend(monkeypatch):
    seen = {}

    def fake_run(image_path, ocr_text=None):
        seen["mode"] = "single"
        return []

    import pipeline.table_pipeline as tp
    monkeypatch.setattr(tp, "run", fake_run)
    monkeypatch.setattr(router, "INGEST_TABLE_MODE", "single")

    router.tables_from_image_verified("x.png", ocr_text="t")
    assert seen["mode"] == "single"
