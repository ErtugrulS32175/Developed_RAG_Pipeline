"""Context assembly: every retrieved passage must reach the model with a usable
source citation, since the answer prompt requires the answer to cite its page.
"""
from pipeline.query import build_context, citation


def _chunk(**kw):
    base = {"type": "text", "text": "govde", "filename": "belge.pdf",
            "page": 26, "headings": [], "table_data": None}
    base.update(kw)
    return base


def test_text_chunk_cites_document_and_page():
    assert citation(_chunk()) == "[belge.pdf | Sayfa 26]"


def test_table_from_a_native_pdf_is_still_cited():
    """The regression this exists for: table chunks used to be passed through
    verbatim on the assumption their text carried its own header. Tables parsed
    out of a native PDF have no such header, so they reached the model with no
    source at all."""
    c = citation(_chunk(type="table", table_data=None))
    assert "belge.pdf" in c and "Sayfa 26" in c and "tablo" in c


def test_verified_table_carries_its_confidence():
    c = citation(_chunk(type="table", table_data={"confidence": 0.87}))
    assert "guven 0.87" in c


def test_section_path_is_included():
    c = citation(_chunk(headings=["Bolum A", "Alt baslik"]))
    assert "Bolum A > Alt baslik" in c


def test_missing_metadata_degrades_gracefully():
    assert citation({"text": "x"}) == "[? | Sayfa 0]"


def test_every_passage_in_the_context_gets_a_citation():
    chunks = [_chunk(page=3), _chunk(type="table", page=7), _chunk(page=11)]
    ctx = build_context(chunks)
    assert ctx.count("belge.pdf") == 3
    for page in (3, 7, 11):
        assert f"Sayfa {page}" in ctx


def test_passages_stay_separated():
    ctx = build_context([_chunk(text="bir"), _chunk(text="iki")])
    assert ctx.count("\n\n---\n\n") == 1
    assert "bir" in ctx and "iki" in ctx


def test_empty_context_is_empty_not_an_error():
    assert build_context([]) == ""
