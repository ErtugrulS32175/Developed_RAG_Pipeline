"""Guard-only ideal-evidence baseline, with entirely invented fixtures.

The synthesis is measurement code, so it needs its own tests: an earlier
version quoted only the expected-answer line and incorrectly blamed the guard
for figures and pages supported by additional passages.
"""
from eval.answer import guard_floor


BLOCKS = [
    (
        "[kurgu-belge.pdf | Sayfa 17]\n"
        "Zeta uretimi 73 000 birimdir."
    ),
    (
        "[kurgu-belge.pdf | Sayfa 18]\n"
        "Gamma uretimi 19 000 birimdir."
    ),
]
SAVED_CONTEXT = guard_floor.BLOCK.join(BLOCKS)
RAG_CONTEXT = guard_floor.legacy_context(SAVED_CONTEXT)


def test_legacy_adapter_preserves_saved_passages_and_adds_only_handles():
    assert RAG_CONTEXT.model_text == guard_floor.BLOCK.join([
        f"[P1] {BLOCKS[0]}",
        f"[P2] {BLOCKS[1]}",
    ])
    assert [
        (passage.handle, passage.page, passage.text)
        for passage in RAG_CONTEXT.passages
    ] == [
        (1, 17, "Zeta uretimi 73 000 birimdir."),
        (2, 18, "Gamma uretimi 19 000 birimdir."),
    ]


def test_ideal_evidence_quotes_a_line_for_each_answer_figure():
    evidence = guard_floor.ideal_evidence(
        "Zeta uretimi 73 000",
        "Sayfa 17'ye gore 73 000, Sayfa 18'e gore 19 000 birimdir.",
        RAG_CONTEXT,
    )

    assert {item["pasaj"] for item in evidence} == {1, 2}
    assert any("73 000" in item["alinti"] for item in evidence)
    assert any("19 000" in item["alinti"] for item in evidence)


def test_ideal_evidence_covers_a_cited_page_even_without_an_extra_figure():
    evidence = guard_floor.ideal_evidence(
        "Zeta uretimi 73 000",
        "Sayfa 18'e gore zeta icin kurgu sonuc bildirilmistir.",
        RAG_CONTEXT,
    )

    assert {item["pasaj"] for item in evidence} == {1, 2}


def test_measurement_separates_rate_derivation_policy(monkeypatch):
    context = (
        "[kurgu-belge.pdf | Sayfa 17]\n"
        "Zeta orani binde 73 seviyesindedir."
    )
    ground_truth = {"key": "Zeta orani binde 73"}
    answer = "Sayfa 17'ye gore oran yuzde 7,3 seviyesindedir."

    def confirmed_correct(_run_dir):
        yield "kurgu", ground_truth, answer, context

    monkeypatch.setattr(guard_floor, "confirmed_correct", confirmed_correct)

    total_on, seen_on, flagged_on = guard_floor.measure(["kurgu"], derive=True)
    total_off, seen_off, flagged_off = guard_floor.measure(["kurgu"], derive=False)

    assert total_on == total_off == 1
    assert seen_on["olculen"] == seen_off["olculen"] == 1
    assert flagged_on["_herhangi"] == 0
    assert flagged_off["_herhangi"] == 1
    assert flagged_off["kaynaksiz_sayi"] == 1
