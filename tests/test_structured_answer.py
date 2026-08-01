"""Answers that cite the line they rest on, and the checks that scoping buys.

Every figure below is invented -- test fixtures must never be copied out of a
real document.

The unscoped checks were measured first: on 187 answers already settled as
correct they raised almost nothing, and on the wrong ones they raised nothing
at all, because "the figure appears somewhere in fifteen passages" is satisfied
by almost any figure. These tests pin down what changes once the answer has to
say WHICH passage, and which line of it.
"""
from pipeline.retrieval.query import build_rag_context
from pipeline.validation.rag.answer_guard import (
    check_structured, parse_structured)

CHUNKS = [
    {"filename": "belge.pdf", "page": 42, "text": "Zeta uretimi 47 000 birimdir."},
    {"filename": "belge.pdf", "page": 43, "text": "Gamma uretimi 8 000 birimdir."},
]
CONTEXT = build_rag_context(CHUNKS, numbered=True)


def _reply(pasaj, alinti, cevap):
    return {"dayanak": [{"pasaj": pasaj, "alinti": alinti}], "cevap": cevap}


# --- the numbered context ---------------------------------------------------

def test_passages_carry_a_handle_and_a_page():
    p = CONTEXT.by_handle()
    assert set(p) == {1, 2}
    assert p[1].page == 42 and "47 000" in p[1].text


def test_an_unnumbered_context_shows_no_handles_to_the_model():
    context = build_rag_context(CHUNKS)
    assert context.numbered is False
    assert "[P1]" not in context.model_text


# --- getting the object back out of whatever the model wrote ----------------

def test_a_bare_object_parses():
    assert parse_structured('{"dayanak": [], "cevap": "yok"}')["cevap"] == "yok"


def test_a_fenced_object_parses():
    """Models wrap JSON in a code fence however firmly they are told not to."""
    text = 'Iste yanit:\n```json\n{"dayanak": [], "cevap": "yok"}\n```\nUmarim yardimci olur.'
    assert parse_structured(text)["cevap"] == "yok"


def test_braces_inside_a_quoted_string_do_not_end_the_object():
    text = '{"dayanak": [], "cevap": "sonuc {47} olarak verilmistir"}'
    assert parse_structured(text)["cevap"] == "sonuc {47} olarak verilmistir"


def test_a_leading_object_that_is_not_the_answer_is_skipped():
    text = '{"not": "dusunuyorum"} sonra: {"dayanak": [], "cevap": "47 000"}'
    assert parse_structured(text)["cevap"] == "47 000"


def test_unparseable_output_is_a_flag_not_an_exception():
    """A malformed reply must degrade to a warning on that answer. Raising here
    would turn a formatting slip into a failed request."""
    assert parse_structured("JSON yazmayi reddediyorum") is None
    assert check_structured("JSON yazmayi reddediyorum", CONTEXT) == \
        [("bicimsiz_yanit", [])]


# --- what scoping catches ---------------------------------------------------

def test_a_quoted_answer_from_the_right_passage_is_clean():
    reply = _reply(1, "Zeta uretimi 47 000 birimdir.", "Sayfa 42'ye gore 47 000 birim.")
    assert check_structured(reply, CONTEXT) == []


def test_a_quote_that_is_not_in_the_cited_passage_is_flagged():
    """The quote is the only part of the answer that can be checked literally.
    A paraphrase presented as a quotation means the evidence was invented."""
    reply = _reply(1, "Zeta uretimi 90 000 birimdir.", "Sayfa 42'ye gore 47 000 birim.")
    assert ("uydurma_alinti", [1]) in check_structured(reply, CONTEXT)


def test_a_passage_that_does_not_exist_is_flagged():
    reply = _reply(9, "her neyse", "Sayfa 42'ye gore 47 000 birim.")
    assert ("uydurma_pasaj", [9]) in check_structured(reply, CONTEXT)


def test_a_figure_from_an_uncited_passage_is_now_caught():
    """THE POINT OF THE WHOLE STEP. Against the full context this answer passes
    -- 8 000 really is in the passages. Against the passage it CITED it does
    not, and that is the difference between checking whether a number exists
    somewhere and checking whether it came from the record that was asked
    about."""
    from pipeline.validation.rag.answer_guard import check
    answer = "Sayfa 42'ye gore 8 000 birim."
    assert check(answer, CONTEXT) == []                       # unscoped: clean
    reply = _reply(1, "Zeta uretimi 47 000 birimdir.", answer)
    assert ("kaynaksiz_sayi", [[8000.0]]) in check_structured(reply, CONTEXT)


def test_a_figure_in_the_passage_but_on_no_quoted_line_is_flagged_softly():
    """The wrong-row shape: the passage holds several records and the answer
    took a figure from one it never quoted."""
    chunks = [{"filename": "belge.pdf", "page": 42,
               "text": "zeta | 47 000\ngamma | 8 000"}]
    ctx = build_rag_context(chunks, numbered=True)
    reply = _reply(1, "zeta | 47 000", "Sayfa 42'ye gore 8 000 birim.")
    names = {n for n, _ in check_structured(reply, ctx)}
    assert names == {"alintisiz_sayi"}


def test_a_page_no_cited_passage_came_from_is_flagged():
    reply = _reply(1, "Zeta uretimi 47 000 birimdir.", "Sayfa 43'e gore 47 000 birim.")
    assert ("kaynaksiz_sayfa", [43]) in check_structured(reply, CONTEXT)


def test_an_abstention_with_no_evidence_raises_nothing():
    reply = {"dayanak": [], "cevap": "Bu bilgi mevcut belgelerde bulunamadı."}
    assert check_structured(reply, CONTEXT) == []


# --- the whole path, with the model faked out -------------------------------

def test_the_structured_path_holds_together_end_to_end(monkeypatch):
    """Unit tests pass while the pieces are wired to each other wrongly, and
    the next run of this path costs rented GPU hours. So drive it once with the
    model faked out: retrieval, numbering, prompt, parse, check."""
    from pipeline.generation import answer as gen
    from pipeline.retrieval import query

    seen = {}

    def fake_complete(prompt):
        seen["prompt"] = prompt
        return ('```json\n{"dayanak": [{"pasaj": 1, '
                '"alinti": "Zeta uretimi 47 000 birimdir."}], '
                '"cevap": "Sayfa 42\'ye gore 47 000 birim."}\n```')

    monkeypatch.setattr(query, "retrieve", lambda q, top_k=None: CHUNKS)
    monkeypatch.setattr(query, "rerank", lambda q, chunks, top_n=None: chunks)
    monkeypatch.setattr(gen, "complete", fake_complete)

    reply = query.ask("zeta uretimi nedir?", structured=True)

    # the model was actually shown handles it could point at, and asked for the
    # evidence before the answer
    assert "[P1]" in seen["prompt"] and "[P2]" in seen["prompt"]
    assert seen["prompt"].index("dayanak") < seen["prompt"].index('"cevap"')

    parsed = parse_structured(reply)
    assert parsed["cevap"].endswith("47 000 birim.")
    assert check_structured(reply, build_rag_context(CHUNKS, numbered=True)) == []


def test_the_plain_path_is_untouched(monkeypatch):
    """The default must keep producing exactly what every earlier run produced,
    or nothing is comparable with anything."""
    from pipeline.generation import answer as gen
    from pipeline.retrieval import query

    seen = {}
    monkeypatch.setattr(query, "retrieve", lambda q, top_k=None: CHUNKS)
    monkeypatch.setattr(query, "rerank", lambda q, chunks, top_n=None: chunks)
    monkeypatch.setattr(gen, "complete",
                        lambda p: seen.setdefault("prompt", p) and "" or "duz cevap")

    assert query.ask("zeta uretimi nedir?") == "duz cevap"
    assert "[P1]" not in seen["prompt"]
    assert "json" not in seen["prompt"].lower()
