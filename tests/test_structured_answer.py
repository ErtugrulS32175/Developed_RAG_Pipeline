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
    ABSTAINED, ANSWERED, DERIVED_CITATION, MODEL_CITATION, REVIEW_REQUIRED,
    PageCitation, check_structured, parse_structured, validate_structured)

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


def test_an_answer_nested_in_a_wrapper_is_not_a_top_level_answer():
    text = '{"sonuc": {"dayanak": [], "cevap": "kurgu"}}'
    assert parse_structured(text) is None


def test_an_answer_wrapped_in_an_array_is_rejected():
    text = '[{"dayanak": [], "cevap": "kurgu"}]'
    assert parse_structured(text) is None


def test_top_level_field_order_does_not_change_schema_meaning():
    text = '{"cevap": "kurgu", "dayanak": []}'
    assert parse_structured(text) == {"dayanak": [], "cevap": "kurgu"}

    reply = {
        "cevap": "Sayfa 42'ye gore 47 000 birim.",
        "dayanak": [{
            "pasaj": 1,
            "alinti": "Zeta uretimi 47 000 birimdir.",
        }],
    }
    assert check_structured(reply, CONTEXT) == []


def test_whitespace_only_answer_and_quote_are_rejected():
    assert parse_structured('{"dayanak": [], "cevap": "   "}') is None
    reply = {"dayanak": [{"pasaj": 1, "alinti": " \n "}],
             "cevap": "kurgu"}
    assert check_structured(reply, CONTEXT) == [("bicimsiz_yanit", [])]


def test_passage_handles_must_be_positive():
    for handle in (0, -1):
        reply = _reply(handle, "kurgu", "kurgu")
        assert check_structured(reply, CONTEXT) == [("bicimsiz_yanit", [])]


def test_non_finite_json_numbers_are_rejected():
    text = '{"dayanak": [{"pasaj": NaN, "alinti": "kurgu"}], "cevap": "kurgu"}'
    assert parse_structured(text) is None


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


def test_a_missing_page_is_annotated_from_one_passage_without_rewriting_answer():
    ctx = build_rag_context([{
        "filename": "kurgu.pdf",
        "page": 953761,
        "text": "KURGU_OMEGA_NESNESI 731641 birimdir.",
    }], numbered=True)
    reply = _reply(
        1,
        "KURGU_OMEGA_NESNESI 731641 birimdir.",
        "KURGU_OMEGA_NESNESI 731641 birimdir.",
    )

    result = validate_structured(reply, ctx)

    assert result.status == ANSWERED
    assert result.answer == reply["cevap"]
    assert result.citations == (
        PageCitation(953761, DERIVED_CITATION),
    )


def test_several_passages_on_one_page_still_have_one_derived_page():
    ctx = build_rag_context([
        {
            "filename": "kurgu.pdf",
            "page": 953762,
            "text": "KURGU_OMEGA_NESNESI ilk satirdadir.",
        },
        {
            "filename": "kurgu.pdf",
            "page": 953762,
            "text": "KURGU_OMEGA_NESNESI ikinci satirdadir.",
        },
    ], numbered=True)
    reply = {
        "dayanak": [
            {"pasaj": 1, "alinti": "KURGU_OMEGA_NESNESI ilk satirdadir."},
            {"pasaj": 2, "alinti": "KURGU_OMEGA_NESNESI ikinci satirdadir."},
        ],
        "cevap": "KURGU_OMEGA_NESNESI dayanaklarla desteklenir.",
    }

    result = validate_structured(reply, ctx)

    assert result.status == ANSWERED
    assert result.diagnostics == ()
    assert result.citations == (
        PageCitation(953762, DERIVED_CITATION),
    )


def test_several_pages_stay_open_even_when_only_one_contains_the_figure():
    """A unique figure-bearing page did not rescue another saved answer."""
    ctx = build_rag_context([
        {
            "filename": "kurgu.pdf",
            "page": 953763,
            "text": "KURGU_OMEGA_NESNESI 731641 birimdir.",
        },
        {
            "filename": "kurgu.pdf",
            "page": 953764,
            "text": "KURGU_OMEGA_NESNESI aciklamasi buradadir.",
        },
    ], numbered=True)
    reply = {
        "dayanak": [
            {
                "pasaj": 1,
                "alinti": "KURGU_OMEGA_NESNESI 731641 birimdir.",
            },
            {
                "pasaj": 2,
                "alinti": "KURGU_OMEGA_NESNESI aciklamasi buradadir.",
            },
        ],
        "cevap": "KURGU_OMEGA_NESNESI 731641 birimdir.",
    }

    result = validate_structured(reply, ctx)

    assert result.status == REVIEW_REQUIRED
    assert result.answer is None
    assert result.citations == ()
    assert result.diagnostics == (("eksik_sayfa", []),)


def test_a_valid_handle_cannot_hide_an_unknown_handle_during_derivation():
    ctx = build_rag_context([{
        "filename": "kurgu.pdf",
        "page": 953761,
        "text": "KURGU_OMEGA_NESNESI 731641 birimdir.",
    }], numbered=True)
    reply = {
        "dayanak": [
            {
                "pasaj": 1,
                "alinti": "KURGU_OMEGA_NESNESI 731641 birimdir.",
            },
            {
                "pasaj": 842753,
                "alinti": "KURGU_OMEGA_NESNESI desteklenir.",
            },
        ],
        "cevap": "KURGU_OMEGA_NESNESI 731641 birimdir.",
    }

    result = validate_structured(reply, ctx)

    assert result.status == REVIEW_REQUIRED
    assert result.citations == ()
    assert {code for code, _ in result.diagnostics} == {
        "eksik_sayfa",
        "uydurma_pasaj",
    }


def test_an_explicit_valid_page_is_marked_as_model_supplied():
    ctx = build_rag_context([{
        "filename": "kurgu.pdf",
        "page": 953761,
        "text": "KURGU_OMEGA_NESNESI 842753 birimdir.",
    }], numbered=True)
    reply = _reply(
        1,
        "KURGU_OMEGA_NESNESI 842753 birimdir.",
        "Sayfa 953761: KURGU_OMEGA_NESNESI 842753 birimdir.",
    )

    result = validate_structured(reply, ctx)

    assert result.status == ANSWERED
    assert result.citations == (
        PageCitation(953761, MODEL_CITATION),
    )


def test_checked_citation_carries_only_internal_database_provenance():
    chunk_id = "00000000-0000-0000-0000-000000000372"
    ctx = build_rag_context([{
        "id": chunk_id,
        "filename": "kurgu.pdf",
        "page": 953761,
        "text": "KURGU_OMEGA_NESNESI 842753 birimdir.",
    }], numbered=True)
    reply = _reply(
        1,
        "KURGU_OMEGA_NESNESI 842753 birimdir.",
        "Sayfa 953761: KURGU_OMEGA_NESNESI 842753 birimdir.",
    )

    result = validate_structured(reply, ctx)

    assert result.status == ANSWERED
    assert result.citations == (
        PageCitation(953761, MODEL_CITATION, chunk_id, "kurgu.pdf"),
    )
    assert chunk_id not in ctx.model_text


def test_two_claimed_chunks_on_one_page_remain_two_evidence_targets():
    first = "00000000-0000-0000-0000-000000000471"
    second = "00000000-0000-0000-0000-000000000472"
    ctx = build_rag_context([
        {"id": first, "filename": "kurgu.pdf", "page": 953762,
         "text": "KURGU_ALPHA ilk satirdadir."},
        {"id": second, "filename": "kurgu.pdf", "page": 953762,
         "text": "KURGU_BETA ikinci satirdadir."},
    ], numbered=True)
    reply = {
        "dayanak": [
            {"pasaj": 1, "alinti": "KURGU_ALPHA ilk satirdadir."},
            {"pasaj": 2, "alinti": "KURGU_BETA ikinci satirdadir."},
        ],
        "cevap": "KURGU_ALPHA ve KURGU_BETA desteklenir.",
    }

    result = validate_structured(reply, ctx)

    assert result.status == ANSWERED
    assert result.citations == (
        PageCitation(953762, DERIVED_CITATION, first, "kurgu.pdf"),
        PageCitation(953762, DERIVED_CITATION, second, "kurgu.pdf"),
    )


def test_an_abstention_with_no_evidence_raises_nothing():
    reply = {"dayanak": [], "cevap": "Bu bilgi mevcut belgelerde bulunamadı."}
    assert check_structured(reply, CONTEXT) == []


def test_abstention_must_be_the_exact_refusal_not_a_substantive_suffix():
    for answer in (
        "Bu bilgi mevcut belgelerde bulunamadi ama zeta vardir.",
        "Bu bilgi mevcut belgelerde bulunamadi 47.",
    ):
        reply = {"dayanak": [], "cevap": answer}
        names = {name for name, _ in check_structured(reply, CONTEXT)}
        assert {"dayanaksiz_yanit", "eksik_sayfa"} <= names


def test_checked_result_has_an_explicit_publication_status():
    clean = _reply(
        1,
        "Zeta uretimi 47 000 birimdir.",
        "Sayfa 42'ye gore 47 000 birim.",
    )
    abstention = {
        "dayanak": [],
        "cevap": "Bu bilgi mevcut belgelerde bulunamadi.",
    }
    unsafe = _reply(
        1,
        "Zeta uretimi 47 000 birimdir.",
        "Sayfa 42'ye gore 8 000 birim.",
    )

    answered = validate_structured(clean, CONTEXT)
    abstained = validate_structured(abstention, CONTEXT)
    review = validate_structured(unsafe, CONTEXT)

    assert (answered.status, answered.answer) == (ANSWERED, clean["cevap"])
    assert (abstained.status, abstained.answer) == (
        ABSTAINED,
        abstention["cevap"],
    )
    assert review.status == REVIEW_REQUIRED
    assert review.answer is None
    assert {code for code, _ in review.diagnostics} == {"kaynaksiz_sayi"}


# --- the whole path, with the model faked out -------------------------------

def test_the_structured_path_holds_together_end_to_end(monkeypatch):
    """Unit tests pass while the pieces are wired to each other wrongly, and
    the next run of this path costs rented GPU hours. So drive it once with the
    model faked out: retrieval, numbering, prompt, parse, check."""
    from pipeline.generation import answer as gen
    from pipeline.retrieval import query

    seen = {}

    def fake_complete(policy, user_content):
        seen["policy"] = policy
        seen["user_content"] = user_content
        return ('```json\n{"dayanak": [{"pasaj": 1, '
                '"alinti": "Zeta uretimi 47 000 birimdir."}], '
                '"cevap": "Sayfa 42\'ye gore 47 000 birim."}\n```')

    monkeypatch.setattr(query, "retrieve", lambda q, top_k=None: CHUNKS)
    monkeypatch.setattr(query, "rerank", lambda q, chunks, top_n=None: chunks)
    monkeypatch.setattr(gen, "complete", fake_complete)

    reply = query.ask("zeta uretimi nedir?", structured=True)

    # the model was actually shown handles it could point at, and asked for the
    # evidence before the answer
    assert "[P1]" in seen["user_content"] and "[P2]" in seen["user_content"]
    assert seen["policy"].index("dayanak") < seen["policy"].index('"cevap"')
    assert "Zeta uretimi" not in seen["policy"]

    parsed = parse_structured(reply)
    assert parsed["cevap"].endswith("47 000 birim.")
    assert check_structured(reply, build_rag_context(CHUNKS, numbered=True)) == []


def test_the_native_public_path_returns_only_a_checked_result(monkeypatch):
    from pipeline.generation import answer as gen
    from pipeline.retrieval import query

    reply = {
        "dayanak": [{
            "pasaj": 1,
            "alinti": "Zeta uretimi 47 000 birimdir.",
        }],
        "cevap": "Sayfa 42'ye gore 47 000 birim.",
    }
    monkeypatch.setattr(query, "retrieve", lambda q, top_k=None: CHUNKS)
    monkeypatch.setattr(query, "rerank", lambda q, chunks, top_n=None: chunks)
    monkeypatch.setattr(gen, "generate_structured", lambda q, c: reply)

    result = query.ask_checked("zeta uretimi nedir?")

    assert result.status == ANSWERED
    assert result.answer == reply["cevap"]
    assert result.diagnostics == ()
    assert result.trace.backend == "native"
    assert result.trace.retrieved_count == len(CHUNKS)
    assert result.trace.reranked_count == len(CHUNKS)
    assert result.trace.context_passage_count == len(CHUNKS)
    assert tuple(stage.name for stage in result.trace.stages) == (
        "retrieve", "rerank", "context", "generate", "validate")


def test_the_llamaindex_public_path_returns_only_a_checked_result(monkeypatch):
    from pipeline.generation import answer as gen
    from pipeline.retrieval import rag_llamaindex

    reply = {
        "dayanak": [{
            "pasaj": 1,
            "alinti": "Zeta uretimi 47 000 birimdir.",
        }],
        "cevap": "Sayfa 42'ye gore 47 000 birim.",
    }
    monkeypatch.setattr(rag_llamaindex, "retrieve", lambda q: CHUNKS)
    monkeypatch.setattr(gen, "generate_structured", lambda q, c: reply)

    result = rag_llamaindex.answer_checked("zeta uretimi nedir?")

    assert result.status == ANSWERED
    assert result.answer == reply["cevap"]
    assert result.diagnostics == ()
    assert result.trace.backend == "llamaindex"
    assert result.trace.retrieved_count == len(CHUNKS)
    assert result.trace.reranked_count is None
    assert result.trace.context_passage_count == len(CHUNKS)
    assert tuple(stage.name for stage in result.trace.stages) == (
        "retrieve", "context", "generate", "validate")


def test_the_plain_path_is_untouched(monkeypatch):
    """The default must keep producing exactly what every earlier run produced,
    or nothing is comparable with anything."""
    from pipeline.generation import answer as gen
    from pipeline.retrieval import query

    seen = {}
    monkeypatch.setattr(query, "retrieve", lambda q, top_k=None: CHUNKS)
    monkeypatch.setattr(query, "rerank", lambda q, chunks, top_n=None: chunks)
    def fake_complete(policy, user_content):
        seen["policy"] = policy
        seen["user_content"] = user_content
        return "duz cevap"

    monkeypatch.setattr(gen, "complete", fake_complete)
    assert query.ask("zeta uretimi nedir?") == "duz cevap"
    assert "[P1]" not in seen["user_content"]
    assert "json" not in (seen["policy"] + seen["user_content"]).lower()
    assert "Zeta uretimi" not in seen["policy"]
