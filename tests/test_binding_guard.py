"""The binding ANNOTATION SIGNAL: what it catches, what it must not do,
and where its known limits sit.

Fixtures reuse the invented, absence-verified vocabulary of the mutation
tests (Alfa/Beta Sirketi, Zeta Endeksi, 47 000 / 88 000, page 42). What is
locked: the comparative rule -- a figure from a WORSE-matching segment while
a better-matching segment offers a different figure is flagged; equal-best
is not; no term signal or no figure occurrence means no flag.

The one-sided guarantee is NARROW and named precisely: header inheritance
cannot create a flag, it only clears one. The signal as a whole can still
annotate a correct answer -- the figure-bearing-header test pins that known
limit instead of hiding it -- which is why two further tests pin the module
OUT of the publication import graph: a false annotation may cost reviewer
minutes, never a withheld answer.
"""
import pytest

from pipeline.retrieval.context import Passage, RagContext
from pipeline.validation.rag.binding_guard import (
    WRONG_BINDING, _expand_aliases, check_binding)

SIRA = ("Alfa Sirketi, Zeta Endeksi = 47 000. "
        "Beta Sirketi, Zeta Endeksi = 88 000.")
OTHER = "Baska bir konu anlatan metin 12 345 gibi bir deger tasir."

QUESTION = "Alfa Sirketi'nin Zeta Endeksi nedir?"


def _ctx(*texts):
    passages = tuple(
        Passage(i + 1, 42, text, f"kurgu-belge s.42/{i + 1}")
        for i, text in enumerate(texts)
    )
    return RagContext(passages=passages, numbered=True)


def test_sibling_row_value_is_flagged():
    flags = check_binding(QUESTION, "Sayfa 42'ye gore 88 000 birim.", _ctx(SIRA))
    assert [code for code, _ in flags] == [WRONG_BINDING]
    assert "88" in flags[0][1][0]


def test_the_requested_row_value_is_clean():
    assert check_binding(QUESTION, "Sayfa 42'ye gore 47 000 birim.", _ctx(SIRA)) == []


def test_inflected_entity_still_matches_through_stems():
    # the question carries "Sirketi'nin"; the passage has bare "Sirketi"
    flags = check_binding("Alfa Sirketinin Zeta Endeksi kactir?",
                          "Cevap 88 000 birimdir.", _ctx(SIRA))
    assert [code for code, _ in flags] == [WRONG_BINDING]


def test_a_figure_absent_from_scope_is_not_this_checks_job():
    assert check_binding(QUESTION, "Cevap 12 345 birimdir.", _ctx(SIRA)) == []


def test_no_term_signal_means_no_flag():
    assert check_binding("Gama Kurulusu hakkinda ne deniyor?",
                         "Cevap 88 000 birimdir.", _ctx(SIRA)) == []


def test_equal_best_segments_do_not_flag_either_value():
    # without the entity, both rows match the question equally well
    question = "Zeta Endeksi kactir?"
    assert check_binding(question, "Cevap 88 000.", _ctx(SIRA)) == []
    assert check_binding(question, "Cevap 47 000.", _ctx(SIRA)) == []


def test_page_citations_are_not_treated_as_figures():
    flags = check_binding(QUESTION, "Sayfa 42'ye gore 88 000 birim.", _ctx(SIRA))
    assert all("42" != token for _, tokens in flags for token in tokens)


def test_cited_handles_scope_the_check():
    context = _ctx(SIRA, OTHER)
    flagged = check_binding(QUESTION, "Sayfa 42'ye gore 88 000 birim.",
                            context, cited_handles=[1])
    assert [code for code, _ in flagged] == [WRONG_BINDING]
    # scoped to a passage with no question terms: silence, not a guess
    assert check_binding(QUESTION, "Cevap 12 345.", context,
                         cited_handles=[2]) == []


def test_scope_defined_alias_reaches_the_long_name_rows():
    """The measured miss: the question says the short name, the rows carry
    only the long official name. The passage defines the alias itself in
    parentheses, so the expansion is derived from scope, not from a list."""
    text = ("Gama Metal Isleri A.S. (Gamis), Zeta Endeksi = 47 000. "
            "Beta Sirketi, Zeta Endeksi = 88 000.")
    context = _ctx(text)
    flags = check_binding("Gamis'in Zeta Endeksi nedir?",
                          "Cevap 88 000 birimdir.", context)
    assert [code for code, _ in flags] == [WRONG_BINDING]
    assert check_binding("Gamis'in Zeta Endeksi nedir?",
                         "Cevap 47 000 birimdir.", context) == []


def test_alias_defined_outside_the_cited_scope_still_counts():
    """The definition usually lives in prose while the row lives in a table;
    citing only the table must not blind the check to the alias."""
    table = ("Gama Metal Isleri A.S., Zeta Endeksi = 47 000. "
             "Beta Sirketi, Zeta Endeksi = 88 000.")
    prose = "Kurulus hakkinda: Gama Metal Isleri A.S. (Gamis) kurgudur."
    flags = check_binding("Gamis'in Zeta Endeksi nedir?",
                          "Cevap 88 000 birimdir.", _ctx(table, prose),
                          cited_handles=[1])
    assert [code for code, _ in flags] == [WRONG_BINDING]


def test_a_question_qualifier_is_cleared_by_the_sibling_rule_not_by_exemption():
    """A year the question states is not blamed -- but nothing special-cases
    it. "2024" carries no label a best-matching segment offers a different
    value under, so the sibling precondition clears it on its own. The
    special case that used to do this job is gone; it was an attack surface
    (see the planted-figure test) and measurement showed it bought nothing."""
    text = ("Alfa Sirketi, Zeta Endeksi = 47 000. "
            "Beta Sirketi 2024, Zeta Endeksi = 88 000.")
    flags = check_binding("2024'te Alfa Sirketi'nin Zeta Endeksi kacti?",
                          "2024 itibariyla 88 000 birim.", _ctx(text))
    assert [code for code, _ in flags] == [WRONG_BINDING]
    assert flags[0][1] == ("88 000",)


def test_prose_outranking_the_value_line_is_not_a_sibling():
    """The mechanism behind most false alarms on CORRECT answers: an
    entity-rich prose segment outscores the true value line. Without a
    same-label sibling figure in the best segment there is nothing to bind
    wrongly, so the check must stay silent."""
    value_line = "Zeta Endeksi 47 000"
    prose = ("Omega Kurumu Zeta Endeksi hakkinda genel degerlendirme "
             "12 345 sayili notta uzun uzun anlatilmistir.")
    flags = check_binding("Omega Kurumu Zeta Endeksi nedir?",
                          "Cevap 47 000 birimdir.", _ctx(value_line, prose))
    assert flags == []


def test_a_figure_planted_in_the_question_cannot_silence_the_check():
    """Auditor attack BG-01: put the wrong value in the question and the
    echo exemption skipped the figure entirely. A first repair made the
    exemption conditional on the answer asserting SOME other figure, and the
    auditor walked straight through it -- adding any second number to the
    answer restored the bypass. The exemption is gone."""
    flags = check_binding("Alfa Sirketi'nin Zeta Endeksi 88 000 midir?",
                          "Evet, 88 000 birimdir.", _ctx(SIRA))
    assert [code for code, _ in flags] == [WRONG_BINDING]
    # the multi-figure variant: a second, unrelated number in the answer
    flags = check_binding("Alfa Sirketi'nin Zeta Endeksi 88 000 midir?",
                          "Evet, 88 000 birimdir; ayrica 12 345 sayili notta gecer.",
                          _ctx(SIRA))
    assert [code for code, _ in flags] == [WRONG_BINDING]
    assert flags[0][1] == ("88 000",)
    # a qualifier stated in both records is still not blamed
    text = ("Alfa Sirketi 2024, Zeta Endeksi = 47 000. "
            "Beta Sirketi 2024, Zeta Endeksi = 88 000.")
    flags = check_binding("2024'te Alfa Sirketi'nin Zeta Endeksi neydi?",
                          "2024 yilinda 88 000 birim.", _ctx(text))
    assert flags[0][1] == ("88 000",)


def test_an_inserted_header_degrades_the_check_to_silence_not_inversion():
    """The header-insertion attack, and the trade three audit rounds forced.

    Every attempt to keep catching this shape -- rank gates on question
    terms, on capitalisation, on unknown content words -- handed the
    auditor a layout where the check flagged the CORRECT answer instead.
    A diagnostic that can be turned against correct answers is worse than
    one that can be muted, so header credit now only ever CLEARS a figure.
    An attacker who controls the context can silence this check (they
    always could, by withholding the row); they can no longer invert it.
    Both directions are pinned: no flag on the wrong answer, and none on
    the correct one either."""
    upper = ("Alfa Sirketi, Zeta Endeksi = 47 000. "
             "Alfa Sirketi Ozeti. "
             "Beta Sirketi, Zeta Endeksi = 88 000.")
    lower = ("alfa sirketi, zeta endeksi = 47 000\n"
             "alfa sirketi ozeti\n"
             "beta sirketi, zeta endeksi = 88 000")
    for text in (upper, lower):
        assert check_binding(QUESTION, "Cevap 88 000 birimdir.", _ctx(text)) == []
        assert check_binding(QUESTION, "Cevap 47 000 birimdir.", _ctx(text)) == []


def test_a_single_word_label_is_in_scope():
    """Auditor attack BG-03: a genuine one-word label was excluded by the
    two-word floor, so that whole sibling shape went unchecked."""
    text = "Alfa Sirketi Endeks 47 000. Beta Sirketi Endeks 88 000."
    flags = check_binding("Alfa Sirketi Endeks nedir?",
                          "Cevap 88 000.", _ctx(text))
    assert [code for code, _ in flags] == [WRONG_BINDING]


def test_an_out_of_scope_alias_definition_does_not_reach_the_scoring():
    """Auditor attack BG-04: a definition elsewhere in the context injected
    terms that re-ranked segments and flagged a CORRECT answer. A
    definition applies only when its long name occurs in the scored scope."""
    table = "Omega Kurumu, Zeta Endeksi = 47 000."
    elsewhere = "Ayri bir konu: Gama Metal Isleri A.S. (Gamis) kurgudur."
    assert check_binding("Gamis'in Zeta Endeksi nedir?",
                         "Cevap 47 000 birimdir.",
                         _ctx(table, elsewhere), cited_handles=[1]) == []


def test_a_header_still_lends_its_entity_to_a_bare_value_line():
    """Auditor attack BG-02, the other side of it: denying the credit
    whenever the value line matched ANY question term broke the ordinary
    card layout, because a value line repeats the attribute ("Zeta
    Endeksi") while the entity sits on the line above. The credit is a
    union now, gated on the line introducing no name of its own."""
    card = ("Omega Kurumu\n"
            "Zeta Endeksi 47 000\n"
            "Beta Kurumu Zeta Endeksi 88 000")
    assert check_binding("Omega Kurumu Zeta Endeksi nedir?",
                         "Cevap 47 000 birimdir.", _ctx(card)) == []


def test_scattered_long_name_words_do_not_open_the_alias_gate():
    """Auditor attack BG-04: requiring one shared word let an unrelated
    definition apply. Here only "Metal" occurs in the scope, as part of a
    different row, and the injected terms re-ranked the segments."""
    table = ("Omega Kurumu, Zeta Endeksi = 47 000. "
             "Metal Bolumu, Zeta Endeksi = 88 000.")
    elsewhere = "Ayri bir konu: Gama Metal Isleri A.S. (Gamis) kurgudur."
    assert check_binding("Gamis'in Zeta Endeksi nedir?",
                         "Cevap 47 000 birimdir.",
                         _ctx(table, elsewhere), cited_handles=[1]) == []


def test_header_inheritance_cannot_create_a_flag():
    """The NARROW guarantee that holds -- and only this one. Inherited
    credit is used solely to clear a figure, so the gate-heuristic
    inversions of rounds 4-7 (cards, inserted headers, unlisted unit words,
    short record codes) all stay clean on the correct answer. It is NOT a
    promise that no layout can annotate a correct answer: the own-term
    ranking still can, and the figure-bearing-header test below pins that
    known limit."""
    question = "Omega Kurumu Zeta Endeksi nedir?"
    shapes = (
        # card + inserted header (the round-6 inversion)
        "Omega Kurumu\nZeta Endeksi 47 000\n"
        "Omega Kurumu Ozeti\nBeta Kurumu Zeta Endeksi 88 000",
        # value line carrying a unit word no stoplist contains (round-7 P1)
        "Omega Kurumu\nZeta Endeksi 47 000 birim\n"
        "Omega Kurumu Ozeti\nBeta Kurumu Zeta Endeksi 88 000",
        # sibling identified only by a short code (round-7 P1)
        "Omega Kurumu\nZeta Endeksi 47 000\n"
        "Omega Kurumu Ozeti\nB2 Zeta Endeksi 88 000",
    )
    for text in shapes:
        assert check_binding(question, "Cevap 47 000 birimdir.",
                             _ctx(text)) == []


def test_a_figure_bearing_header_is_a_known_false_annotation_limit():
    """Auditor finding BG-R8: a header carrying ANY figure -- a year, a
    table number -- cuts the lenient chain, the value line under it loses
    the header's entity terms even for suppression, and a fuller sibling
    row wins the own ranking. The correct answer gets annotated and the
    wrong one does not.

    This test PINS the limit, it does not bless it. These are ordinary
    table shapes, which is exactly why the module is an annotation signal
    and not a validator: fixing this inside flat text means another guess
    about what a figure-bearing line IS, and four audit rounds showed where
    those guesses end. If this behaviour ever changes, this test failing is
    the announcement that the limit moved."""
    question = "Omega Kurumu Zeta Endeksi nedir?"
    year_header = ("Omega Kurumu 2024 Ozeti\n"
                   "Zeta Endeksi 47 000\n"
                   "Beta Kurumu Zeta Endeksi 88 000")
    numbered_header = ("Omega Kurumu\nCizelge 7\n"
                       "Zeta Endeksi 47 000\n"
                       "Beta Kurumu Zeta Endeksi 88 000")
    for text in (year_header, numbered_header):
        wrong = check_binding(question, "Cevap 88 000.", _ctx(text))
        right = check_binding(question, "Cevap 47 000.", _ctx(text))
        assert wrong == []                                   # the miss
        assert [code for code, _ in right] == [WRONG_BINDING]  # the false annotation


def test_binding_guard_stays_out_of_the_publication_import_graph():
    """The module docstring's one enforceable promise: this signal never
    reaches a publish/withhold decision. A fresh interpreter imports the
    production guard and the API; if either ever pulls binding_guard into
    its import graph, the printed list stops being empty. Run in a
    subprocess because this test module itself imports binding_guard, so
    the current interpreter's module table proves nothing."""
    import os
    import subprocess
    import sys
    code = (
        "import sys\n"
        "import pipeline.validation.rag.answer_guard\n"
        "import pipeline.api.app\n"
        "print([m for m in sys.modules if 'binding_guard' in m])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr  # the imports must succeed
    assert result.stdout.strip() == "[]"


def test_no_pipeline_module_references_binding_guard_even_lazily():
    """Round-9 P2: the subprocess test pins only the EAGER import graph --
    a lazy import inside a function body, or importlib with a string, would
    sail through it. The annotation-only boundary is an architectural
    invariant, so it is enforced statically: outside binding_guard.py
    itself, the token "binding_guard" may not appear anywhere under
    pipeline/, not as an import, not as a string, not in a comment. The
    string form is banned deliberately -- a module path written as a string
    already defeated an import checker once in this project."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "pipeline"
    offenders = [
        str(path.relative_to(root.parent))
        for path in sorted(root.rglob("*.py"))
        if path.name != "binding_guard.py"
        and "binding_guard" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_an_abbreviation_dot_is_not_a_record_boundary():
    """Auditor finding BG-R5-02: splitting at "A.S. " cut the long name in
    half, so the alias gate -- which requires the name as a phrase inside
    one segment -- never saw it, no expansion happened, and the wrong
    sibling passed in silence. The worst variant puts the abbreviation
    INSIDE the name, where the cut lands between its content words."""
    table = ("Gama A.S. Metal Isleri Zeta Endeksi = 47 000. "
             "Beta Sirketi, Zeta Endeksi = 88 000.")
    prose = "Kurulus hakkinda: Gama A.S. Metal Isleri (Gamis) kurgudur."
    flags = check_binding("Gamis'in Zeta Endeksi nedir?",
                          "Cevap 88 000 birimdir.", _ctx(table, prose),
                          cited_handles=[1])
    assert [code for code, _ in flags] == [WRONG_BINDING]
    assert check_binding("Gamis'in Zeta Endeksi nedir?",
                         "Cevap 47 000 birimdir.", _ctx(table, prose),
                         cited_handles=[1]) == []


def test_a_lower_case_passage_is_still_split_into_records():
    """Found by attacking our own repair rather than by an audit: the
    boundary required an upper-case start, so a passage normalised to lower
    case never split at all. One segment means no comparison, and the check
    returned silence -- which reads exactly like a clean result. Nothing in
    the output distinguished "nothing to report" from "never ran"."""
    flat = ("alfa sirketi, zeta endeksi = 47 000. "
            "beta sirketi, zeta endeksi = 88 000.")
    flags = check_binding(QUESTION, "Cevap 88 000 birimdir.", _ctx(flat))
    assert [code for code, _ in flags] == [WRONG_BINDING]


def test_an_alias_phrase_must_sit_inside_one_segment():
    """Auditor finding BG-04, second round: flattening a passage into one
    word sequence puts the end of one record beside the start of the next,
    and the halves of a long name landing either side of that join read as
    a phrase the document never says -- flagging a correct answer."""
    split = ("Omega Kurumu Gama. "
             "Metal Isleri Bolumu, Zeta Endeksi = 88 000. "
             "Omega Kurumu, Zeta Endeksi = 47 000.")
    prose = "Tanim: Gama Metal Isleri A.S. (Gamis) kurgudur."
    assert check_binding("Gamis'in Zeta Endeksi nedir?",
                         "Cevap 47 000 birimdir.",
                         _ctx(split, prose), cited_handles=[1]) == []


def test_a_year_in_the_label_keeps_two_columns_apart():
    """Auditor finding BG-03: the frame deleted digits, so "Endeks 2023"
    and "Endeks 2024" became one label and a value from the other year
    counted as a sibling -- flagging a correct answer."""
    text = ("Endeks 2023 47 000. "
            "Omega Kurumu Endeks 2024 88 000.")
    assert check_binding("Omega Kurumu 2023 Endeksi nedir?",
                         "Cevap 47 000 birimdir.", _ctx(text)) == []


def test_a_footnote_parenthesis_is_never_read_as_an_alias():
    text = "Alfa Sirketi (2), Zeta Endeksi = 47 000. Beta Sirketi, Zeta Endeksi = 88 000."
    expanded = _expand_aliases(frozenset({"alfa"}),
                               _ctx(text).passages, _ctx(text).passages)
    assert expanded == frozenset({"alfa"})


def test_requires_a_rag_context():
    with pytest.raises(TypeError):
        check_binding(QUESTION, "cevap", "duz metin baglam")
