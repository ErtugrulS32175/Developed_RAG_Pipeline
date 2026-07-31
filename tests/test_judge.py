"""Three-state judging: what may be accepted, what may only be rejected, and
what has to go to a person.

Every figure and every stem below is invented -- test fixtures must never be
copied out of a real document. The morphology is real Turkish applied to
nonsense words, which is the point: the suffix rules have to work on a stem the
author has never seen.

Nothing here needs the embedding service. Tier 3 is exercised by supplying the
similarity directly, so the suite stays runnable with no containers up.
"""
import pytest

from eval.answer.judge import (DOGRU, INCELE, YANLIS, expand_cardinals, expand_months,
                        judge, normalize, notation_match, stems)


# --- suffix stripping -------------------------------------------------------

def test_a_word_reduces_to_several_candidate_stems():
    """Which ending a word carries is ambiguous from the spelling: "zetaya"
    could be losing "-a" or "-ya". Committing to one is how a stem became a
    two-letter fragment that matched nothing."""
    assert "zeta" in stems("zetaya")
    assert "zeta" in stems("zetadan")
    assert "zeta" in stems("zetalarin")


def test_a_short_word_is_left_alone():
    """Without the floor, a three-letter word loses two characters and what
    remains matches almost anything."""
    assert stems("ay") == {"ay"}
    assert "g" not in stems("gun")


def test_an_inflected_word_meets_its_bare_form():
    assert stems("zetadan") & stems("zeta")


# --- Turkish number words ---------------------------------------------------

def test_number_words_become_digits():
    assert expand_cardinals("kirk yedi bin zeta") == "47000 zeta"
    assert expand_cardinals("iki yuz elli bin") == "250000"
    assert expand_cardinals("on iki gamma") == "12 gamma"


def test_an_inflected_number_word_still_counts():
    """Only the last word of a Turkish number phrase inflects."""
    assert expand_cardinals("dokuzu") == "9"


def test_a_magnitude_word_after_a_digit_is_left_alone():
    """'47 bin' is a figure with a magnitude word and is already handled as one.
    Expanding the 'bin' here would invent a separate 1000 that no correct
    answer contains."""
    assert expand_cardinals("47 bin zeta") == "47 bin zeta"


def test_a_bare_bir_is_not_read_as_the_number_one():
    """It is the indefinite article far more often than it is a quantity."""
    assert expand_cardinals("bir zeta bulundu") == "bir zeta bulundu"
    assert expand_cardinals("bir milyon zeta") == "1000000 zeta"


def test_number_words_that_do_not_descend_are_not_one_number():
    """Turkish builds a number from larger to smaller -- 'yirmi bes', never
    'bes yirmi'. Found by the mismatched-pair control: a fraction was being
    read as the sum of its parts, and the invented figure matched an unrelated
    answer."""
    assert expand_cardinals("dokuzda biri") != "10"
    assert expand_cardinals("uc bes") == "uc bes"
    assert expand_cardinals("yirmi bes") == "25"


def test_an_inflected_bir_is_not_the_quantity_one():
    """'biri' is 'one of them'; 'birim' is a unit. Neither is a figure."""
    assert expand_cardinals("zetalarin biri") == "zetalarin biri"


def test_a_rate_is_not_a_quantity():
    """'binde' is 'per thousand'. Reading its stem as a scale word would turn
    an expected rate into an expected amount."""
    assert expand_cardinals("binde yedi") == "binde 7"


# --- dates ------------------------------------------------------------------

def test_a_written_month_meets_a_dotted_date():
    assert expand_months("15 mart 1977 tarihli") == "15.3.1977 tarihli"


def test_a_month_name_without_a_date_around_it_is_untouched():
    assert expand_months("mart ayinda") == "mart ayinda"


# --- what notation matching may accept --------------------------------------

def test_a_number_word_meets_the_same_figure_in_digits():
    assert notation_match("toplam 47 000 zeta odenir", "kirk yedi bin zeta")


def test_a_compound_number_word_meets_its_digits():
    assert notation_match("47 000 zeta ile 940 000 zeta arasinda",
                          "kirk yedi bin zetadan dokuz yuz kirk bin")


def test_an_inflected_unit_meets_its_bare_form():
    assert notation_match("kapasitesi 47 000 zetadir", "47 bin zeta")


def test_a_written_date_meets_a_dotted_one():
    assert notation_match("15 Mart 1977 tarihli kapanislara gore", "15.3.1977")


def test_a_connective_the_answer_drops_is_excused():
    """The expected answer names a bound ('... zetaya kadar'); an answer to
    'what is the upper limit' need not repeat the bound marker."""
    assert notation_match("ust siniri 940 zetadir", "dokuz yuz kirk zetaya kadar")


# --- what it must not accept ------------------------------------------------

def test_a_different_figure_is_never_accepted():
    assert not notation_match("toplam 8 000 zeta", "kirk yedi bin zeta")


def test_the_unit_still_has_to_be_there():
    """Loose word matching must not become no word matching."""
    assert not notation_match("toplam 47 000 gamma", "kirk yedi bin zeta")


def test_a_word_that_carries_the_meaning_is_not_excused():
    """The anchor case: an expected 'en gec' against an answered 'en cok' has
    the same figure and opposite meanings. It has to stay unresolved."""
    assert not notation_match("en cok dokuz zeta icinde", "en gec dokuz zeta")


# --- the three states -------------------------------------------------------

def test_an_exact_match_is_correct_without_any_similarity():
    durum, why = judge("47 bin zeta", "Sayfa 42'ye gore 47 bin zeta.", "47 bin zeta")
    assert durum == DOGRU and why == "birebir"


def test_a_notation_variant_is_correct():
    durum, why = judge("kirk yedi bin zeta", "Sayfa 42: 47 000 zetadir.", "kirk yedi bin zeta")
    assert durum == DOGRU and why == "notasyon"


def test_an_unrelated_answer_is_wrong():
    durum, why = judge("kirk yedi bin zeta", "Sayfa 42: 8 000 gamma.",
                       "kirk yedi bin zeta", sim=0.11)
    assert durum == YANLIS and "0.11" in why


def test_a_plausible_paraphrase_goes_to_a_person():
    durum, _ = judge("en gec dokuz zeta", "en cok dokuz zeta icinde",
                     "en gec dokuz zeta", sim=0.95)
    assert durum == INCELE


def test_high_similarity_never_counts_as_correct():
    """The load-bearing rule. One adjudicated pair differs by a single word,
    means the opposite, and embeds at 0.57 -- no threshold separates it from a
    genuine rephrasing, so similarity may reject and never accept."""
    durum, _ = judge("en gec dokuz zeta", "en cok dokuz zeta", "en gec dokuz zeta",
                     sim=0.999)
    assert durum != DOGRU


def test_without_a_reference_nothing_is_called_wrong():
    durum, why = judge("kirk yedi bin zeta", "Sayfa 42: 8 000 gamma.", None)
    assert durum == INCELE and "referans" in why


def test_an_unreachable_embedding_service_sends_answers_to_review(monkeypatch):
    """Scoring must degrade to 'ask a person', never to 'wrong'."""
    monkeypatch.setattr("eval.answer.judge.similarity", lambda a, b: None)
    durum, why = judge("kirk yedi bin zeta", "Sayfa 42: 8 000 gamma.", "kirk yedi bin zeta")
    assert durum == INCELE and "servis" in why


# --- how the states reach the metrics ---------------------------------------

def _q(**kw):
    base = {"q": "soru?", "pages": [42], "key": "kirk yedi bin zeta",
            "answer": "kirk yedi bin zeta"}
    base.update(kw)
    return base


def test_an_unsettled_answer_is_not_blamed_on_the_generator():
    from eval.answer.rag_answer_eval import score_one
    r = score_one(_q(), "Sayfa 42: en cok kirk yedi bin gamma.", "olcum kirk yedi bin zeta",
                  sim=0.9)
    assert r["durum"] == INCELE
    assert r["hata"] is None          # not yet anybody's fault
    assert r["cevap_dogru"] is False  # and not counted as correct either


def test_accuracy_is_reported_as_a_band():
    """Reporting only what the scorer could confirm made one run look far worse
    than it was; reporting the upper bound alone would flatter it."""
    from eval.answer.rag_answer_eval import score_one, summarize
    rows = [
        score_one(_q(), "Sayfa 42: 47 000 zeta", "olcum kirk yedi bin zeta"),
        score_one(_q(), "Sayfa 42: en cok kirk yedi bin gamma", "olcum kirk yedi bin zeta",
                  sim=0.9),
        score_one(_q(), "Sayfa 42: 8 000 gamma", "olcum kirk yedi bin zeta", sim=0.1),
    ]
    m = summarize(rows)
    assert m["cevap_dogrulugu"] == round(1 / 3, 4)
    assert m["incele_orani"] == round(1 / 3, 4)
    assert m["ust_sinir"] == round(2 / 3, 4)
    assert m["hata_dagilimi"] == {"uretim_yanlis": 1}


@pytest.mark.parametrize("text", ["", None])
def test_empty_input_does_not_raise(text):
    assert notation_match(text or "", "kirk yedi bin zeta") is False
    assert normalize(text or "") == ""
