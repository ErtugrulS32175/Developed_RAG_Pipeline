"""Answer grading and, more importantly, fault attribution: a wrong answer has
to be blamed on the half of the system that actually caused it.

Every figure below is invented and verified absent from the corpus -- test data
must never be lifted from a real document.
"""
from eval.rag_answer_eval import abstained, cited_pages, score_one, summarize
from eval.rag_eval import contains_key


def _q(**kw):
    base = {"q": "soru?", "pages": [42], "key": "555 birim"}
    base.update(kw)
    return base


# --- contains_key ---

def test_plain_substring_match():
    assert contains_key("toplamda 555 birim vardir", "555 birim") is True


def test_case_and_spacing_are_ignored():
    assert contains_key("Toplam   555   BIRIM", "555 birim") is True


def test_number_formatting_does_not_matter():
    """8.765 / 8765 / 8,765 are the same figure written three ways."""
    assert contains_key("deger 8.765 birim", "8765") is True
    assert contains_key("deger 8765 birim", "8.765") is True


def test_digits_are_not_spliced_across_neighbouring_numbers():
    """A whole-string digit compare would turn '87 65' into '8765' and match a
    figure that is not in the text at all."""
    assert contains_key("kalem 87 ve kalem 65", "8765") is False


def test_absent_key_is_false_and_missing_key_is_none():
    assert contains_key("baska bir metin", "555 birim") is False
    assert contains_key("herhangi bir metin", None) is None


# --- citation parsing ---

def test_reads_the_page_the_prompt_asks_for():
    assert cited_pages("Sayfa 7'ye göre deger 5'tir.") == {7}
    assert cited_pages("sayfa 42 ve Sayfa 96") == {42, 96}
    assert cited_pages("hicbir atif yok") == set()


def test_abstention_is_recognised():
    assert abstained("Bu bilgi mevcut belgelerde bulunamadı.")
    assert not abstained("Cevap 555 birimdir.")


# --- attribution ---

def test_missing_from_context_is_blamed_on_retrieval():
    r = score_one(_q(), "Bu bilgi mevcut belgelerde bulunamadı.", "alakasiz baglam")
    assert r["hata"] == "retrieval"
    assert r["ctx_var"] is False


def test_present_in_context_but_refused_is_blamed_on_generation():
    r = score_one(_q(), "Bu bilgi mevcut belgelerde bulunamadı.", "olcum 555 birim")
    assert r["hata"] == "uretim_cekimser"


def test_present_in_context_but_answered_wrong_is_blamed_on_generation():
    r = score_one(_q(), "Sayfa 42'ye göre 444 birimdir.", "olcum 555 birim")
    assert r["hata"] == "uretim_yanlis"
    assert r["cevap_dogru"] is False


def test_correct_answer_has_no_fault():
    r = score_one(_q(), "Sayfa 42'ye göre 555 birimdir.", "olcum 555 birim")
    assert r["hata"] is None
    assert r["cevap_dogru"] and r["sayfa_dogru"]


def test_right_answer_with_the_wrong_page_is_still_a_citation_failure():
    """The regression the page fix exists for: before it, every answer cited
    page 1 no matter which page it came from."""
    r = score_one(_q(), "Sayfa 1'e göre 555 birimdir.", "olcum 555 birim")
    assert r["cevap_dogru"] is True
    assert r["sayfa_dogru"] is False
    assert r["hata"] is None          # the answer itself was right


# --- summary ---

def test_summary_counts_each_dimension_separately():
    rows = [
        score_one(_q(), "Sayfa 42: 555 birim", "555 birim"),      # tam dogru
        score_one(_q(), "Sayfa 1: 555 birim", "555 birim"),        # sayfa yanlis
        score_one(_q(), "bulunamadı", "alakasiz"),                 # retrieval
    ]
    m = summarize(rows)
    assert m["n"] == 3
    assert m["ctx_recall"] == round(2 / 3, 4)
    assert m["cevap_dogrulugu"] == round(2 / 3, 4)
    assert m["sayfa_dogrulugu"] == round(1 / 3, 4)
    assert m["hata_dagilimi"] == {"retrieval": 1}


def test_empty_run_does_not_divide_by_zero():
    assert summarize([]) == {"n": 0}


def test_figures_and_prose_are_reported_separately():
    """One number over both hides whichever half is weaker -- a figure is copied
    or it is not, while prose can be fluent and still miss."""
    rows = [
        score_one(_q(type="sayisal"), "Sayfa 42: 555 birim", "555 birim"),
        score_one(_q(type="sayisal"), "bulunamadı", "555 birim"),
        score_one(_q(type="metin"), "Sayfa 42: 555 birim", "555 birim"),
    ]
    m = summarize(rows)
    assert m["tipe_gore"]["sayisal"]["cevap_dogrulugu"] == 0.5
    assert m["tipe_gore"]["metin"]["cevap_dogrulugu"] == 1.0
    assert "tipe_gore" not in m["tipe_gore"]["metin"]      # no infinite nesting


def test_no_type_split_when_every_question_is_one_type():
    rows = [score_one(_q(type="sayisal"), "Sayfa 42: 555 birim", "555 birim")]
    assert "tipe_gore" not in summarize(rows)
