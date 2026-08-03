"""Three-state answer judging: correct, wrong, or needs a human.

The scorer this replaces forced a binary verdict, and a human read of one run's
rejections found that 13 of 15 were the scorer's fault rather than the system's.
The precision is deeply asymmetric: most historical rejections were unjust,
while adversarial probes found that unordered word matching could accept
swapped mappings and negation. The response is conservative acceptance plus a
review state, not forcing every unresolved pair into "wrong".

Three tiers, and only the first two may ACCEPT:

  1. the folded key as an actual phrase, with matching negation polarity
  2. NOTATION identities: number words, month names and conservative
     inflectional suffixes. Multi-figure mappings preserve order.
  3. embedding similarity, which may only REJECT. It separates an answer that
     is unrelated to the reference from one that might be a valid paraphrase,
     and hands the latter to a person.

**Tier 3 is barred from accepting**, and that is the load-bearing decision. An
adjudicated case turns on a single adverb: the expected answer and the given
one differ by one short word, carry opposite meanings, and embed almost
identically. No threshold separates that pair, so a similarity score is
evidence of relatedness only, never of correctness.

Tier 2 is reached only after every FIGURE in the expected answer has been
found, so its word-level looseness cannot let a wrong number through -- the
numbers are checked strictly and the surrounding words loosely, which is the
right way round for answers that are mostly figures.
"""
import logging
import math
import os
import re

# The notation layer -- number words, month names, suffixes -- lives in
# pipeline/lang/tr_notation, because the answer guard that ships with the product
# needs exactly the same vocabulary and must not reach into the eval package
# for it. What stays here is the POLICY: which differences are forgiven, and
# what may accept versus only reject. Re-exported so existing callers and tests
# keep working unchanged.
from pipeline.lang.tr_notation import (  # noqa: F401
    expand_cardinals, expand_months, fold, normalize, number_forms, numbers,
    stems, _MAGNITUDE, _SCALE, _THOUSANDS_SPACE)

logger = logging.getLogger(__name__)


# Words the expected answer may carry that a correct answer need not repeat.
# Deliberately tiny, and deliberately EXCLUDING the comparatives (en, cok, az,
# gec, fazla): those are exactly what distinguishes one adjudicated case's
# correct reading from its incorrect one, so dropping them would auto-accept a
# wrong answer.
_FUNCTION = {"kadar", "ile", "ve", "arasi", "arasinda", "icin", "gore",
             "olarak", "uzerinden", "yaklasik", "civarinda"}


_UNIT = re.compile(
    r"\d[\d.,]*(?:\s*(?:" + "|".join(_MAGNITUDE) + r"))?|[a-z]+"
)
_NEGATORS = {"degil", "yok", "hayir"}
_KEY_CASE_SUFFIXES = {
    "a", "e", "ya", "ye", "i", "u", "yi", "yu", "na", "ne",
    "da", "de", "ta", "te", "dan", "den", "tan", "ten",
}
_ANSWER_COPULA_SUFFIXES = {"dir", "dur", "tir", "tur"}
_ANSWER_CASE_SUFFIXES = {"da", "de", "ta", "te", "dan", "den", "tan", "ten"}
_SENTENCE_BOUNDARY = re.compile(r"(?:[!?;]+|\.(?=\s|$))\s*|\n+")
_REFUTED_CLAIM = re.compile(
    r"\b(?:iddia|ifade|sav|soylem|denme)[a-z]*\b.{0,80}"
    r"\b(?:yanlis[a-z]*|dogru\s+degil[a-z]*|gercek\s+disi[a-z]*)\b"
)
_LEADING_REFUTATION = re.compile(
    r"^\s*(?:(?:yanlis[a-z]*|gercek\s+disi[a-z]*|dogru\s+olmayan)"
    r"(?:\s+olan)?\s+(?:su|iddia|ifade|sav)[a-z]*"
    r"|su\s+(?:iddia|ifade|sav)[a-z]*\s+yanlis[a-z]*)\s*:"
)
_VERBAL_REFUTATION = re.compile(
    r"\b(?:denemez|soylenemez|savunulamaz|kabul\s+edilemez)\b"
)


def _word_match(expected, actual):
    """Conservative Turkish inflection match for an accepting tier.

    Short reductions are not accepted by set intersection alone. They are too
    ambiguous: ordinary words can lose a final vowel and collide with an
    unrelated root. Only measured, directional case/copula forms below may use
    them; everything else goes to review.
    """
    if expected == actual:
        return True
    common = stems(expected) & stems(actual)
    for root in common:
        if not expected.startswith(root) or not actual.startswith(root):
            continue
        if len(root) >= 4:
            return True
        expected_suffix = expected[len(root):]
        actual_suffix = actual[len(root):]
        # Direction matters. A curated key may inflect a short root that the
        # answer gives bare; the reverse is ambiguous with a different lexeme
        # ("short root" -> "root + vowel") and is not accepted.
        if actual == root and expected_suffix in _KEY_CASE_SUFFIXES:
            return True
        # A copula is longer and semantically transparent; possessive/final
        # vowel suffixes remain excluded because the adversarial set shows they
        # collide with unrelated words.
        if expected == root and actual_suffix in (
                _ANSWER_COPULA_SUFFIXES | _ANSWER_CASE_SUFFIXES):
            return True
    return False


def _semantic_units(text):
    """Ordered word/figure units, with notation-equivalent figures grouped."""
    prepared = _THOUSANDS_SPACE.sub("", normalize(text or ""))
    units = []
    for token in _UNIT.findall(prepared):
        if token[0].isdigit():
            units.extend(("figure", frozenset(forms))
                         for forms in number_forms(token) if forms)
        else:
            units.append(("word", token))
    return units


def _has_negation(text):
    for kind, token in _semantic_units(text):
        if kind != "word":
            continue
        if token in _NEGATORS or any(
                root in _NEGATORS for root in stems(token)):
            return True
    return False


def _same_polarity(text, key):
    return _has_negation(text) == _has_negation(key)


def _sentences(text):
    """Folded sentence-sized scopes; a proposition must match within one."""
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(fold(text or ""))
        if sentence.strip()
    ]


def _refutes_claim(sentence):
    """Whether a matching sentence explicitly disowns the proposition in it."""
    folded = fold(sentence or "")
    return bool(
        _REFUTED_CLAIM.search(folded)
        or _LEADING_REFUTATION.search(folded)
        or _VERBAL_REFUTATION.search(folded)
    )


def literal_match(text, key):
    """Tier 1: an unrefuted key phrase in one proposition-sized scope."""
    if not key:
        return False
    needle = fold(key)
    pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    )
    return any(
        _same_polarity(sentence, key)
        and not _refutes_claim(sentence)
        and pattern.search(sentence)
        for sentence in _sentences(text)
    )


def notation_match(text, key):
    """Does `text` state `key`, allowing for how Turkish writes the same fact?

    Every figure still has to be there exactly; only the words around it are
    compared loosely, by stem, with a short list of connectives excused.
    """
    if not key:
        return False
    if literal_match(text, key):
        return True

    expected = [
        unit for unit in _semantic_units(key)
        if not (unit[0] == "word" and unit[1] in _FUNCTION)
    ]
    if not expected:
        return False

    # A one-figure key often puts its unit after the figure while a full
    # Turkish sentence puts the unit before it. There is no competing mapping
    # to swap in that case, so order adds false reviews without adding safety.
    # With multiple figures, order is load-bearing: it keeps labels attached to
    # their values and rejects swapped mappings.
    unordered = sum(kind == "figure" for kind, _ in expected) == 1
    for sentence in _sentences(text):
        if not _same_polarity(sentence, key) or _refutes_claim(sentence):
            continue
        actual = _semantic_units(sentence)
        if unordered:
            remaining = list(actual)
            for expected_kind, expected_value in expected:
                for index, (actual_kind, actual_value) in enumerate(remaining):
                    if actual_kind != expected_kind:
                        continue
                    if expected_kind == "figure":
                        matched = bool(expected_value & actual_value)
                    else:
                        matched = _word_match(expected_value, actual_value)
                    if matched:
                        remaining.pop(index)
                        break
                else:
                    break
            else:
                return True
            continue

        position = 0
        for expected_kind, expected_value in expected:
            matched = False
            while position < len(actual):
                actual_kind, actual_value = actual[position]
                position += 1
                if actual_kind != expected_kind:
                    continue
                if expected_kind == "figure":
                    matched = bool(expected_value & actual_value)
                else:
                    matched = _word_match(expected_value, actual_value)
                if matched:
                    break
            if not matched:
                break
        else:
            return True
    return False


def accepts_without_similarity(key, answer):
    """Whether deterministic tiers 1-2 may safely accept this answer."""
    return literal_match(answer, key) or notation_match(answer, key)


# --- semantic tier ----------------------------------------------------------

# Separates "unrelated" from "possibly a paraphrase". Calibrated, not guessed:
# eval/calibrate_judge.py sweeps it against hand-adjudicated cases with
# mismatched question/answer pairs as negatives. Measured: it sends 98% of
# unrelated pairs to "wrong" while keeping every adjudicated case on the right
# side, with roughly equal room on both sides -- the sweep's top passing value
# cleared one of those cases by six thousandths and was rejected for it.
SIM_THRESHOLD = float(os.getenv("JUDGE_SIM_THRESHOLD", "0.50"))


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("embedding dimensions differ")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def similarity(text_a, text_b):
    """Cosine between two texts under the embedding model already serving the
    index. None when the service is unreachable -- scoring must not require it,
    or the test suite would depend on a running container."""
    try:
        from pipeline.index.embeddings import embed_dense
        return cosine(embed_dense(text_a), embed_dense(text_b))
    except Exception:
        logger.exception("answer similarity could not be computed")
        return None


# --- verdict ----------------------------------------------------------------

DOGRU, INCELE, YANLIS = "dogru", "incele", "yanlis"


def judge(key, answer, reference=None, sim=None):
    """(verdict, reason) for one answer.

    `sim` may be supplied by a caller that batches its own embedding calls; left
    as None it is computed here, and if that is impossible the answer goes to
    review rather than being called wrong.
    """
    if literal_match(answer, key):
        return DOGRU, "birebir"
    if notation_match(answer, key):
        return DOGRU, "notasyon"
    if not reference:
        return INCELE, "referans yok"
    if sim is None:
        sim = similarity(answer or "", reference)
    if sim is None:
        return INCELE, "gomu servisi yok"
    try:
        sim = float(sim)
    except (TypeError, ValueError):
        return INCELE, "gecersiz benzerlik"
    if not math.isfinite(sim):
        return INCELE, "gecersiz benzerlik"
    if sim >= SIM_THRESHOLD:
        return INCELE, f"benzerlik {sim:.2f}"
    return YANLIS, f"benzerlik {sim:.2f}"
