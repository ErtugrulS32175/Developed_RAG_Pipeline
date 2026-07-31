"""Three-state answer judging: correct, wrong, or needs a human.

The scorer this replaces forced a binary verdict, and a human read of one run's
rejections found that 13 of 15 were the scorer's fault rather than the system's.
The precision is deeply asymmetric: a deterministic match has never once been
found accepting a wrong answer, while most of its rejections are unjust. So the
fix is not another string rule -- it is to stop treating "no match" as "wrong".

Three tiers, and only the first two may ACCEPT:

  1. the deterministic match (`contains_key`) -- unchanged, still the ceiling
  2. NOTATION identities: number words, month names, inflectional suffixes.
     Same fact, different spelling.
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
import os
import re

from eval.retrieval.rag_eval import contains_key
# The notation layer -- number words, month names, suffixes -- lives in
# pipeline/lang/tr_notation, because the answer guard that ships with the product
# needs exactly the same vocabulary and must not reach into the eval package
# for it. What stays here is the POLICY: which differences are forgiven, and
# what may accept versus only reject. Re-exported so existing callers and tests
# keep working unchanged.
from pipeline.lang.tr_notation import (  # noqa: F401
    expand_cardinals, expand_months, fold, normalize, number_forms, numbers,
    stems, _SCALE)


# Words the expected answer may carry that a correct answer need not repeat.
# Deliberately tiny, and deliberately EXCLUDING the comparatives (en, cok, az,
# gec, fazla): those are exactly what distinguishes one adjudicated case's
# correct reading from its incorrect one, so dropping them would auto-accept a
# wrong answer.
_FUNCTION = {"kadar", "ile", "ve", "arasi", "arasinda", "icin", "gore",
             "olarak", "uzerinden", "yaklasik", "civarinda"}


_FUNCTION_STEMS = set().union(*(stems(w) for w in _FUNCTION))


_WORDS = re.compile(r"[a-z]+")


def notation_match(text, key):
    """Does `text` state `key`, allowing for how Turkish writes the same fact?

    Every figure still has to be there exactly; only the words around it are
    compared loosely, by stem, with a short list of connectives excused.
    """
    if not key:
        return False
    text_n, key_n = normalize(text), normalize(key)
    if contains_key(text_n, key_n):
        return True

    key_figures = number_forms(key_n)
    if key_figures:
        found = numbers(text_n)
        if not all(forms & found for forms in key_figures):
            return False

    text_stems = set().union(*(stems(w) for w in _WORDS.findall(text_n))) \
        if _WORDS.search(text_n) else set()
    for word in key_n.split():
        if any(ch.isdigit() for ch in word) or word in _SCALE:
            continue                          # a figure, checked above
        letters = "".join(_WORDS.findall(word))
        if not letters:
            continue
        roots = stems(letters)
        if roots & _FUNCTION_STEMS:
            continue
        if not roots & text_stems:
            return False
    return True


# --- semantic tier ----------------------------------------------------------

# Separates "unrelated" from "possibly a paraphrase". Calibrated, not guessed:
# eval/calibrate_judge.py sweeps it against hand-adjudicated cases with
# mismatched question/answer pairs as negatives. Measured: it sends 98% of
# unrelated pairs to "wrong" while keeping every adjudicated case on the right
# side, with roughly equal room on both sides -- the sweep's top passing value
# cleared one of those cases by six thousandths and was rejected for it.
SIM_THRESHOLD = float(os.getenv("JUDGE_SIM_THRESHOLD", "0.50"))


def cosine(a, b):
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
        return None


# --- verdict ----------------------------------------------------------------

DOGRU, INCELE, YANLIS = "dogru", "incele", "yanlis"


def judge(key, answer, reference=None, sim=None):
    """(verdict, reason) for one answer.

    `sim` may be supplied by a caller that batches its own embedding calls; left
    as None it is computed here, and if that is impossible the answer goes to
    review rather than being called wrong.
    """
    if contains_key(answer, key):
        return DOGRU, "birebir"
    if notation_match(answer, key):
        return DOGRU, "notasyon"
    if not reference:
        return INCELE, "referans yok"
    if sim is None:
        sim = similarity(answer or "", reference)
    if sim is None:
        return INCELE, "gomu servisi yok"
    if sim >= SIM_THRESHOLD:
        return INCELE, f"benzerlik {sim:.2f}"
    return YANLIS, f"benzerlik {sim:.2f}"
