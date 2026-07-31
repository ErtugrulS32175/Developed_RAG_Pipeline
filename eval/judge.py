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

from eval.rag_eval import contains_key, fold, number_forms, numbers

# --- Turkish notation -------------------------------------------------------
# Every table below is keyed on FOLDED text, so it is already lowercase and
# stripped of diacritics: "uc" for three, "subat" for February.

_ONES = {"sifir": 0, "bir": 1, "iki": 2, "uc": 3, "dort": 4,
         "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9}
_TENS = {"on": 10, "yirmi": 20, "otuz": 30, "kirk": 40, "elli": 50,
         "altmis": 60, "yetmis": 70, "seksen": 80, "doksan": 90}
_SCALE = {"yuz": 100, "bin": 1000, "milyon": 10 ** 6,
          "milyar": 10 ** 9, "trilyon": 10 ** 12}

_MONTHS = {"ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5,
           "haziran": 6, "temmuz": 7, "agustos": 8, "eylul": 9,
           "ekim": 10, "kasim": 11, "aralik": 12}

# Case, possessive, plural and copula endings, longest first. Turkish is
# agglutinative, so the same fact appears as "gun", "gune" and "gunden"
# depending on the sentence around it -- an expected answer and a correct answer
# routinely differ by nothing else.
_SUFFIXES = sorted(
    ["nden", "ndan", "larin", "lerin", "lari", "leri", "imiz", "iniz",
     "umuz", "unuz", "nin", "nun", "lar", "ler", "den", "dan", "ten", "tan",
     "dir", "dur", "tir", "tur", "yle", "yla",
     # the buffer -y- forms, for stems that end in a vowel ("zeta" -> "zetaya")
     "ya", "ye", "yi", "yu",
     "de", "da", "te", "ta", "le", "la", "in", "un", "im", "um",
     "si", "su", "ni", "nu", "ne", "na", "i", "u", "e", "a"],
    key=len, reverse=True)

# Words the expected answer may carry that a correct answer need not repeat.
# Deliberately tiny, and deliberately EXCLUDING the comparatives (en, cok, az,
# gec, fazla): those are exactly what distinguishes one adjudicated case's
# correct reading from its incorrect one, so dropping them would auto-accept a
# wrong answer.
_FUNCTION = {"kadar", "ile", "ve", "arasi", "arasinda", "icin", "gore",
             "olarak", "uzerinden", "yaklasik", "civarinda"}


def stems(word):
    """Every form `word` could reduce to once inflection is stripped.

    A SET rather than one answer, because which ending a word carries is often
    ambiguous from the spelling alone: "gune" ends in both "-e" and "-ne", and
    committing to the longer one leaves a two-letter fragment that matches
    nothing. Keeping both readings and comparing by intersection removes that
    whole class of mistake, at the cost of a little looseness that the figure
    check upstream already contains.

    Bounded three ways -- at most three rounds, never below two characters, and
    only from words of three or more -- because Turkish stacks suffixes but a
    short word is usually already a stem.
    """
    found = {word}
    frontier = [word]
    for _ in range(3):
        nxt = []
        for current in frontier:
            if len(current) < 3:
                continue
            for suffix in _SUFFIXES:
                if current.endswith(suffix) and len(current) - len(suffix) >= 2:
                    shorter = current[:-len(suffix)]
                    if shorter not in found:
                        found.add(shorter)
                        nxt.append(shorter)
        frontier = nxt
    return found


_FUNCTION_STEMS = set().union(*(stems(w) for w in _FUNCTION))


def _classify(token):
    """(kind, value) if this token is a number word, else None.

    A scale word counts only in its bare form: "bin" is a thousand, but "binde"
    is a rate ("per thousand") and treating its stem as a scale would turn an
    expected rate into an expected quantity.
    """
    if token in _ONES:
        return "unit", _ONES[token]
    if token in _TENS:
        return "unit", _TENS[token]
    if token in _SCALE:
        return "scale", _SCALE[token]
    for root in stems(token):
        if root == "bir":
            # "biri", "birinci", "birim" -- an inflected "bir" is a pronoun or
            # another word entirely, never the quantity one
            return None
        if root in _ONES:
            return "unit", _ONES[root]
        if root in _TENS:
            return "unit", _TENS[root]
    return None


def _run_value(items):
    """Value of a run of number words: 'iki yuz elli bin' -> 250000, or None if
    the run is not a number phrase at all.

    Turkish builds a number from LARGER to smaller -- "yirmi bes", never "bes
    yirmi" -- so two units that do not descend are two separate words that
    happen to be number-like, and adding them invents a figure. Found by the
    mismatched-pair control: a fraction was being read as the sum of its parts.
    """
    total = current = 0
    previous = None
    for kind, value in items:
        if kind == "unit":
            if previous is not None and value >= previous:
                return None
            previous = value
            current += value
        elif value == 100:                    # multiplies what came before it
            current = (current or 1) * 100
            previous = None
        else:                                 # closes a group off
            total += (current or 1) * value
            current, previous = 0, None
    return total + current


def expand_cardinals(text):
    """Rewrite runs of Turkish number words as digits.

    A source writes "kirk yedi bin" where an answer writes "47 000", and the
    figure machinery only sees the second. Two runs are left alone on purpose:
    a run of scale words only -- "47 bin" is a digit with a magnitude word and
    is already handled as one, and expanding the "bin" would invent a separate
    1000 that no correct answer contains -- and a lone "bir", which is the
    indefinite article far more often than it is the number one.
    """
    out, run = [], []

    def flush():
        if not run:
            return
        tokens = [t for t, _ in run]
        items = [c for _, c in run]
        value = _run_value(items) if any(k == "unit" for k, _ in items) else None
        if value is None or tokens == ["bir"]:
            out.extend(tokens)
        else:
            # str, not a format spec: 'g' turns a million into 1e+06, which no
            # figure parser downstream reads back as a million
            out.append(str(value))
        run.clear()

    for token in text.split():
        found = _classify(token)
        if found:
            run.append((token, found))
            continue
        flush()
        out.append(token)
    flush()
    return " ".join(out)


_DATE_WORDS = re.compile(
    r"(?<!\d)(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})(?!\d)")


def expand_months(text):
    """'15 mart 1977' -> '15.3.1977', so a written date can meet a dotted one."""
    return _DATE_WORDS.sub(
        lambda m: f"{int(m.group(1))}.{_MONTHS[m.group(2)]}.{m.group(3)}", text)


def normalize(text):
    return expand_cardinals(expand_months(fold(text)))


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
        from pipeline.embeddings import embed_dense
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
