"""How Turkish writes the same fact more than one way.

Numbers, dates and word endings only -- no opinion about whether two texts mean
the same thing. That judgement is a policy and lives with whoever is making it:
the answer scorer wants to be generous so it does not call a correct answer
wrong, while the answer guard wants to be strict so an unsupported figure
cannot slip through. Opposite goals, one shared vocabulary. Keeping the
vocabulary here lets each side tune its own rules without silently changing the
other's, and keeps the product from importing its own test tooling.

Every table below is keyed on FOLDED text, so it is already lowercase and
stripped of diacritics: "uc" for three, "subat" for February.
"""
import re
import unicodedata

_MARKDOWN = re.compile(r"[*_`~#]+")


def fold(text):
    """Normalise for comparison: lowercase, drop markdown emphasis, strip
    Turkish diacritics.

    Both halves are needed and both were learned the hard way. A model bolds
    the salient words, which breaks a contiguous match on the plain phrase; and
    it drops the circumflex that a source carries, because models normalise it
    away. Neither is a wrong answer, but both scored as one. The cost is that
    comparison becomes diacritic-blind -- fine here, where the question is
    whether the fact was conveyed, not how it was spelled.
    """
    # DELETED, not replaced with a space: a model bolds the STEM of a word and
    # leaves the suffix outside the emphasis. In an agglutinative language that
    # is the normal case, so substituting a space would split the word in two.
    s = _MARKDOWN.sub("", str(text).lower())
    # Accept correct Turkish dotless-i and the legacy y-acute produced by an
    # old encoding error in saved outputs. Both represent the same letter here.
    s = s.replace("ı", "i").replace("\u00fd", "i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


# --- figures ----------------------------------------------------------------

_MAGNITUDE = {"bin": 1e3, "milyon": 1e6, "milyar": 1e9, "trilyon": 1e12}
# a separator between digit groups is a THOUSANDS separator only when exactly
# three digits follow; otherwise it is a decimal point (512.7 stays 512.7)
# a space is never a decimal point, so it can be resolved before tokenising
_THOUSANDS_SPACE = re.compile(r"(?<=\d) (?=\d{3}(?!\d))")
_THOUSANDS_DOT = re.compile(r"(?<=\d)\.(?=\d{3}(?!\d))")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_FIGURE = re.compile(r"(\d[\d.,]*\d|\d)\s*(bin|milyon|milyar|trilyon)?")


def _readings(token):
    """Every value a numeric token could denote, given that the separator
    convention is not reliable.

    Turkish writes 1.234,56 and English writes 1,234.56, and a model answering
    a Turkish question uses either -- often both in one answer. A group of
    exactly three digits after a separator is therefore genuinely ambiguous
    between a thousands group and a three-place decimal, so both readings are
    kept and the caller accepts a match on either.

    Measured: reading "3,927" as 3.92 (two decimal places, comma never a
    thousands separator) turned correct answers into failures.
    """
    turkish = _THOUSANDS_DOT.sub("", token).replace(",", ".")
    english = _THOUSANDS_COMMA.sub("", token)
    out = set()
    for s in (turkish, english):
        try:
            out.add(round(float(s), 4))
        except ValueError:
            pass
    return out


def number_forms(text):
    """Each figure in `text` as the set of values it could denote.

    A figure written with a magnitude word denotes two things at once, and
    which one an expected answer means depends on where the unit was stated.
    "48.213 milyon" is both the figure 48213 and the amount 4.8213e10; an
    expected answer often carries the figure alone, because the unit sat in the
    question. Expanding the magnitude word and keeping ONLY the expanded value
    makes those two forms unable to meet.

    This was measured, not theorised: five correct answers in one run scored
    wrong for exactly this reason, moving a reported accuracy from 0.88 to 0.59
    and sending the diagnosis off after a generation defect that did not exist.
    """
    out = []
    for token, magnitude in _FIGURE.findall(_THOUSANDS_SPACE.sub("", fold(text))):
        forms = _readings(token)
        if not forms:
            # not a single value under either convention -- a composite such as
            # a date or an article number. Its PARTS are the figures; without
            # this a dotted date contributed nothing at all.
            for part in re.split(r"[.,]", token):
                if part.isdigit():
                    out.append({float(part)})
            continue
        if magnitude:
            forms |= {round(v * _MAGNITUDE[magnitude], 4) for v in forms}
        out.append(forms)
    return out


def numbers(text):
    """Every value any figure in `text` could denote, flattened.

    A source writes the magnitude as a word and a model answers in digits. Same
    figure, and scoring that as a wrong answer is simply wrong. Financial
    Turkish leans on bin/milyon/milyar constantly, so comparing digits alone
    silently fails a large share of correct answers.
    """
    return {v for forms in number_forms(text) for v in forms}


# --- number words -----------------------------------------------------------

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
# depending on the sentence around it -- an expected answer and a correct
# answer routinely differ by nothing else.
_SUFFIXES = sorted(
    ["nden", "ndan", "larin", "lerin", "lari", "leri", "imiz", "iniz",
     "umuz", "unuz", "nin", "nun", "lar", "ler", "den", "dan", "ten", "tan",
     "dir", "dur", "tir", "tur", "yle", "yla",
     # the buffer -y- forms, for stems that end in a vowel ("zeta" -> "zetaya")
     "ya", "ye", "yi", "yu",
     "de", "da", "te", "ta", "le", "la", "in", "un", "im", "um",
     "si", "su", "ni", "nu", "ne", "na", "i", "u", "e", "a"],
    key=len, reverse=True)


def stems(word):
    """Every form `word` could reduce to once inflection is stripped.

    A SET rather than one answer, because which ending a word carries is often
    ambiguous from the spelling alone: "gune" ends in both "-e" and "-ne", and
    committing to the longer one leaves a two-letter fragment that matches
    nothing. Keeping both readings and comparing by intersection removes that
    whole class of mistake.

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
    """Fold, then rewrite number words and written dates into digits."""
    return expand_cardinals(expand_months(fold(text)))
