"""Does the answer's figure come from the record the question asked about?

The existing checks verify PRESENCE: every figure the answer states occurs
somewhere in the cited passages. The dominant real failure passes that test
untouched -- the model reads the right passage and binds a NEIGHBOURING
record's genuine value to the requested entity. Measured on the mutation
harness: zero of twenty-two wrong-binding mutants raise any existing flag.

This check verifies BINDING. A passage is split into record-scale segments,
each segment is scored by how many of the question's distinctive terms it
contains, and the answer's figure is expected to come from the segment that
matches the question BEST. The comparison is relative on purpose: sibling
rows share their label ("X, Alan = 1. Y, Alan = 2."), so an absolute
"segment mentions the question" rule would accept the sibling -- what
separates the right row from its neighbour is the entity term, and only a
comparison against the best-matching segment sees that.

One-sided like every other check here: a raised flag means the figure sits in
a worse-matching segment while a better-matching one offers a different
figure. No flag means no binding evidence AGAINST the answer -- not proof of
correctness. When the question's terms appear nowhere in the scope, the check
stays silent rather than guessing.

STATUS: an EXPERIMENTAL ANNOTATION SIGNAL, not a validator. Four audit
rounds established the boundary of what this mechanism family can promise.
The narrow guarantee that holds: header inheritance cannot CREATE a flag --
inherited credit is used solely to clear a figure, never to rank a segment.
The broad guarantee that does NOT hold: the other flat-text heuristics (own
term ranking, sentence segmentation, alias expansion, label frames, and the
cut of the lenient chain at any figure-bearing line) can still annotate a
CORRECT answer. Measured: 1 of 183 settled-correct answers on the saved
runs. Constructed: a header that carries any figure -- a year, a table
number -- cuts the inheritance chain, and the value line under it loses to
a fuller sibling row (pinned as a known limit in the tests). Silence-side
limits: an inserted header, a card whose sibling shares question words, a
dotted multi-letter abbreviation, a sentence ending in a one-letter token.

Flat text cannot carry a row relation losslessly: the segmenter must GUESS
record boundaries and the scorer must GUESS which words are entities, and
each guess has a wrong side. The real validator is a row-binding check over
the EXTRACTION's row/cell structure, where "same row" is a fact instead of
an inference.

NOT wired into the publication policy, and the import graph is pinned by a
test: this signal's output reaches human annotation and measurement
ledgers, never a publish/withhold decision.
"""
import re

from pipeline.lang.tr_notation import fold, normalize, numbers, stems
from pipeline.retrieval.context import RagContext

WRONG_BINDING = "yanlis_baglama"

# A record boundary in flattened table text and prose alike: a sentence end
# followed by whitespace, or a line break. Turkish thousands dots ("3.579
# milyar") survive because they are not followed by whitespace.
#
# The boundary deliberately does NOT require an uppercase start. It used to,
# and that made the whole check inert on a passage normalised to lower case:
# nothing split, one segment, no comparison possible, silence reported as a
# clean result. A check that goes quiet on a formatting change is worse than
# one that is merely imperfect, because nothing announces it.
#
# A dot closing a SINGLE LETTER is an abbreviation ("A.S.", "T.A.S."), not a
# sentence end. Splitting there cut a company name in half, and the alias
# gate -- which rightly demands the name as a phrase inside ONE segment --
# could then never see it: no expansion, no signal, and a wrong sibling
# value sailed through in silence. Digits stay out of the lookbehind so
# "= 47 000. Beta" still splits.
_SEGMENT_BOUNDARY = re.compile(
    r"(?<!\b[A-Za-zÇĞİÖŞÜçğıöşü])[.;!?]\s+|\n+")
_ANSWER_PAGE = re.compile(r"sayfa\s*\d+", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_FIGURE_TOKEN = re.compile(
    r"(?<![a-zA-Z0-9])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d[\d.,]*\d|\d)"
    r"(?:\s*(?:bin|milyon|milyar|trilyon))?"
)

# Words that carry no binding signal: question machinery, units, magnitudes
# and the attribute fillers every question in this corpus shares. Ordinary
# Turkish function words only -- nothing document-derived.
_STOPWORDS = frozenset("""
    nedir kactir kacti kac hangi kadar oldu olan olarak gore icin ile veya
    yuzde oran orani oraninda milyon milyar bin trilyon adet ton kwh mwp
    yil yili yilinda yilindaki sonu sonunda itibariyla toplam tam tamamen
    ne zaman nasil neden nerede neresidir kimdir mi mu tane sayi sayisi
    """.split())


def _question_terms(question: str) -> frozenset:
    terms = set()
    for word in _WORD.findall(fold(question)):
        if len(word) < 3 or word.isdigit() or word in _STOPWORDS:
            continue
        terms.add(word)
    return frozenset(terms)


def _segments(text: str):
    for piece in _SEGMENT_BOUNDARY.split(text):
        piece = piece.strip()
        if piece:
            yield piece


def _stem_union(text: str) -> set:
    union = set()
    for word in _WORD.findall(fold(text)):
        if len(word) >= 3:
            union |= stems(word)
    return union


def _matched_terms(terms: frozenset, stem_union: set) -> frozenset:
    return frozenset(term for term in terms if stems(term) & stem_union)


def _answer_figures(answer: str):
    """Figure form-sets the answer states, page citations stripped."""
    body = _ANSWER_PAGE.sub(" ", answer)
    for match in _FIGURE_TOKEN.finditer(body):
        token = match.group()
        forms = frozenset(numbers(normalize(token)))
        if forms:
            yield token.strip(), forms


# "Long Official Name A.S. (Alias)" -- the passage defines its own aliases.
# The alias group is letters only: a footnote reference "(2)" or "(1)" must
# never be read as a name, and a mixed token is not an alias either.
_ALIAS_DEFINITION = re.compile(
    r"((?:[A-ZÇĞİÖŞÜ][\w&.çğıöşü-]*[\s.]+){1,5}[A-ZÇĞİÖŞÜ][\w&.çğıöşü-]*)"
    r"\s*\(\s*([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]{2,})\s*\)"
)


def _content_words(text: str) -> list:
    """Folded words that carry identity: function words and fragments out."""
    return [word for word in _WORD.findall(fold(text))
            if len(word) >= 3 and word not in _STOPWORDS]


def _phrase_present(words: list, passages) -> bool:
    """Does the long name occur as a RUN, not merely word by word?

    Requiring only one shared word was far too weak: an alias defined
    somewhere else in the context ("Gama Metal Isleri A.S. (Gamis)") opened
    the gate as soon as any single word of it -- "metal" -- turned up
    anywhere in the scored scope, and the injected terms then re-ranked the
    segments and flagged a CORRECT answer. A name is a sequence, so the
    scope must carry the sequence. Both sides are reduced to content words
    first, so "A.S." or a linking "ve" between the parts does not break it.

    The run is sought inside ONE SEGMENT, not across a whole passage. A
    passage flattened into a single word sequence puts the end of one record
    next to the start of the next, and the halves of a long name landing on
    either side of that join read as a phrase that nothing in the document
    actually says -- which flagged a correct answer.
    """
    target = [stems(word) for word in words]
    for passage in passages:
        for segment in _segments(passage.text):
            sequence = _content_words(segment)
            for start in range(len(sequence) - len(target) + 1):
                if all(target[i] & stems(sequence[start + i])
                       for i in range(len(target))):
                    return True
    return False


def _expand_aliases(terms: frozenset, definition_passages, scope_passages) -> frozenset:
    """Add the long-name words of an alias the question uses.

    The dominant alias miss looked like this: the question says the short
    name, the table rows carry only the long official name, so the
    discriminating term never matches and every row ties. The documents
    define the mapping in parentheses themselves, so the expansion is
    derived at check time -- no hardcoded knowledge, and a question using
    no alias expands to nothing.

    Definitions are read from the WHOLE context (the parenthetical usually
    sits in prose while the row sits in a table), but a definition is only
    APPLIED when its long name actually occurs in the SCOPE being scored --
    as a phrase, not as scattered words. Without that, an unrelated
    definition elsewhere in the context injects terms that re-rank the
    scored segments and flag a correct answer."""
    expanded = set(terms)
    for passage in definition_passages:
        for match in _ALIAS_DEFINITION.finditer(passage.text):
            alias = fold(match.group(2))
            if not any(stems(alias) & stems(term) for term in terms):
                continue
            words = _content_words(match.group(1))
            if not words:
                continue
            # the long name must be present where we are scoring, or the
            # definition is about something this scope never mentions
            if not _phrase_present(words, scope_passages):
                continue
            expanded.update(words)
    return frozenset(expanded)


def _label_frame(text: str, index: int) -> str:
    """The folded words immediately before a value -- its label, roughly.

    Duplicated from the mutation harness rather than imported: pipeline code
    must not depend on eval tooling, and the two copies are allowed to drift
    if measurement and product ever need different framings.

    Digits stay in the frame. Dropping them merged labels that a document
    keeps apart -- "Endeks 2023" and "Endeks 2024", "Tablo 1 Toplam" and
    "Tablo 2 Toplam" all collapsed to one label, so a value from a different
    year or a different table counted as a sibling of the one asked about.
    """
    window = fold(text[max(0, index - 64):index])
    words = re.sub(r"[^a-z0-9]+", " ", window).split()
    return " ".join(words[-3:])


def _frames_share_label(one: str, other: str) -> bool:
    """Two frames carry the same label when their trailing words agree.

    Frame equality is too strict: the 3-word window drags the PRECEDING
    entity word in ("gamis zeta endeksi" vs "sirketi zeta endeksi"), so
    sibling rows of different entities would never match. The label proper
    is the trailing words.

    Two guards keep a short frame from faking a label. The comparison uses
    as many trailing words as the SHORTER frame has, so a genuine one-word
    label ("Endeks 47 000" / "Endeks 88 000") is in scope -- an earlier
    two-word floor silently excluded that whole shape. And the shared
    suffix must carry a distinctive word: matching on function words alone
    would make every figure in a passage a sibling of every other."""
    a, b = one.split(), other.split()
    if not a or not b:
        return False
    width = min(len(a), len(b), 2)
    suffix = a[-width:]
    if suffix != b[-width:]:
        return False
    return any(word not in _STOPWORDS for word in suffix)


def _segment_frames(segment: str) -> dict:
    """label frame -> union of figure forms appearing under it (2+ digits)."""
    frames = {}
    for match in _FIGURE_TOKEN.finditer(segment):
        token = match.group()
        if len(re.sub(r"\D", "", token)) < 2:
            continue
        forms = frozenset(numbers(normalize(token)))
        if not forms:
            continue
        frame = _label_frame(segment, match.start())
        frames[frame] = frames.get(frame, frozenset()) | forms
    return frames


def check_binding(question: str, answer: str, context: RagContext,
                  cited_handles=None) -> list:
    """Diagnostics for figures bound to the wrong record.

    ``cited_handles`` scopes the check to claimed evidence when the caller has
    it (structured answers); otherwise the whole context is the scope, which
    is the only honest option for plain answers.
    """
    if not isinstance(context, RagContext):
        raise TypeError("check_binding requires a RagContext")
    terms = _question_terms(question or "")
    if not terms:
        return []

    if cited_handles is None:
        passages = context.passages
    else:
        wanted = set(cited_handles)
        passages = tuple(p for p in context.passages if p.handle in wanted)
    terms = _expand_aliases(terms, context.passages, passages)

    # THE DESIGN RULE, learned across three audit rounds: header inheritance
    # may only SUPPRESS a flag, never help create one. Every earlier version
    # let an adjacent header's terms raise a segment in the RANKING, gated by
    # some lexical test of what the segment "looked like" -- no question
    # term, then no unknown capitalised word, then no unknown content word --
    # and every gate's boundary became the next inversion: the check flagged
    # the CORRECT answer and cleared the wrong one on layouts as ordinary as
    # a value line carrying a unit word. Ranking now uses only what a
    # segment says ITSELF, which no inserted header can change; the header's
    # terms count solely toward clearing a figure.
    #
    # This closes the INHERITANCE-side inversions only. The own-term ranking
    # can still put a fuller sibling above the true value line -- the
    # figure-bearing-header limit in the module docstring is exactly that
    # shape -- so the module remains an annotation signal, not a validator.
    scored = []  # (own_count, lenient_count, figure_forms_in_segment, frames)
    for passage in passages:
        previous_lenient, previous_had_figures = frozenset(), True
        for segment in _segments(passage.text):
            own = _matched_terms(terms, _stem_union(segment))
            forms = frozenset(numbers(normalize(segment)))
            # Card and infographic layouts put the entity on one line and the
            # value on the next ("Omega Kurumu" / "Zeta Yili 1907"); the
            # union below is that value line's benefit of the doubt.
            lenient = own
            if not previous_had_figures:
                lenient = own | previous_lenient
            scored.append((len(own), len(lenient), forms, _segment_frames(segment)))
            previous_lenient, previous_had_figures = lenient, bool(forms)
    if not scored:
        return []

    best = max(own for own, _, _, _ in scored)
    if best == 0:
        return []  # the question's terms appear nowhere: no signal, no flag
    best_entries = [entry for entry in scored if entry[0] == best]
    best_forms = frozenset().union(*(entry[2] for entry in best_entries))

    # No exemption for figures the QUESTION also states. An earlier version
    # skipped them as mere echoes of the question's qualifier ("2024'te ...
    # kac artti?"), which handed an attacker the whole check: plant the wrong
    # value in the question, have the answer affirm it, and the figure was
    # never examined. Measured on both saved runs, the exemption bought
    # nothing -- a qualifier that genuinely appears under every record's label
    # is already in the best-matching segment, so the sibling precondition
    # clears it without any special case.
    misbound = []
    for token, forms in _answer_figures(answer):
        if any(forms & entry[2] for entry in scored) is False:
            continue  # absent from scope entirely: kaynaksiz_sayi's job
        # own decides whether the figure's segment fell short of the best;
        # LENIENT decides whether it might merely be a value line under its
        # header -- and lenient credit reaching the bar always clears, never
        # accuses.
        figure_best = max(
            (entry[0] for entry in scored if forms & entry[2]), default=0)
        figure_lenient = max(
            (entry[1] for entry in scored if forms & entry[2]), default=0)
        # the figure's label frames, wherever it occurs in scope
        figure_frames = set()
        for entry in scored:
            for frame, frame_forms in entry[3].items():
                if forms & frame_forms:
                    figure_frames.add(frame)
        # SIBLING PRECONDITION: a best-matching segment must offer a
        # DIFFERENT figure under one of the SAME label frames. That is the
        # defining shape of a wrong-row binding. Without it, prose and
        # infographic layouts -- where the entity and its value never share
        # a segment -- flooded the check with flags on CORRECT answers: an
        # unrelated prose segment can always out-score the true value line,
        # but it cannot fake carrying the same label over a different value.
        sibling = any(
            (frame_forms - forms)
            and any(_frames_share_label(frame, ff) for ff in figure_frames)
            for entry in best_entries
            for frame, frame_forms in entry[3].items()
        )
        if (figure_best < best and figure_lenient < best
                and not (forms & best_forms) and sibling):
            misbound.append(token)

    return [(WRONG_BINDING, tuple(sorted(misbound)))] if misbound else []
