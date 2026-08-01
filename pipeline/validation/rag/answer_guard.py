"""Deterministic checks on an answer against the passages it was given.

The question these exist to answer is not "is this answer good" -- no cheap
check can tell -- but "did this answer stay inside its evidence". Two things
can be verified with no model and no ground truth:

  * every FIGURE the answer states appears somewhere in the passages
  * every PAGE the answer cites is a page that was actually supplied

Both are one-sided. Passing means nothing was invented outright; it does NOT
mean the figure was taken from the right row, which is the failure this whole
line of work is aimed at and which needs the answer to say WHICH passage it
used. That comes next. What these buy now is a measurement: run them over
answers already known to be correct and the false-flag rate falls out, and that
rate decides whether a check can ever be promoted from a warning to a block.

Everything here is a FLAG, never a gate. A check whose false-positive rate is
unknown must not be allowed to refuse an answer.
"""
import json
import re

from pipeline.lang.tr_notation import _readings, fold, normalize, number_forms, numbers
from pipeline.retrieval.context import RagContext

# the citation shape the answer prompt asks the model to produce
_ANSWER_PAGE = re.compile(r"sayfa\s*\d+", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"sayfa\s*(\d+)", re.IGNORECASE)


def context_pages(context):
    """Pages actually supplied, taken from trusted retrieval metadata."""
    if not isinstance(context, RagContext):
        raise TypeError("context_pages requires a RagContext")
    return {passage.page for passage in context.passages
            if passage.page is not None}


def cited_pages(answer):
    return {int(n) for n in _PAGE_NUMBER.findall(answer or "")}


# Turkish states a small rate as a fraction phrase -- "binde yedi" is seven in a
# thousand -- and a model helpfully restates it as a percentage. That percentage
# is nowhere in the source, so without this every such answer is flagged.
_RATE_DIVISOR = {"yuzde": 100, "binde": 1000, "on binde": 10_000,
                 "yuz binde": 100_000, "milyonda": 1_000_000}
_RATE = re.compile(r"\b(yuz binde|on binde|milyonda|binde|yuzde)\s+(\d[\d.,]*)")


def derived_figures(context):
    """Values a passage implies without stating: a rate phrase as a fraction and
    as a percentage.

    Only these two, and only from an explicit phrase. Deriving more -- sums,
    differences, ratios between arbitrary cells -- would accept almost any
    figure and empty the check of meaning. An answer that adds up several
    numbers therefore still gets flagged, which is right: nothing verified that
    arithmetic either.

    A caution measured on real answers, and the reason `derive` is optional: a
    model restated a rate an order of magnitude wrong, and the wrong value was
    covered anyway because an UNRELATED passage elsewhere in the same context
    implied it. Deriving over the whole context makes the supported set large
    enough to absorb real errors. It stops being a problem once an answer says
    which passage it used, and the check can be scoped to that passage.
    """
    out = set()
    for word, token in _RATE.findall(context or ""):
        for value in _readings(token):
            out.add(round(value / _RATE_DIVISOR[word], 10))
            out.add(round(value * 100 / _RATE_DIVISOR[word], 10))
    return out


def unsupported_figures(answer, context, minimum=0, derive=True):
    """Figures stated by the answer that no passage contains.

    Page citations are stripped first: "Sayfa 72" is the model pointing at its
    source, not a claim about the world, and counting it as one would flag
    every correctly cited answer.

    Both sides are normalised the same way, so a figure written as a word on
    one side and as digits on the other is not a mismatch. `minimum` drops
    small numbers, which are mostly clause and item markers rather than data.

    `derive` decides whether a rate restated as a percentage counts as
    supported, and the choice is genuinely open -- see the note on
    derived_figures. Measured both ways it trades detection for quiet.
    """
    body = _ANSWER_PAGE.sub(" ", answer or "")
    context_n = normalize(context or "")
    supported = numbers(context_n)
    if derive:
        supported = supported | derived_figures(context_n)
    missing = []
    for forms in number_forms(normalize(body)):
        if not forms or forms & supported:
            continue
        if minimum and max(forms) < minimum:
            continue
        missing.append(sorted(forms))
    return missing


def unsupported_pages(answer, context):
    """Pages the answer cites that were never in front of it."""
    return sorted(cited_pages(answer) - context_pages(context))


# --- structured answers -----------------------------------------------------

_BLOCK = "\n\n---\n\n"


def _json_containers(text):
    """Yield complete top-level JSON containers found in model prose.

    A whole container is consumed before looking for another one. This is
    important for strictness: an otherwise valid answer object nested inside
    an array or wrapper object must not be mistaken for the requested top-level
    response.
    """
    position = 0
    while position < len(text):
        openings = [
            index for index in (text.find("{", position),
                                text.find("[", position))
            if index != -1
        ]
        if not openings:
            return
        start = min(openings)
        stack = []
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not stack or stack[-1] != expected:
                    # Consume the malformed candidate rather than inspecting a
                    # nested object as if it were a top-level model response.
                    position = index + 1
                    break
                stack.pop()
                if not stack:
                    yield text[start:index + 1]
                    position = index + 1
                    break
        else:
            # An unclosed leading container makes the boundary ambiguous.
            # Failing closed is safer than recovering an object from inside it.
            return


def _object_without_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON field")
        obj[key] = value
    return obj


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _valid_structured(obj):
    """Whether *obj* exactly matches the model-answer contract."""
    if not isinstance(obj, dict) or set(obj) != {"dayanak", "cevap"}:
        return False
    if not isinstance(obj["dayanak"], list):
        return False
    if not isinstance(obj["cevap"], str) or not obj["cevap"].strip():
        return False

    for evidence in obj["dayanak"]:
        if not isinstance(evidence, dict) or \
                set(evidence) != {"pasaj", "alinti"}:
            return False
        handle = evidence["pasaj"]
        if type(handle) is not int or handle < 1:
            return False
        quote = evidence["alinti"]
        if not isinstance(quote, str) or not quote.strip():
            return False
    return True


def parse_structured(text):
    """The answer object a model was asked for, or None if it did not produce one.

    Models wrap JSON in prose or a code fence however firmly they are told not
    to, so complete JSON containers are located before parsing. Discovery is
    tolerant; acceptance is not. Only the exact requested object is returned,
    with duplicate fields, wrapper objects and array wrappers rejected.
    JSON object-member order is not part of the schema; evidence-first remains
    a generation instruction, not an uncalibrated reason to reject an answer.
    Returning None rather than raising matters: a malformed reply is a flag on
    that answer, never a failed request.
    """
    if not isinstance(text, str) or not text:
        return None
    for candidate in _json_containers(text):
        try:
            obj = json.loads(
                candidate,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError):
            continue
        if _valid_structured(obj):
            return obj
    return None


def _quoted(parsed):
    return " \n ".join(item["alinti"] for item in parsed["dayanak"])


def check_structured(reply, context, minimum=0, derive=True):
    """Flags for an answer that was asked to cite its evidence.

    Everything here is scoped to what the answer SAID it used, which is the
    whole reason for asking. The unscoped version of these checks was measured
    against fifteen passages at once and the supported set came out wide enough
    to cover a figure the answer had got wrong by a factor of ten.

    ``context`` must be the exact ``RagContext`` used for generation. Parsing
    the model-visible string here would let document text mint a fake handle or
    page, so a string is a programmer error rather than a tolerated legacy
    input.
    """
    if not isinstance(context, RagContext):
        raise TypeError("check_structured requires a RagContext")
    if not context.numbered:
        raise ValueError("structured answers require a numbered RagContext")

    if isinstance(reply, dict):
        parsed = reply if _valid_structured(reply) else None
    else:
        parsed = parse_structured(reply) if isinstance(reply, str) else None
    if parsed is None:
        return [("bicimsiz_yanit", [])]

    answer = parsed["cevap"]
    known = context.by_handle()
    flags = []

    cited, bad_handles, bad_quotes = [], [], []
    for item in parsed["dayanak"]:
        handle = item["pasaj"]
        if handle not in known:
            bad_handles.append(handle)
            continue
        cited.append(handle)
        quote = fold(item["alinti"])
        if quote and quote not in fold(known[handle].text):
            bad_quotes.append(handle)

    if bad_handles:
        flags.append(("uydurma_pasaj", bad_handles))
    if bad_quotes:
        flags.append(("uydurma_alinti", sorted(set(bad_quotes))))

    scope = _BLOCK.join(known[h].text for h in cited) if cited else ""
    figures = unsupported_figures(answer, scope, minimum, derive)
    if figures:
        flags.append(("kaynaksiz_sayi", figures))
    else:
        # softer, and the one aimed at the wrong-row failure: the figure is in
        # the passage but not on any line the answer quoted, so nothing shows
        # it came from the record that was asked about
        loose = unsupported_figures(answer, _quoted(parsed), minimum, derive)
        if loose:
            flags.append(("alintisiz_sayi", loose))

    pages = {known[h].page for h in cited} - {None}
    stray = sorted(cited_pages(answer) - pages) if pages else []
    if stray:
        flags.append(("kaynaksiz_sayfa", stray))
    return flags


def check(answer, context, minimum=0, derive=True):
    """Every flag raised against one answer; empty means nothing was invented.

    Returned as data rather than text so a caller decides what to show. The
    figures themselves are document content, so a report that aggregates over
    many answers should count these, not print them.
    """
    if not isinstance(context, RagContext):
        raise TypeError("check requires a RagContext")

    flags = []
    passage_text = _BLOCK.join(passage.text for passage in context.passages)
    figures = unsupported_figures(answer, passage_text, minimum, derive)
    if figures:
        flags.append(("kaynaksiz_sayi", figures))
    pages = unsupported_pages(answer, context)
    if pages:
        flags.append(("kaynaksiz_sayfa", pages))
    return flags
