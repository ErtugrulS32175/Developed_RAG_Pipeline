"""Step 2B: the offline mutation harness, DEVELOPMENT clusters only.

Builds one synthetic wrong-answer CANDIDATE per eligible development case and
class, plus an untouched paired control, and replays both through the same
frozen guard policies as ``guard_ledger``. Every mutation is deterministic
(seeded by the case's stable id) and takes provenance from ``RagContext`` only.

The classes are not epistemically equal, and the report says so. Classes with
a deterministic eligibility basis assign their expected label from executable
invariants before any human looks at them. The three RELATION classes cannot:
no executable check proves "a different real record" or "two genuine
label/value relations", so their mutants stay candidates pending the
contract's human construct-validity audit and never enter a structural-recall
denominator.

Locked clusters are refused at the door: this module is the mutator developer's
tool, and per the measurement contract the developer never touches locked
content. The final gate waits for a fresh holdout; everything this prints is a
DEVELOPMENT number.

The persisted report carries counts and fingerprints only.
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval.answer.adversarial_feasibility import (
    ARITHMETIC_RESTATEMENT,
    CLASS_ORDER,
    CORRUPTED_QUOTE,
    FORCED_ANSWER_UNANSWERABLE,
    INVENTED_PROVENANCE,
    LABEL_VALUE_SWAP,
    QUESTION_ANSWER_MISMATCH,
    UNSUPPORTED_FIGURE,
    WRONG_ROW,
    BaseCase,
    DETECTORS,
    is_locked,
    _can_force_unanswerable,
    _formatted_forms,
    _numeric_relation_peer,
    _shared_passages,
    _shares_key_passage,
    _validated_evidence,
    checked_output_path,
    load_cases,
    repository_head,
    write_report,
)
from eval.answer.guard_ledger import (
    PLAIN,
    POLICIES,
    STRUCTURED_DERIVED,
    STRUCTURED_EXPLICIT,
    replay_flags,
)
from eval.answer.judge import accepts_without_similarity
from eval.retrieval.rag_eval import contains_key
from pipeline.lang.tr_notation import fold, normalize, numbers
from pipeline.retrieval.context import RagContext

PROTOCOL_VERSION = "adversarial_dev_mutation_v1"

# The feasibility module's figure pattern runs on NORMALIZED text, where
# thousands groups are already collapsed. This harness edits the RAW answer and
# scans RAW passages -- there "47 000"-style spaced thousands are one figure
# and must be captured as one span, or the mutator would replace half a number.
_RAW_FIGURE = re.compile(
    r"(?<![a-zA-Z0-9])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d[\d.,/]*\d|\d)"
    r"(?:\s*(?:bin|milyon|milyar|trilyon))?"
)

# Preregistered per-class expectation, fixed in the measurement contract before
# any mutation output existed. "caught" names the diagnostic that must fire;
# None means the structural expectation is that the current guard CANNOT see
# the mutation (the wrong-binding blind spot) -- a pass there confirms the
# prediction rather than refuting the harness.
EXPECTED_DIAGNOSTIC = {
    # Per the frozen contract, only these three classes carry a universally
    # correct diagnostic code. The other semantic classes report the safety
    # catch and whichever codes caused it, WITHOUT crediting any particular
    # code as semantic understanding -- an earlier draft required
    # kaynaksiz_sayi for arithmetic/forced too, which redefined diagnostic
    # recall outside the contract.
    UNSUPPORTED_FIGURE: "kaynaksiz_sayi",
    CORRUPTED_QUOTE: "uydurma_alinti",
    INVENTED_PROVENANCE: "uydurma_pasaj",
    ARITHMETIC_RESTATEMENT: None,
    FORCED_ANSWER_UNANSWERABLE: None,
    WRONG_ROW: None,
    LABEL_VALUE_SWAP: None,
    QUESTION_ANSWER_MISMATCH: None,
}

# Provenance violations live in the evidence fields; the plain policy never
# reads those, so replaying them under it would measure nothing.
EVIDENCE_ONLY_CLASSES = {CORRUPTED_QUOTE, INVENTED_PROVENANCE}

# The relation classes inherit the feasibility inventory's epistemics: their
# detectors are upper bounds, and no executable invariant can PROVE "a
# different real record" or "two genuine label/value relations" -- that is
# precisely why the contract routes them through a human construct-validity
# audit. The label-frame check screens out degenerate constructions (two bare
# numbers side by side), but what survives it is a CANDIDATE wrong answer,
# never a confirmed one, and must not enter a structural-recall denominator
# until a human confirms it.
CANDIDATE_ROLE = "candidate_pending_human_confirmation"
CONFIRMED_ROLE = "confirmed_structural"


def population_role(class_name: str) -> str:
    if DETECTORS[class_name]["basis"] == "manual_upper_bound":
        return CANDIDATE_ROLE
    return CONFIRMED_ROLE


@dataclass(frozen=True)
class Mutant:
    case_id: str
    class_name: str
    answer: str
    evidence: tuple
    context: RagContext
    invariants: tuple


def _require_development(case: BaseCase) -> None:
    """Verify the split from the IDENTITY, never from a stored flag.

    A forged record could carry locked=False beside a locked id, and a
    malformed id would previously sail through -- ``is_locked`` recomputes the
    split and rejects anything that is not a 64-hex digest. The stored flag is
    then cross-checked so an inconsistent record is refused rather than
    silently reinterpreted."""
    locked = is_locked(case.stable_id)
    if locked or case.locked:
        raise ValueError("locked case reached the mutator")


def _label_frame(text: str, index: int) -> str:
    """The folded words immediately before a value -- its label, roughly.

    This is the executable stand-in for a row/label relation: two values that
    share a frame sit under the SAME label in sibling records, two values with
    different non-trivial frames carry DIFFERENT labels in one scope. It is a
    structural check, not semantics -- the human construct-validity audit in
    the contract remains the arbiter -- but it refuses the degenerate case a
    bare "any other number nearby" rule accepted."""
    window = fold(text[max(0, index - 64):index])
    words = re.sub(r"[^a-z]+", " ", window).split()
    return " ".join(words[-3:])


def _framed_tokens(text: str):
    """Every raw figure token with its span and label frame."""
    for match in _RAW_FIGURE.finditer(text):
        token = match.group()
        if token.count(".") + token.count(",") + token.count("/") > 1:
            continue
        forms = frozenset(numbers(token))
        if forms:
            yield token, forms, _label_frame(text, match.start())


def _quote_around(text: str, token: str) -> str:
    """A verbatim slice of the passage that contains the token."""
    index = text.find(token)
    if index < 0:
        raise ValueError("token not found in passage text")
    return text[max(0, index - 60):index + len(token) + 60]


def _supported_span(case: BaseCase, validated):
    """Locate the answer's key figure: present in the key AND in a quote."""
    key_values = numbers(normalize(case.key))
    quote_values = numbers(normalize(" ".join(
        claim.quote for claim, _ in validated
    )))
    for match in _RAW_FIGURE.finditer(case.answer):
        token = match.group()
        if token.count(".") + token.count(",") + token.count("/") > 1:
            continue
        forms = frozenset(numbers(token))
        if forms and forms & key_values and forms & quote_values:
            return match.start(), match.end(), forms
    return None


def _absent_figure(case: BaseCase, avoid) -> str | None:
    """A deterministic invented figure provably absent from the context."""
    context_values = numbers(normalize(case.context.model_text))
    seed = int(case.stable_id[:12], 16)
    for step in range(64):
        token = str(100003 + (seed + step * 7919) % 8_999_938)
        forms = frozenset(numbers(token))
        if forms and forms.isdisjoint(context_values) and forms.isdisjoint(avoid):
            return token
    return None


def _replace(answer: str, start: int, end: int, token: str) -> str:
    return answer[:start] + token + answer[end:]


def _claims_as_rows(claims) -> tuple:
    return tuple({"pasaj": handle, "alinti": quote} for handle, quote in claims)


def mutate_unsupported_figure(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    validated = _validated_evidence(case)
    if not validated:
        return None
    span = _supported_span(case, validated)
    if span is None:
        return None
    start, end, forms = span
    token = _absent_figure(case, forms)
    if token is None:
        return None
    return Mutant(
        case.stable_id, UNSUPPORTED_FIGURE,
        _replace(case.answer, start, end, token),
        _claims_as_rows((c.handle, c.quote) for c, _ in validated),
        case.context,
        ("figure_absent_from_context", "original_figure_was_supported"),
    )


def mutate_arithmetic_restatement(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    validated = _validated_evidence(case)
    if not validated:
        return None
    span = _supported_span(case, validated)
    if span is None:
        return None
    start, end, forms = span
    context_values = numbers(normalize(case.context.model_text))
    for value in sorted(forms):
        if not value:
            continue
        for factor in (10, 0.1):
            transformed = value * factor
            transformed_forms = _formatted_forms(transformed)
            if (
                transformed_forms
                and transformed_forms.isdisjoint(forms)
                and transformed_forms.isdisjoint(context_values)
            ):
                token = f"{transformed:.4f}".rstrip("0").rstrip(".")
                return Mutant(
                    case.stable_id, ARITHMETIC_RESTATEMENT,
                    _replace(case.answer, start, end, token),
                    _claims_as_rows((c.handle, c.quote) for c, _ in validated),
                    case.context,
                    ("magnitude_shift_absent_from_context",),
                )
    return None


def mutate_wrong_row(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    validated = _validated_evidence(case)
    if not validated:
        return None
    span = _supported_span(case, validated)
    if span is None:
        return None
    start, end, key_forms = span
    for passage in case.context.passages:
        if not contains_key(passage.text, case.key):
            continue
        # the requested record's own frame: the label under which the key sits
        key_frame = next(
            (frame for _, forms, frame in _framed_tokens(passage.text)
             if forms & key_forms),
            "",
        )
        if len(key_frame.split()) < 2:
            continue
        for token, forms, frame in _framed_tokens(passage.text):
            # a SIBLING record: same label, different value. Any other nearby
            # number is not a row relation and must not count as one.
            if forms.isdisjoint(key_forms) and frame == key_frame:
                return Mutant(
                    case.stable_id, WRONG_ROW,
                    _replace(case.answer, start, end, token),
                    ({"pasaj": passage.handle,
                      "alinti": _quote_around(passage.text, token)},),
                    case.context,
                    ("sibling_value_shares_the_key_label_frame",
                     "sibling_value_genuinely_in_cited_passage"),
                )
    return None


def mutate_label_value_swap(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    validated = _validated_evidence(case)
    if not validated:
        return None
    span = _supported_span(case, validated)
    if span is None:
        return None
    start, end, key_forms = span
    for peer in peers:
        _require_development(peer)
        if not _numeric_relation_peer(case, peer):
            continue
        peer_values = numbers(normalize(peer.key))
        for shared in _shared_passages(case, peer):
            framed = list(_framed_tokens(shared.text))
            key_frame = next(
                (frame for _, forms, frame in framed if forms & key_forms), "")
            if len(key_frame.split()) < 2:
                continue
            for token, forms, frame in framed:
                # a swap needs TWO labelled relations in one scope: the peer
                # value must sit under its own non-trivial label, and that
                # label must differ from the key's -- two bare numbers side by
                # side are not a label/value swap.
                if (
                    forms & peer_values
                    and forms.isdisjoint(key_forms)
                    and len(frame.split()) >= 2
                    and frame != key_frame
                ):
                    return Mutant(
                        case.stable_id, LABEL_VALUE_SWAP,
                        _replace(case.answer, start, end, token),
                        ({"pasaj": shared.handle,
                          "alinti": _quote_around(shared.text, token)},),
                        case.context,
                        ("two_distinct_label_frames_in_shared_passage",
                         "swapped_value_genuinely_in_shared_passage"),
                    )
    return None


def mutate_question_answer_mismatch(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    for peer in peers:
        _require_development(peer)
        if not _shares_key_passage(case, peer):
            continue
        if accepts_without_similarity(peer.key, peer.answer) is not True:
            continue
        shared = next(iter(_shared_passages(case, peer)), None)
        if shared is None:
            continue
        return Mutant(
            case.stable_id, QUESTION_ANSWER_MISMATCH,
            peer.answer,
            ({"pasaj": shared.handle, "alinti": shared.text},),
            case.context,
            ("peer_answer_correct_for_peer", "peer_target_in_shared_passage"),
        )
    return None


def mutate_forced_answer_unanswerable(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    if not _can_force_unanswerable(case):
        return None
    remaining = tuple(
        passage for passage in case.context.passages
        if not contains_key(passage.text, case.key)
    )
    stripped = RagContext(passages=remaining, numbered=case.context.numbered)
    surviving_handles = {passage.handle for passage in remaining}
    validated = _validated_evidence(case) or ()
    evidence = _claims_as_rows(
        (claim.handle, claim.quote)
        for claim, _ in validated
        if claim.handle in surviving_handles
    )
    return Mutant(
        case.stable_id, FORCED_ANSWER_UNANSWERABLE,
        case.answer,
        evidence,
        stripped,
        ("key_absent_from_every_remaining_passage",),
    )


def mutate_corrupted_quote(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    validated = _validated_evidence(case)
    if not validated:
        return None
    claim, passage = validated[0]
    if any(ch.isdigit() for ch in claim.quote):
        corrupted = "".join(
            str((int(ch) + 1) % 10) if ch.isdigit() else ch
            for ch in claim.quote
        )
    else:
        words = claim.quote.split()
        if len(words) < 2:
            return None
        words[0], words[1] = words[1], words[0]
        corrupted = " ".join(words)
    if fold(corrupted) in fold(passage.text):
        return None
    rest = _claims_as_rows((c.handle, c.quote) for c, _ in validated[1:])
    return Mutant(
        case.stable_id, CORRUPTED_QUOTE,
        case.answer,
        ({"pasaj": claim.handle, "alinti": corrupted},) + rest,
        case.context,
        ("quote_no_longer_in_claimed_passage", "answer_unchanged"),
    )


def mutate_invented_provenance(case: BaseCase, peers) -> Mutant | None:
    _require_development(case)
    validated = _validated_evidence(case)
    if not validated:
        return None
    claim, _ = validated[0]
    invented = max(p.handle for p in case.context.passages) + 1
    rest = _claims_as_rows((c.handle, c.quote) for c, _ in validated[1:])
    return Mutant(
        case.stable_id, INVENTED_PROVENANCE,
        case.answer,
        ({"pasaj": invented, "alinti": claim.quote},) + rest,
        case.context,
        ("handle_not_supplied_by_context", "answer_unchanged"),
    )


MUTATORS = {
    UNSUPPORTED_FIGURE: mutate_unsupported_figure,
    ARITHMETIC_RESTATEMENT: mutate_arithmetic_restatement,
    WRONG_ROW: mutate_wrong_row,
    LABEL_VALUE_SWAP: mutate_label_value_swap,
    QUESTION_ANSWER_MISMATCH: mutate_question_answer_mismatch,
    FORCED_ANSWER_UNANSWERABLE: mutate_forced_answer_unanswerable,
    CORRUPTED_QUOTE: mutate_corrupted_quote,
    INVENTED_PROVENANCE: mutate_invented_provenance,
}


def development_cases(cases) -> tuple:
    """Filter by RECOMPUTED identity and refuse records that lie about it."""
    kept = []
    for case in cases:
        locked = is_locked(case.stable_id)  # rejects malformed ids outright
        if locked != case.locked:
            raise ValueError("case split flag disagrees with its stable id")
        if not locked:
            kept.append(case)
    return tuple(kept)


def _row(answer: str, evidence) -> dict:
    return {"cevap": answer, "dayanak": list(evidence)}


def build_mutants(dev_cases) -> dict:
    """Every mutant this harness can construct, grouped by class."""
    produced = {name: [] for name in CLASS_ORDER}
    for case in dev_cases:
        _require_development(case)
    for case in dev_cases:
        peers = tuple(
            peer for peer in dev_cases
            if peer.set_name == case.set_name and peer.stable_id != case.stable_id
        )
        for name, mutator in MUTATORS.items():
            mutant = mutator(case, peers)
            if mutant is not None:
                produced[name].append((case, mutant))
    return produced


def replay(produced) -> dict:
    report = {}
    for name in CLASS_ORDER:
        pairs = produced[name]
        policies = (
            (STRUCTURED_DERIVED, STRUCTURED_EXPLICIT)
            if name in EVIDENCE_ONLY_CLASSES
            else POLICIES
        )
        expected = EXPECTED_DIAGNOSTIC[name]
        per_policy = {}
        for policy in policies:
            counts = Counter()
            flags_seen = Counter()
            for case, mutant in pairs:
                mutant_flags = replay_flags(
                    _row(mutant.answer, mutant.evidence), mutant.context, policy)
                control_flags = replay_flags(
                    _row(case.answer, _claims_as_rows(
                        (c.handle, c.quote) for c in case.evidence)),
                    case.context, policy)
                codes = {code for code, _ in mutant_flags}
                counts["n"] += 1
                counts["caught"] += bool(codes)
                counts["expected_diagnostic"] += bool(expected and expected in codes)
                counts["published"] += not codes
                counts["control_review"] += bool(control_flags)
                counts["control_published"] += not control_flags
                for code in codes:
                    flags_seen[code] += 1
            per_policy[policy] = {
                "counts": dict(counts),
                # sorted so two identical runs serialize byte-identically:
                # set iteration order fed this Counter, and dict order follows
                # insertion, so an unsorted dump differed between runs
                "flags": {code: flags_seen[code] for code in sorted(flags_seen)},
            }
        report[name] = {
            "mutants": len(pairs),
            "basis": DETECTORS[name]["basis"],
            "population_role": population_role(name),
            "structural_recall_eligible": population_role(name) == CONFIRMED_ROLE,
            "expected_diagnostic": expected,
            "policies": per_policy,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?",
                        default="output/RAG_Outputs/run2/native")
    parser.add_argument("--question-dir", default="data/rag_eval")
    parser.add_argument(
        "--output", default="output/eval/adversarial_dev_mutation_v1.json")
    args = parser.parse_args()

    cases, source_metadata = load_cases(Path(args.run_dir), Path(args.question_dir))
    dev = development_cases(cases)
    produced = build_mutants(dev)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "repository_head": repository_head(),
        "evaluation_layer": "generator_guard",
        "population": "development_clusters_only",
        "development_cases": len(dev),
        "locked_cases_excluded": len(cases) - len(dev),
        "source": source_metadata,
        "classes": replay(produced),
    }
    path = checked_output_path(Path(args.output))
    write_report(report, path)

    print(f"development kumeleri: {len(dev)}  (locked haric: {len(cases) - len(dev)})")
    print(f"{'sinif':<30}{'mutant':>7}   politika bazinda yakalanan/n")
    for name in CLASS_ORDER:
        body = report["classes"][name]
        cells = []
        for policy, result in body["policies"].items():
            counts = result["counts"]
            cells.append(f"{policy.split('_')[-1]}:{counts.get('caught', 0)}"
                         f"/{counts.get('n', 0)}")
        label = name + (" *" if body["population_role"] == CANDIDATE_ROLE else "")
        print(f"{label:<30}{body['mutants']:>7}   {'  '.join(cells)}")
    print(f"\nyazildi: {path}")
    print("NOT: bunlar DEVELOPMENT sayilari; kapi karari yeni holdout ister.")
    print("  *  ADAY populasyon: yapisal on-eleme gecti ama gercek kayit/etiket")
    print("     iliskisi insan onayi bekliyor; structural-recall paydasina girmez.")


if __name__ == "__main__":
    main()
