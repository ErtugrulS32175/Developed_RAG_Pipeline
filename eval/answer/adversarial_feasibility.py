"""Read-only feasibility inventory for the adversarial answer evaluation.

This is Step 2A, not a mutator and not a guard replay. It answers only whether
saved base-question clusters have enough structure to construct each
preregistered error class. Saved guard flags and publication decisions are
deliberately excluded from every eligibility input.

The persisted report contains aggregate counts and fingerprints only. Questions,
answers, quotations, figures, pages and source filenames remain under ignored
data/output paths.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from eval.answer.guard_floor import legacy_context
from eval.answer.judge import accepts_without_similarity
from eval.retrieval.rag_eval import contains_key
from pipeline.lang.tr_notation import fold, normalize, number_forms, numbers
from pipeline.retrieval.context import Passage, RagContext


PROTOCOL_VERSION = "adversarial_feasibility_v1"
REVIEW_PACKAGE_VERSION = "adversarial_feasibility_review_v1"
TARGET_LOCKED_CLUSTERS = 30
LOCKED_SHARE = 0.6875
DEVELOPMENT_HEX = frozenset("01234")
_SWEEP_SUFFIX = re.compile(r"_k\d+$")
_ANSWER_PAGE = re.compile(r"sayfa\s*\d+", re.IGNORECASE)
_FIGURE_TOKEN = re.compile(
    r"(?<![a-z0-9])(?:\d[\d.,/]*\d|\d)"
    r"(?:\s*(?:bin|milyon|milyar|trilyon))?"
)

WRONG_ROW = "wrong_row"
LABEL_VALUE_SWAP = "label_value_swap"
ARITHMETIC_RESTATEMENT = "arithmetic_restatement"
UNSUPPORTED_FIGURE = "unsupported_figure"
QUESTION_ANSWER_MISMATCH = "question_answer_mismatch"
FORCED_ANSWER_UNANSWERABLE = "forced_answer_unanswerable"
CORRUPTED_QUOTE = "corrupted_quote"
INVENTED_PROVENANCE = "invented_provenance"

CLASS_ORDER = (
    WRONG_ROW,
    LABEL_VALUE_SWAP,
    ARITHMETIC_RESTATEMENT,
    UNSUPPORTED_FIGURE,
    QUESTION_ANSWER_MISMATCH,
    FORCED_ANSWER_UNANSWERABLE,
    CORRUPTED_QUOTE,
    INVENTED_PROVENANCE,
)

DETECTORS = {
    WRONG_ROW: {
        "detector_id": "table_question_upper_bound_v1",
        "basis": "manual_upper_bound",
    },
    LABEL_VALUE_SWAP: {
        "detector_id": "curated_same_passage_numeric_pair_v1",
        "basis": "manual_upper_bound",
    },
    ARITHMETIC_RESTATEMENT: {
        "detector_id": "supported_safe_scale_transform_v1",
        "basis": "deterministic_lower_bound",
    },
    UNSUPPORTED_FIGURE: {
        "detector_id": "supported_mutable_figure_v1",
        "basis": "deterministic_lower_bound",
    },
    QUESTION_ANSWER_MISMATCH: {
        "detector_id": "curated_same_passage_target_v1",
        "basis": "manual_upper_bound",
    },
    FORCED_ANSWER_UNANSWERABLE: {
        "detector_id": "remove_all_key_passages_v1",
        "basis": "deterministic_lower_bound",
    },
    CORRUPTED_QUOTE: {
        "detector_id": "validated_saved_quote_v1",
        "basis": "deterministic_lower_bound",
    },
    INVENTED_PROVENANCE: {
        "detector_id": "validated_saved_handle_v1",
        "basis": "deterministic_lower_bound",
    },
}


@dataclass(frozen=True)
class EvidenceClaim:
    handle: int
    quote: str


@dataclass(frozen=True)
class BaseCase:
    stable_id: str
    locked: bool
    set_name: str
    question: str
    key: str
    answer: str
    question_type: str
    context: RagContext
    evidence_shape_valid: bool
    evidence: tuple[EvidenceClaim, ...]
    source_row_fingerprint: str


def _canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_question_id(set_name: str, question: str) -> str:
    return _digest(f"{set_name}\0{question}")


def is_locked(stable_id: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{64}", stable_id):
        raise ValueError("stable id must be a lowercase SHA-256 hex digest")
    return stable_id[0] not in DEVELOPMENT_HEX


def expected_additional_questions(cluster_deficit: int) -> int:
    if type(cluster_deficit) is not int or cluster_deficit < 0:
        raise ValueError("cluster deficit must be a non-negative integer")
    return math.ceil(cluster_deficit / LOCKED_SHARE)


def _claims(raw) -> tuple[bool, tuple[EvidenceClaim, ...]]:
    if type(raw) is not list:
        return False, ()
    claims = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"pasaj", "alinti"}:
            return False, ()
        handle, quote = item["pasaj"], item["alinti"]
        if type(handle) is not int or handle < 1:
            return False, ()
        if not isinstance(quote, str) or not quote.strip():
            return False, ()
        claims.append(EvidenceClaim(handle, quote))
    return True, tuple(claims)


def make_case(set_name: str, row: dict, question_record: dict) -> BaseCase:
    """Build one immutable eligibility input from explicitly allowed fields."""
    question = row.get("soru")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("saved row has no question")
    if question_record.get("q") != question:
        raise ValueError("saved row and question record disagree")
    key = question_record.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("question record has no key")
    answer = row.get("cevap")
    context_text = row.get("baglam")
    if not isinstance(answer, str) or not isinstance(context_text, str):
        raise ValueError("saved row answer/context must be strings")

    shape_valid, evidence = _claims(row.get("dayanak"))
    stable_id = stable_question_id(set_name, question)
    allowed_source = {
        "set": set_name,
        "question": question,
        "key": key,
        "type": question_record.get("type"),
        "answer": answer,
        "context": context_text,
        "evidence": row.get("dayanak"),
    }
    return BaseCase(
        stable_id=stable_id,
        locked=is_locked(stable_id),
        set_name=set_name,
        question=question,
        key=key,
        answer=answer,
        question_type=str(question_record.get("type") or ""),
        context=legacy_context(context_text),
        evidence_shape_valid=shape_valid,
        evidence=evidence,
        source_row_fingerprint=_digest(_canonical(allowed_source)),
    )


def _selected_result_paths(run_dir: Path) -> dict[str, Path]:
    selected = {}
    for path in sorted(run_dir.glob("rag_answers_*.json")):
        set_name = path.stem.removeprefix("rag_answers_")
        if _SWEEP_SUFFIX.search(set_name):
            continue
        if set_name in selected:
            raise ValueError("duplicate saved result set")
        selected[set_name] = path
    if not selected:
        raise ValueError("no non-sweep saved result sets found")
    return selected


def _content_group_fingerprint(named_bodies) -> str:
    """Fingerprint over (name, bytes) pairs read ONCE by the caller.

    The earlier shape took paths and re-read them here -- so what was
    PARSED and what was FINGERPRINTED were two separate reads, and an audit
    probe that swapped the file in between produced a report whose numbers
    came from the old content and whose fingerprint blessed the new. One
    read, one buffer, both uses."""
    items = [
        {
            "content_sha256": _digest(body),
            "logical_name_sha256": _digest(name),
        }
        for name, body in sorted(named_bodies, key=lambda item: item[0])
    ]
    return _digest(_canonical(items))


def load_cases(run_dir: Path, question_dir: Path):
    """Load scoreable saved sets without consuming saved guard decisions.

    Every file is read EXACTLY ONCE: the same bytes are parsed and
    fingerprinted, so a file swapped mid-run cannot leave the numbers from
    one content and the fingerprint from another.

    Within a set the coverage is EXACT: an audit probe fed a two-question
    set with a one-row result file and the missing question simply
    vanished from the population -- denominators shrank with no error.
    Every question now needs exactly one result row. (Whether the
    DIRECTORY may hold question sets with no results is the caller's
    contract: the shared eval directory legitimately carries holdout sets
    no development run has answered, so the dev harness tolerates them,
    while the locked runner separately requires its question directory to
    be exactly the measured population.) Errors carry COUNTS, never
    names -- set names are content."""
    result_paths = _selected_result_paths(run_dir)
    cases = []
    result_bodies = []
    question_bodies = []
    for set_name, result_path in sorted(result_paths.items()):
        question_path = question_dir / f"{set_name}.json"
        if not question_path.is_file():
            raise ValueError("matching question set is missing")
        question_bytes = question_path.read_bytes()
        question_bodies.append((question_path.name, question_bytes))
        questions = json.loads(question_bytes.decode("utf-8"))
        if not isinstance(questions, list):
            raise ValueError("question set must be a list")
        by_question = {}
        for record in questions:
            if not isinstance(record, dict):
                raise ValueError("question record must be an object")
            question = record.get("q")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question record has no question")
            if question in by_question:
                raise ValueError("duplicate question in question set")
            by_question[question] = record

        result_bytes = result_path.read_bytes()
        result_bodies.append((result_path.name, result_bytes))
        payload = json.loads(result_bytes.decode("utf-8"))
        rows = payload.get("sorular") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("saved result has no question rows")
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("saved result row must be an object")
            question = row.get("soru")
            if question not in by_question:
                raise ValueError("saved question is absent from its question set")
            if question in seen:
                raise ValueError("duplicate saved question row")
            seen.add(question)
            cases.append(make_case(set_name, row, by_question[question]))
        if seen != set(by_question):
            raise ValueError(
                f"sonuc dosyasi soru setini tam kapsamiyor: "
                f"{len(set(by_question) - seen)} soru sonucsuz")

    stable_ids = [case.stable_id for case in cases]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("stable base-question ids are not unique")

    metadata = {
        "result_files": len(result_paths),
        "question_files": len(question_bodies),
        "raw_result_files_fingerprint": _content_group_fingerprint(
            result_bodies
        ),
        "question_files_fingerprint": _content_group_fingerprint(
            question_bodies
        ),
        "eligibility_input_fingerprint": _digest(_canonical(sorted(
            case.source_row_fingerprint for case in cases
        ))),
        "split_manifest_fingerprint": _digest(_canonical(sorted(
            (case.stable_id, "locked" if case.locked else "development")
            for case in cases
        ))),
    }
    return tuple(cases), metadata


def _validated_evidence(case: BaseCase):
    if not case.evidence_shape_valid or not case.evidence:
        return None
    known = case.context.by_handle()
    validated = []
    for claim in case.evidence:
        passage = known.get(claim.handle)
        if passage is None or fold(claim.quote) not in fold(passage.text):
            return None
        validated.append((claim, passage))
    return tuple(validated)


def _supported_figure_forms(case: BaseCase):
    if not accepts_without_similarity(case.key, case.answer):
        return ()
    evidence = _validated_evidence(case)
    if not evidence:
        return ()
    key_values = numbers(normalize(case.key))
    quote_values = numbers(normalize(" ".join(
        claim.quote for claim, _ in evidence
    )))
    if not key_values or not quote_values:
        return ()

    body = normalize(_ANSWER_PAGE.sub(" ", case.answer))
    supported = []
    for match in _FIGURE_TOKEN.finditer(body):
        token = match.group()
        if token.count(".") + token.count(",") + token.count("/") > 1:
            continue
        forms = frozenset(numbers(token))
        if forms and forms & key_values and forms & quote_values:
            supported.append(forms)
    return tuple(supported)


def _formatted_forms(value: float):
    if not math.isfinite(value):
        return set()
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return numbers(text)


def _has_safe_scale_transform(case: BaseCase) -> bool:
    evidence = _validated_evidence(case)
    forms_by_figure = _supported_figure_forms(case)
    if not evidence or not forms_by_figure:
        return False
    scope_values = numbers(normalize(" ".join(
        passage.text for _, passage in evidence
    )))
    for forms in forms_by_figure:
        for value in forms:
            if not math.isfinite(value) or value == 0:
                continue
            for transformed in (value * 10, value / 10):
                transformed_forms = _formatted_forms(transformed)
                if (
                    transformed_forms
                    and transformed_forms.isdisjoint(forms)
                    and transformed_forms.isdisjoint(scope_values)
                ):
                    return True
    return False


def _can_force_unanswerable(case: BaseCase) -> bool:
    if not accepts_without_similarity(case.key, case.answer):
        return False
    carrying = [
        passage
        for passage in case.context.passages
        if contains_key(passage.text, case.key)
    ]
    remaining = [
        passage
        for passage in case.context.passages
        if passage not in carrying
    ]
    if not carrying or not remaining:
        return False
    remaining_text = "\n\n".join(passage.text for passage in remaining)
    return contains_key(remaining_text, case.key) is False


def _shares_key_passage(case: BaseCase, peer: BaseCase) -> bool:
    if case.set_name != peer.set_name or case.stable_id == peer.stable_id:
        return False
    if accepts_without_similarity(case.key, peer.key):
        return False
    return any(
        contains_key(passage.text, case.key)
        and contains_key(passage.text, peer.key)
        for passage in case.context.passages
    )


def _numeric_relation_peer(case: BaseCase, peer: BaseCase) -> bool:
    case_values = numbers(normalize(case.key))
    peer_values = numbers(normalize(peer.key))
    return bool(
        case_values
        and peer_values
        and case_values.isdisjoint(peer_values)
        and _shares_key_passage(case, peer)
    )


def _shared_passages(case: BaseCase, peer: BaseCase):
    return tuple(
        passage
        for passage in case.context.passages
        if contains_key(passage.text, case.key)
        and contains_key(passage.text, peer.key)
    )


def eligible_sets(cases: tuple[BaseCase, ...]):
    """Confirmed lower bounds and manual candidate upper bounds by class."""
    by_set = {}
    for case in cases:
        by_set.setdefault(case.set_name, []).append(case)

    confirmed = {name: set() for name in CLASS_ORDER}
    candidates = {name: set() for name in CLASS_ORDER}
    for case in cases:
        valid_evidence = _validated_evidence(case)
        supported_figures = _supported_figure_forms(case)
        peers = by_set[case.set_name]

        if supported_figures:
            confirmed[UNSUPPORTED_FIGURE].add(case.stable_id)
        if _has_safe_scale_transform(case):
            confirmed[ARITHMETIC_RESTATEMENT].add(case.stable_id)
        if _can_force_unanswerable(case):
            confirmed[FORCED_ANSWER_UNANSWERABLE].add(case.stable_id)
        if valid_evidence:
            confirmed[CORRUPTED_QUOTE].add(case.stable_id)
            confirmed[INVENTED_PROVENANCE].add(case.stable_id)

        if case.question_type == "tablo" and contains_key(
                case.context.model_text, case.key):
            candidates[WRONG_ROW].add(case.stable_id)
        if any(_numeric_relation_peer(case, peer) for peer in peers):
            candidates[LABEL_VALUE_SWAP].add(case.stable_id)
        if any(_shares_key_passage(case, peer) for peer in peers):
            candidates[QUESTION_ANSWER_MISMATCH].add(case.stable_id)

    for name, spec in DETECTORS.items():
        if spec["basis"] != "manual_upper_bound":
            candidates[name] = set(confirmed[name])
    return confirmed, candidates


def build_manual_review_package(
        cases: tuple[BaseCase, ...], aggregate_report: dict):
    """Private locked-candidate packet for an independent eligibility reviewer.

    The caller must persist this only below ignored ``output/`` and must never
    print it. The aggregate report receives only a fingerprint and counts.
    """
    _, candidates = eligible_sets(cases)
    by_id = {case.stable_id: case for case in cases}
    by_set = {}
    for case in cases:
        by_set.setdefault(case.set_name, []).append(case)

    private_cases = {}
    private_passages = {}

    def include_case(case: BaseCase):
        private_cases.setdefault(case.stable_id, {
            "question": case.question,
            "key": case.key,
            "question_type": case.question_type,
        })

    def include_passage(case: BaseCase, passage: Passage):
        passage_id = _digest(_canonical({
            "case_id": case.stable_id,
            "handle": passage.handle,
            "page": passage.page,
            "citation": passage.citation,
            "text": passage.text,
        }))
        private_passages.setdefault(passage_id, {
            "case_id": case.stable_id,
            "handle": passage.handle,
            "page": passage.page,
            "citation": passage.citation,
            "text": passage.text,
        })
        return passage_id

    entries = {name: [] for name in (
        WRONG_ROW,
        LABEL_VALUE_SWAP,
        QUESTION_ANSWER_MISMATCH,
    )}
    locked_ids = {
        case.stable_id for case in cases if case.locked
    }
    for name in entries:
        for case_id in sorted(candidates[name] & locked_ids):
            case = by_id[case_id]
            include_case(case)
            peer_cases = []
            passage_ids = []
            if name == WRONG_ROW:
                passage_ids = [
                    include_passage(case, passage)
                    for passage in case.context.passages
                ]
            else:
                predicate = (
                    _numeric_relation_peer
                    if name == LABEL_VALUE_SWAP
                    else _shares_key_passage
                )
                peer_cases = sorted(
                    (
                        peer
                        for peer in by_set[case.set_name]
                        if predicate(case, peer)
                    ),
                    key=lambda peer: peer.stable_id,
                )
                for peer in peer_cases:
                    include_case(peer)
                    passage_ids.extend(
                        include_passage(case, passage)
                        for passage in _shared_passages(case, peer)
                    )

            entries[name].append({
                "case_id": case.stable_id,
                "candidate_peer_ids": [
                    peer.stable_id for peer in peer_cases
                ],
                "passage_ids": sorted(set(passage_ids)),
                "decision": None,
                "selected_peer_id": None,
                "reviewer_note": None,
            })

    return {
        "package_version": REVIEW_PACKAGE_VERSION,
        "private_document_content": True,
        "do_not_commit_or_print": True,
        "reviewer_role": "eligibility_reviewer_not_mutator_implementer",
        "review_order": "stable_id_ascending",
        "allowed_decisions": ["eligible", "reject"],
        "source_eligibility_input_fingerprint": aggregate_report[
            "source"
        ]["eligibility_input_fingerprint"],
        "stopping_rules": {
            name: {
                "gate": aggregate_report["classes"][name][
                    "manual_gate_review_plan"
                ],
                "descriptive": aggregate_report["classes"][name].get(
                    "manual_descriptive_review_plan"
                ),
            }
            for name in entries
        },
        "entry_counts": {
            name: len(class_entries)
            for name, class_entries in entries.items()
        },
        "entries": entries,
        "cases": dict(sorted(private_cases.items())),
        "passages": dict(sorted(private_passages.items())),
    }


def attach_review_metadata(aggregate_report: dict, review_package: dict):
    """Return aggregate report metadata without copying private review text."""
    report = dict(aggregate_report)
    report["manual_review_package"] = {
        "package_version": review_package["package_version"],
        "fingerprint": _digest(_canonical(review_package)),
        "entry_counts": review_package["entry_counts"],
        "private_document_content_in_aggregate": False,
    }
    return report


def _class_report(
        name: str,
        cases: tuple[BaseCase, ...],
        confirmed_ids: set[str],
        candidate_ids: set[str],
):
    locked_ids = {case.stable_id for case in cases if case.locked}
    confirmed_locked = len(confirmed_ids & locked_ids)
    candidate_locked = len(candidate_ids & locked_ids)
    confirmed_deficit = max(
        0,
        TARGET_LOCKED_CLUSTERS - confirmed_locked,
    )
    basis = DETECTORS[name]["basis"]
    if basis == "manual_upper_bound":
        minimum_deficit = max(
            0,
            TARGET_LOCKED_CLUSTERS - candidate_locked,
        )
        if candidate_locked < TARGET_LOCKED_CLUSTERS:
            status = "underpowered"
        else:
            status = "pending_manual_confirmation"
        exact_deficit_known = candidate_ids == confirmed_ids
    else:
        minimum_deficit = confirmed_deficit
        status = (
            "candidate_powered"
            if confirmed_locked >= TARGET_LOCKED_CLUSTERS
            else "underpowered"
        )
        exact_deficit_known = True

    report = {
        **DETECTORS[name],
        "status": status,
        "total_base_question_clusters": len(cases),
        "locked_base_question_clusters": len(locked_ids),
        "confirmed_eligible_total_clusters": len(confirmed_ids),
        "confirmed_eligible_locked_clusters": confirmed_locked,
        "confirmed_locked_cluster_deficit": confirmed_deficit,
        "candidate_upper_bound_total_clusters": len(candidate_ids),
        "candidate_upper_bound_locked_clusters": candidate_locked,
        "minimum_unavoidable_locked_cluster_deficit": minimum_deficit,
        "exact_deficit_known": exact_deficit_known,
    }
    if basis == "manual_upper_bound":
        gate_review_required = candidate_locked >= TARGET_LOCKED_CLUSTERS
        report["manual_gate_review_plan"] = {
            "review_order": "stable_id_ascending",
            "review_required": gate_review_required,
            "confirmation_target": (
                TARGET_LOCKED_CLUSTERS if gate_review_required else None
            ),
            "stop_at_confirmation": (
                TARGET_LOCKED_CLUSTERS if gate_review_required else None
            ),
            "stop_at_rejection": (
                candidate_locked - TARGET_LOCKED_CLUSTERS + 1
                if gate_review_required else None
            ),
        }
        if name == WRONG_ROW:
            descriptive_target = 10
            descriptive_review_required = candidate_locked >= descriptive_target
            report["manual_descriptive_review_plan"] = {
                "review_order": "stable_id_ascending",
                "review_required": descriptive_review_required,
                "confirmation_target": (
                    descriptive_target
                    if descriptive_review_required else None
                ),
                "stop_at_confirmation": (
                    descriptive_target
                    if descriptive_review_required else None
                ),
                "stop_at_rejection": (
                    candidate_locked - descriptive_target + 1
                    if descriptive_review_required else None
                ),
                "changes_gate_status": False,
            }
    if exact_deficit_known:
        report["expected_additional_eligible_base_questions"] = (
            expected_additional_questions(confirmed_deficit)
        )
    else:
        report["expected_additional_eligible_base_questions"] = None
        report["expected_additional_eligible_base_questions_floor"] = (
            expected_additional_questions(minimum_deficit)
        )
    return report


def build_report(cases: tuple[BaseCase, ...], source_metadata: dict):
    confirmed, candidates = eligible_sets(cases)
    locked = sum(case.locked for case in cases)
    detector_fingerprint = _digest(_canonical(DETECTORS))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_layer": "generator_guard",
        "three_layer_scope": {
            "retrieval": "recorded_input_condition_not_remeasured",
            "generator_guard": "eligibility_inventory",
            "end_to_end": "not_measured",
        },
        "guard_results_used_for_eligibility": False,
        "target_locked_clusters_per_class": TARGET_LOCKED_CLUSTERS,
        "development_hex_digits": "0-4",
        "locked_hex_digits": "5-f",
        "intended_development_share": 0.3125,
        "intended_locked_share": LOCKED_SHARE,
        "total_base_question_clusters": len(cases),
        "development_base_question_clusters": len(cases) - locked,
        "locked_base_question_clusters": locked,
        "detector_fingerprint": detector_fingerprint,
        "implementation_fingerprint": _digest(Path(__file__).read_bytes()),
        "source": source_metadata,
        "classes": {
            name: _class_report(
                name,
                cases,
                confirmed[name],
                candidates[name],
            )
            for name in CLASS_ORDER
        },
    }


# Git and output/ operations are ANCHORED here, never to the process's
# working directory: a probe invoked the runner from a different (clean)
# repository and the clean-tree check blessed the WRONG repo while HEAD
# named a commit of code that was not running.
REPO_ROOT = Path(__file__).resolve().parents[2]


def repository_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(
        r"[0-9a-f]{40}", value
    ) else None


def checked_output_path(path: Path) -> Path:
    output_root = (REPO_ROOT / "output").resolve()
    anchored = path if path.is_absolute() else REPO_ROOT / path
    resolved = anchored.resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("report path must stay under output/") from exc
    if resolved.exists():
        raise FileExistsError("report path already exists")
    return resolved


def write_report(report: dict, path: Path):
    resolved = checked_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--question-dir", default="data/rag_eval")
    parser.add_argument(
        "--output",
        default="output/eval/adversarial_feasibility_v1.json",
    )
    parser.add_argument(
        "--review-output",
        default="output/eval/adversarial_feasibility_review_v1.json",
    )
    args = parser.parse_args()

    output_path = checked_output_path(Path(args.output))
    review_output_path = checked_output_path(Path(args.review_output))
    if output_path == review_output_path:
        raise ValueError("aggregate and private review paths must differ")
    cases, source = load_cases(
        Path(args.run_dir),
        Path(args.question_dir),
    )
    source["repository_head"] = repository_head()
    report = build_report(cases, source)
    review_package = build_manual_review_package(cases, report)
    report = attach_review_metadata(report, review_package)
    write_report(review_package, review_output_path)
    write_report(report, output_path)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
