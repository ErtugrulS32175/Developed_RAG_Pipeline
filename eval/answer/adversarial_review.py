"""Apply opaque eligibility decisions to an adversarial feasibility report.

The private review package is used only to verify its fingerprint and the
ordered candidate ids. Case text, passages and reviewer notes are never copied
to the derived aggregate report.

This step measures construction feasibility. It does not replay a guard and
does not establish a production or enterprise release gate.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from eval.answer.adversarial_feasibility import (
    CLASS_ORDER,
    LABEL_VALUE_SWAP,
    QUESTION_ANSWER_MISMATCH,
    TARGET_LOCKED_CLUSTERS,
    WRONG_ROW,
    checked_output_path,
    expected_additional_questions,
    write_report,
)


DECISION_VERSION = "adversarial_eligibility_decisions_v1"
REVIEWED_REPORT_VERSION = "adversarial_feasibility_reviewed_v1"
MANUAL_CLASSES = (
    WRONG_ROW,
    LABEL_VALUE_SWAP,
    QUESTION_ANSWER_MISMATCH,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ordered_candidate_ids(review_package: dict, class_name: str):
    entries = review_package.get("entries", {}).get(class_name)
    if not isinstance(entries, list):
        raise ValueError(f"review package has no {class_name} entries")
    ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("review entry must be an object")
        ids.append(_require_sha256(entry.get("case_id"), "case id"))
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("review candidates must be unique and hash ordered")
    return tuple(ids)


def _id_set(values, label: str):
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    result = {_require_sha256(value, label) for value in values}
    if len(result) != len(values):
        raise ValueError(f"{label} contains duplicates")
    return result


def _first_stop(candidate_ids, rejected, unresolved, plan):
    target = plan.get("confirmation_target")
    rejection_limit = plan.get("stop_at_rejection")
    if type(target) is not int or target < 1:
        raise ValueError("review plan has no confirmation target")
    if type(rejection_limit) is not int or rejection_limit < 1:
        raise ValueError("review plan has no rejection limit")

    confirmations = 0
    rejections = 0
    for index, case_id in enumerate(candidate_ids, start=1):
        if case_id in rejected:
            rejections += 1
        elif case_id not in unresolved:
            confirmations += 1
        if confirmations == target:
            return {
                "outcome": "confirmation_target_reached",
                "at_reviewed_candidate": index,
            }
        if rejections == rejection_limit:
            return {
                "outcome": "rejection_limit_reached",
                "at_reviewed_candidate": index,
            }
    return {
        "outcome": "not_reached",
        "at_reviewed_candidate": None,
    }


def _review_one_class(
        class_name: str,
        base_class: dict,
        review_package: dict,
        decision: dict,
):
    candidate_ids = _ordered_candidate_ids(review_package, class_name)
    reviewed_prefix = decision.get("reviewed_prefix")
    if (
        type(reviewed_prefix) is not int
        or reviewed_prefix < 0
        or reviewed_prefix > len(candidate_ids)
    ):
        raise ValueError("reviewed prefix is outside the candidate list")

    reviewed_ids = set(candidate_ids[:reviewed_prefix])
    rejected = _id_set(decision.get("rejected_ids"), "rejected ids")
    unresolved = _id_set(decision.get("unresolved_ids"), "unresolved ids")
    if rejected & unresolved:
        raise ValueError("a review decision cannot be rejected and unresolved")
    if not rejected | unresolved <= reviewed_ids:
        raise ValueError("review decisions must stay inside the reviewed prefix")

    eligible = reviewed_prefix - len(rejected) - len(unresolved)
    unreviewed = len(candidate_ids) - reviewed_prefix
    if class_name == WRONG_ROW:
        plan = base_class.get("manual_descriptive_review_plan")
        gate_status = "underpowered"
    else:
        plan = base_class.get("manual_gate_review_plan")
        gate_status = (
            "candidate_powered"
            if eligible >= TARGET_LOCKED_CLUSTERS
            else "underpowered"
        )
    if not isinstance(plan, dict) or not plan.get("review_required"):
        raise ValueError("manual review plan is missing or inactive")

    first_stop = _first_stop(
        candidate_ids[:reviewed_prefix],
        rejected,
        unresolved,
        plan,
    )
    stop_index = first_stop["at_reviewed_candidate"]
    stop_ids = set(candidate_ids[:stop_index]) if stop_index else set()
    confirmed_at_stop = (
        len(stop_ids - rejected - unresolved)
        if stop_index else 0
    )
    rejected_at_stop = len(stop_ids & rejected)
    unresolved_at_stop = len(stop_ids & unresolved)
    over_reviewed = (
        reviewed_prefix - stop_index
        if stop_index is not None
        else 0
    )

    eligible_universe_exact = unreviewed == 0 and not unresolved
    target_deficit = max(0, TARGET_LOCKED_CLUSTERS - eligible)
    target_deficit_exact = (
        target_deficit == 0 or eligible_universe_exact
    )
    result = copy.deepcopy(base_class)
    result.update({
        "status": gate_status,
        "manual_review": {
            "candidate_locked_clusters": len(candidate_ids),
            "reviewed_prefix_clusters": reviewed_prefix,
            "confirmed_eligible_locked_clusters": eligible,
            "rejected_locked_clusters": len(rejected),
            "unresolved_locked_clusters": len(unresolved),
            "unreviewed_locked_clusters": unreviewed,
            "eligible_universe_exact": eligible_universe_exact,
            "first_preregistered_stop": first_stop,
            "confirmed_at_first_stop": confirmed_at_stop,
            "rejected_at_first_stop": rejected_at_stop,
            "unresolved_at_first_stop": unresolved_at_stop,
            "reviewed_after_first_stop": over_reviewed,
            "confirmed_after_first_stop": (
                eligible - confirmed_at_stop
                if stop_index is not None else 0
            ),
            "protocol_clean": (
                over_reviewed == 0 and not unresolved
            ),
        },
        "confirmed_eligible_total_clusters": eligible,
        "confirmed_eligible_total_clusters_is_lower_bound": True,
        "confirmed_eligible_locked_clusters": eligible,
        "confirmed_locked_cluster_deficit": target_deficit,
        "minimum_unavoidable_locked_cluster_deficit": target_deficit,
        "exact_deficit_known": target_deficit_exact,
        "expected_additional_eligible_base_questions": (
            expected_additional_questions(target_deficit)
            if target_deficit_exact else None
        ),
    })
    if target_deficit_exact:
        result.pop(
            "expected_additional_eligible_base_questions_floor",
            None,
        )
    else:
        result["expected_additional_eligible_base_questions_floor"] = (
            expected_additional_questions(
                max(
                    0,
                    TARGET_LOCKED_CLUSTERS
                    - int(base_class[
                        "candidate_upper_bound_locked_clusters"
                    ]),
                )
            )
        )
    return result


def apply_review_decisions(
        aggregate: dict,
        review_package: dict,
        decisions: dict,
):
    """Validate and merge opaque reviewer decisions into aggregate counts."""
    if decisions.get("decision_version") != DECISION_VERSION:
        raise ValueError("unsupported decision version")
    if set(decisions.get("classes", {})) != set(MANUAL_CLASSES):
        raise ValueError("decisions must cover exactly the manual classes")
    if aggregate.get("manual_review_package", {}).get(
            "fingerprint") != _digest(_canonical(review_package)):
        raise ValueError("private review package fingerprint mismatch")
    if decisions.get("review_package_fingerprint") != aggregate[
            "manual_review_package"]["fingerprint"]:
        raise ValueError("decision and review package fingerprints disagree")
    if review_package.get(
            "source_eligibility_input_fingerprint") != aggregate.get(
                "source", {}).get("eligibility_input_fingerprint"):
        raise ValueError("review package source fingerprint mismatch")

    integrity = decisions.get("holdout_integrity")
    required_integrity = {
        "locked_content_returned_to_implementer",
        "mutation_rules_frozen_before_return",
        "reviewer_withdrawn_from_mutator_work",
    }
    if not isinstance(integrity, dict) or set(integrity) != required_integrity:
        raise ValueError("holdout integrity declaration is incomplete")
    if any(type(integrity[name]) is not bool for name in required_integrity):
        raise ValueError("holdout integrity declarations must be boolean")

    classes = copy.deepcopy(aggregate.get("classes"))
    if not isinstance(classes, dict) or set(classes) != set(CLASS_ORDER):
        raise ValueError("aggregate class inventory is incomplete")
    for class_name in MANUAL_CLASSES:
        classes[class_name] = _review_one_class(
            class_name,
            classes[class_name],
            review_package,
            decisions["classes"][class_name],
        )

    blinding_breached = (
        integrity["locked_content_returned_to_implementer"]
        and not integrity["mutation_rules_frozen_before_return"]
    )
    over_review = {
        name: classes[name]["manual_review"][
            "reviewed_after_first_stop"
        ]
        for name in MANUAL_CLASSES
        if classes[name]["manual_review"][
            "reviewed_after_first_stop"
        ] > 0
    }
    powered = sum(
        item["status"] == "candidate_powered"
        for item in classes.values()
    )
    underpowered = sum(
        item["status"] == "underpowered"
        for item in classes.values()
    )
    protocol_deviations = []
    if over_review:
        protocol_deviations.append({
            "code": "manual_review_continued_after_preregistered_stop",
            "classes": over_review,
        })
    unresolved_review = {
        name: classes[name]["manual_review"][
            "unresolved_locked_clusters"
        ]
        for name in MANUAL_CLASSES
        if classes[name]["manual_review"][
            "unresolved_locked_clusters"
        ] > 0
    }
    if unresolved_review:
        protocol_deviations.append({
            "code": "review_returned_unregistered_unresolved_disposition",
            "classes": unresolved_review,
        })
    if integrity["locked_content_returned_to_implementer"]:
        protocol_deviations.append({
            "code": "locked_content_returned_beyond_aggregate_decisions",
        })

    report = {
        "report_version": REVIEWED_REPORT_VERSION,
        "protocol_version": aggregate.get("protocol_version"),
        "evaluation_layer": "generator_guard",
        "three_layer_scope": copy.deepcopy(
            aggregate.get("three_layer_scope")
        ),
        "source_aggregate_fingerprint": _digest(_canonical(aggregate)),
        "review_package_fingerprint": decisions[
            "review_package_fingerprint"
        ],
        "decision_fingerprint": _digest(_canonical(decisions)),
        "guard_results_used_for_eligibility": False,
        "eligibility_summary": {
            "candidate_powered_classes": powered,
            "underpowered_classes": underpowered,
            "total_classes": len(classes),
        },
        "holdout_integrity": {
            **integrity,
            "blinding_breached_before_mutator_implementation": (
                blinding_breached
            ),
            "eligibility_measurement_valid": True,
            "current_locked_allowed_for_final_guard_recall_gate": (
                not blinding_breached
            ),
            "current_locked_use": (
                "feasibility_and_development_only"
                if blinding_breached
                else "frozen_evaluation"
            ),
            "new_unseen_holdout_required_before_guard_recall_gate": (
                blinding_breached
            ),
        },
        "protocol_deviations": protocol_deviations,
        "classes": classes,
        "private_document_content_in_report": False,
    }
    return report


def _load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default="output/eval/adversarial_feasibility_v1.json",
    )
    parser.add_argument(
        "--review-package",
        default="output/eval/adversarial_feasibility_review_v1.json",
    )
    parser.add_argument(
        "--decisions",
        default="output/eval/adversarial_feasibility_decisions_v1.json",
    )
    parser.add_argument(
        "--output",
        default="output/eval/adversarial_feasibility_reviewed_v1.json",
    )
    args = parser.parse_args()

    output_path = checked_output_path(Path(args.output))
    report = apply_review_decisions(
        _load_json(Path(args.aggregate)),
        _load_json(Path(args.review_package)),
        _load_json(Path(args.decisions)),
    )
    write_report(report, output_path)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
