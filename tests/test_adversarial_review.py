import json

import pytest

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
)
from eval.answer.adversarial_review import (
    DECISION_VERSION,
    _canonical,
    _digest,
    apply_review_decisions,
)


def _ids(prefix, count):
    return [prefix + f"{index:063x}" for index in range(count)]


def _inputs():
    class_counts = {
        WRONG_ROW: 11,
        LABEL_VALUE_SWAP: 34,
        QUESTION_ANSWER_MISMATCH: 52,
    }
    entries = {
        name: [
            {
                "case_id": case_id,
                "private_marker": "KURGU_OZEL_INCELEME_METNI",
            }
            for case_id in _ids(prefix, class_counts[name])
        ]
        for name, prefix in (
            (WRONG_ROW, "5"),
            (LABEL_VALUE_SWAP, "6"),
            (QUESTION_ANSWER_MISMATCH, "7"),
        )
    }
    source_fingerprint = "a" * 64
    package = {
        "source_eligibility_input_fingerprint": source_fingerprint,
        "entries": entries,
        "cases": {"private": "KURGU_GIZLI_SORU"},
        "passages": {"private": "KURGU_GIZLI_PASAJ"},
    }
    package_fingerprint = _digest(_canonical(package))

    manual_classes = {
        WRONG_ROW: {
            "status": "underpowered",
            "candidate_upper_bound_locked_clusters": 11,
            "manual_gate_review_plan": {"review_required": False},
            "manual_descriptive_review_plan": {
                "review_required": True,
                "confirmation_target": 10,
                "stop_at_confirmation": 10,
                "stop_at_rejection": 2,
            },
        },
        LABEL_VALUE_SWAP: {
            "status": "pending_manual_confirmation",
            "candidate_upper_bound_locked_clusters": 34,
            "manual_gate_review_plan": {
                "review_required": True,
                "confirmation_target": 30,
                "stop_at_confirmation": 30,
                "stop_at_rejection": 5,
            },
        },
        QUESTION_ANSWER_MISMATCH: {
            "status": "pending_manual_confirmation",
            "candidate_upper_bound_locked_clusters": 52,
            "manual_gate_review_plan": {
                "review_required": True,
                "confirmation_target": 30,
                "stop_at_confirmation": 30,
                "stop_at_rejection": 23,
            },
        },
    }
    exact_classes = {
        name: {
            "status": "candidate_powered",
            "confirmed_eligible_locked_clusters": 30,
        }
        for name in (
            ARITHMETIC_RESTATEMENT,
            UNSUPPORTED_FIGURE,
            FORCED_ANSWER_UNANSWERABLE,
            CORRUPTED_QUOTE,
            INVENTED_PROVENANCE,
        )
    }
    aggregate = {
        "protocol_version": "kurgu_protokol",
        "three_layer_scope": {
            "retrieval": "recorded_input_condition_not_remeasured",
            "generator_guard": "eligibility_inventory",
            "end_to_end": "not_measured",
        },
        "source": {
            "eligibility_input_fingerprint": source_fingerprint,
        },
        "manual_review_package": {
            "fingerprint": package_fingerprint,
        },
        "classes": {
            name: (manual_classes | exact_classes)[name]
            for name in CLASS_ORDER
        },
    }
    decisions = {
        "decision_version": DECISION_VERSION,
        "review_package_fingerprint": package_fingerprint,
        "holdout_integrity": {
            "locked_content_returned_to_implementer": True,
            "mutation_rules_frozen_before_return": False,
            "reviewer_withdrawn_from_mutator_work": True,
        },
        "classes": {
            WRONG_ROW: {
                "reviewed_prefix": 11,
                "rejected_ids": [entries[WRONG_ROW][7]["case_id"]],
                "unresolved_ids": [],
            },
            LABEL_VALUE_SWAP: {
                "reviewed_prefix": 34,
                "rejected_ids": [],
                "unresolved_ids": [],
            },
            QUESTION_ANSWER_MISMATCH: {
                "reviewed_prefix": 31,
                "rejected_ids": [],
                "unresolved_ids": [
                    entries[QUESTION_ANSWER_MISMATCH][13]["case_id"]
                ],
            },
        },
    }
    return aggregate, package, decisions


def test_review_overlay_applies_stopping_rules_and_exact_deficit():
    aggregate, package, decisions = _inputs()

    report = apply_review_decisions(aggregate, package, decisions)

    assert report["eligibility_summary"] == {
        "candidate_powered_classes": 7,
        "underpowered_classes": 1,
        "total_classes": 8,
    }
    wrong = report["classes"][WRONG_ROW]
    assert wrong["status"] == "underpowered"
    assert wrong["confirmed_eligible_locked_clusters"] == 10
    assert wrong["confirmed_eligible_total_clusters"] == 10
    assert wrong["confirmed_eligible_total_clusters_is_lower_bound"] is True
    assert wrong["confirmed_locked_cluster_deficit"] == 20
    assert wrong["minimum_unavoidable_locked_cluster_deficit"] == 20
    assert wrong["exact_deficit_known"] is True
    assert wrong["expected_additional_eligible_base_questions"] == 30
    assert wrong["manual_review"]["first_preregistered_stop"] == {
        "outcome": "confirmation_target_reached",
        "at_reviewed_candidate": 11,
    }

    label = report["classes"][LABEL_VALUE_SWAP]
    assert label["status"] == "candidate_powered"
    assert label["confirmed_eligible_locked_clusters"] == 34
    assert label["manual_review"]["reviewed_after_first_stop"] == 4
    assert label["manual_review"]["confirmed_at_first_stop"] == 30
    assert label["manual_review"]["confirmed_after_first_stop"] == 4
    assert label["manual_review"]["protocol_clean"] is False

    mismatch = report["classes"][QUESTION_ANSWER_MISMATCH]
    assert mismatch["status"] == "candidate_powered"
    assert mismatch["confirmed_eligible_locked_clusters"] == 30
    assert mismatch["manual_review"]["unresolved_locked_clusters"] == 1
    assert mismatch["manual_review"]["unreviewed_locked_clusters"] == 21
    assert mismatch["manual_review"]["confirmed_at_first_stop"] == 30
    assert mismatch["manual_review"]["unresolved_at_first_stop"] == 1
    assert mismatch["manual_review"]["protocol_clean"] is False
    assert mismatch["manual_review"]["first_preregistered_stop"] == {
        "outcome": "confirmation_target_reached",
        "at_reviewed_candidate": 31,
    }


def test_locked_content_exposure_invalidates_final_blind_gate_not_feasibility():
    aggregate, package, decisions = _inputs()

    report = apply_review_decisions(aggregate, package, decisions)

    assert report["holdout_integrity"] == {
        "locked_content_returned_to_implementer": True,
        "mutation_rules_frozen_before_return": False,
        "reviewer_withdrawn_from_mutator_work": True,
        "blinding_breached_before_mutator_implementation": True,
        "eligibility_measurement_valid": True,
        "current_locked_allowed_for_final_guard_recall_gate": False,
        "current_locked_use": "feasibility_and_development_only",
        "new_unseen_holdout_required_before_guard_recall_gate": True,
    }
    assert report["protocol_deviations"] == [
        {
            "code": "manual_review_continued_after_preregistered_stop",
            "classes": {LABEL_VALUE_SWAP: 4},
        },
        {
            "code": "review_returned_unregistered_unresolved_disposition",
            "classes": {QUESTION_ANSWER_MISMATCH: 1},
        },
        {
            "code": "locked_content_returned_beyond_aggregate_decisions",
        },
    ]


def test_derived_aggregate_copies_no_private_review_content():
    aggregate, package, decisions = _inputs()

    encoded = json.dumps(
        apply_review_decisions(aggregate, package, decisions),
        ensure_ascii=False,
    )

    assert "KURGU_OZEL_INCELEME_METNI" not in encoded
    assert "KURGU_GIZLI_SORU" not in encoded
    assert "KURGU_GIZLI_PASAJ" not in encoded


def test_decisions_outside_reviewed_prefix_are_rejected():
    aggregate, package, decisions = _inputs()
    decisions["classes"][QUESTION_ANSWER_MISMATCH][
        "rejected_ids"
    ] = [package["entries"][QUESTION_ANSWER_MISMATCH][40]["case_id"]]

    with pytest.raises(ValueError, match="reviewed prefix"):
        apply_review_decisions(aggregate, package, decisions)


def test_review_package_fingerprint_must_match():
    aggregate, package, decisions = _inputs()
    package["cases"]["private"] = "KURGU_DEGISTIRILMIS_GIZLI_SORU"

    with pytest.raises(ValueError, match="fingerprint"):
        apply_review_decisions(aggregate, package, decisions)


def test_holdout_integrity_declaration_is_mandatory():
    aggregate, package, decisions = _inputs()
    decisions["holdout_integrity"].pop(
        "locked_content_returned_to_implementer"
    )

    with pytest.raises(ValueError, match="integrity"):
        apply_review_decisions(aggregate, package, decisions)
