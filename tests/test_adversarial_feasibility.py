import json
from dataclasses import replace

import pytest

from eval.answer.adversarial_feasibility import (
    ARITHMETIC_RESTATEMENT,
    CORRUPTED_QUOTE,
    FORCED_ANSWER_UNANSWERABLE,
    INVENTED_PROVENANCE,
    LABEL_VALUE_SWAP,
    QUESTION_ANSWER_MISMATCH,
    UNSUPPORTED_FIGURE,
    WRONG_ROW,
    _class_report,
    attach_review_metadata,
    build_manual_review_package,
    build_report,
    checked_output_path,
    eligible_sets,
    expected_additional_questions,
    load_cases,
    make_case,
)
from pipeline.retrieval.query import build_rag_context


PRIMARY_KEY = "KURGU_LAMBDA_KAYDI 864217 birimdir."
PEER_KEY = "KURGU_SIGMA_KAYDI 975328 birimdir."


def _context():
    return build_rag_context(
        [
            {
                "filename": "kurgu.pdf",
                "page": 963521,
                "text": PRIMARY_KEY,
            },
            {
                "filename": "kurgu.pdf",
                "page": 963522,
                "text": PEER_KEY,
            },
            {
                "filename": "kurgu.pdf",
                "page": 963523,
                "text": f"{PRIMARY_KEY}\n{PEER_KEY}",
            },
        ],
        numbered=True,
    )


def _case(question, key, answer, handle, quote, question_type="tablo"):
    context = _context()
    return make_case(
        "kurgu",
        {
            "soru": question,
            "cevap": answer,
            "baglam": context.model_text,
            "dayanak": [{"pasaj": handle, "alinti": quote}],
        },
        {
            "q": question,
            "key": key,
            "type": question_type,
        },
    )


def test_exact_detectors_and_manual_candidate_detectors_stay_separate():
    primary = _case(
        "Birinci kurgu kaydinin degeri nedir?",
        PRIMARY_KEY,
        PRIMARY_KEY,
        1,
        PRIMARY_KEY,
    )
    peer = _case(
        "Ikinci kurgu kaydinin degeri nedir?",
        PEER_KEY,
        PEER_KEY,
        2,
        PEER_KEY,
    )

    confirmed, candidates = eligible_sets((primary, peer))

    for name in (
        UNSUPPORTED_FIGURE,
        ARITHMETIC_RESTATEMENT,
        FORCED_ANSWER_UNANSWERABLE,
        CORRUPTED_QUOTE,
        INVENTED_PROVENANCE,
    ):
        assert primary.stable_id in confirmed[name]
        assert primary.stable_id in candidates[name]

    for name in (WRONG_ROW, LABEL_VALUE_SWAP, QUESTION_ANSWER_MISMATCH):
        assert primary.stable_id not in confirmed[name]
        assert primary.stable_id in candidates[name]


def test_saved_guard_fields_cannot_change_eligibility_input_or_fingerprint():
    context = _context()
    row = {
        "soru": "Birinci kurgu kaydinin degeri nedir?",
        "cevap": PRIMARY_KEY,
        "baglam": context.model_text,
        "dayanak": [{"pasaj": 1, "alinti": PRIMARY_KEY}],
        "guard_status": "answered",
        "bayraklar": [],
    }
    question = {"q": row["soru"], "key": PRIMARY_KEY, "type": "tablo"}
    first = make_case("kurgu", row, question)

    row["guard_status"] = "review_required"
    row["bayraklar"] = [["kurgu_tanisi", []]]
    row["durum"] = "yanlis"
    second = make_case("kurgu", row, question)

    assert first == second
    assert first.source_row_fingerprint == second.source_row_fingerprint


def _locked_cases(template, count):
    return tuple(
        replace(
            template,
            stable_id="5" + f"{index:063x}",
            locked=True,
        )
        for index in range(count)
    )


def test_upper_bound_cannot_make_a_manual_class_candidate_powered():
    template = _case(
        "Birinci kurgu kaydinin degeri nedir?",
        PRIMARY_KEY,
        PRIMARY_KEY,
        1,
        PRIMARY_KEY,
    )
    cases = _locked_cases(template, 30)
    candidate_ids = {case.stable_id for case in cases}

    report = _class_report(
        LABEL_VALUE_SWAP,
        cases,
        confirmed_ids=set(),
        candidate_ids=candidate_ids,
    )

    assert report["status"] == "pending_manual_confirmation"
    assert report["exact_deficit_known"] is False
    assert report["expected_additional_eligible_base_questions"] is None
    assert report["manual_gate_review_plan"] == {
        "review_order": "stable_id_ascending",
        "review_required": True,
        "confirmation_target": 30,
        "stop_at_confirmation": 30,
        "stop_at_rejection": 1,
    }


def test_manual_gate_stops_when_remaining_candidates_cannot_reach_target():
    template = _case(
        "Birinci kurgu kaydinin degeri nedir?",
        PRIMARY_KEY,
        PRIMARY_KEY,
        1,
        PRIMARY_KEY,
    )
    cases = _locked_cases(template, 52)
    candidate_ids = {case.stable_id for case in cases}

    report = _class_report(
        QUESTION_ANSWER_MISMATCH,
        cases,
        confirmed_ids=set(),
        candidate_ids=candidate_ids,
    )

    assert report["manual_gate_review_plan"]["stop_at_confirmation"] == 30
    assert report["manual_gate_review_plan"]["stop_at_rejection"] == 23


def test_underpowered_upper_bound_has_separate_descriptive_stopping_rule():
    template = _case(
        "Birinci kurgu kaydinin degeri nedir?",
        PRIMARY_KEY,
        PRIMARY_KEY,
        1,
        PRIMARY_KEY,
    )
    cases = _locked_cases(template, 30)
    candidate_ids = {case.stable_id for case in cases[:11]}

    report = _class_report(
        WRONG_ROW,
        cases,
        confirmed_ids=set(),
        candidate_ids=candidate_ids,
    )

    assert report["status"] == "underpowered"
    assert report["minimum_unavoidable_locked_cluster_deficit"] == 19
    assert report["expected_additional_eligible_base_questions_floor"] == 28
    assert report["manual_gate_review_plan"]["review_required"] is False
    assert report["manual_descriptive_review_plan"] == {
        "review_order": "stable_id_ascending",
        "review_required": True,
        "confirmation_target": 10,
        "stop_at_confirmation": 10,
        "stop_at_rejection": 2,
        "changes_gate_status": False,
    }


def test_exact_class_reaches_candidate_powered_only_on_confirmed_clusters():
    template = _case(
        "Birinci kurgu kaydinin degeri nedir?",
        PRIMARY_KEY,
        PRIMARY_KEY,
        1,
        PRIMARY_KEY,
        question_type="sayisal",
    )
    cases = _locked_cases(template, 30)
    confirmed_ids = {case.stable_id for case in cases}

    report = _class_report(
        UNSUPPORTED_FIGURE,
        cases,
        confirmed_ids=confirmed_ids,
        candidate_ids=confirmed_ids,
    )

    assert report["status"] == "candidate_powered"
    assert report["confirmed_locked_cluster_deficit"] == 0
    assert report["expected_additional_eligible_base_questions"] == 0


def test_expected_additional_question_estimate_is_explicitly_split_based():
    assert expected_additional_questions(0) == 0
    assert expected_additional_questions(5) == 8
    with pytest.raises(ValueError):
        expected_additional_questions(-1)


def test_loader_excludes_sweeps_and_raw_guard_fields_from_eligibility_hash(
        tmp_path):
    run_dir = tmp_path / "run"
    question_dir = tmp_path / "questions"
    run_dir.mkdir()
    question_dir.mkdir()
    context = _context()
    row = {
        "soru": "Birinci kurgu kaydinin degeri nedir?",
        "cevap": PRIMARY_KEY,
        "baglam": context.model_text,
        "dayanak": [{"pasaj": 1, "alinti": PRIMARY_KEY}],
        "guard_status": "answered",
    }
    payload = {"sorular": [row]}
    result_path = run_dir / "rag_answers_kurgu.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "rag_answers_kurgu_k15.json").write_text(
        json.dumps({"sorular": []}),
        encoding="utf-8",
    )
    (question_dir / "kurgu.json").write_text(
        json.dumps([{
            "q": row["soru"],
            "key": PRIMARY_KEY,
            "type": "sayisal",
            "pages": [963521],
        }]),
        encoding="utf-8",
    )

    cases, first = load_cases(run_dir, question_dir)
    row["guard_status"] = "review_required"
    payload["sorular"][0]["bayraklar"] = [["kurgu_tanisi", []]]
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    _, second = load_cases(run_dir, question_dir)

    assert len(cases) == 1
    assert first["result_files"] == 1
    assert first["raw_result_files_fingerprint"] != (
        second["raw_result_files_fingerprint"]
    )
    assert first["eligibility_input_fingerprint"] == (
        second["eligibility_input_fingerprint"]
    )


def test_aggregate_report_does_not_persist_private_case_text():
    primary = _case(
        "Birinci kurgu kaydinin degeri nedir?",
        PRIMARY_KEY,
        PRIMARY_KEY,
        1,
        PRIMARY_KEY,
    )
    report = build_report((primary,), {"repository_head": None})
    encoded = json.dumps(report, ensure_ascii=False)

    assert report["evaluation_layer"] == "generator_guard"
    assert report["three_layer_scope"] == {
        "retrieval": "recorded_input_condition_not_remeasured",
        "generator_guard": "eligibility_inventory",
        "end_to_end": "not_measured",
    }
    assert report["guard_results_used_for_eligibility"] is False
    assert primary.question not in encoded
    assert primary.key not in encoded
    assert primary.answer not in encoded


def test_private_review_package_is_ordered_and_aggregate_keeps_only_metadata():
    primary = replace(
        _case(
            "Birinci kurgu kaydinin degeri nedir?",
            PRIMARY_KEY,
            PRIMARY_KEY,
            1,
            PRIMARY_KEY,
        ),
        stable_id="5" + "0" * 63,
        locked=True,
    )
    peer = replace(
        _case(
            "Ikinci kurgu kaydinin degeri nedir?",
            PEER_KEY,
            PEER_KEY,
            2,
            PEER_KEY,
        ),
        stable_id="6" + "0" * 63,
        locked=True,
    )
    report = build_report(
        (primary, peer),
        {"eligibility_input_fingerprint": "a" * 64},
    )

    package = build_manual_review_package((primary, peer), report)
    aggregate = attach_review_metadata(report, package)
    private_text = json.dumps(package, ensure_ascii=False)
    aggregate_text = json.dumps(aggregate, ensure_ascii=False)

    assert package["private_document_content"] is True
    assert package["entry_counts"] == {
        WRONG_ROW: 2,
        LABEL_VALUE_SWAP: 2,
        QUESTION_ANSWER_MISMATCH: 2,
    }
    assert [
        item["case_id"]
        for item in package["entries"][LABEL_VALUE_SWAP]
    ] == sorted((primary.stable_id, peer.stable_id))
    assert all(
        item["decision"] is None
        for entries in package["entries"].values()
        for item in entries
    )
    assert PRIMARY_KEY in private_text
    assert PRIMARY_KEY not in aggregate_text
    assert aggregate["manual_review_package"][
        "private_document_content_in_aggregate"
    ] is False


def test_report_path_must_stay_under_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    allowed = checked_output_path(tmp_path / "output" / "report.json")
    assert allowed == (tmp_path / "output" / "report.json").resolve()
    with pytest.raises(ValueError):
        checked_output_path(tmp_path / "report.json")

    allowed.parent.mkdir()
    allowed.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        checked_output_path(allowed)
