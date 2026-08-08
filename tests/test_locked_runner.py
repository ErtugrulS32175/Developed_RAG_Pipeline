"""The locked runner must fail closed, measure only locked clusters, and
leak nothing it read.

Fixtures use the invented, absence-verified vocabulary of the mutation
tests. LOCKED ids are built with the real ``stable_question_id`` so the
identity arithmetic is the production one, not a lookalike.
"""
import hashlib
import json

import pytest

from eval.answer import locked_runner
from eval.answer.adversarial_feasibility import stable_question_id
from eval.answer.adversarial_mutate import (
    LockedGate,
    build_mutants,
    mutate_unsupported_figure,
    verify_contract_chain,
)
from eval.answer.locked_runner import (
    _assert_content_free,
    _assert_report_schema,
    _gate_summary,
    locked_cases,
    one_sided_upper_bound,
    run_locked,
)
from pipeline.retrieval.context import Passage
from tests.test_adversarial_mutate import DEV_ID, LOCKED_ID, ZETA, _case


def _chain(tmp_path, monkeypatch=None,
           base_body="Protocol: `kurgu_protokol_v9`\n"):
    """A complete contract chain in a temp directory, plus its statement.

    The runner pins the real addendum's version names and base hash; a
    synthetic chain can only exist under test by pointing those pins at the
    fixture -- which is itself a proof that the pins bite."""
    base = tmp_path / "adversarial_contract.md"
    base.write_text(base_body, encoding="utf-8")
    addendum = tmp_path / "adversarial_contract_ek.md"
    addendum.write_text("Protocol: `kurgu_protokol_v9.1`\n", encoding="utf-8")
    # tmp_path doubles as the question directory; the runner requires it
    # to hold exactly as many question files as were measured (1 here)
    (tmp_path / "kurguseti.json").write_text("[]", encoding="utf-8")
    statement = {
        "expected_version": "kurgu_protokol_v9.1",
        "expected_contract_sha256": hashlib.sha256(
            base.read_bytes()).hexdigest(),
        "expected_addenda": {
            "adversarial_contract_ek.md": hashlib.sha256(
                addendum.read_bytes()).hexdigest()},
    }
    if monkeypatch is not None:
        monkeypatch.setattr(locked_runner, "REQUIRED_EFFECTIVE_VERSION",
                            "kurgu_protokol_v9.1")
        monkeypatch.setattr(locked_runner, "REQUIRED_BASE_VERSION",
                            "kurgu_protokol_v9")
        monkeypatch.setattr(locked_runner, "REQUIRED_BASE_SHA256",
                            statement["expected_contract_sha256"])
    return statement


# --- the addendum's mandatory first regression test ------------------------

def test_incomplete_chain_loads_nothing(tmp_path, monkeypatch):
    """contract_complete=False must stop the run BEFORE any case file is
    opened. Loading is instrumented: if the gate ever lets execution reach
    it, this test fails on the sentinel, not on a downstream error."""
    orphan = tmp_path / "adversarial_contract_ek.md"
    orphan.write_text("Protocol: `kurgu_protokol_v9.1`\n", encoding="utf-8")

    loaded = []
    monkeypatch.setattr(locked_runner, "load_cases",
                        lambda *a, **k: loaded.append(1))
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    monkeypatch.setattr(locked_runner, "REQUIRED_EFFECTIVE_VERSION",
                        "kurgu_protokol_v9.1")
    monkeypatch.setattr(locked_runner, "REQUIRED_BASE_SHA256", "x" * 64)
    with pytest.raises(ValueError):
        run_locked(tmp_path, tmp_path, "kurgu_protokol_v9.1", "x" * 64, {},
                   **_screening_args(monkeypatch))
    assert loaded == []


def test_every_wrong_statement_refuses(tmp_path, monkeypatch):
    """Wrong version, wrong base hash, missing addendum, tampered addendum
    hash, and an UNDECLARED addendum on disk: each one refuses."""
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)

    good = dict(statement)
    verify_contract_chain(good["expected_version"],
                          good["expected_contract_sha256"],
                          good["expected_addenda"])  # sanity: chain holds

    with pytest.raises(ValueError):
        verify_contract_chain("baska_surum",
                              good["expected_contract_sha256"],
                              good["expected_addenda"])
    with pytest.raises(ValueError):
        verify_contract_chain(good["expected_version"], "f" * 64,
                              good["expected_addenda"])
    with pytest.raises(ValueError):
        verify_contract_chain(good["expected_version"],
                              good["expected_contract_sha256"], {})
    tampered = {"adversarial_contract_ek.md": "0" * 64}
    with pytest.raises(ValueError):
        verify_contract_chain(good["expected_version"],
                              good["expected_contract_sha256"], tampered)
    extra = dict(good["expected_addenda"])
    extra["adversarial_contract_yok.md"] = "1" * 64
    with pytest.raises(ValueError):
        verify_contract_chain(good["expected_version"],
                              good["expected_contract_sha256"], extra)


# --- membership inversion --------------------------------------------------

def _locked_case(**overrides):
    question = overrides.pop("question", "kurgu kilitli soru nedir?")
    set_name = overrides.pop("set_name", "kurgu")
    stable_id = stable_question_id(set_name, question)
    if not locked_runner.is_locked(stable_id):
        # nudge deterministically into the locked half of the split
        question += " (ek)"
        stable_id = stable_question_id(set_name, question)
    assert locked_runner.is_locked(stable_id)
    return _case(stable_id, question=question, set_name=set_name, **overrides)


def test_the_gate_inverts_membership():
    gate = LockedGate("kurgu_protokol_v9.1", "a" * 64, ())
    locked = _locked_case()
    dev = _case(DEV_ID)
    # without a gate: locked refused (the standing rule)
    with pytest.raises(ValueError):
        mutate_unsupported_figure(locked, ())
    # with a gate: locked accepted, DEVELOPMENT refused
    assert mutate_unsupported_figure(locked, (), gate) is not None
    with pytest.raises(ValueError):
        mutate_unsupported_figure(dev, (), gate)
    with pytest.raises(ValueError):
        build_mutants((dev,), gate)


def test_a_non_gate_object_is_refused():
    class Fake:
        pass

    with pytest.raises(ValueError):
        mutate_unsupported_figure(_locked_case(), (), Fake())


def test_locked_cases_filter_recomputes_identity():
    import dataclasses

    kept = locked_cases((_case(DEV_ID), _case(LOCKED_ID)))
    assert [case.stable_id for case in kept] == [LOCKED_ID]
    forged = dataclasses.replace(_case(LOCKED_ID), locked=False)
    with pytest.raises(ValueError):
        locked_cases((forged,))


# --- gate arithmetic -------------------------------------------------------

def test_provenance_classes_are_scored_on_the_expected_diagnostic():
    """Auditor finding, round 14: the gate counted ANY flag as a catch. On
    a probe where all 30 mutants were held for SOME reason but the correct
    diagnostic fired 0 times, the scorer reported missed=0 and a 9.5% upper
    bound where true diagnostic recall was 0/30 -- overstating the guard's
    value. The contract's basis decides: classes with a preregistered
    diagnostic count only that code; None-classes keep the safety catch."""
    template = {name: {"population_role": "confirmed_structural",
                       "structural_recall_eligible": True, "policies": {}}
                for name in locked_runner.CLASS_ORDER}
    template["unsupported_figure"]["policies"] = {
        "plain": {"counts": {"n": 30, "caught": 30, "expected_diagnostic": 0}}}
    template["forced_answer_unanswerable"]["policies"] = {
        "plain": {"counts": {"n": 10, "caught": 7, "expected_diagnostic": 0}}}
    summary = _gate_summary(template)

    provenance = summary["unsupported_figure"]["policies"]["plain"]
    assert provenance["recall_basis"] == "expected_diagnostic"
    assert provenance["caught"] == 0 and provenance["missed"] == 30
    assert provenance["caught_any_flag"] == 30  # the gap stays visible
    assert provenance["miss_rate_upper_95"] == 1.0

    safety = summary["forced_answer_unanswerable"]["policies"]["plain"]
    assert safety["recall_basis"] == "safety_catch"
    assert safety["caught"] == 7 and safety["missed"] == 3


def test_upper_bound_reproduces_the_contract_anchor():
    assert one_sided_upper_bound(0, 30) == pytest.approx(0.095, abs=5e-4)
    assert one_sided_upper_bound(0, 10) == pytest.approx(0.2589, abs=5e-4)
    assert one_sided_upper_bound(2, 2) == 1.0
    with pytest.raises(ValueError):
        one_sided_upper_bound(-1, 30)
    with pytest.raises(ValueError):
        one_sided_upper_bound(3, 2)
    with pytest.raises(ValueError):
        one_sided_upper_bound(0, 0)


# --- the report leaks nothing ----------------------------------------------

def test_a_report_carrying_case_text_is_refused():
    case = _locked_case()
    clean = {"classes": {"wrong_row": {"clusters": 3, "caught": 3}},
             "ids": [case.stable_id]}
    _assert_content_free(clean, (case,))  # counts and opaque ids pass

    for leak in (case.answer, case.question, ZETA,
                 case.answer[8:31],           # even a partial quote
                 case.answer.upper(),         # re-cased copy: fold catches it
                 ):
        dirty = dict(clean)
        dirty["aciklama"] = f"kume su nedenle elendi: {leak}"
        with pytest.raises(ValueError):
            _assert_content_free(dirty, (case,))


SOURCE_STATEMENT = {"raw_result_files_fingerprint": "e" * 64,
                    "question_files_fingerprint": "f" * 64}
# the schema requires the source section to be EXACTLY what load_cases
# emits -- six fields, no more, no fewer (round 18: removable counters
# and smuggled extra fingerprints both passed the loose version)
SOURCE_METADATA = {"result_files": 1, "question_files": 1,
                   "eligibility_input_fingerprint": "a" * 64,
                   "split_manifest_fingerprint": "b" * 64,
                   **SOURCE_STATEMENT}


def _screening_args(monkeypatch, locked_cases=1):
    """The FULL owner manifest the runner now demands -- nothing optional.

    The clean-tree check is patched, not parameterised: round 17 removed
    the ``require_clean_tree`` keyword because a public bypass on a locked
    gate is a hole, whoever it was built for."""
    monkeypatch.setattr(locked_runner, "repository_head", lambda: "c" * 40)
    monkeypatch.setattr(locked_runner, "_require_clean_tree", lambda: None)
    return {
        "expected_source_fingerprints": dict(SOURCE_STATEMENT),
        "expected_locked_cases": locked_cases,
        "expected_head": "c" * 40,
    }


def _screened_report(tmp_path, monkeypatch):
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    locked = _locked_case(
        passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
        evidence=[(1, ZETA)])
    monkeypatch.setattr(
        locked_runner, "load_cases",
        lambda *a, **k: ((locked, _case(DEV_ID)), dict(SOURCE_METADATA)))
    return run_locked(tmp_path, tmp_path, statement["expected_version"],
                      statement["expected_contract_sha256"],
                      statement["expected_addenda"],
                      **_screening_args(monkeypatch))


def test_the_schema_refuses_what_the_blacklist_could_not(
        tmp_path, monkeypatch):
    """Rounds 14+16: a SHORT key, a set name, a negative count, a NaN
    bound, an EMPTY class table -- each slipped some earlier check. The
    allowlist schema refuses them by construction, and the valid baseline
    is a REAL run's report, so the schema can never drift away from what
    the runner actually writes."""
    import copy

    baseline = _screened_report(tmp_path, monkeypatch)
    _assert_report_schema(baseline)  # sanity: the real shape passes

    def poke(mutate):
        report = copy.deepcopy(baseline)
        mutate(report)
        with pytest.raises(ValueError):
            _assert_report_schema(report)

    # an unknown top field -- however innocent-looking
    poke(lambda r: r.__setitem__(
        "aciklama", "kume elendi cunku anahtar 47 000 idi"))
    # a free string where a version belongs
    poke(lambda r: r["contract"].__setitem__(
        "contract_version", "sayfa 42 kurgu kume"))
    # an unknown source field cannot smuggle a set name
    poke(lambda r: r["source"].__setitem__("set_name", "kurgu"))
    # an incomplete chain can never be a locked report
    poke(lambda r: r["contract"].__setitem__("contract_complete", False))
    # round 16: a negative counter is arithmetic that cannot have happened
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         ["plain"].update({"caught": -1, "missed": 2}))
    # round 16: inconsistent cell arithmetic
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         ["plain"].__setitem__("missed", 5))
    # round 16: a NaN bound is not a bound
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         ["plain"].update({"miss_rate_upper_95": float("nan")}))
    # round 16: an empty or thinned class table is a report about nothing
    poke(lambda r: r["classes"].pop("wrong_row"))
    poke(lambda r: r["gate_summary"].pop("wrong_row"))
    # round 16: a missing mandatory source fingerprint
    poke(lambda r: r["source"].pop("question_files_fingerprint"))
    # round 17: an ABBREVIATED hash answered for a sha256 / a HEAD
    poke(lambda r: r["contract"].__setitem__("contract_sha256", "abc123" * 2))
    poke(lambda r: r.__setitem__("repository_head", "c" * 12))
    poke(lambda r: r.__setitem__("repository_head", None))
    poke(lambda r: r["source"].__setitem__(
        "raw_result_files_fingerprint", "e" * 12))
    # round 17: {} used to be a valid cell -- every field is mandatory now
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         ["plain"].pop("meets_cluster_floor"))
    # round 17: the frozen matrix admits neither missing nor extra cells
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         .pop("plain"))
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]
         .pop("plain"))
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         .__setitem__("binding_probe", dict(
             r["gate_summary"]["unsupported_figure"]["policies"]["plain"])))
    # round 17: a bound below the cluster floor is a number the frozen
    # text says may not exist -- presence itself is refused
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         ["plain"].__setitem__("miss_rate_upper_95", 0.095))
    # round 17: the floor flag must match its own cluster count
    poke(lambda r: r["gate_summary"]["unsupported_figure"]["policies"]
         ["plain"].__setitem__("meets_cluster_floor", True))
    # round 17: a role or basis that contradicts the class contract
    poke(lambda r: r["gate_summary"]["wrong_row"]
         .update({"population_role": "confirmed_structural",
                  "structural_recall_eligible": True}))
    poke(lambda r: r["classes"]["unsupported_figure"]
         .__setitem__("expected_diagnostic", "baska_kod"))
    # round 17: the two sections may not contradict each other
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]["plain"]
         ["counts"].__setitem__("n", 5))
    # round 18, ten independent inconsistencies a real-report fixture
    # still accepted -- each now refused:
    # 1-2: the source statement is an exact set (no removable counters,
    # no smuggled extra fingerprints)
    poke(lambda r: r["source"].pop("result_files"))
    poke(lambda r: r["source"].pop("question_files"))
    poke(lambda r: r["source"].__setitem__("ek_fingerprint", "c" * 64))
    # 3: classes-side eligibility can be neither deleted nor inverted
    poke(lambda r: r["classes"]["wrong_row"].pop("structural_recall_eligible"))
    poke(lambda r: r["classes"]["wrong_row"]
         .__setitem__("structural_recall_eligible", True))
    # 4: mutant / published / control counters are re-derived
    poke(lambda r: r["classes"]["unsupported_figure"]
         .__setitem__("mutants", 9))
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]["plain"]
         ["counts"].__setitem__("published", 7))
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]["plain"]
         ["counts"].__setitem__("control_review", 9))
    # 5: a deleted or contradictory flag distribution
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]["plain"]
         ["flags"].clear())
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]["plain"]
         .pop("flags"))
    # 6: an extra gate-body field
    poke(lambda r: r["gate_summary"]["unsupported_figure"]
         .__setitem__("aciklama_alani", 1))
    # counters must be a complete set when non-empty
    poke(lambda r: r["classes"]["unsupported_figure"]["policies"]["plain"]
         ["counts"].pop("control_published"))


def test_git_operations_are_anchored_to_the_repo_not_the_cwd(monkeypatch):
    """Round 18: clean-tree and HEAD ran git in the CWD -- called from a
    different clean repository, the wrong repo was blessed and a foreign
    HEAD reported. Both now pin cwd to this module's repository root."""
    import subprocess

    import eval.answer.adversarial_feasibility as feas

    seen = {}

    class _Clean:
        returncode = 0
        stdout = ""

    def recording_run(args, **kwargs):
        seen[tuple(args)] = kwargs.get("cwd")
        return _Clean()

    monkeypatch.setattr(subprocess, "run", recording_run)
    locked_runner._require_clean_tree()
    feas.repository_head()
    for args, cwd in seen.items():
        assert cwd == feas.REPO_ROOT, args


def test_a_two_character_set_name_cannot_ride_out(tmp_path):
    """Round 18: the first name gate started at three characters and a
    two-character locked set name slipped through. Two is the floor now;
    purely numeric stays out because counts are numbers."""
    case = _locked_case(set_name="k7")
    clean = {"classes": {"wrong_row": {"clusters": 3}}}
    _assert_content_free(clean, (case,))
    dirty = dict(clean, kaynak="k7")
    with pytest.raises(ValueError):
        _assert_content_free(dirty, (case,))


def test_a_passage_source_cannot_ride_out(tmp_path):
    """Passage sources carry document identity (file + page label); the
    content gate covers them as whole tokens like set names."""
    case = _locked_case(
        passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
        evidence=[(1, ZETA)])
    clean = {"classes": {"wrong_row": {"clusters": 3}}}
    _assert_content_free(clean, (case,))
    dirty = dict(clean, kaynak_etiketi="kurgu-belge s.42")
    with pytest.raises(ValueError):
        _assert_content_free(dirty, (case,))


def test_a_bound_value_must_be_the_exact_recomputed_binomial():
    """Round 17: the schema accepted any finite number in [0,1] as a
    bound. The value is re-derived now -- a hand-edited 'better' bound is
    a refused report."""
    cell = {"clusters": 30, "caught": 30, "caught_any_flag": 30,
            "recall_basis": "expected_diagnostic", "missed": 0,
            "meets_cluster_floor": True,
            "miss_rate_upper_95": one_sided_upper_bound(0, 30)}
    locked_runner._check_policy_cell(cell, "kurgu", True,
                                     "expected_diagnostic")  # exact: passes
    for wrong in (0.05, 0.0949, 0.5):
        broken = dict(cell, miss_rate_upper_95=wrong)
        with pytest.raises(ValueError, match="yeniden hesaplanan"):
            locked_runner._check_policy_cell(broken, "kurgu", True,
                                             "expected_diagnostic")
    # and a diagnostic count above the any-flag count cannot have happened
    impossible = dict(cell, caught=31, clusters=31, caught_any_flag=30,
                      missed=0)
    with pytest.raises(ValueError):
        locked_runner._check_policy_cell(impossible, "kurgu", True,
                                         "expected_diagnostic")


def test_an_unmeasured_question_set_in_the_directory_refuses(
        tmp_path, monkeypatch):
    """Round 17: a question set with NO result file was silently
    not-loaded -- absent from every fingerprint, invisible to every
    count. The locked runner requires its question directory to BE the
    measured population, nothing more."""
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    locked = _locked_case(
        passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
        evidence=[(1, ZETA)])
    monkeypatch.setattr(
        locked_runner, "load_cases",
        lambda *a, **k: ((locked,), dict(SOURCE_METADATA)))
    (tmp_path / "kurgudisi.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="birebir degil"):
        run_locked(tmp_path, tmp_path, statement["expected_version"],
                   statement["expected_contract_sha256"],
                   statement["expected_addenda"],
                   **_screening_args(monkeypatch))


def test_the_clean_tree_check_has_no_public_bypass():
    """Round 17: ``require_clean_tree=False`` was a public switch on a
    locked gate. The signature no longer carries it; tests patch the
    checker, production callers cannot."""
    import inspect

    parameters = inspect.signature(run_locked).parameters
    assert "require_clean_tree" not in parameters


def test_a_set_name_cannot_ride_out_in_the_report():
    """Round 17: the sliding fragment windows start at 12 characters and a
    SHORT set name slipped between them. Names are content at any length."""
    case = _locked_case()
    clean = {"classes": {"wrong_row": {"clusters": 3}},
             "ids": [case.stable_id]}
    _assert_content_free(clean, (case,))
    dirty = dict(clean, kaynak_kumesi=case.set_name)
    with pytest.raises(ValueError):
        _assert_content_free(dirty, (case,))
    # embedded in prose it is still the name
    prose = dict(clean, aciklama=f"su kumeden geldi: {case.set_name} (3)")
    with pytest.raises(ValueError):
        _assert_content_free(prose, (case,))


def test_the_report_head_is_the_single_verified_snapshot(
        tmp_path, monkeypatch):
    """Round 17: HEAD was read twice -- verified once, reported from a
    SECOND read, and a probe that moved HEAD between them put an
    unverified commit in the report. One snapshot serves both."""
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    locked = _locked_case(
        passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
        evidence=[(1, ZETA)])
    monkeypatch.setattr(
        locked_runner, "load_cases",
        lambda *a, **k: ((locked,), dict(SOURCE_METADATA)))
    monkeypatch.setattr(locked_runner, "_require_clean_tree", lambda: None)
    heads = iter(["c" * 40, "d" * 40, "e" * 40])
    monkeypatch.setattr(locked_runner, "repository_head",
                        lambda: next(heads))
    report = run_locked(tmp_path, tmp_path, statement["expected_version"],
                        statement["expected_contract_sha256"],
                        statement["expected_addenda"],
                        expected_source_fingerprints=dict(SOURCE_STATEMENT),
                        expected_locked_cases=1,
                        expected_head="c" * 40)
    assert report["repository_head"] == "c" * 40  # never the second read


def test_run_locked_end_to_end_on_synthetic_locked_cases(tmp_path, monkeypatch):
    """The happy path, fully synthetic: complete chain, locked-only
    population, mutants built, the FROZEN A/B/C matrix for every class,
    bounds only where structural recall is eligible AND the cluster floor
    is met, and the written report free of case text."""
    report = _screened_report(tmp_path, monkeypatch)
    assert report["population"] == "locked_clusters_only"
    assert report["locked_cases"] == 1
    assert report["development_cases_excluded"] == 1
    summary = report["gate_summary"]
    caught = summary["unsupported_figure"]["policies"]["plain"]
    # ONE cluster: the counters are right, but no bound may exist below the
    # contract's 30-cluster floor -- an auditor probe showed 29/29 passing
    # as gate-eligible before this field existed
    assert caught == {"clusters": 1, "caught": 1, "caught_any_flag": 1,
                      "recall_basis": "expected_diagnostic", "missed": 0,
                      "meets_cluster_floor": False}
    # the FROZEN matrix: A/B/C present for EVERY class, evidence-only ones
    # included -- "A cannot see this class" is a mandatory recorded result
    for name in ("corrupted_quote", "invented_provenance"):
        assert set(summary[name]["policies"]) == {
            "plain", "structured_derived", "structured_explicit_page"}
    # and no experimental binding cells in the gate table
    for name, body in summary.items():
        assert all("binding" not in policy for policy in body["policies"])
    # candidate classes carry no bound, by contract
    for policy_cell in summary["wrong_row"]["policies"].values():
        assert "miss_rate_upper_95" not in policy_cell
    assert not summary["wrong_row"]["structural_recall_eligible"]
    # addendum entries carry their DECLARED protocol -- round 16: it used
    # to vanish, leaving the effective version unexplained
    addenda = report["contract"]["addenda"]
    assert addenda["adversarial_contract_ek.md"]["protocol"] == (
        "kurgu_protokol_v9.1")
    # and nothing the runner read appears in the serialized report
    blob = json.dumps(report, ensure_ascii=False)
    assert ZETA[:20] not in blob
    assert "kurgu kilitli soru" not in blob


def test_the_cluster_floor_gates_the_bound():
    """29 clusters with zero misses is NOT a gate number: the contract's
    floor is 30, and the auditor's 29/29 probe passed before this test."""
    template = {name: {"population_role": "confirmed_structural",
                       "structural_recall_eligible": True, "policies": {}}
                for name in locked_runner.CLASS_ORDER}
    template["unsupported_figure"]["policies"] = {
        "plain": {"counts": {"n": 29, "caught": 29,
                             "expected_diagnostic": 29}}}
    template["forced_answer_unanswerable"]["policies"] = {
        "plain": {"counts": {"n": 30, "caught": 30,
                             "expected_diagnostic": 0}}}
    summary = _gate_summary(template)
    below = summary["unsupported_figure"]["policies"]["plain"]
    assert below["meets_cluster_floor"] is False
    assert "miss_rate_upper_95" not in below
    at_floor = summary["forced_answer_unanswerable"]["policies"]["plain"]
    assert at_floor["meets_cluster_floor"] is True
    assert at_floor["miss_rate_upper_95"] == pytest.approx(0.095, abs=5e-4)


def test_statements_outside_the_addendum_pins_are_refused(tmp_path, monkeypatch):
    """The addendum fixes the version names and base hash; an owner
    statement that contradicts the pins refuses before the disk is read."""
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    args = _screening_args(monkeypatch)
    with pytest.raises(ValueError):
        run_locked(tmp_path, tmp_path, "baska_surum",
                   statement["expected_contract_sha256"],
                   statement["expected_addenda"], **args)
    with pytest.raises(ValueError):
        run_locked(tmp_path, tmp_path, statement["expected_version"],
                   "f" * 64, statement["expected_addenda"], **args)


def test_every_missing_or_wrong_manifest_statement_refuses(
        tmp_path, monkeypatch):
    """Round 16: optional verification is no verification. Missing source
    fingerprints, a missing case count, a missing HEAD, a wrong
    fingerprint, a wrong count and a wrong HEAD each refuse -- the silent
    96-to-95 drop the auditor demonstrated cannot pass any of these."""
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    locked = _locked_case(
        passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
        evidence=[(1, ZETA)])
    monkeypatch.setattr(
        locked_runner, "load_cases",
        lambda *a, **k: ((locked,), dict(SOURCE_METADATA)))

    def attempt(**overrides):
        args = _screening_args(monkeypatch)
        args.update(overrides)
        return run_locked(tmp_path, tmp_path,
                          statement["expected_version"],
                          statement["expected_contract_sha256"],
                          statement["expected_addenda"], **args)

    attempt()  # sanity: the complete manifest passes
    for broken in (
        {"expected_source_fingerprints": None},
        {"expected_source_fingerprints": {
            "raw_result_files_fingerprint": "e" * 64}},   # half a statement
        {"expected_source_fingerprints": {
            "raw_result_files_fingerprint": "0" * 64,
            "question_files_fingerprint": "f" * 64}},     # wrong value
        {"expected_locked_cases": None},
        {"expected_locked_cases": 2},                     # the silent 95
        {"expected_head": None},
        {"expected_head": "d" * 40},
    ):
        with pytest.raises(ValueError):
            attempt(**broken)


def test_a_dirty_tree_refuses_the_screening_run(monkeypatch):
    """HEAD only names the running code when the tree is clean."""
    import subprocess

    class Dirty:
        returncode = 0
        stdout = " M pipeline/kurgu.py\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Dirty())
    with pytest.raises(ValueError):
        locked_runner._require_clean_tree()


def test_the_report_carries_the_verified_snapshot_not_the_disk(
        tmp_path, monkeypatch):
    """Auditor finding, round 14 (TOCTOU): the contract was verified, then
    the report re-read the disk -- an addendum swapped in between was
    reported as if the owner had approved it. The report must carry the
    GATE's snapshot; here the swap happens while cases load, and the
    report must still show the verified hash and version."""
    statement = _chain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "eval.answer.adversarial_mutate._CONTRACT_DIR", tmp_path)
    verified_sha = statement["expected_addenda"]["adversarial_contract_ek.md"]

    locked = _locked_case(
        passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
        evidence=[(1, ZETA)])

    def load_and_swap(*_a, **_k):
        (tmp_path / "adversarial_contract_ek.md").write_text(
            "Protocol: `kurgu_protokol_v9.9`\n", encoding="utf-8")
        return ((locked,), dict(SOURCE_METADATA))

    monkeypatch.setattr(locked_runner, "load_cases", load_and_swap)
    report = run_locked(tmp_path, tmp_path, statement["expected_version"],
                        statement["expected_contract_sha256"],
                        statement["expected_addenda"],
                        **_screening_args(monkeypatch))
    contract = report["contract"]
    assert contract["effective_protocol_version"] == "kurgu_protokol_v9.1"
    assert contract["addenda"]["adversarial_contract_ek.md"]["sha256"] == (
        verified_sha)
