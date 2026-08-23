"""PACKAGE B3 -- the runner, as the thing that joins the phases.

NO REAL MODEL IS CALLED ANYWHERE IN THIS FILE, and it is proven rather
than asserted: every binary is a shim written into `tmp_path`, an autouse
recorder judges every process the loop starts, and anything that is
neither a stub, the fixture's git, nor the interpreter the acceptance
registry names fails the test that started it.

WHAT IS TESTED HERE AND NOT ELSEWHERE. Each phase already has its own
battery, and none of them can see the ORDER: the change-set suite proves
a candidate is what the model said it is, the acceptance suite proves a
receipt describes the bytes the commands saw, the application suite
proves a move is transactional. What none of them can prove is that the
runner asked for those things in the one order that makes them mean
anything -- so almost every test below is about a SEQUENCE, a COUNT, or a
thing that must NOT have happened.

EVERY ATTACK TEST PROVES FOUR THINGS SEPARATELY: that the scenario it
meant to build really exists, that the gate it aims at was really
reached, that the refusal came from the gate it names, and how many
processes, mirrors and applications happened. This project has produced
false green evidence from every one of those being assumed.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

import test_agent_loop_b2_changes as legacy
import test_agent_loop_contract as base
from tools.agent_loop import (acceptance, application, audit, changes,
                              contract, execution, flat_workspace, runner,
                              runner_events, schemas)
from tools.agent_loop import process as process_module

REPO = Path(__file__).resolve().parent.parent
STUB_HOLDER = "sahte-bin"

# Reused rather than rebuilt: one definition of "a private workspace root"
# for the whole phase, so two suites cannot drift apart on what a run is
# allowed to touch.
private_runner_root = legacy.private_runner_root

_PASSING_TEST = "def test_kurgu():\n    assert True\n"
_SECOND = "pipeline/ikinci.py"
_FIRST = "pipeline/kurgu.py"


# =====================================================================
# THE WORLD
# =====================================================================

def _git(repo, *args):
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def build_world(tmp_path, index=0, **task_overrides):
    """A real repository, a bound task manifest and two fake binaries.

    NOT a flat workspace: the runner builds its own, and a fixture that
    pre-built one would be testing a workspace nobody's run owns.

    `core.autocrlf` is pinned off for the reason the application suite
    documents at length -- `Path.write_text` spells `\\n` as `\\r\\n` on
    Windows while this machine's system config normalises it back out on
    `git add`, so every tracked file would differ from the baseline the
    flat workspace materialises from raw git objects, and the drift
    precondition would refuse every candidate for a difference the
    fixture invented."""
    repo = tmp_path / f"kurgu-depo-{index}"
    (repo / "pipeline").mkdir(parents=True)
    for argv in (["init", "-q"], ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"],
                 ["config", "core.autocrlf", "false"]):
        _git(repo, *argv)
    (repo / _FIRST).write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pipeline" / "test_gecer.py").write_text(_PASSING_TEST,
                                                     encoding="utf-8")
    (repo / ".gitignore").write_text(f"{contract.STATE_DIR_NAME}/\n",
                                     encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "kurgu")
    baseline = _git(repo, "rev-parse", "HEAD")

    bin_dir = tmp_path / STUB_HOLDER
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / f"cagrilar-{index}.jsonl"
    binaries = {
        "implementer": base._write_fake(
            bin_dir / f"sahte_claude_{index}{base.SHIM_SUFFIX}", record,
            base._implementer_body()),
        "evaluator": base._write_fake(
            bin_dir / f"sahte_codex_{index}{base.SHIM_SUFFIX}", record,
            base._emits(base._code_audit_reply())),
    }
    task = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "objective": "kurgu hedef",
        "baseline_sha": baseline,
        "allowed_paths": ["pipeline/"],
        "forbidden_paths": ["contracts/", "data/"],
        "acceptance_commands": [{"command_id": "pytest_selected",
                                 "paths": ["pipeline/test_gecer.py"]}],
        "acceptance_criteria": ["kurgu olcut"],
        "max_implementation_rounds": 1,
        "max_repair_rounds": 1,
        "max_wall_clock_minutes": 5,
        "max_budget_usd": 1.0,
        "max_output_bytes": 65536,
        "leak_policy": {"command_id": "leak_scan", "max_hard_findings": 0},
        "dirty_tree_allowlist": ["kurgu-task.json"],
    }
    task.update(task_overrides)
    task_path = repo / "kurgu-task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return types.SimpleNamespace(
        repo=repo, task=task, task_path=task_path, baseline=baseline,
        binaries=binaries, record=record,
        digest=hashlib.sha256(task_path.read_bytes()).hexdigest(),
        state_dir=repo / contract.STATE_DIR_NAME)


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def retask(world_obj, **overrides):
    """Rewrite the manifest and rebind the digest the run will be issued."""
    payload = dict(world_obj.task, **overrides)
    world_obj.task_path.write_text(json.dumps(payload), encoding="utf-8")
    world_obj.task = payload
    world_obj.digest = hashlib.sha256(
        world_obj.task_path.read_bytes()).hexdigest()
    return world_obj


def evaluator(world_obj, body):
    return base._write_fake(world_obj.binaries["evaluator"],
                            world_obj.record, body)


def implementer(world_obj, body):
    return base._write_fake(world_obj.binaries["implementer"],
                            world_obj.record, body)


def run(world_obj, **kwargs):
    return runner.run(world_obj.task_path, repo=world_obj.repo,
                      binaries=world_obj.binaries, **kwargs)


def calls(world_obj):
    """Every fake-model invocation the run recorded, in order."""
    if not world_obj.record.exists():
        return []
    return [json.loads(line) for line
            in world_obj.record.read_text(encoding="utf-8").splitlines()]


def model_calls(world_obj):
    return [call for call in calls(world_obj)
            if call["argv"] != ["--version"]]


def evaluator_calls(world_obj):
    return [call for call in model_calls(world_obj) if "exec" in call["argv"]]


def implementer_calls(world_obj):
    return [call for call in model_calls(world_obj)
            if "--print" in call["argv"]]


@pytest.fixture(autouse=True)
def only_fake_models_may_run(tmp_path, monkeypatch):
    """THE claim this file rests on, enforced rather than repeated.

    Three programs are legitimate and no fourth is: a stub under
    `tmp_path`, the git the fixture and the flat workspace run, and the
    INTERPRETER the acceptance registry names -- `python -m pytest` is a
    gate the loop is supposed to run, and it lives wherever this machine
    keeps its interpreter. Everything else is a real binary somebody
    reached, which is the failure this fixture exists to catch."""
    launched, started, hepsi = [], [], []
    real_popen = process_module.subprocess.Popen

    class _Recorder:
        DEVNULL = subprocess.DEVNULL
        PIPE = subprocess.PIPE
        TimeoutExpired = subprocess.TimeoutExpired
        SubprocessError = subprocess.SubprocessError
        CREATE_NEW_PROCESS_GROUP = getattr(subprocess,
                                           "CREATE_NEW_PROCESS_GROUP", 0)

        @staticmethod
        def run(argv, **kwargs):
            hepsi.append(list(argv))
            return subprocess.run(argv, **kwargs)

        @staticmethod
        def Popen(argv, **kwargs):                 # noqa: N802 -- stdlib
            hepsi.append(list(argv))
            if STUB_HOLDER in str(argv[0]):
                launched.append(list(argv))
            process = real_popen(argv, **kwargs)
            started.append(process)
            return process

    monkeypatch.setattr(process_module, "subprocess", _Recorder)
    yield launched
    for process in started:
        if process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=15)
            except Exception:                      # noqa: BLE001
                pass
    root = str(tmp_path).casefold()
    interpreter = str(Path(sys.executable).resolve().parent).casefold()
    # The programs the FROZEN REGISTRY names, read from the contract
    # rather than listed here: an acceptance command really does start
    # `python` and `bash`, and a program the registry does not name is
    # still a real binary somebody reached.
    izinli = {"git", "git.exe", "taskkill", "taskkill.exe"}
    izinli |= {entry["argv"][0].casefold()
               for entry in contract.COMMAND_REGISTRY.values()}
    strayed = [argv[0] for argv in hepsi
               if not str(argv[0]).casefold().startswith(root)
               and not str(argv[0]).casefold().startswith(interpreter)
               and Path(argv[0]).name.casefold() not in izinli]
    assert strayed == [], f"tmp_path disinda bir program calistirildi: {strayed}"
    assert [p.pid for p in started if p.poll() is None] == []


@pytest.fixture
def spy(monkeypatch):
    """An ORDERED log of the three seams whose sequence is the contract.

    Recorded by wrapping the module attributes the runner looks up at
    call time, so nothing about the runner's own control flow is
    consulted -- what is measured is what it actually called and when."""
    order = []
    real = {"acceptance": acceptance.run_acceptance,
            "audit_run": None,
            "apply": application.apply_accepted_candidate}
    from tools.agent_loop import audit as audit_module
    real["audit_run"] = audit_module.run_evaluator

    def wrap(name, target, module, attribute):
        def wrapper(*args, **kwargs):
            order.append(name)
            return target(*args, **kwargs)
        monkeypatch.setattr(module, attribute, wrapper)
        return wrapper

    wrap("acceptance", real["acceptance"], acceptance, "run_acceptance")
    wrap("audit", real["audit_run"], audit_module, "run_evaluator")
    wrap("apply", real["apply"], application, "apply_accepted_candidate")
    return order


# The evaluator answers approved on round 1 and asks for one repair on
# round 0. Spelled once, because six tests need exactly this shape.
def _two_round_evaluator(finding=None, **overrides):
    first = base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[finding or base._code_finding()],
        next_action="await_repair", **overrides)
    return ("reply = " + repr([first, base._code_audit_reply()]) + "\n"
            "emit(reply[min(round_index(), 1)])\n")


def _two_file_implementer():
    """Round 0 edits TWO files; the repair re-edits only ONE of them.

    The delta and the cumulative set have to be DIFFERENT sets or the
    candidate-versus-delta test below proves nothing: here the repair's
    delta is `{_SECOND}` while the candidate is `{_FIRST, _SECOND}`, so
    applying the delta would silently leave the first round's file
    behind.

    RETARGETED FOR B10-R1, intent unchanged. The earlier shape had the
    repair ADD a file the first round never touched, which the repair
    scope gate now refuses on purpose -- a repair may narrow a candidate,
    never widen it. Making the first round touch both files keeps the two
    sets distinct without asking for the thing that is now forbidden."""
    first = base._implementer_reply(changed_files=[_FIRST, _SECOND])
    second = base._implementer_reply(changed_files=[_SECOND])
    return ("if in_run():\n"
            "    if round_index() == 0:\n"
            f"        edit({_FIRST!r}, 'VALUE = 2\\n')\n"
            f"        edit({_SECOND!r}, 'VALUE = 3\\n')\n"
            "    else:\n"
            f"        edit({_SECOND!r}, 'VALUE = 9\\n')\n"
            "reply = " + repr([first, second]) + "\n"
            "emit(reply[min(round_index(), 1)])\n")


# =====================================================================
# A. POSITIVE CONTROL -- a runner that refused everything would pass
#    every negative test in this file
# =====================================================================

def test_a_clean_run_applies_the_candidate_and_approves(world, spy):
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert result.stop_reason == contract.StopReason.COMPLETED
    assert result.acceptance_passed is True
    assert result.applied_files == (_FIRST,)
    assert (world.repo / _FIRST).read_text(encoding="utf-8") == "VALUE = 2\n"
    assert spy == ["acceptance", "audit", "apply"]
    assert len(implementer_calls(world)) == 1
    assert len(evaluator_calls(world)) == 1


def test_the_happy_visited_sequence_is_exactly_the_frozen_one(world):
    result = run(world)
    assert result.visited == [
        contract.State.PREFLIGHT, contract.State.IMPLEMENTING,
        contract.State.ACCEPTANCE, contract.State.AUDITING,
        contract.State.APPROVED]


def test_the_repair_visited_sequence_is_exactly_the_frozen_one(world):
    evaluator(world, _two_round_evaluator())
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert result.repair_rounds == 1
    assert result.visited == [
        contract.State.PREFLIGHT, contract.State.IMPLEMENTING,
        contract.State.ACCEPTANCE, contract.State.AUDITING,
        contract.State.REPAIRING, contract.State.ACCEPTANCE_2,
        contract.State.FINAL_AUDITING, contract.State.APPROVED]


# =====================================================================
# B. THE AUTHORITY CHAIN -- what may call what, and in which order
# =====================================================================

def test_the_runner_never_reaches_the_implementer_adapter_directly(world):
    """READ FROM THE AST, not from the text: a comment naming the
    forbidden call would satisfy a substring search, and this rule is the
    one that keeps every model edit behind the change-set module's
    workspace binding, manifest binding, main-checkout guard and path
    authorisation."""
    source = (REPO / "tools" / "agent_loop" / "runner.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    imports = [name.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module
               and node.module.startswith("tools.agent_loop")
               for name in node.names]
    assert "execution" not in imports, "runner adapteri dogrudan iceri aldi"

    called = {ast.unparse(node.func) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    assert "execution.run_implementer" not in called
    assert {"changes.run_verified_implementation",
            "changes.run_verified_repair"} <= called

    # AND the same thing measured over a real run: exactly one implementer
    # process, launched through the seam that verifies what it did.
    result = run(world)
    assert result.state == contract.State.APPROVED
    assert len(implementer_calls(world)) == 1


def test_apply_is_never_called_before_acceptance_and_the_audit(world, spy):
    """The order IS the safety property. An application that ran before
    the gates is an unreviewed, untested candidate in the operator's
    checkout, and every downstream check would still pass."""
    evaluator(world, _two_round_evaluator())
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert spy == ["acceptance", "audit", "acceptance", "audit", "apply"]
    assert spy.index("apply") == len(spy) - 1
    assert spy.count("apply") == 1


def test_a_forged_acceptance_report_never_reaches_the_checkout(world,
                                                               monkeypatch):
    """The exact-typed forgery B2B-C2-R1 measured: every digest honestly
    derived, `passed=True`, and ZERO acceptance commands behind it. The
    class proves nothing about the RUN -- only the receipt does."""
    forged = {}

    def never_runs(**kwargs):
        candidate = changes.derive_candidate_changes(
            repo=kwargs["repo"], state_dir=kwargs["state_dir"],
            task_path=kwargs["task_path"],
            manifest_digest=kwargs["manifest_digest"],
            run_id=kwargs["run_id"], workspace_id=kwargs["workspace_id"],
            baseline_sha=kwargs["baseline_sha"])
        report = acceptance.AcceptanceReport(
            run_id=kwargs["run_id"], workspace_id=kwargs["workspace_id"],
            baseline_sha=kwargs["baseline_sha"], passed=True,
            command_results=tuple(
                acceptance.AcceptanceCommandResult(
                    command_id=command_id, passed=True, exit_code=0,
                    duration_ms=1, stdout_bytes=0, stderr_bytes=0,
                    event=contract.EventCode.ACCEPTANCE_FINISHED)
                for command_id, _ in candidate.acceptance_commands),
            total_duration_ms=1,
            event=contract.EventCode.ACCEPTANCE_FINISHED,
            manifest_digest=candidate.task_digest,
            candidate_fingerprint=candidate.fingerprint,
            command_plan_digest=acceptance.command_plan_digest(
                candidate.acceptance_commands),
            receipt_id="f" * 32)
        forged["report"] = report
        return report

    monkeypatch.setattr(acceptance, "run_acceptance", never_runs)
    before = (world.repo / _FIRST).read_text(encoding="utf-8")
    result = run(world)

    # the setup really happened: a report was manufactured and it was
    # exactly the type the application layer type-checks for
    assert type(forged.get("report")) is acceptance.AcceptanceReport
    assert forged["report"].passed is True
    # and no acceptance ever ran, so there is no receipt behind it
    assert not acceptance.receipt_path(world.state_dir).exists()
    assert result.state != contract.State.APPROVED
    assert result.stop_reason == contract.StopReason.PATH_NOT_ALLOWED
    assert (world.repo / _FIRST).read_text(encoding="utf-8") == before


def test_a_pending_receipt_never_reaches_the_checkout(world, monkeypatch):
    """A crashed or interrupted acceptance leaves `pending`, and pending
    is not permission. The report object is the REAL one this time --
    only the file behind it is rewound."""
    real = acceptance.run_acceptance
    seen = {}

    def then_rewind(**kwargs):
        report = real(**kwargs)
        path = acceptance.receipt_path(kwargs["state_dir"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        seen["before"] = payload["status"]
        payload["status"] = acceptance.STATUS_PENDING
        path.write_text(json.dumps(payload), encoding="utf-8")
        return report

    monkeypatch.setattr(acceptance, "run_acceptance", then_rewind)
    before = (world.repo / _FIRST).read_text(encoding="utf-8")
    result = run(world)

    assert seen.get("before") == acceptance.STATUS_PASSED, "senaryo kurulmadi"
    assert json.loads(acceptance.receipt_path(world.state_dir).read_text(
        encoding="utf-8"))["status"] == acceptance.STATUS_PENDING
    assert result.state != contract.State.APPROVED
    assert result.stop_reason == contract.StopReason.PATH_NOT_ALLOWED
    assert (world.repo / _FIRST).read_text(encoding="utf-8") == before


def test_a_failing_acceptance_gate_never_reaches_the_audit_or_the_checkout(
        world, spy):
    """A red gate ends the run. It is not a finding for the evaluator to
    weigh: the commands the task named are the contract, and a candidate
    that fails them is not a candidate.

    RETARGETED FOR B10-R1, intent unchanged. The scenario now breaks the
    file so pytest cannot even COLLECT it, which is a failure shape the
    diagnostic classifier refuses on purpose. That keeps this test about
    the class it has always been about: a red gate nobody can classify
    ends the run, with no audit and nothing applied. The repairable class
    -- ordinary FAILED nodes -- takes the repair road and is proven in
    section H below."""
    implementer(world, "if in_run():\n"
                       "    edit('pipeline/test_gecer.py',"
                       " 'def test_kurgu( :\\n')\n"
                + base._emits(base._implementer_reply(
                    changed_files=["pipeline/test_gecer.py"])))
    before = (world.repo / "pipeline" / "test_gecer.py").read_text(
        encoding="utf-8")
    result = run(world)

    assert len(implementer_calls(world)) == 1, "senaryo kurulmadi"
    assert spy == ["acceptance"], "kabul kapisindan sonrasi kosuldu"
    assert result.acceptance_passed is False
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.ACCEPTANCE_FAILED
    assert (world.repo / "pipeline" / "test_gecer.py").read_text(
        encoding="utf-8") == before


def test_the_state_is_never_approved_without_an_application(world,
                                                            monkeypatch):
    """`approved` is a claim that the candidate is IN the checkout. A run
    whose application refused must not be able to spell it."""
    def refuses(**kwargs):
        raise application.ApplicationRefused(
            "kurgu ret", reason=contract.StopReason.PATH_NOT_ALLOWED)

    monkeypatch.setattr(application, "apply_accepted_candidate", refuses)
    result = run(world)

    assert result.state == contract.State.FAILED
    assert contract.State.APPROVED not in result.visited
    persisted = json.loads(
        (world.state_dir / "state.json").read_text(encoding="utf-8"))
    assert persisted["state"] == contract.State.FAILED


# =====================================================================
# C. THE EVALUATOR IS READ-ONLY, AND IT IS PROVEN AFTERWARDS
# =====================================================================

def test_an_evaluator_that_edits_an_allowed_path_blocks_the_application(
        world, spy):
    """The hard version of the read-only rule. Writing OUTSIDE the
    allowed paths is refused by the path gate, which is a different
    mechanism -- an edit to a file the task authorises produces a
    perfectly legal candidate that simply is not the one that was
    audited, and only the before/after evidence can see it."""
    evaluator(world, "edit('pipeline/kurgu.py', 'VALUE = 99\\n')\n"
                     + base._emits(base._code_audit_reply()))
    before = (world.repo / _FIRST).read_text(encoding="utf-8")
    result = run(world)

    assert len(evaluator_calls(world)) == 1, "denetci hic calismadi"
    assert result.state == contract.State.FAILED
    assert result.stop_reason == \
        contract.StopReason.EVALUATOR_MODIFIED_WORKSPACE
    assert "apply" not in spy
    assert (world.repo / _FIRST).read_text(encoding="utf-8") == before


def test_an_evaluator_that_writes_into_the_main_checkout_blocks(world, spy):
    """The candidate is not the only root at risk. The evaluator runs
    read-only in a flat tree, so a write into the operator's checkout is
    an absolute path it had no business knowing -- and it is measured by
    the same before/after evidence rather than by trusting the sandbox
    flag on the command line."""
    stray = world.repo / "pipeline" / "sizdi.py"
    evaluator(world, f"edit({str(stray)!r}, 'SIZDI = 1\\n')\n"
                     + base._emits(base._code_audit_reply()))
    result = run(world)

    assert stray.exists(), "senaryo kurulmadi: ana agaca yazilmadi"
    assert len(evaluator_calls(world)) == 1
    assert result.stop_reason == \
        contract.StopReason.EVALUATOR_MODIFIED_WORKSPACE
    assert "apply" not in spy


# =====================================================================
# D. THE ROUND LIMITS -- one implementation, one repair, no third patch
# =====================================================================

def test_a_repair_applies_the_whole_candidate_and_not_its_delta(world):
    """What a later step applies is everything that separates the
    reference tree from the final one. Returning the repair's DELTA
    instead would put half a candidate in the operator's checkout -- the
    first round's file would silently be left behind."""
    implementer(world, _two_file_implementer())
    evaluator(world, _two_round_evaluator())
    result = run(world)

    assert len(implementer_calls(world)) == 2, "onarim turu kosulmadi"
    assert result.state == contract.State.APPROVED
    assert set(result.applied_files) == {_FIRST, _SECOND}
    assert (world.repo / _FIRST).read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (world.repo / _SECOND).read_text(encoding="utf-8") == "VALUE = 9\n"


def test_a_final_audit_asking_for_changes_never_starts_a_second_repair(world):
    """THE THIRD-PATCH EDGE DOES NOT EXIST. Both audits ask for changes
    about a NEW mechanism each time, so the second-patch rule alone would
    not fire -- the round budget has to."""
    evaluator(world,
              "reply = " + repr(base._code_audit_reply(
                  status=contract.Status.CHANGES_REQUESTED,
                  findings=[base._code_finding()],
                  next_action="await_repair")) + "\n"
              "reply['findings'][0]['mechanism_id'] = "
              "'kurgu-mekanizma-%d' % round_index()\n"
              "emit(reply)\n")
    result = run(world)

    assert len(evaluator_calls(world)) == 2, "ikinci denetim kosulmadi"
    assert len(implementer_calls(world)) == 2, "tek onarim kosulmadi"
    assert result.repair_rounds == 1
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.REPAIR_ROUNDS_EXHAUSTED
    assert contract.State.APPROVED not in result.visited


def test_the_same_mechanism_twice_stops_before_the_round_budget_does(world):
    """Two separate limits, and this one fires FIRST. A run that reported
    `repair_rounds_exhausted` here would be telling the operator to buy
    another round, when what the evidence says is that the mechanism
    needs a design change."""
    evaluator(world, base._emits(base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._code_finding(mechanism_id="kurgu-mekanizma-a")],
        next_action="await_repair")))
    result = run(world)

    assert len(evaluator_calls(world)) == 2, "ikinci denetim kosulmadi"
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == \
        contract.StopReason.REPEATED_MECHANISM_FAILURE
    assert result.repair_rounds == 1


def test_a_finding_about_a_file_this_run_did_not_change_blocks(world, spy):
    """Scope widening is a human gate, not something a repair round may
    do quietly: the file is inside `allowed_paths`, so nothing but the
    candidate's own change set can tell that the repair would reach
    outside what the acceptance gate measured."""
    evaluator(world, base._emits(base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._code_finding(file="pipeline/baska.py",
                                     mechanism_id="kapsam-disi")],
        next_action="await_repair")))
    result = run(world)

    assert len(evaluator_calls(world)) == 1
    assert result.stop_reason == contract.StopReason.OUT_OF_SCOPE_FINDING
    assert result.repair_rounds == 0
    assert len(implementer_calls(world)) == 1, "onarim yine de kosuldu"
    assert "apply" not in spy


# =====================================================================
# E. THE HUMAN GATES AND THE THINGS THE LOOP MAY NEVER DO
# =====================================================================

@pytest.mark.parametrize("gate", ["git_commit", "git_push", "git_add",
                                  "dependency_install", "production_deploy"])
def test_a_gated_action_performs_nothing_at_all(world, gate):
    """It stops and asks BEFORE anything exists: no workspace, no model
    process, no mirror, no application. A gate answered by doing the
    thing and asking afterwards is not a gate."""
    head = _git(world.repo, "rev-parse", "HEAD")
    result = run(world, _test_request_gate=gate)

    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.USER_APPROVAL_REQUIRED
    assert gate in result.pending_approval
    assert model_calls(world) == []
    assert result.workspace_id is None
    assert _git(world.repo, "rev-parse", "HEAD") == head
    assert _git(world.repo, "status", "--porcelain") == "?? kurgu-task.json"


def test_no_version_control_argv_is_ever_built_for_add_commit_or_push(
        world, monkeypatch):
    """Measured over EVERY process the run started, not over the runner's
    source. The loop runs git for exactly one thing -- reading the
    baseline's objects -- and a read is not a write."""
    seen = []
    real_run = subprocess.run
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: (seen.append(list(argv)), real_run(argv, **kw))[1])
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda argv, **kw: (seen.append(list(argv)),
                            real_popen(argv, **kw))[1])
    result = run(world)

    assert result.state == contract.State.APPROVED, "senaryo tamamlanmadi"
    assert seen, "hicbir surec kaydedilmedi"
    forbidden = {"add", "commit", "push", "rebase", "reset", "checkout",
                 "branch", "tag", "remote", "stash"}
    offenders = [argv for argv in seen
                 if Path(argv[0]).name.casefold().startswith("git")
                 and forbidden.intersection(argv[1:])]
    assert offenders == [], f"surum kontrolu yazma cagrisi: {offenders}"


def test_the_loop_leaves_head_and_the_index_exactly_where_they_were(world):
    head = _git(world.repo, "rev-parse", "HEAD")
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert _git(world.repo, "rev-parse", "HEAD") == head
    assert _git(world.repo, "diff", "--cached", "--name-only") == ""


# =====================================================================
# F. BUDGET AND DEADLINE -- checked BEFORE a process exists
# =====================================================================

def test_a_zero_budget_starts_no_model_process(world):
    retask(world, max_budget_usd=0.0)
    result = run(world)

    assert result.stop_reason == contract.StopReason.BUDGET_EXHAUSTED
    assert model_calls(world) == []
    assert result.workspace_id is None


class _JumpingClock:
    """The real clock, except that time leaps once the deadline is set."""

    def __init__(self, real, jump):
        self.real = real
        self.jump = jump
        self.calls = 0

    def monotonic(self):
        self.calls += 1
        return 0.0 if self.calls == 1 else self.jump

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_a_spent_deadline_starts_no_model_process(world, monkeypatch):
    clock = _JumpingClock(time, 10 ** 6)
    monkeypatch.setattr(runner, "time", clock)
    result = run(world)

    assert clock.calls >= 2, "senaryo kurulmadi: saat hic sorulmadi"
    assert result.stop_reason == contract.StopReason.WALL_CLOCK_EXCEEDED
    assert model_calls(world) == []
    assert result.workspace_id is None


def test_every_model_call_is_bounded_by_what_is_left_of_the_run(world):
    """`min(call limit, remaining)`. A phase that took its own full
    timeout from a fresh clock would let a five-minute run spend twenty
    minutes across four phases."""
    retask(world, max_wall_clock_minutes=1, model_call_timeout_seconds=7200)
    seen = []
    from tools.agent_loop import audit as audit_module
    real = audit_module.run_evaluator

    def spy_evaluator(binary, **kwargs):
        seen.append(kwargs["timeout_seconds"])
        return real(binary, **kwargs)

    audit_module.run_evaluator, saved = spy_evaluator, audit_module.run_evaluator
    try:
        run(world)
    finally:
        audit_module.run_evaluator = saved

    assert seen, "denetci hic cagrilmadi"
    assert all(limit <= 60 for limit in seen), seen


# =====================================================================
# G. THE PRIVACY BOUNDARY, MEASURED OVER WHAT THE RUN PRODUCED
# =====================================================================

def _locked_evaluator_body():
    """A schema-valid LOCKED reply, built from the ids the runner issued.

    It has to be built in the child: the envelope is textless, so its
    `run_id` and both id fields are opaque and the per-call schema pins
    them to exactly what was minted for this run."""
    return ("reply = " + repr(base._locked_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._locked_finding()], next_action="await_repair")) + "\n"
        "reply['run_id'] = os.environ['AGENT_LOOP_LOCKED_RUN_ID']\n"
        "reply['findings'][0]['finding_id'] = "
        "issued('AGENT_LOOP_FINDING_IDS')[0]\n"
        "reply['findings'][0]['mechanism_id'] = "
        "issued('AGENT_LOOP_MECHANISM_IDS')[0]\n"
        "emit(reply)\n")


def test_a_locked_finding_crosses_as_a_class_and_a_count_and_nothing_else(
        world, monkeypatch):
    """Measured in the THREE places a passage could travel: the state
    directory, the event journal, and the repair prompt -- which is the
    one that actually reaches the other model and the one no artefact
    check can see."""
    prompts = []
    real = changes.run_verified_repair

    def spy_repair(binary, **kwargs):
        prompts.append(kwargs["prompt"])
        return real(binary, **kwargs)

    monkeypatch.setattr(changes, "run_verified_repair", spy_repair)
    evaluator(world, _locked_evaluator_body())
    result = run(world, audit_kind=contract.AuditKind.LOCKED)

    assert len(evaluator_calls(world)) == 2, "ikinci kilitli denetim kosulmadi"
    # EXACTLY the repeated-mechanism reason, not either terminal code.
    # The ids are minted ONCE FOR THE RUN, so the second audit can name
    # the mechanism the first one named -- and if they were minted per
    # call it could not, the run would report the round budget instead,
    # and this test would be green about a rule nothing can trip.
    assert result.stop_reason == \
        contract.StopReason.REPEATED_MECHANISM_FAILURE
    assert prompts, "onarim istemi hic kurulmadi"

    findings = runner_events.read_findings(world.state_dir)
    recorded = findings["rounds"][0]["findings"][0]
    assert recorded["error_class"] == contract.LockedFindingClass.WRONG_ROW
    assert recorded["case_count"] == 3
    assert set(recorded) <= set(runner_events.LOCKED_RECORD_FIELDS)

    written = " ".join(path.read_text(encoding="utf-8", errors="ignore")
                       for path in world.state_dir.rglob("*")
                       if path.is_file())
    haystack = written + " " + " ".join(prompts)
    assert contract.LockedFindingClass.WRONG_ROW in haystack
    for banned in contract.NEVER_TRANSFERABLE:
        assert banned not in haystack, banned


def test_no_model_authored_prose_is_written_into_the_state_directory(world):
    """A CODE finding's `claim` and `required_action` may reach the
    IMPLEMENTER -- it can already open the file they are about -- but a
    500-character model-authored sentence in the run's own record is a
    document-sized hole in the most frequently written artefact."""
    evaluator(world, _two_round_evaluator(
        finding=base._code_finding(claim="KURGU-GIZLI-IDDIA",
                                   required_action="KURGU-GIZLI-EYLEM")))
    result = run(world)

    assert result.state == contract.State.APPROVED, "senaryo tamamlanmadi"
    findings = runner_events.read_findings(world.state_dir)
    assert findings["rounds"][0]["findings"], "bulgu hic kaydedilmedi"
    written = " ".join(path.read_text(encoding="utf-8", errors="ignore")
                       for path in world.state_dir.rglob("*")
                       if path.is_file())
    assert "KURGU-GIZLI-IDDIA" not in written
    assert "KURGU-GIZLI-EYLEM" not in written


def test_the_result_object_carries_no_path_no_output_and_no_prose(world):
    result = run(world)
    rendered = repr(result)

    assert result.state == contract.State.APPROVED
    for leaked in (str(world.repo), str(world.state_dir), str(sys.executable),
                   "VALUE =", "kurgu ozet"):
        assert leaked not in rendered, leaked


# =====================================================================
# H. INTERRUPTION, RECOVERY AND RESIDUE
# =====================================================================

def test_an_interrupt_leaves_the_state_resumable_and_the_workspace_alive(
        world):
    """A resumable run needs BOTH halves: a state document that is not
    terminal, and the workspace that document names still on disk."""
    result = run(world, _test_interrupt_after=contract.State.IMPLEMENTING)

    assert result.stop_reason == contract.StopReason.INTERRUPTED
    assert result.state == contract.State.IMPLEMENTING
    persisted = json.loads(
        (world.state_dir / "state.json").read_text(encoding="utf-8"))
    assert persisted["state"] not in contract.TERMINAL_STATES
    assert "stop_reason" not in persisted
    assert flat_workspace.holder_for(result.workspace_id).is_dir()


def test_resume_recovers_the_same_run_from_the_backup(world):
    finished = run(world)
    assert finished.state == contract.State.APPROVED, "senaryo tamamlanmadi"
    state_path = world.state_dir / "state.json"
    state_path.write_text('{"protocol_version": "1.0", "run_',
                          encoding="utf-8")

    resumed = runner.resume(repo=world.repo, binaries=world.binaries)

    assert resumed.recovered_from_backup is True
    assert resumed.run_id == finished.run_id
    assert resumed.workspace_id == finished.workspace_id
    assert resumed.baseline_sha == finished.baseline_sha
    assert resumed.state == contract.State.APPROVED
    # and the live document is readable again, not merely reported
    assert json.loads(state_path.read_text(encoding="utf-8"))["run_id"] == \
        finished.run_id


def test_a_terminal_success_leaves_no_workspace_behind(world,
                                                       private_runner_root):
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert not flat_workspace.holder_for(result.workspace_id).exists()
    assert list(private_runner_root.iterdir()) == []
    assert application.find_pending_applications(world.repo) == ()


def test_a_second_runner_is_refused_before_it_builds_anything(world):
    """The lock is the OUTERMOST boundary: a second instance must be
    refused before a workspace, a mirror, a model process or an
    application exists."""
    with runner.single_instance_lock(world.repo):
        with pytest.raises(runner.LockHeld):
            run(world)
    assert model_calls(world) == []
    assert not (world.state_dir / "state.json").exists()


# =====================================================================
# B4-R2 -- THE FAILURE CLASS MUST REACH THE JOURNAL
# =====================================================================
#
# MEASURED ON A REAL RUN. The first controlled task stopped with
# `stop_reason: preflight_failed` AFTER `preflight_ok` had already been
# recorded, and the journal held nothing at all between
# `model_call_started` and the transition to `blocked`. Reconstructing
# which gate refused took a stubbed replay of the whole change-set seam,
# because the one artefact built to answer that question was silent.
#
# THE MECHANISM. Every `execution.AdapterError` carries a closed
# `.event`; every `changes.ChangeSetError` carries a closed `.reason`
# and NO event. `_translate` read only `.event`, so an entire family of
# refusals -- every evidence gate in the change-set seam -- produced no
# journal line whatsoever.

def _journal(world_obj):
    path = runner_events.events_path(world_obj.state_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_an_evidence_refusal_is_recorded_as_a_closed_failure_event(world):
    """The exact shape of the blocked run, reproduced: a change-set
    evidence gate refuses, and the journal must say so."""
    def refuse(*args, **kwargs):
        raise changes.EvidenceUnavailable("dosya sistemi kaniti alinamadi")

    original = changes.run_verified_implementation
    changes.run_verified_implementation = refuse
    try:
        result = run(world)
    finally:
        changes.run_verified_implementation = original

    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED
    codes = [entry["event"] for entry in _journal(world)]
    # the scenario really happened: the call was reached
    assert contract.EventCode.MODEL_CALL_STARTED in codes
    # THE REPAIR: the refusal is no longer invisible
    assert contract.EventCode.PREFLIGHT_FAILED in codes, \
        "kanit reddi gunluge hic yazilmadi"
    # and it is DISTINGUISHABLE from the two failures it is not
    assert contract.EventCode.MODEL_CALL_FINISHED not in codes
    assert contract.EventCode.SCHEMA_VIOLATION not in codes


def test_a_declaration_mismatch_is_recorded_as_a_schema_violation(world):
    """The same family, a different closed reason: the journal must
    separate 'the evidence could not be read' from 'the reply did not
    describe what happened'."""
    def refuse(*args, **kwargs):
        raise changes.DeclarationMismatch("bildirim gerceklesenle uyusmuyor")

    original = changes.run_verified_implementation
    changes.run_verified_implementation = refuse
    try:
        result = run(world)
    finally:
        changes.run_verified_implementation = original

    assert result.stop_reason == contract.StopReason.SCHEMA_VIOLATION
    codes = [entry["event"] for entry in _journal(world)]
    assert contract.EventCode.SCHEMA_VIOLATION in codes
    assert contract.EventCode.PREFLIGHT_FAILED not in codes


def test_the_failure_event_carries_no_free_text_and_no_path(world):
    """A journal line is a closed record. The refusal's SENTENCE, the
    exception type and any path must not travel into it -- the event
    code is the whole payload."""
    secret = "kanit-alinamadi-gizli-cumle"

    def refuse(*args, **kwargs):
        raise changes.EvidenceUnavailable(secret)

    original = changes.run_verified_implementation
    changes.run_verified_implementation = refuse
    try:
        run(world)
    finally:
        changes.run_verified_implementation = original

    raw = runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    assert secret not in raw
    assert "EvidenceUnavailable" not in raw
    assert str(world.repo) not in raw
    # B4-R4 added closed diagnostic fields, so the allowlist grows -- but
    # it stays an ALLOWLIST, and the event schema closes the record too.
    allowed = {"ts", "run_id", "event", "state", "command_id", "exit_code",
               "duration_ms", "bytes_truncated", "failure_code", "role",
               "requested_models", "stdout_bytes", "stderr_bytes",
               "cleanup_complete"}
    for entry in _journal(world):
        assert set(entry) <= allowed, sorted(set(entry) - allowed)


# =====================================================================
# B4-R4 -- WHICH MECHANISM FAILED, not merely that one did
# =====================================================================
#
# MEASURED ON TWO REAL RUNS. One ended `schema_violation`, the next
# `model_process_failed`, and neither code named the defect: the first
# is raised BOTH by the adapter refusing a reply and by the change-set
# gate refusing a declaration the filesystem contradicts, and the second
# covers a non-zero exit, a container that could not be built, and a
# process tree that outlived its call. Diagnosing either meant writing a
# probe, which is the cost this section removes.
#
# The stop reason answers "may the run continue". The failure code
# answers "which mechanism broke". Both are recorded now.

def _run_raising(world_obj, failure):
    """Run with the implementer seam raising an EXACT lower-layer
    object -- not a stand-in, so the code under test sees the real
    type it will meet in production."""
    def refuse(*args, **kwargs):
        raise failure

    original = changes.run_verified_implementation
    changes.run_verified_implementation = refuse
    try:
        return run(world_obj)
    finally:
        changes.run_verified_implementation = original


def _terminal_event(world_obj, code):
    entries = [entry for entry in _journal(world_obj)
               if entry["event"] == code]
    assert entries, f"{code} olayi hic yazilmadi"
    return entries[-1]


def test_a_process_failure_records_its_code_and_its_measurements(world):
    """The numbers are the LOWER LAYER'S OWN -- copied, never invented."""
    result = _run_raising(world, execution.ProcessFailed(
        "model sureci sifirdan farkli koda dondu", exit_code=3,
        duration_ms=1234, stdout_bytes=77, stderr_bytes=9,
        cleanup_complete=True))
    assert result.stop_reason == contract.StopReason.MODEL_PROCESS_FAILED
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_PROCESS_FAILED
    assert entry["role"] == contract.Role.IMPLEMENTER
    assert entry["exit_code"] == 3
    assert entry["duration_ms"] == 1234
    assert entry["stdout_bytes"] == 77 and entry["stderr_bytes"] == 9
    assert entry["cleanup_complete"] is True
    assert entry["failure_code"] in contract.ALL_FAILURE_CODES


def test_a_schema_violation_and_a_declaration_mismatch_are_told_apart(
        tmp_path, world):
    """THE AMBIGUITY THIS SECTION EXISTS FOR. Both carry stop reason
    `schema_violation` and event `schema_violation`; they are two
    different defects and must not share one code.

    TWO WORLDS, because a terminal state belongs to the run that wrote
    it: a second `runner.run` in the same checkout is refused by
    preflight, and reusing one world here would go green on that refusal
    instead of on the thing being measured."""
    result = _run_raising(world, execution.SchemaViolation(
        "yanit sema disi (alan: kok)", exit_code=0, duration_ms=11,
        stdout_bytes=5, stderr_bytes=0, cleanup_complete=True))
    assert result.stop_reason == contract.StopReason.SCHEMA_VIOLATION
    adapter = _terminal_event(world, contract.EventCode.SCHEMA_VIOLATION)
    assert adapter["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_SCHEMA_VIOLATION

    other = build_world(tmp_path, index=1)
    result = _run_raising(other, changes.DeclarationMismatch(
        "bildirim gerceklesenle uyusmuyor"))
    assert result.stop_reason == contract.StopReason.SCHEMA_VIOLATION
    gate = _terminal_event(other, contract.EventCode.SCHEMA_VIOLATION)
    assert gate["failure_code"] == \
        contract.FailureCode.CHANGE_DECLARATION_MISMATCH
    # same reason, same event, DIFFERENT mechanism
    assert gate["failure_code"] != adapter["failure_code"]
    assert gate.get("exit_code") is None, "olcumu olmayan katman sayi uydurdu"


def test_a_containment_failure_is_its_own_code_and_has_no_exit_code(world):
    """The container could not be built, so no process ever ran -- and
    that is visible in two ways: a code of its own, and NO `exit_code`,
    because a process that never started never returned one.

    The byte and duration fields are present as ZEROS, and that is the
    lower layer's own default rather than a measurement. The runner
    copies what the layer supplies and does not decide which of its
    zeros are meaningful: guessing that a `0` means 'absent' is exactly
    the silent-zero reading this project keeps paying for. The honest
    discriminator is `exit_code` and the code itself."""
    _run_raising(world, execution.ContainmentFailed(
        "surec kapsayicisi kurulamadi"))
    entry = _terminal_event(world, contract.EventCode.PREFLIGHT_FAILED)
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_CONTAINMENT_FAILED
    assert entry["failure_code"] != \
        contract.FailureCode.IMPLEMENTER_PROCESS_FAILED
    assert "exit_code" not in entry, "hic baslamayan surec cikis kodu bildirdi"
    assert entry["stdout_bytes"] == 0 and entry["stderr_bytes"] == 0


def test_an_unproven_cleanup_is_recorded_and_still_counts_a_survivor(world):
    """`cleanup_complete=False` must reach the journal AND keep the
    surviving-children count it already produced -- a flag that replaced
    the count would hide the process."""
    result = _run_raising(world, execution.ProcessFailed(
        "model sureci sifirdan farkli koda dondu", exit_code=1,
        duration_ms=7, stdout_bytes=0, stderr_bytes=0,
        cleanup_complete=False))
    assert result.surviving_children == 1
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert entry["cleanup_complete"] is False


def test_a_successful_model_event_carries_no_failure_code(world):
    """The healthy road is unchanged: a failure code on a success would
    make the field meaningless."""
    result = run(world)
    assert result.state == contract.State.APPROVED
    for entry in _journal(world):
        assert "failure_code" not in entry
        assert "role" not in entry
    finished = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert finished["exit_code"] == 0
    assert isinstance(finished["duration_ms"], int)


def test_no_message_path_or_note_reaches_the_journal(world):
    """The failure carries a sentinel, an absolute path and a note. The
    journal may show that it happened and which mechanism -- nothing
    else."""
    sentinel = "GIZLI-ARIZA-CUMLESI"
    failure = execution.ProcessFailed(
        f"{sentinel} {world.repo}", exit_code=2, duration_ms=1,
        stdout_bytes=0, stderr_bytes=0)
    failure.add_note(f"{sentinel} not")
    _run_raising(world, failure)
    raw = runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    assert sentinel not in raw
    assert "ProcessFailed" not in raw, "istisna sinif adi gunluge sizdi"
    assert str(world.repo) not in raw
    assert "not" not in raw.lower().replace("notification", ""), \
        "not/cause metni gunluge sizdi"
    # no path separator of either kind survives in a closed record
    assert "\\" not in raw and "/" not in raw
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_PROCESS_FAILED


def test_an_unclassified_failure_invents_no_code(world):
    """A type the table does not name is not ours to label. It keeps the
    behaviour it already had -- an unclassified exception flies out of
    the runner untouched -- and no code is guessed for it."""
    class Bilinmeyen(RuntimeError):
        pass

    with pytest.raises(Bilinmeyen):
        _run_raising(world, Bilinmeyen("siniflandirilmamis"))
    for entry in _journal(world):
        assert "failure_code" not in entry


def test_the_failure_vocabulary_and_the_event_schema_agree(world):
    """The closed sets are pinned together: a code the schema refuses
    could never be written, and a journal line is validated before the
    handle is opened."""
    from jsonschema import Draft202012Validator

    from tools.agent_loop import schemas as schema_module

    # 10 -> 14 in B4-R6: the four error-envelope classes. 14 -> 29 in
    # B4-R8: the whole evaluator road, which had no code at all. The
    # COUNT is pinned rather than derived so that adding a code stays a
    # decision somebody makes on purpose -- this assertion is what caught
    # both additions, which is the whole reason it is a literal.
    assert len(set(contract.ALL_FAILURE_CODES)) == 29
    assert len(contract.ALL_FAILURE_CODES) == 29, "yinelenen basarisizlik kodu"
    assert set(contract.FAILURE_CODES.values()) <= \
        set(contract.ALL_FAILURE_CODES)
    properties = schema_module.EVENT_SCHEMA["properties"]
    assert properties["failure_code"]["enum"] == list(
        contract.ALL_FAILURE_CODES)
    assert properties["role"]["enum"] == list(contract.ALL_ROLES)
    assert schema_module.EVENT_SCHEMA["additionalProperties"] is False
    validator = Draft202012Validator(schema_module.EVENT_SCHEMA)
    assert not validator.is_valid(
        {"ts": "t", "run_id": "kosu-abc", "event": "model_call_finished",
         "failure_code": "boyle-bir-kod-yok"})


# =====================================================================
# B4-R5 -- THE PLAN-ONLY GATE, WHERE THE RUNNER USES IT
# =====================================================================
#
# The authority itself is proven in `test_agent_loop_plan_auth.py`. What
# only this suite can prove is WHERE it is consulted: before anything is
# built, and again immediately before each model call.

def test_an_api_key_stops_the_run_before_anything_is_built(world, monkeypatch):
    """Preflight's refusal costs nothing: no workspace, no state
    document, no model process. The counters are asserted rather than
    the reason alone, because "refused" and "refused before spending"
    are different claims."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "kurgu-anahtar-degeri")
    result = run(world)
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED
    assert model_calls(world) == []
    assert not (world.state_dir / "state.json").exists()
    assert result.workspace_id is None
    root = flat_workspace.runner_temp_root()
    assert not root.exists() or list(root.iterdir()) == []


def test_the_gate_is_re_proven_immediately_before_every_model_call(
        world, monkeypatch):
    """A preflight answer describes the moment preflight ran. Each call
    asks again -- and when the answer changes, THAT call does not
    happen.

    The second half is the load-bearing one: failing the gate that
    precedes the evaluator must leave the implementer's single call
    intact and the evaluator's count at zero."""
    seen = []
    genuine = runner.plan_auth.assert_plan_only

    def counting(**kwargs):
        seen.append(kwargs)
        return genuine(**kwargs)

    monkeypatch.setattr(runner.plan_auth, "assert_plan_only", counting)
    result = run(world)
    assert result.state == contract.State.APPROVED
    # preflight, then one before the implementer, one before the evaluator
    assert len(seen) == 3
    assert {tuple(sorted(call)) for call in seen} == {
        ("evaluator_binary", "implementer_binary")}
    # the EXACT canonical binaries, never re-derived from the task
    for call in seen:
        assert call["implementer_binary"] == world.binaries["implementer"]
        assert call["evaluator_binary"] == world.binaries["evaluator"]

    # a SECOND world: a terminal state belongs to the run that wrote it,
    # so a second `runner.run` in the same checkout would be refused by
    # preflight and go green on the wrong refusal
    other = build_world(world.repo.parent, index=3)
    calls = {"n": 0}

    def failing(**kwargs):
        calls["n"] += 1
        if calls["n"] >= 3:            # preflight, implementer, THEN fail
            raise runner.plan_auth.PlanAuthRefused(
                "ortamda API anahtari var: ANTHROPIC_API_KEY")
        return genuine(**kwargs)

    monkeypatch.setattr(runner.plan_auth, "assert_plan_only", failing)
    result = run(other)
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED
    assert len(implementer_calls(other)) == 1
    assert evaluator_calls(other) == []


# =====================================================================
# B4-R6 -- THE ENVELOPE'S LIMIT REACHES THE JOURNAL
# =====================================================================
#
# The adapter now distinguishes the CLI's own terminal error envelopes.
# What only this suite can prove is that the distinction SURVIVES the
# trip into the journal -- through the importless `(module, class)`
# table, because the runner still may not import `execution`.

def test_an_envelope_limit_reaches_the_journal_as_its_own_code(world):
    """The whole point of B4-R6: an operator reading the journal learns
    WHICH ceiling ended the call, without a probe."""
    result = _run_raising(world, execution.MaxBudgetReached(
        "model sureci bildirilen bir sinirda durdu", exit_code=1,
        duration_ms=102327, stdout_bytes=1563, stderr_bytes=0,
        cleanup_complete=True))
    assert result.stop_reason == contract.StopReason.MODEL_PROCESS_FAILED
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_MAX_BUDGET_REACHED
    # NOT the generic one -- that was the whole complaint
    assert entry["failure_code"] != \
        contract.FailureCode.IMPLEMENTER_PROCESS_FAILED
    assert entry["role"] == contract.Role.IMPLEMENTER
    assert entry["exit_code"] == 1 and entry["duration_ms"] == 102327
    assert entry["stdout_bytes"] == 1563 and entry["stderr_bytes"] == 0
    assert entry["cleanup_complete"] is True

    raw = runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    # the vendor's own spelling never travels; only this package's code
    assert "error_max_budget_usd" not in raw
    assert "MaxBudgetReached" not in raw


def test_a_schema_violation_reaches_the_journal_with_its_two_closed_words(
        world):
    """B5-R3, at the layer that matters. A real run recorded
    `implementer_schema_violation` and nothing else, because the
    adapter's diagnosis lived in a SENTENCE and sentences do not
    travel. These two words do."""
    result = _run_raising(world, execution.SchemaViolation(
        "yanit sema disi (alan: next_action)",
        schema_issue=contract.SchemaIssue.REQUIRED,
        schema_field=contract.SchemaField.NEXT_ACTION,
        exit_code=0, duration_ms=219171, stdout_bytes=6766, stderr_bytes=0,
        cleanup_complete=True))
    assert result.stop_reason == contract.StopReason.SCHEMA_VIOLATION
    entry = _terminal_event(world, contract.EventCode.SCHEMA_VIOLATION)
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_SCHEMA_VIOLATION
    assert entry["schema_issue"] == contract.SchemaIssue.REQUIRED
    assert entry["schema_field"] == contract.SchemaField.NEXT_ACTION
    assert entry["role"] == contract.Role.IMPLEMENTER
    assert entry["exit_code"] == 0 and entry["stdout_bytes"] == 6766

    raw = runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    # the adapter's own sentence is NOT what travelled
    assert "yanit sema disi" not in raw
    assert "alan:" not in raw


def test_a_word_outside_the_contract_never_reaches_the_journal(world):
    """The allowlist is a membership test, not a copy. A failure that
    carries invented words -- or model text under those names -- writes
    neither."""
    sentinel = "GIZLI-MODEL-ALANI"
    failure = execution.SchemaViolation(
        "yanit sema disi (alan: kok)", exit_code=0, duration_ms=5,
        stdout_bytes=1, stderr_bytes=0, cleanup_complete=True)
    # set AFTER construction, bypassing the exception's own check, so
    # this test measures the RUNNER's gate rather than the adapter's
    object.__setattr__(failure, "schema_issue", sentinel)
    object.__setattr__(failure, "schema_field", sentinel)
    _run_raising(world, failure)

    entry = _terminal_event(world, contract.EventCode.SCHEMA_VIOLATION)
    assert "schema_issue" not in entry
    assert "schema_field" not in entry
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_SCHEMA_VIOLATION
    raw = runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    assert sentinel not in raw


def test_a_failure_that_is_not_a_schema_violation_writes_neither_word(world):
    """The two words belong to one mechanism. A process failure must not
    acquire them, or an operator reading the journal would look for a
    schema problem that never happened."""
    _run_raising(world, execution.ProcessFailed(
        "model sureci sifirdan farkli koda dondu", exit_code=1,
        duration_ms=5, stdout_bytes=0, stderr_bytes=0,
        cleanup_complete=True))
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert "schema_issue" not in entry
    assert "schema_field" not in entry


def test_the_generic_process_failure_still_reads_as_generic(world):
    """The regression half: refining four envelopes must not have
    reclassified everything else."""
    _run_raising(world, execution.ProcessFailed(
        "model sureci sifirdan farkli koda dondu", exit_code=1,
        duration_ms=5, stdout_bytes=0, stderr_bytes=0,
        cleanup_complete=True))
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert entry["failure_code"] == \
        contract.FailureCode.IMPLEMENTER_PROCESS_FAILED


# =====================================================================
# B4-R8 -- THE EVALUATOR ROAD REACHES THE JOURNAL TOO
# =====================================================================
#
# The first run that ever got to `auditing` failed there, and the
# terminal event carried role `evaluator` with NO failure code: the
# table named only the implementer's family. What only this suite can
# prove is that the evaluator's own classes now survive the trip --
# through the importless `(module, class)` key, because the runner still
# may not import either adapter.

def _run_with_failing_evaluator(world_obj, failure, monkeypatch):
    """Run with the EVALUATOR seam raising an exact `audit` object. The
    implementer and acceptance phases run for real, so the failure
    arrives where the measured one did."""
    def refuse(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runner.audit, "run_evaluator", refuse)
    return run(world_obj)


def test_an_evaluator_failure_reaches_the_journal_with_its_own_code(
        world, monkeypatch):
    """The measured gap, closed: the same shape of failure that produced
    a codeless terminal event now names its mechanism."""
    result = _run_with_failing_evaluator(world, audit.RepositoryRefused(
        "denetci sureci bildirilen bir nedenle durdu", exit_code=1,
        duration_ms=4255, stdout_bytes=0, stderr_bytes=733,
        cleanup_complete=True), monkeypatch)

    assert result.state == contract.State.FAILED
    assert result.stop_reason == contract.StopReason.MODEL_PROCESS_FAILED
    assert len(implementer_calls(world)) == 1, "senaryo kurulmadi"
    entry = _terminal_event(world, contract.EventCode.MODEL_CALL_FINISHED)
    assert entry["failure_code"] == \
        contract.FailureCode.EVALUATOR_REPOSITORY_REFUSED
    assert entry["role"] == contract.Role.EVALUATOR
    assert entry["exit_code"] == 1 and entry["duration_ms"] == 4255
    assert entry["stdout_bytes"] == 0 and entry["stderr_bytes"] == 733
    assert entry["cleanup_complete"] is True

    raw = runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    # neither the vendor's sentence nor this package's class name travels
    assert "trusted directory" not in raw
    assert "skip-git-repo-check" not in raw
    assert "RepositoryRefused" not in raw


def test_the_two_roads_never_share_a_code(tmp_path, world, monkeypatch):
    """THE AMBIGUITY THIS SECTION EXISTS FOR. `audit.ProcessFailed` and
    `execution.ProcessFailed` are the same NAME, and a name-only lookup
    would file an evaluator failure as an implementer one -- a
    confidently wrong code an operator would act on.

    TWO WORLDS, because a terminal state belongs to the run that wrote
    it: a second `runner.run` in the same checkout is refused by
    preflight and would go green on that refusal instead."""
    _run_with_failing_evaluator(world, audit.ProcessFailed(
        "denetci sureci sifirdan farkli koda dondu", exit_code=1,
        duration_ms=7, stdout_bytes=0, stderr_bytes=11,
        cleanup_complete=True), monkeypatch)
    evaluator_side = _terminal_event(
        world, contract.EventCode.MODEL_CALL_FINISHED)["failure_code"]

    other = build_world(tmp_path, index=4)
    _run_raising(other, execution.ProcessFailed(
        "model sureci sifirdan farkli koda dondu", exit_code=1,
        duration_ms=7, stdout_bytes=0, stderr_bytes=11,
        cleanup_complete=True))
    implementer_side = _terminal_event(
        other, contract.EventCode.MODEL_CALL_FINISHED)["failure_code"]

    assert evaluator_side == contract.FailureCode.EVALUATOR_PROCESS_FAILED
    assert implementer_side == contract.FailureCode.IMPLEMENTER_PROCESS_FAILED
    assert evaluator_side != implementer_side
    # and no code may serve both roads, whatever it is called
    implementer_codes = {code for (module, _), code
                         in contract.FAILURE_CODES.items()
                         if module == "execution"}
    evaluator_codes = {code for (module, _), code
                       in contract.FAILURE_CODES.items() if module == "audit"}
    assert implementer_codes and evaluator_codes
    assert not implementer_codes & evaluator_codes


# =====================================================================
# B7-R1 -- BOTH IMPLEMENTER ROADS CARRY THE NEW TRANSPORT
#
# The initial implementation and the verified repair reach the model
# through the SAME adapter (`changes.py` calls `execution.run_implementer`
# on both), so one change covers both. That is a claim about wiring, and
# a claim about wiring is worth an assertion rather than a reading.
# =====================================================================

def test_both_implementer_roads_send_the_projected_transport_schema(world):
    """A repair round really happens here, so the argv of the SECOND
    implementer call is evidence about the repair road rather than a
    repetition of the first."""
    evaluator(world, base._emits(base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._code_finding(mechanism_id="kurgu-mekanizma-a")],
        next_action="await_repair")))
    run(world)

    argvs = implementer_calls(world)
    assert len(argvs) >= 2, f"onarim yolu kosulmadi: {len(argvs)}"
    for index, call in enumerate(argvs):
        argv = call["argv"]
        token = argv[argv.index("--json-schema") + 1]
        sent = json.loads(token)
        assert sent == schemas.CLAUDE_TRANSPORT_SCHEMA, \
            f"{index}. cagri eski semayi tasidi"
        # the field the adapter derives is not asked of the model on
        # EITHER road
        assert "next_action" not in sent["properties"]
        assert "next_action" not in sent.get("required", ())


def test_both_implementer_roads_carry_the_protocol_matrix(world):
    """The instruction that replaces the removed field must reach the
    child on the repair road too -- a repair that never heard it would
    write the field and be refused."""
    evaluator(world, base._emits(base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._code_finding(mechanism_id="kurgu-mekanizma-a")],
        next_action="await_repair")))
    run(world)

    prompts = [call["stdin"] for call in implementer_calls(world)]
    assert len(prompts) >= 2, "onarim yolu kosulmadi"
    for index, prompt in enumerate(prompts):
        for line in execution.PROTOCOL_MATRIX:
            assert line in prompt, f"{index}. cagri protokol satirini tasimadi"


# =====================================================================
# H. B10-R1 -- ONE REPAIR FOR A PROVEN PYTEST ACCEPTANCE FAILURE
#
# The budget is ONE repair and it now has TWO possible spenders. Every
# test here is about the same question from a different side: can the two
# spenders ever add up to two patches.
# =====================================================================

_TEST_FILE = "pipeline/test_gecer.py"
_ACCEPTANCE_SENTINEL = "KURGU-SIZINTI-4711-GIZLI-DEGER"


def _failing_then_fixed_implementer(sentinel=_ACCEPTANCE_SENTINEL):
    """Round 0 leaves the selected test red; the repair makes it green.

    The failing assertion carries a sentinel in its MESSAGE, which is
    exactly the text a diagnostic must never carry: it is in the child's
    stdout, in the traceback and in the summary line, and it must appear
    in no field, no prompt, no journal record and no state document."""
    reply = base._implementer_reply(changed_files=[_TEST_FILE])
    # The two file contents are built as VALUES and then `repr`d whole.
    # Hand-quoting them inside the shim source is how a sentinel carrying
    # quotes turns into a syntax error in the stub, which then fails its
    # own `--version` probe and refuses the run in preflight -- measured.
    red = "def test_kurgu():\n    assert False, %r\n" % (sentinel,)
    # DIFFERENT from the fixture's baseline content on purpose: a repair
    # that restored the baseline byte for byte would leave an EMPTY
    # candidate, and this test is about a candidate that survives its
    # second gate and gets applied.
    green = "def test_kurgu():\n    assert 1 == 1\n"
    return ("if in_run():\n"
            "    if round_index() == 0:\n"
            f"        edit({_TEST_FILE!r}, {red!r})\n"
            "    else:\n"
            f"        edit({_TEST_FILE!r}, {green!r})\n"
            + base._emits(reply))


def test_a_proven_pytest_failure_spends_the_one_repair_and_approves(world,
                                                                    spy):
    """THE POSITIVE CONTROL for this whole section. An ordinary red test
    is now a repairable failure: the implementer gets one more turn,
    acceptance runs AGAIN on the repaired tree, and only then does the
    evaluator see the candidate for the first time."""
    implementer(world, _failing_then_fixed_implementer())
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert result.stop_reason == contract.StopReason.COMPLETED
    assert result.implementation_rounds == 1
    assert result.repair_rounds == 1
    assert result.evaluator_rounds == 1
    assert result.acceptance_passed is True
    assert set(result.applied_files) == {_TEST_FILE}
    assert len(implementer_calls(world)) == 2, "onarim turu kosulmadi"
    assert len(evaluator_calls(world)) == 1, "denetim bir kez kosmali"
    # the evaluator judged the REPAIRED tree, never the red one
    assert spy.index("acceptance") < spy.index("audit")
    assert spy.count("acceptance") == 2


def test_the_acceptance_repair_visited_sequence_is_the_frozen_one(world):
    """The order is the contract. ACCEPTANCE is entered once, REPAIRING
    once, then ACCEPTANCE_2 and FINAL_AUDITING -- and AUDITING is never
    entered at all, because the first candidate never passed its gate."""
    implementer(world, _failing_then_fixed_implementer())
    result = run(world)

    assert result.visited == [
        contract.State.PREFLIGHT, contract.State.IMPLEMENTING,
        contract.State.ACCEPTANCE, contract.State.REPAIRING,
        contract.State.ACCEPTANCE_2, contract.State.FINAL_AUDITING,
        contract.State.APPROVED]
    assert contract.State.AUDITING not in result.visited


def test_the_acceptance_repair_prompt_carries_identities_and_nothing_else(
        world):
    """WHAT THE MODEL IS TOLD. A file, a test name and a count -- and not
    the assertion message that was sitting in the same stdout line."""
    implementer(world, _failing_then_fixed_implementer())
    run(world)

    repair_prompt = implementer_calls(world)[1]["stdin"]
    assert "KABUL BULGULARI (kapali):" in repair_prompt
    assert f"  - {_TEST_FILE}::test_kurgu x1" in repair_prompt
    assert "assertion silme, gevsetme, skip veya xfail ekleme yok" \
        in repair_prompt
    assert "yalnizca onceki adayin degistirdigi dosyalarda onarim yap" \
        in repair_prompt
    # the sentinel travelled through the child's stdout, the traceback and
    # the summary line, and reached none of this
    assert _ACCEPTANCE_SENTINEL not in repair_prompt
    for yasak in ("Traceback", "AssertionError", "assert False",
                  str(world.repo), "stdout", "stderr"):
        assert yasak not in repair_prompt, f"{yasak!r} isteme sizdi"


def test_no_raw_acceptance_text_reaches_any_durable_record(world):
    """The identities live in the running process. Everything on disk is
    a closed word, a count or a boolean."""
    implementer(world, _failing_then_fixed_implementer())
    run(world)

    state_text = (world.state_dir / "state.json").read_text(encoding="utf-8")
    journal = (world.state_dir / "events.jsonl").read_text(encoding="utf-8")
    receipt = acceptance.receipt_path(world.state_dir).read_text(
        encoding="utf-8")
    for artefact, isim in ((state_text, "state"), (journal, "journal"),
                           (receipt, "receipt")):
        assert _ACCEPTANCE_SENTINEL not in artefact, f"{isim} sentinel tasidi"
        assert "test_kurgu" not in artefact, f"{isim} test adini tasidi"
        assert _TEST_FILE not in artefact, f"{isim} test yolunu tasidi"


def test_the_journal_records_the_closed_kind_count_and_decision(world):
    """A closed word, an exact count and a boolean -- and the schema that
    refuses anything else."""
    implementer(world, _failing_then_fixed_implementer())
    run(world)

    records = [json.loads(line) for line
               in (world.state_dir / "events.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]
    marked = [record for record in records
              if "acceptance_failure_kind" in record]
    assert len(marked) == 1, "kapali ozet tam bir kez yazilmali"
    (record,) = marked
    assert record["acceptance_failure_kind"] == \
        contract.AcceptanceFailureKind.REPAIRABLE_TESTS
    assert record["acceptance_failure_count"] == 1
    assert record["acceptance_repair_requested"] is True
    assert record["event"] == contract.EventCode.ACCEPTANCE_FINISHED
    # the schema is what makes "nothing else" true, so it is the schema
    # that is asked rather than a list repeated here
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schemas.EVENT_SCHEMA)
    for record in records:
        validator.validate(record)
        assert set(record) <= set(schemas.EVENT_SCHEMA["properties"])
    assert schemas.EVENT_SCHEMA["additionalProperties"] is False


def test_an_unrepairable_failure_is_recorded_as_such_and_stops(world, spy):
    """The other half of the journal contract: a failure nobody can
    classify is written down as `not_repairable`, with a zero count and
    no repair requested, and the run stops exactly as it always did."""
    implementer(world, "if in_run():\n"
                       f"    edit({_TEST_FILE!r}, 'def test_kurgu( :\\n')\n"
                + base._emits(base._implementer_reply(
                    changed_files=[_TEST_FILE])))
    result = run(world)

    (record,) = [json.loads(line) for line
                 in (world.state_dir / "events.jsonl").read_text(
                     encoding="utf-8").splitlines()
                 if line.strip() and "acceptance_failure_kind" in line]
    assert record["acceptance_failure_kind"] == \
        contract.AcceptanceFailureKind.NOT_REPAIRABLE
    assert record["acceptance_failure_count"] == 0
    assert record["acceptance_repair_requested"] is False
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.ACCEPTANCE_FAILED
    assert result.repair_rounds == 0
    assert len(implementer_calls(world)) == 1, "onarim istenmemeliydi"
    assert spy == ["acceptance"]


def test_a_zero_repair_budget_refuses_the_acceptance_repair(world, spy):
    """`max_repair_rounds = 0` is a budget, not a preference. The failure
    is just as repairable; there is simply nothing to spend."""
    retask(world, max_repair_rounds=0)
    implementer(world, _failing_then_fixed_implementer())
    result = run(world)

    assert len(implementer_calls(world)) == 1, "butcesiz onarim cagrildi"
    assert result.repair_rounds == 0
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.ACCEPTANCE_FAILED
    assert contract.State.REPAIRING not in result.visited
    assert spy == ["acceptance"]
    (record,) = [json.loads(line) for line
                 in (world.state_dir / "events.jsonl").read_text(
                     encoding="utf-8").splitlines()
                 if line.strip() and "acceptance_failure_kind" in line]
    # the evidence was still classified; only the decision differs
    assert record["acceptance_failure_kind"] == \
        contract.AcceptanceFailureKind.REPAIRABLE_TESTS
    assert record["acceptance_repair_requested"] is False


def test_a_second_acceptance_failure_is_terminal_with_no_third_call(world,
                                                                    spy):
    """THE SECOND PATCH DOES NOT EXIST. The repair does not fix the test,
    the second gate is red, and the run stops -- no third implementer
    call, no evaluator, nothing applied."""
    reply = base._implementer_reply(changed_files=[_TEST_FILE])
    # Both rounds leave the test red, but they write DIFFERENT text: a
    # repair that changed nothing would be refused for its declaration
    # rather than for its second gate, which is a different test.
    red_first = "def test_kurgu():\n    assert False, 'bir'\n"
    red_again = "def test_kurgu():\n    assert False, 'iki'\n"
    implementer(world, "if in_run():\n"
                       "    if round_index() == 0:\n"
                       f"        edit({_TEST_FILE!r}, {red_first!r})\n"
                       "    else:\n"
                       f"        edit({_TEST_FILE!r}, {red_again!r})\n"
                + base._emits(reply))
    result = run(world)

    assert len(implementer_calls(world)) == 2, "tek onarim kosmaliydi"
    assert len(evaluator_calls(world)) == 0, "denetim hic kosmamaliydi"
    assert result.repair_rounds == 1
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.ACCEPTANCE_FAILED
    assert result.applied_files == ()
    assert spy.count("acceptance") == 2
    assert "audit" not in spy


def test_an_evaluator_asking_for_changes_after_an_acceptance_repair_stops(
        world):
    """The two spenders share ONE budget. The acceptance road already
    spent it, so the first evaluator finding that asks for changes stops
    the run instead of buying a second patch."""
    implementer(world, _failing_then_fixed_implementer())
    evaluator(world, base._emits(base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._code_finding(mechanism_id="kurgu-mekanizma-yeni")],
        next_action="await_repair")))
    result = run(world)

    assert result.repair_rounds == 1, "ikinci onarim satin alindi"
    assert len(implementer_calls(world)) == 2
    assert len(evaluator_calls(world)) == 1
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.REPAIR_ROUNDS_EXHAUSTED
    assert contract.State.APPROVED not in result.visited


def test_the_second_acceptance_can_never_ask_for_a_repair(world):
    """STRUCTURAL, not behavioural. The second acceptance is called
    without the flag that makes a repair possible, and the frozen table
    has no edge out of ACCEPTANCE_2 into REPAIRING. Either alone would be
    enough; both is the point."""
    road = ast.parse(textwrap.dedent(
        inspect.getsource(runner._Run._repair_road)))
    calls_with_flag = [
        node for node in ast.walk(road)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "may_repair" for keyword in node.keywords)]
    assert calls_with_flag == [], "ikinci kabul onarim isteyebiliyor"
    assert contract.State.REPAIRING not in \
        contract.ALLOWED_TRANSITIONS[contract.State.ACCEPTANCE_2]
    assert contract.State.REPAIRING in \
        contract.ALLOWED_TRANSITIONS[contract.State.ACCEPTANCE]


def test_a_repair_may_not_touch_a_file_the_candidate_never_touched(world):
    """A REPAIR MAY NARROW A CANDIDATE, NEVER WIDEN IT. The file below is
    inside `allowed_paths`, so the task's own authorisation would let the
    model write it -- and the round's bound is strictly tighter, because
    whatever asked for the repair was about the files the candidate had
    already changed.

    The refusal is typed, nothing is applied, and no third call happens."""
    first = base._implementer_reply(changed_files=[_FIRST])
    second = base._implementer_reply(changed_files=[_SECOND])
    implementer(world, "if in_run():\n"
                       "    if round_index() == 0:\n"
                       f"        edit({_FIRST!r}, 'VALUE = 2\\n')\n"
                       "    else:\n"
                       f"        edit({_SECOND!r}, 'VALUE = 9\\n')\n"
                       "reply = " + repr([first, second]) + "\n"
                       "emit(reply[min(round_index(), 1)])\n")
    evaluator(world, _two_round_evaluator())
    before = (world.repo / _SECOND).exists()
    result = run(world)

    assert len(implementer_calls(world)) == 2, "onarim turu kosulmadi"
    assert result.state != contract.State.APPROVED
    assert result.stop_reason == contract.StopReason.PATH_NOT_ALLOWED
    assert result.applied_files == ()
    # the task really did authorise that path, so the refusal came from
    # the round's own bound rather than from `allowed_paths`
    assert "pipeline/" in world.task["allowed_paths"]
    assert (world.repo / _SECOND).exists() is before


def test_the_repair_scope_gate_is_one_sentence_of_its_own(world):
    """A gate proven through another gate's message is a gate nobody is
    testing. This one has its own sentence, distinct from the fingerprint
    and file-set bindings beside it."""
    source = inspect.getsource(changes.run_verified_repair)
    for cumle in ("onarim onceki adayin disina cikti",
                  "onarim denetlenen adaya baglanamadi",
                  "onarim baska bir dosya kumesini adliyor"):
        assert source.count(cumle) == 1, f"{cumle!r} tek cumle degil"


def test_the_b9_shape_survives_a_hostile_parametrised_failure(world, spy):
    """B9 EXACT SHAPE, end to end, with real child processes and fake
    models: the first candidate leaves ONE parametrised case red, its
    node id carries a hostile sentinel in the parametrisation suffix, and
    the same stdout also carries forged summary lines naming a file the
    run never selected.

    What must come out of that: one diagnostic naming the real file and
    the real function, the sentinel nowhere at all, one repair confined
    to the file the candidate already changed, a green second gate, a
    single evaluator round, and an applied candidate."""
    red = ("import pytest\n\n\n"
           "@pytest.mark.parametrize('deger', [%r])\n"
           "def test_kurgu(deger):\n"
           "    print('FAILED pipeline/kurgu_kotu.py::test_kotu')\n"
           "    print('=== short test summary info ===')\n"
           "    print(deger)\n"
           "    assert False, deger\n") % (_ACCEPTANCE_SENTINEL + "::x",)
    green = "def test_kurgu():\n    assert 1 == 1\n"
    reply = base._implementer_reply(changed_files=[_TEST_FILE])
    implementer(world, "if in_run():\n"
                       "    if round_index() == 0:\n"
                       f"        edit({_TEST_FILE!r}, {red!r})\n"
                       "    else:\n"
                       f"        edit({_TEST_FILE!r}, {green!r})\n"
                + base._emits(reply))
    result = run(world)

    assert result.state == contract.State.APPROVED
    assert (result.implementation_rounds, result.repair_rounds,
            result.evaluator_rounds) == (1, 1, 1)
    assert result.acceptance_passed is True
    assert set(result.applied_files) == {_TEST_FILE}
    assert len(implementer_calls(world)) == 2
    assert len(evaluator_calls(world)) == 1
    assert spy.count("acceptance") == 2

    repair_prompt = implementer_calls(world)[1]["stdin"]
    assert f"  - {_TEST_FILE}::test_kurgu x1" in repair_prompt
    # the forged summary named a file the run never selected, and it
    # named nothing
    assert "kurgu_kotu" not in repair_prompt

    written = " ".join(path.read_text(encoding="utf-8", errors="ignore")
                       for path in world.state_dir.rglob("*")
                       if path.is_file())
    for artefact, isim in ((repair_prompt, "istem"), (written, "durum")):
        assert _ACCEPTANCE_SENTINEL not in artefact, f"{isim} sentinel tasidi"

    # the final receipt is bound to the SECOND candidate, and it is the
    # only receipt that can carry a candidate into the checkout
    receipt = json.loads(acceptance.receipt_path(world.state_dir).read_text(
        encoding="utf-8"))
    assert receipt["status"] == acceptance.STATUS_PASSED
    assert (world.repo / _TEST_FILE).read_text(encoding="utf-8") == green


def test_both_repair_sources_use_one_seam_and_one_counter(world):
    """Whatever owed the repair, what follows is identical: the same
    change-set seam, the same budget counter, the same two gates after
    it."""
    source = inspect.getsource(runner._Run)
    assert source.count("changes.run_verified_repair") == 1
    assert source.count('self.rounds["repair"] += 1') == 1
    assert source.count("_repair_road") == 3   # definition and two callers


def test_requested_models_reach_argv_state_and_the_closed_journal(world):
    """The evidence says REQUESTED, not effective. Both persisted copies
    and each argv are fed from the same role-specific model value."""
    retask(world, implementer={"model": "opus"},
           evaluator={"model": "gpt-5.4"})

    result = run(world)
    assert result.state == contract.State.APPROVED
    state_doc = json.loads((world.state_dir / "state.json").read_text(
        encoding="utf-8"))
    assert state_doc["requested_models"] == {
        "implementer": "opus", "evaluator": "gpt-5.4"}

    starts = [entry for entry in _journal(world)
              if entry["event"] == contract.EventCode.MODEL_CALL_STARTED]
    assert [entry["requested_models"] for entry in starts] == [
        {"implementer": "opus"}, {"evaluator": "gpt-5.4"}]
    implementer_argv = implementer_calls(world)[0]["argv"]
    evaluator_argv = evaluator_calls(world)[0]["argv"]
    assert implementer_argv[implementer_argv.index("--model") + 1] == "opus"
    assert evaluator_argv[evaluator_argv.index("--model") + 1] == "gpt-5.4"

    persisted = (world.state_dir / "state.json").read_text(encoding="utf-8")
    persisted += runner_events.events_path(world.state_dir).read_text(
        encoding="utf-8")
    assert "reported_model" not in persisted
    assert "effective_model" not in persisted
