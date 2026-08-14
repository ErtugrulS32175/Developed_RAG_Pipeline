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
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import test_agent_loop_b2_changes as legacy
import test_agent_loop_contract as base
from tools.agent_loop import (acceptance, application, changes, contract,
                              execution, flat_workspace, runner,
                              runner_events)
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
    """Round 0 edits one file, the repair ADDS another.

    The two rounds must move DIFFERENT files, or the repair's delta
    cannot be told apart from the first round's work -- and telling them
    apart is the whole point of the candidate-versus-delta test below."""
    first = base._implementer_reply(changed_files=[_FIRST])
    second = base._implementer_reply(changed_files=[_SECOND])
    return ("if in_run():\n"
            "    if round_index() == 0:\n"
            f"        edit({_FIRST!r}, 'VALUE = 2\\n')\n"
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
    weigh and it is not something a repair round may argue with: the
    commands the task named are the contract, and a candidate that fails
    them is not a candidate."""
    implementer(world, "if in_run():\n"
                       "    edit('pipeline/test_gecer.py',"
                       " 'def test_kurgu():\\n    assert False\\n')\n"
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
               "stdout_bytes", "stderr_bytes", "cleanup_complete"}
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

    assert len(set(contract.ALL_FAILURE_CODES)) == 10
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
