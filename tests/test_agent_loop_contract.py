"""RED PHASE -- the agent loop's contract, before the runner exists.

Written BEFORE any runner, deliberately. Every requirement the loop is
supposed to enforce is named here first, so the implementation has one
target it cannot quietly miss.

NO REAL MODEL IS EVER CALLED, and that is now PROVEN rather than
asserted. The first draft of this file created fake `claude` and `codex`
executables and then handed them to nothing -- there was no seam to hand
them to, so a runner written against it would have found the real
binaries on PATH and spent real money while these tests claimed
otherwise. The binaries are a mandatory argument now, the package
contains no discovery function, and one test actually EXECUTES a fake
through the built argv and reads back what it recorded.

WHY PART OF THIS FILE IS STILL RED. `tools.agent_loop.runner` does not
exist -- Phase A is the contract, the schemas, the CLI seam and this
battery; Phase B is the runner. `_runner()` fails with one legible
message so every red test says the same true thing. Everything that can
be proven WITHOUT a runner is green today, including the seam itself.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import cli, contract, flat_workspace, schemas

REPO = Path(__file__).resolve().parent.parent
BACKSLASH = chr(92)

# What a fake binary is CALLED. `launch_contained` runs argv[0] directly
# and so does preflight's handshake, so a bare `.py` is not something
# either can start -- WinError 193 on Windows, and no interpreter named
# at all on POSIX. The shim shape is B2's, reused rather than reinvented.
SHIM_SUFFIX = ".cmd" if os.name == "nt" else ".sh"


def _runner():
    """The production runner, or a clear explanation of its absence.

    XFAIL, NOT FAIL. These tests are red BY DESIGN until Phase B lands,
    and `pytest.fail` made that design permanently red in CI -- both
    remotes run the whole suite on every push to main, so a deliberately
    unfinished battery turned the build red for everyone until the
    runner existed. `xfail` reports exactly the same fact ("this cannot
    run yet") without claiming the build is broken.

    It also cleans itself up: the moment `tools.agent_loop.runner`
    imports, this helper returns it and every test below runs for real.
    There is no marker left behind to forget -- which is the failure
    mode of decorating twenty-six tests by hand."""
    try:
        from tools.agent_loop import runner
    except ImportError as missing:            # pragma: no cover -- phase A
        pytest.xfail(
            f"tools.agent_loop.runner yok (Faz B henuz yazilmadi): {missing}")
    return runner


# ---------------------------------------------------------------------
# Fake CLIs -- real processes, scripted replies, zero cost
# ---------------------------------------------------------------------

_FAKE = '''\
import json, os, sys

ARGV = sys.argv[1:]
if ARGV == ["--version"]:
    # PREFLIGHT'S HANDSHAKE, and it must do nothing at all. It spends no
    # tokens, so it is not a model call and is deliberately not recorded
    # -- the tests that assert "no model was called" assert the record
    # file does not EXIST. It must also touch no tree: the working
    # directory here is the operator's own checkout.
    sys.stdout.write("kurgu 0.0\\n")
    sys.exit(0)

with open(__RECORD__, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": ARGV}) + "\\n")


def flag(name):
    return ARGV[ARGV.index(name) + 1] if name in ARGV else None


def round_index():
    return int(os.environ.get("AGENT_LOOP_ROUND", "0"))


def issued(name):
    return [item for item in os.environ.get(name, "").split(",") if item]


def emit(payload):
    """Answer the way THIS CLI actually answers.

    `codex exec` writes its final message into the file named by
    `--output-last-message`; `claude --print` answers on stdout with a
    RESULT ENVELOPE. The fake reads its OWN argv to decide, so nothing
    here invents a stdout road for the evaluator that the real binary
    does not have.

    THE ENVELOPE IS THE MEASURED ONE (B4-R3), taken from a real
    `claude 2.1.220` call: `type`/`subtype`/`is_error`, the payload
    under `structured_output`, the same payload rendered as text under
    `result`, plus the identifiers and usage the adapter must refuse to
    carry anywhere. A fake that kept printing a bare payload would be
    testing a protocol no binary speaks.

    The run id is the RUNNER'S, read from the environment it exported
    for this call. A recording with a hard-coded id describes a run that
    never happened, and the change-set gate refuses it by name."""
    payload = dict(payload)
    run_id = os.environ.get("AGENT_LOOP_RUN_ID")
    if run_id and payload.get("run_id") == "kurgu-run-1":
        payload["run_id"] = run_id
    text = json.dumps(payload)
    target = flag("--output-last-message")
    if target is None:
        sys.stdout.write(json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "terminal_reason": "completed", "stop_reason": "tool_use",
            "num_turns": 2, "duration_ms": 7523, "duration_api_ms": 5011,
            "total_cost_usd": 0.056757, "permission_denials": [],
            "session_id": "00000000-0000-4000-8000-000000000000",
            "uuid": "00000000-0000-4000-8000-000000000001",
            "usage": {"input_tokens": 4, "output_tokens": 5},
            "modelUsage": {}, "result": text, "structured_output": payload}))
        sys.stdout.flush()
    else:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)


def in_run():
    """Is a RUNNER driving this call?

    The implementer fake edits a file, and an edit is relative to
    whatever directory it was started in. Inside a run that is the
    candidate tree; outside one -- the two tests that execute a built
    argv directly -- it would be whatever the test happened to be
    standing in. So the edit is gated on the runner's own exported
    identity rather than on the caller remembering a `cwd`."""
    return bool(os.environ.get("AGENT_LOOP_RUN_ID"))


def edit(relative, text):
    with open(relative, "w", encoding="utf-8") as handle:
        handle.write(text)


try:
    sys.stdin.read()
except Exception:
    pass
__BODY__
'''


def _write_fake(path: Path, record: Path, body: str) -> Path:
    """An executable shim plus the python that does the work.

    `path` is the SHIM -- the thing `binaries` carries and the thing both
    `launch_contained` and the preflight handshake start -- and the
    helper is written beside it under the same stem. A test rewriting a
    fake therefore passes the shim it was given straight back in."""
    helper = path.with_suffix(".py")
    helper.write_text(
        _FAKE.replace("__RECORD__", repr(str(record))).replace(
            "__BODY__", body),
        encoding="utf-8")
    if os.name == "nt":
        path.write_text(f'@echo off\r\n"{sys.executable}" "{helper}" %*\r\n',
                        encoding="ascii")
    else:
        path.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{helper}" "$@"\n',
            encoding="ascii")
        path.chmod(0o755)
    return path


def _emits(payload: dict) -> str:
    return f"emit({payload!r})"


def _implementer_body(**overrides) -> str:
    """What the implementer fake DOES, not merely what it says.

    A reply that declares a changed file while changing nothing is a
    declaration mismatch -- the change-set gate refusing a lie -- so a
    fake that only printed would fail every test in this file for one
    reason that has nothing to do with what the test is about.

    The content carries the ROUND because a repair that rewrites a file
    with the bytes already in it has an EMPTY delta, and the repair seam
    refuses a reply declaring a file this call did not move."""
    return ("if in_run():\n"
            "    edit('pipeline/kurgu.py',"
            " 'VALUE = %d\\n' % (round_index() + 2))\n"
            + _emits(_implementer_reply(**overrides)))


def _implementer_reply(**overrides) -> dict:
    base = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "kurgu-run-1",
        "role": contract.Role.IMPLEMENTER,
        "status": contract.Status.IMPLEMENTED,
        "summary": "kurgu degisiklik",
        "changed_files": ["pipeline/kurgu.py"],
        "next_action": "await_acceptance",
    }
    base.update(overrides)
    return base


def _code_audit_reply(**overrides) -> dict:
    base = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "kurgu-run-1",
        "role": contract.Role.EVALUATOR,
        "audit_kind": contract.AuditKind.CODE,
        "status": contract.Status.APPROVED,
        "summary": "kurgu denetim",
        "next_action": "stop",
    }
    base.update(overrides)
    return base


def _code_finding(**overrides) -> dict:
    base = {
        "finding_id": "kurgu-bulgu-1",
        "mechanism_id": "kurgu-mekanizma-a",
        "severity": "high",
        "claim": "kurgu iddia",
        "reproduction_result": "reproduced",
        "required_action": "kurgu duzeltme",
    }
    base.update(overrides)
    return base


def _locked_audit_reply(**overrides) -> dict:
    base = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "0" * 32,
        "role": contract.Role.EVALUATOR,
        "audit_kind": contract.AuditKind.LOCKED,
        "status": contract.Status.APPROVED,
        "summary_code": contract.SummaryCode.CRITERIA_MET,
        "next_action": "stop",
    }
    base.update(overrides)
    return base


def _locked_finding(**overrides) -> dict:
    base = {
        "finding_id": "f" * 32,
        "mechanism_id": "e" * 32,
        "severity": "high",
        "error_class": contract.LockedFindingClass.WRONG_ROW,
        "case_count": 3,
    }
    base.update(overrides)
    return base


@pytest.fixture
def workspace(tmp_path):
    """A throwaway git repo, fake binaries, and the seam that joins them."""
    repo = tmp_path / "kurgu-depo"
    # NOT under tools/: that is an ancestor of the control plane, so a
    # task pointed at it is refused -- and every red test below would
    # then die in task validation instead of reaching the behaviour it
    # was written for. Red for the wrong reason still counts as red.
    (repo / "pipeline").mkdir(parents=True)
    for argv in (["init", "-q"], ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"],
                 # A REAL CHECKOUT AGREES WITH ITS OWN BLOBS. This one
                 # is written through `Path.write_text`, which spells
                 # `\n` as `\r\n` on Windows, and this machine's
                 # system-level `core.autocrlf` would normalise it back
                 # out on `git add` -- so every tracked file would differ
                 # from the baseline the flat workspace materialises from
                 # raw git objects, and the application layer's drift
                 # precondition would correctly refuse a candidate for a
                 # difference the fixture invented. The refusal itself is
                 # not worked around anywhere.
                 ["config", "core.autocrlf", "false"]):
        subprocess.run(["git", *argv], cwd=repo, check=True)
    (repo / "pipeline" / "kurgu.py").write_text("VALUE = 1\n",
                                                encoding="utf-8")
    # THE ACCEPTANCE GATE HAS TO BE ABLE TO PASS. `run_acceptance` runs
    # the registry's real argv against a mirror of the candidate, and a
    # tree with no test in it makes pytest exit 5 -- so every run would
    # stop at `acceptance_failed` and the happy path could not exist.
    (repo / "pipeline" / "test_gecer.py").write_text(
        "def test_kurgu():\n    assert True\n", encoding="utf-8")
    # THE STATE DIRECTORY IS GIT-IGNORED, which the contract requires of
    # any repository this loop runs in and which the real checkout is
    # separately tested for. The lock is the run's first outer boundary,
    # so `.agent-loop/run.lock` exists before preflight looks at the
    # tree -- and an un-ignored state directory would make every run
    # refuse itself as a dirty worktree.
    (repo / ".gitignore").write_text(f"{contract.STATE_DIR_NAME}/\n",
                                     encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "kurgu"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()

    bin_dir = tmp_path / "sahte-bin"
    bin_dir.mkdir()
    record = tmp_path / "cagrilar.jsonl"
    binaries = {
        "implementer": _write_fake(bin_dir / f"sahte_claude{SHIM_SUFFIX}",
                                   record, _implementer_body()),
        "evaluator": _write_fake(bin_dir / f"sahte_codex{SHIM_SUFFIX}",
                                 record, _emits(_code_audit_reply())),
    }
    task = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "objective": "kurgu hedef",
        "baseline_sha": head,
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
        # The manifest itself lives in the repository and is written
        # AFTER the baseline commit, so it is untracked -- and preflight's
        # dirty-tree gate is what the allowlist exists for. Nothing else
        # is excused: the dirty-tree test edits a tracked source file and
        # is still refused.
        "dirty_tree_allowlist": ["kurgu-task.json"],
    }
    task_path = repo / "kurgu-task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return {"repo": repo, "head": head, "task": task, "task_path": task_path,
            "binaries": binaries, "record": record, "record_fake": _write_fake}


@pytest.fixture(autouse=True)
def private_runner_root(tmp_path, monkeypatch):
    """Every test gets its OWN flat-workspace root.

    The runner really builds workspaces now, so a battery pointed at the
    shared runner-owned temp directory would create, list and delete
    holders a real agent loop could be using. This is the fixture B2 and
    B3's other suites already use, for the same reason."""
    private = tmp_path / "runner-koku"
    private.mkdir()
    monkeypatch.setattr(flat_workspace, "runner_temp_root", lambda: private)
    yield private
    shutil.rmtree(private, ignore_errors=True)


def test_the_base_workspace_task_is_valid(workspace):
    """The fixture's own task must PASS the schema, or every red test
    below dies in task validation instead of reaching the behaviour it
    was written for -- and the battery still looks uniformly red, for
    the wrong reason. Round-A3 made `tools/` illegal (it is an ancestor
    of the control plane) and this fixture was still using it."""
    Draft202012Validator(schemas.TASK_SCHEMA).validate(workspace["task"])
    Draft202012Validator(schemas.IMPLEMENTER_RESULT_SCHEMA).validate(
        _implementer_reply())
    Draft202012Validator(schemas.CODE_AUDIT_RESULT_SCHEMA).validate(
        _code_audit_reply())


# =====================================================================
# THE CLI SEAM -- green today, and the reason the fakes are real
# =====================================================================

def test_the_package_has_no_cli_discovery_at_all():
    """No PATH fallback anywhere. A runner that can discover a binary is
    a runner that can find the REAL one during a test and bill for it,
    so the ability simply does not exist -- a caller who forgets to pass
    a binary gets a TypeError, not an invoice."""
    offenders = []
    for module in (REPO / "tools" / "agent_loop").rglob("*.py"):
        body = module.read_text(encoding="utf-8")
        for probe in ("shutil.which", "find_executable", 'os.environ["PATH"]',
                      "os.getenv('PATH')", 'os.getenv("PATH")'):
            if probe in body:
                offenders.append(f"{module.name}:{probe}")
    assert not offenders, f"CLI kesfi bulundu: {offenders}"


def test_both_argv_builders_require_a_binary():
    """Mandatory, positional, no default."""
    for builder in (cli.build_implementer_argv, cli.build_evaluator_argv):
        import inspect

        first = list(inspect.signature(builder).parameters.values())[0]
        assert first.name == "binary"
        assert first.default is inspect.Parameter.empty


def test_the_implementer_argv_carries_every_required_flag(tmp_path):
    argv = cli.build_implementer_argv(
        tmp_path / "sahte_claude.py",
        budget_usd=1.0, allowed_tools=["Edit", "Read"])
    for flag in contract.CLAUDE_REQUIRED_FLAGS:
        assert flag in argv, f"eksik bayrak: {flag}"
    assert argv[0] == str(tmp_path / "sahte_claude.py")


def test_the_evaluator_argv_is_read_only_and_never_asks_for_approval(tmp_path):
    """`codex exec` REJECTS `-a/--ask-for-approval` -- verified against
    the installed binary -- so the approval policy travels as the config
    override that `exec` does accept. Mandating the flag that does not
    exist would break every evaluator call."""
    argv = cli.build_evaluator_argv(
        tmp_path / "sahte_codex.py", repo=tmp_path,
        schema_path=tmp_path / "s.json",
        last_message_path=tmp_path / "o.txt")
    for flag in contract.CODEX_REQUIRED_FLAGS:
        assert flag in argv, f"eksik bayrak: {flag}"
    assert argv[argv.index("--sandbox") + 1] == \
        contract.CODEX_SANDBOX_READ_ONLY
    flag, value = contract.CODEX_APPROVAL_OVERRIDE
    assert value in argv and argv[argv.index(value) - 1] == flag
    assert "--ask-for-approval" not in argv and "-a" not in argv


@pytest.mark.parametrize(
    "alan",
    ["binary", "repo", "schema_path", "last_message_path"],
    ids=["ikili", "depo", "sema-yolu", "son-mesaj-yolu"])
@pytest.mark.parametrize(
    "kotu",
    [b"/kurgu/yol", None, 5, True, ["/kurgu/yol"],
     "/kurgu/" + chr(0xD800)],
    ids=["bayt", "none", "sayi", "bool", "liste", "yalniz-vekil"])
def test_the_evaluator_builder_converts_every_token_exactly_once(
        tmp_path, alan, kotu):
    """B3: this builder was the LAST deferred `str()` conversion in the
    module. The implementer road was hardened and this one kept calling
    `str()` on the binary, the repository and both file paths -- so
    `b'/x'` became the literal `b'/x'` with quotes in it, `None` became
    `'None'`, and an object was free to answer differently the second
    time it was asked.

    The lone-surrogate case is why the type gate alone is not enough:
    `type(value) is str` is TRUE of it, and the failure would otherwise
    surface inside `Popen` as an OS error carrying the path."""
    saglikli = {"repo": tmp_path, "schema_path": tmp_path / "s.json",
                "last_message_path": tmp_path / "o.txt"}
    cagri = dict(saglikli, binary=tmp_path / "sahte_codex.py")
    cagri[alan] = kotu
    binary = cagri.pop("binary")
    with pytest.raises(cli.UnsafeInvocation) as refusal:
        cli.build_evaluator_argv(binary, **cagri)
    # the refused value is caller input and this text travels into
    # reports; the FIELD may be named, the value may not
    metin = str(refusal.value) + repr(refusal.value)
    assert "kurgu" not in metin, "ret metni reddedilen degeri tasiyor"


def test_the_evaluator_builder_still_builds_from_exact_inputs(tmp_path):
    """POSITIVE CONTROL: a builder that refuses everything would satisfy
    the rule above. Both `str` and `PathLike` remain acceptable, and the
    token that lands on argv is the ONE conversion's result."""
    argv = cli.build_evaluator_argv(
        str(tmp_path / "sahte_codex.py"), repo=tmp_path,
        schema_path=tmp_path / "s.json",
        last_message_path=str(tmp_path / "o.txt"), model="kurgu-model-1")
    assert all(type(token) is str for token in argv)
    assert argv[argv.index("--output-schema") + 1] == str(tmp_path / "s.json")
    assert argv[argv.index("--model") + 1] == "kurgu-model-1"


def test_the_evaluator_model_obeys_the_frozen_task_grammar(tmp_path):
    """`model` reached argv through a bare `str()` behind a truthiness
    test, so it was the one flag value with no grammar at all -- while
    the implementer builder next door had enforced the task schema's
    pattern all along. One grammar, both roads."""
    saglikli = {"repo": tmp_path, "schema_path": tmp_path / "s.json",
                "last_message_path": tmp_path / "o.txt"}
    binary = tmp_path / "sahte_codex.py"
    for bad in (_Taklitci("kurgu-model", "; rm -rf /"), "BUYUK-HARF",
                "-bastan-tire", 5, b"m", ""):
        with pytest.raises(cli.UnsafeInvocation):
            cli.build_evaluator_argv(binary, model=bad, **saglikli)
    # absent stays absent rather than becoming the literal "None"
    assert "--model" not in cli.build_evaluator_argv(binary, **saglikli)


@pytest.mark.parametrize(
    "extra",
    [["--dangerously-skip-permissions"],
     ["--dangerously-bypass-approvals-and-sandbox"],
     ["--sandbox", "danger-full-access"],
     ["--permission-mode", "bypassPermissions"]],
    ids=["claude-bypass", "codex-bypass", "yazilabilir-sandbox",
         "izin-atlama"])
def test_a_bypass_flag_is_refused_on_the_built_argv(extra):
    """Checked on the BUILT argv, not on the caller's intent: a rule that
    inspects inputs misses whatever the builder actually emitted."""
    with pytest.raises(cli.UnsafeInvocation):
        cli.assert_safe_argv(["kurgu", *extra])


def test_an_empty_tool_allowlist_is_refused(tmp_path):
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv(tmp_path / "k.py",
                                   budget_usd=1.0, allowed_tools=[])


def test_a_fake_binary_really_runs_through_the_built_argv(workspace, tmp_path):
    """THE SEAM, end to end and without a runner. The argv the builder
    produces is executed, the fake answers, and the recording proves the
    call reached the fake rather than anything on PATH.

    ARGV[0] IS STARTED DIRECTLY, with no interpreter in front of it,
    because that is what `launch_contained` and preflight's handshake
    both do -- prefixing `sys.executable` here would prove a fake nobody
    launches that way.

    THE ANSWER IS READ FROM THE FILE. `codex exec` writes its final
    message into `--output-last-message`, the adapter reads it from
    there, and a fake that printed a reply on stdout instead would be
    proving a road the real evaluator does not have."""
    last_message = tmp_path / "o.txt"
    argv = cli.build_evaluator_argv(
        workspace["binaries"]["evaluator"], repo=workspace["repo"],
        schema_path=tmp_path / "s.json", last_message_path=last_message)
    done = subprocess.run(argv, input="", text=True, capture_output=True,
                          cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "", "denetci yanitini stdout'a basti"
    reply = json.loads(last_message.read_text(encoding="utf-8"))
    Draft202012Validator(schemas.CODE_AUDIT_RESULT_SCHEMA).validate(reply)
    recorded = [json.loads(line) for line in
                workspace["record"].read_text(encoding="utf-8").splitlines()]
    assert recorded, "sahte CLI cagrilmadi"
    assert "exec" in recorded[-1]["argv"]
    assert contract.CODEX_SANDBOX_READ_ONLY in recorded[-1]["argv"]


def test_the_implementer_fake_really_runs_with_schema_budget_and_tools(
        workspace, tmp_path):
    """The implementer's half of the seam proof. The evaluator side was
    executed for real while this one was only inspected as a list -- so
    the claim "no real Claude is reachable" rested on argv shape alone.
    Here the fake is EXECUTED, the prompt goes over stdin, and the flags
    the run depends on are read back out of what the process recorded."""
    argv = cli.build_implementer_argv(
        workspace["binaries"]["implementer"],
        budget_usd=0.25, allowed_tools=["Edit", "Read"])
    # STARTED DIRECTLY and in a directory of its own: argv[0] is the
    # shim, which is what the loop actually launches, and a fake run
    # anywhere else would edit whatever tree happened to be the working
    # directory.
    done = subprocess.run(argv, input="kurgu istem", text=True,
                          capture_output=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    # The fake answers with the RESULT ENVELOPE the real CLI was
    # measured to produce (B4-R3), so the payload is unwrapped here
    # exactly as the adapter unwraps it -- and it is the AUTHORITATIVE
    # schema that judges the payload, never the transport copy that
    # travelled on the argv.
    envelope = json.loads(done.stdout)
    assert envelope["type"] == "result"
    assert envelope["subtype"] == "success"
    assert envelope["is_error"] is False
    Draft202012Validator(schemas.AUTHORITATIVE_RESULT_SCHEMA).validate(
        envelope["structured_output"])

    recorded = [json.loads(line) for line in
                workspace["record"].read_text(encoding="utf-8").splitlines()]
    assert recorded, "sahte implementer cagrilmadi"
    seen = recorded[-1]["argv"]
    assert "--print" in seen
    # the INLINE canonical TRANSPORT schema, byte-identical after a real
    # process boundary -- the py-fake is executed via sys.executable, so
    # the token survives argv quoting exactly. What travels is the copy
    # the API can compile; the acceptance authority stays local (B4-R2).
    assert seen[seen.index("--json-schema") + 1] == \
        schemas.IMPLEMENTER_TRANSPORT_BINDING.canonical_json
    assert seen[seen.index("--max-budget-usd") + 1] == "0.25"
    assert seen[seen.index("--allowedTools") + 1] == "Edit"
    # the prompt is nowhere on the command line
    assert "kurgu istem" not in " ".join(seen)


def test_neither_builder_can_be_pointed_at_a_real_binary_by_accident():
    """There is no default and no discovery, so reaching the real CLI
    takes an explicit path -- which a test never provides."""
    import inspect

    for builder in (cli.build_implementer_argv, cli.build_evaluator_argv):
        source = inspect.getsource(builder)
        assert "claude" not in source.lower() or "binary" in source
        assert "PATH" not in source


@pytest.mark.parametrize(
    "tools", [["Bash"], ["Agent"], ["WebFetch"], ["Read", "Bash"]],
    ids=["bash", "agent", "web", "karisik"])
def test_the_implementer_can_never_be_handed_bash_or_an_agent(tools):
    """A Claude holding Bash can `git add`, `git commit`, `git push`,
    install a dependency or reach the network -- every one of them a
    human gate the runner never sees, because the gate lives in the
    runner and the action happened inside the model's own tool call.
    The implementer reads and edits; the RUNNER runs the commands."""
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude",
                                   budget_usd=1.0, allowed_tools=tools)


def test_the_implementer_tool_list_is_a_contract_constant_not_an_argument():
    assert contract.IMPLEMENTER_ALLOWED_TOOLS == (
        "Read", "Glob", "Grep", "Edit", "Write")
    for banned in ("Bash", "Agent", "WebFetch", "WebSearch"):
        assert banned in contract.IMPLEMENTER_FORBIDDEN_TOOLS
        assert banned not in contract.IMPLEMENTER_ALLOWED_TOOLS
    argv = cli.build_implementer_argv("/kurgu/claude",
                                      budget_usd=1.0)
    tail = argv[argv.index("--allowedTools") + 1:]
    assert tail[:len(contract.IMPLEMENTER_ALLOWED_TOOLS)] == list(
        contract.IMPLEMENTER_ALLOWED_TOOLS)


# =====================================================================
# R2B -- ONE canonical schema binding: bytes, hash, argv, validator
# =====================================================================
#
# `--json-schema` takes INLINE JSON -- measured against the installed
# CLI -- and the builder used to pass a caller-chosen FILE PATH there,
# while the validator used a separate live dictionary. Nothing tied the
# two. The binding below is the one authority: canonical bytes, their
# SHA-256, the exact argv value, and the validator, all from the same
# serialization.

def test_canonical_json_is_order_independent_compact_and_deterministic():
    import hashlib

    first = schemas.canonical_json(
        {"b": 1, "a": [1, 2], "c": {"y": 0, "x": 1}})
    second = schemas.canonical_json(
        {"c": {"x": 1, "y": 0}, "a": [1, 2], "b": 1})
    assert first == second, "ekleme sirasi baytlari degistirdi"
    assert " " not in first and chr(10) not in first, "kompakt degil"
    assert first.encode("utf-8").isascii()
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == \
        hashlib.sha256(second.encode("utf-8")).hexdigest()


def test_canonical_json_refuses_what_json_cannot_carry():
    for poison in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            schemas.canonical_json({"deger": poison})


def test_a_non_schema_document_cannot_become_a_binding():
    from jsonschema import SchemaError

    with pytest.raises(SchemaError):
        schemas.SchemaBinding({"type": "boyle-bir-tur-yok"})


def test_two_fresh_processes_agree_on_the_binding_hash():
    """Determinism ACROSS interpreters, not just within one: dict
    iteration order, environment and platform must not move the bytes,
    or a hash recorded today stops naming the schema tomorrow."""
    code = ("from tools.agent_loop import schemas;"
            "print(schemas.IMPLEMENTER_SCHEMA_BINDING.sha256)")
    hashes = set()
    for _ in range(2):
        done = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        hashes.add(done.stdout.strip())
    assert hashes == {schemas.IMPLEMENTER_SCHEMA_BINDING.sha256}
    assert re.fullmatch(r"[0-9a-f]{64}",
                        schemas.IMPLEMENTER_SCHEMA_BINDING.sha256)


def test_the_binding_outlives_mutation_of_its_source_dictionary():
    """The mutable source dictionary must not stay a second live
    authority once the binding exists."""
    import hashlib

    source = {"type": "object", "properties": {"a": {"type": "string"}}}
    binding = schemas.SchemaBinding(source)
    before = (binding.canonical_json, binding.sha256)
    source["properties"]["a"] = {"type": "integer"}
    assert (binding.canonical_json, binding.sha256) == before
    binding.validate({"a": "metin"})        # judged by the OLD schema
    assert hashlib.sha256(binding.canonical_bytes).hexdigest() == \
        binding.sha256


def test_the_binding_refuses_attribute_rewrites():
    """R2B-R1: the binding's fields were ordinary writable attributes,
    and an evaluator probe rewrote the canonical text and the hash
    TOGETHER -- the substitute entered the argv, the hash comparison
    agreed with itself, and a permissive validator accepted an
    arbitrary reply. Assignment and deletion refuse now, on the shared
    binding AND on a fresh instance, and the values are proven
    unchanged after every attempt. (`object.__setattr__` remains a
    language-level bypass; the claim is the NORMAL API, and the
    docstring says so.)"""
    fresh = schemas.SchemaBinding({"type": "object"})
    for target in (schemas.IMPLEMENTER_SCHEMA_BINDING, fresh):
        witness = (target.canonical_json, target.canonical_bytes,
                   target.sha256)
        for field in ("canonical_json", "canonical_bytes", "sha256",
                      "_validator", "yeni_alan"):
            with pytest.raises(AttributeError):
                setattr(target, field, "{}")
        for field in ("canonical_json", "sha256"):
            with pytest.raises(AttributeError):
                delattr(target, field)
        assert (target.canonical_json, target.canonical_bytes,
                target.sha256) == witness


def test_the_implementer_argv_schema_is_inline_not_a_path(tmp_path):
    import hashlib
    import os

    argv = cli.build_implementer_argv(tmp_path / "sahte_claude.py",
                                      budget_usd=1.0)
    assert argv.count("--json-schema") == 1
    token = argv[argv.index("--json-schema") + 1]
    # B4-R2: the TRANSPORT binding is what travels. It is still inline,
    # still canonical and still hash-pinned -- only weaker, and never an
    # acceptance authority.
    assert token == schemas.IMPLEMENTER_TRANSPORT_BINDING.canonical_json
    assert json.loads(token) == schemas.CLAUDE_TRANSPORT_SCHEMA
    # os.path.exists, NOT Path.exists: on Linux a 3KB "filename" makes
    # pathlib raise ENAMETOOLONG instead of answering False
    assert not os.path.exists(token), "sema hala bir dosya yolu"
    assert hashlib.sha256(token.encode("utf-8")).hexdigest() == \
        schemas.IMPLEMENTER_TRANSPORT_BINDING.sha256


# =====================================================================
# B2A -- the CALL BOUNDARY: validate once, canonicalize once, use only
# the canonical value
# =====================================================================
#
# ONE mechanism, inventoried across every builder input: a value is
# CHECKED in one representation and USED in another, and a Python
# object can make those two disagree. The allowlist compared an object
# with `==` and the argv then took its `__str__` -- so `Read` passed the
# gate and `Bash` reached the command line.


class _Taklitci:
    """Equal to a permitted value; converts to a forbidden one."""

    def __init__(self, gorunen, gercek):
        self.gorunen, self.gercek = gorunen, gercek

    def __eq__(self, other):
        return other == self.gorunen

    def __hash__(self):
        return hash(self.gorunen)

    def __str__(self):
        return self.gercek


def test_a_refused_tool_name_is_never_echoed_back(tmp_path):
    """B2A-R1: the rejected names were formatted straight into the
    exception text, so arbitrary caller input travelled into whatever a
    report writes. A count and the FROZEN allowlist may leave; the
    input may not."""
    nobetci = "KURGU-GIZLI-ARAC-NOBETCISI-" + "z" * 8
    with pytest.raises(cli.UnsafeInvocation) as refusal:
        cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                   allowed_tools=[nobetci])
    metin = str(refusal.value) + repr(refusal.value)
    assert nobetci not in metin, "ret metni reddedilen arac adini tasiyor"
    # the contract's own allowlist is not caller input and may appear
    assert "Read" in metin


@pytest.mark.parametrize("yasak", ["Bash", "Agent", "WebFetch", "WebSearch"])
def test_a_deceptive_tool_object_can_never_reach_the_argv(yasak, monkeypatch):
    """THE critical one. A Claude holding Bash can `git push`, install a
    dependency or reach the network -- every one of them a human gate
    the runner never sees. The allowlist has to be asked of an exact
    string, because only an exact string cannot answer differently the
    second time it is asked.

    TWO layers refuse this today: the allowlist's own exact-type check
    and the final argv net. So the second half asks the allowlist ALONE
    -- a mutation run showed the first assertion staying green with the
    exact-type check deleted, because the net behind it caught the
    object instead. A layer whose removal changes nothing observable is
    a layer nobody is testing."""
    sahte = _Taklitci("Read", yasak)
    assert sahte == "Read" and str(sahte) == yasak, "senaryo kurulmadi"
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                   allowed_tools=[sahte])

    monkeypatch.setattr(cli, "assert_safe_argv", lambda argv: list(argv))
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                   allowed_tools=[sahte])


def test_exact_tool_strings_still_build_an_argv():
    """POSITIVE CONTROL: a builder that refuses everything would pass
    every test above."""
    argv = cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                      allowed_tools=["Edit", "Read"])
    tail = argv[argv.index("--allowedTools") + 1:]
    assert tail[:2] == ["Edit", "Read"]
    assert all(type(token) is str for token in argv)


def test_a_repeated_tool_is_refused_rather_than_left_ambiguous():
    """Repeated flags have no agreed CLI meaning here, so the duplicate
    is a caller mistake rather than something to guess at."""
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                   allowed_tools=["Read", "Read"])


def test_the_stdin_flag_must_be_exactly_true():
    """Truthiness is not a type check: an object with `__bool__` is not
    the caller promising the prompt stays off the command line."""
    class DogruGibi:
        def __bool__(self):
            return True

    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                   prompt_is_stdin=DogruGibi())
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                   prompt_is_stdin=1)


def test_the_permission_mode_is_pinned_to_the_agreed_exact_string():
    """The deceptive object was already caught downstream -- but only
    because `bypassPermissions` happens to be on the forbidden-value
    list. An unlisted alternate mode went straight through, so the mode
    is pinned by exact value here, in front of that net."""
    sahte = _Taklitci(cli.IMPLEMENTER_PERMISSION_MODE, "plan")
    for mode in (sahte, "plan", "bypassPermissions", b"acceptEdits", None):
        with pytest.raises(cli.UnsafeInvocation):
            cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                       permission_mode=mode)
    argv = cli.build_implementer_argv(
        "/kurgu/claude", budget_usd=1.0,
        permission_mode=cli.IMPLEMENTER_PERMISSION_MODE)
    assert argv[argv.index("--permission-mode") + 1] == \
        cli.IMPLEMENTER_PERMISSION_MODE


def test_assert_safe_argv_refuses_a_token_that_is_not_an_exact_string():
    """The final net. It used to `str()` whatever it was handed, which
    is the deferred conversion this whole package removes: by the time
    argv exists there must be nothing left to convert."""
    for token in (_Taklitci("guvenli", "Bash"), 5, None, b"kurgu",
                  Path("/kurgu")):
        with pytest.raises(cli.UnsafeInvocation):
            cli.assert_safe_argv(["kurgu", token])
    assert cli.assert_safe_argv(["kurgu", "--print"]) == ["kurgu", "--print"]


@pytest.mark.parametrize(
    "kotu", [0, 0.0, -5, -0.001, float("nan"), float("inf"),
             float("-inf"), 101, 100.0001, True],
    ids=["sifir-int", "sifir", "negatif-int", "negatif", "nan", "sonsuz",
         "eksi-sonsuz", "tavan-ustu-int", "tavan-ustu", "bool"])
def test_the_public_builder_enforces_the_budget_bounds_itself(kotu):
    """B2A-R1: the bounds lived only in `execution`, and the builder is
    a PUBLIC callable that tests hand straight to a subprocess -- called
    directly it spelled `101`, `0`, `-5`, `nan` and `inf` onto the
    command line. A rule enforced on one of two roads is a rule for
    people who take that road.

    The refused value never appears in the message: it is caller input
    and this text travels into reports."""
    with pytest.raises(cli.UnsafeInvocation) as refusal:
        cli.build_implementer_argv("/kurgu/claude", budget_usd=kotu)
    metin = str(refusal.value) + repr(refusal.value)
    assert str(kotu) not in metin, "ret metni reddedilen degeri tasiyor"


@pytest.mark.parametrize("iyi", [0.375, 1, 100, 100.0])
def test_the_builder_still_spells_an_in_range_budget(iyi):
    """The boundary in the other direction, including the schema
    maximum exactly -- or the rule above is just "refuse budgets"."""
    argv = cli.build_implementer_argv("/kurgu/claude", budget_usd=iyi)
    assert argv[argv.index("--max-budget-usd") + 1] == str(iyi)


def test_one_budget_authority_serves_both_roads():
    """The rule may not be copied into two modules: a second copy is a
    second place to forget. Both the builder and the adapter refuse the
    same value, and the adapter's own ceiling IS the CLI's."""
    from tools.agent_loop import execution as execution_module

    assert execution_module.MAX_BUDGET_USD is cli.MAX_BUDGET_USD
    assert cli.MAX_BUDGET_USD == schemas.TASK_SCHEMA["properties"][
        "max_budget_usd"]["maximum"]
    assert cli.exact_budget(0.375) == 0.375
    with pytest.raises(cli.UnsafeInvocation):
        cli.exact_budget(cli.MAX_BUDGET_USD + 1)


def test_the_builder_refuses_a_model_outside_the_frozen_schema():
    """`model` crosses into argv too. The pattern and the length come
    from the task schema, not from a second grammar invented here."""
    rule = schemas.TASK_SCHEMA["properties"]["implementer"]["properties"][
        "model"]
    for bad in (_Taklitci("kurgu-model", "; rm -rf /"), "BUYUK-HARF",
                "-bastan-tire", "a" * (rule["maxLength"] + 1), 5, b"m"):
        with pytest.raises(cli.UnsafeInvocation):
            cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                       model=bad)
    argv = cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0,
                                      model="kurgu-model-1")
    assert argv[argv.index("--model") + 1] == "kurgu-model-1"
    assert "--model" not in cli.build_implementer_argv("/kurgu/claude",
                                                       budget_usd=1.0)


def test_the_builder_refuses_a_budget_that_is_not_an_exact_number():
    """Defence in depth: the adapter canonicalises the budget, and the
    builder still refuses anything that could stringify into a
    different number than the one that was checked."""
    class YalanciButce(float):
        def __str__(self):
            return "999999"

    for bad in (YalanciButce(1.0), True, "1.0", None):
        with pytest.raises(cli.UnsafeInvocation):
            cli.build_implementer_argv("/kurgu/claude", budget_usd=bad)
    argv = cli.build_implementer_argv("/kurgu/claude", budget_usd=0.375)
    assert argv[argv.index("--max-budget-usd") + 1] == "0.375"


def test_the_builder_converts_a_path_like_binary_exactly_once():
    """`__fspath__` is the one conversion; `__str__` is never consulted
    afterwards, so the two cannot name different programs."""
    class IkiYuzlu:
        def __fspath__(self):
            return "/kurgu/denetlenen"

        def __str__(self):
            return "/kurgu/calisan"

    argv = cli.build_implementer_argv(IkiYuzlu(), budget_usd=1.0)
    assert argv[0] == "/kurgu/denetlenen"
    assert type(argv[0]) is str


def test_no_implementer_schema_is_read_from_or_written_to_disk():
    """Structural: `schema_path` is gone from the implementer builder
    and no file API appears in the CLI module -- a schema file would
    reintroduce the read/change/use ambiguity R2B removed. (Evaluator
    `--output-schema` is B3's business and out of scope.)"""
    import inspect

    builder = inspect.getsource(cli.build_implementer_argv)
    assert "schema_path" not in builder
    body = Path(cli.__file__).read_text(encoding="utf-8")
    for file_api in (".read_text", ".read_bytes", ".write_text",
                     ".write_bytes", "open("):
        assert file_api not in body, f"cli.py dosya APIsi kullaniyor: " \
            f"{file_api}"


def test_a_failed_model_process_has_its_own_closed_stop_reason():
    """A process that failed after preflight must not blame preflight,
    and a terminal state cannot carry an open-ended or missing reason."""
    reason = contract.StopReason.MODEL_PROCESS_FAILED
    assert reason == "model_process_failed"
    assert reason in contract.ALL_STOP_REASONS
    assert reason != contract.StopReason.PREFLIGHT_FAILED


# =====================================================================
# THE CONTROL PLANE -- a running loop may not rewrite its own rules
# =====================================================================

@pytest.mark.parametrize("path", list(contract.CONTROL_PLANE_PATHS))
def test_a_task_cannot_grant_edit_rights_over_the_loop_itself(workspace,
                                                               path):
    """`allowed_paths: ["tools/agent_loop/"]` would let the implementer
    rewrite the command registry, the forbidden-flag list, the schemas
    and the tests that judge it -- and then pass against the rules it
    had just written. Naming `contract_change` as a human gate does not
    prevent that; refusing the permission does."""
    task = dict(workspace["task"], allowed_paths=[path])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)


@pytest.mark.parametrize(
    "ancestor", ["tools/", "tests/", "eval/", "scripts/", "eval/tools/"],
    ids=["tools", "tests", "eval", "scripts", "eval-tools"])
def test_a_task_cannot_grant_a_parent_of_the_control_plane_either(
        workspace, ancestor):
    """`tools/` was accepted because only the exact protected prefix was
    refused -- and `tools/` contains `tools/agent_loop/`. So do `tests/`,
    `eval/` and `scripts/`, which hold the loop's tests, the leak
    scanner and the gate script. Permission over a parent is permission
    over the child."""
    task = dict(workspace["task"], allowed_paths=[ancestor])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)


def test_the_verification_tools_are_part_of_the_control_plane():
    """A task that can edit the leak scanner can silence it; one that
    can edit the gate script can make the gate pass."""
    covered = " ".join(contract.CONTROL_PLANE_PATHS)
    assert "eval/tools/leak_scan.py" in covered
    assert "scripts/p0_gate.sh" in covered
    assert contract.TASK_MANIFEST_IS_IMMUTABLE is True


def test_phase_b_is_frozen_to_flat_workspace_isolation():
    """Hashing after the fact detects damage that has already happened
    and been left in the operator's tree. The implementer runs in a FLAT
    WORKSPACE materialised from the baseline's raw git objects: two
    independent trees with no `.git` in either, the model confined to the
    implementer one, and only a verified allowed-path change coming back.

    The disposable git worktree this replaced is not merely deprecated --
    it carried a `.git` link into the model's reach, which is how git
    lost the evidence job twice. The old constant is gone rather than
    left beside the new one, because two frozen design statements about
    the same question is one too many."""
    assert contract.IMPLEMENTER_RUNS_IN_FLAT_WORKSPACE is True
    assert not hasattr(contract, "IMPLEMENTER_RUNS_IN_DISPOSABLE_WORKTREE"), \
        "eski normatif sabit hala duruyor"


def test_the_control_plane_covers_the_loop_and_its_own_tests():
    """If the tests were editable the loop could delete the test that
    catches it editing them."""
    covered = " ".join(contract.CONTROL_PLANE_PATHS)
    assert "tools/agent_loop/" in covered
    assert "test_agent_loop_contract.py" in covered
    assert contract.CONTROL_PLANE_VIOLATION_IS_TERMINAL is True
    assert contract.StopReason.CONTROL_PLANE_MODIFIED in         contract.ALL_STOP_REASONS


# =====================================================================
# =====================================================================
# THE COMMAND REGISTRY -- a task names a command, never spells one
# =====================================================================

def test_a_task_cannot_spell_its_own_argv(workspace):
    """`["git", "push"]` is a well-formed argv list, which is exactly why
    writing one may not be a task-file privilege."""
    task = dict(workspace["task"])
    task["acceptance_commands"] = [{"command_id": "pytest_full",
                                    "argv": ["git", "push"]}]
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)


def test_an_unregistered_command_id_is_refused(workspace):
    task = dict(workspace["task"],
                acceptance_commands=[{"command_id": "kurgu-bilinmeyen"}])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)
    with pytest.raises(cli.UnsafeInvocation):
        cli.resolve_registry_command("kurgu-bilinmeyen",
                                     contract.COMMAND_REGISTRY)


def test_the_registry_runs_no_version_control_shell_or_installer():
    """Defence in depth over the registry itself, so a future "just one
    small command" cannot slip past review."""
    offenders = []
    for command_id, entry in contract.COMMAND_REGISTRY.items():
        program = entry["argv"][0].lower()
        if program in contract.REGISTRY_FORBIDDEN_PROGRAMS:
            offenders.append(command_id)
        joined = " ".join(entry["argv"]).lower()
        for banned in ("git ", "powershell", "pip install", "npm ", "curl "):
            if banned in joined:
                offenders.append(f"{command_id}:{banned.strip()}")
    assert not offenders, f"kayitta yasak program: {offenders}"


@pytest.mark.parametrize(
    "path", ["../disari.py", "/mutlak.py", "-rf",
             f"tools{BACKSLASH}..{BACKSLASH}gizli.py"],
    ids=["ust-dizin", "mutlak", "bayrak-gibi", "ters-bolu"])
def test_a_registry_path_argument_cannot_escape_or_become_a_flag(path):
    with pytest.raises(cli.UnsafeInvocation):
        cli.resolve_registry_command("pytest_selected",
                                     contract.COMMAND_REGISTRY, paths=[path])


def test_a_command_that_takes_no_paths_refuses_them():
    with pytest.raises(cli.UnsafeInvocation):
        cli.resolve_registry_command("pytest_full", contract.COMMAND_REGISTRY,
                                     paths=["tests/test_x.py"])


def test_the_human_gates_are_not_task_configuration(workspace):
    """A task that could set `user_gates: []` could authorise its own
    commit. The gates are a contract constant; the schema has no field
    for them at all."""
    assert "user_gates" not in schemas.TASK_SCHEMA["properties"]
    task = dict(workspace["task"], user_gates=[])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)
    for gate in ("git_add", "git_commit", "git_push"):
        assert gate in contract.USER_APPROVAL_REQUIRED


def test_model_configuration_cannot_smuggle_cli_flags(workspace):
    """`implementer`/`evaluator` were free-form objects, so a task could
    carry the very flags the contract forbids past validation."""
    task = dict(workspace["task"],
                implementer={"flags": ["--dangerously-skip-permissions"]})
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)


# =====================================================================
# SCHEMAS -- the four reproduced fail-opens, kept as regressions
# =====================================================================

def test_every_schema_compiles_and_refuses_unknown_fields():
    leaky = []
    for name, schema in schemas.ALL_SCHEMAS.items():
        Draft202012Validator.check_schema(schema)
        if schema.get("additionalProperties") is not False:
            leaky.append(name)
    assert not leaky, f"bilinmeyen alani kabul eden sema: {leaky}"


def test_no_schema_declares_a_dialect_url():
    """The dialect is chosen in code by instantiating
    `Draft202012Validator`, so the `$schema` URL carried no information
    -- and it carried a four-digit year, a filename-fragment class in
    this corpus, which made the leak scanner fail closed. Removing a
    redundant key beats teaching the scanner an exception."""
    # The SCHEMA OBJECTS, not the source text: the module's docstring
    # explains the absence and says the word, which a text search
    # cannot tell apart from a declaration.
    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    for name, schema in schemas.ALL_SCHEMAS.items():
        assert "$schema" not in set(keys(schema)), f"{name} dialect ilan ediyor"
    code = re.sub(r'""".*?"""', "",
                  (REPO / "tools" / "agent_loop" / "schemas.py").read_text(
                      encoding="utf-8"), flags=re.S)
    assert "json-schema.org" not in code


def test_an_implementer_cannot_return_the_evaluators_verdict():
    """A model grading its own work. The Python constant said so from
    the start; the SCHEMA did not, and the schema is the enforcement."""
    payload = _implementer_reply(status=contract.Status.APPROVED,
                                 next_action="stop")
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.IMPLEMENTER_RESULT_SCHEMA).validate(
            payload)


def test_an_evaluator_cannot_claim_to_have_implemented_anything():
    payload = _code_audit_reply(status=contract.Status.IMPLEMENTED)
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.CODE_AUDIT_RESULT_SCHEMA).validate(
            payload)


def test_an_evaluator_result_has_no_changed_files_field():
    """An evaluator reporting edited files has edited files."""
    assert "changed_files" not in \
        schemas.CODE_AUDIT_RESULT_SCHEMA["properties"]


@pytest.mark.parametrize(
    "path",
    ["/mutlak/yol.py", "C:/mutlak/yol.py", "../disari.py",
     "tools/../../disari.py",
     f"tools{BACKSLASH}..{BACKSLASH}contracts{BACKSLASH}x.md",
     f"tools{BACKSLASH}gizli.py"],
    ids=["mutlak", "surucu", "ust-dizin", "gomulu-ust-dizin",
         "ters-bolu-kacisi", "ters-bolu-ayrac"])
def test_a_path_escaping_the_repo_is_rejected(path):
    """The backslash cases are the ones the first pattern let through:
    it rejected `../` and drive letters and never looked at `\\`."""
    payload = _implementer_reply(changed_files=[path])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.IMPLEMENTER_RESULT_SCHEMA).validate(
            payload)


@pytest.mark.parametrize(
    ("payload", "why"),
    [(_code_audit_reply(status=contract.Status.CHANGES_REQUESTED,
                        next_action="await_repair"), "bulgusuz-ret"),
     (_code_audit_reply(status=contract.Status.APPROVED,
                        findings=[_code_finding()]), "bulgulu-onay"),
     (_implementer_reply(status=contract.Status.BLOCKED, next_action="stop"),
      "gerekcesiz-blok"),
     (_implementer_reply(next_action="stop"), "yanlis-sonraki-adim")],
    ids=lambda value: value if isinstance(value, str) else "")
def test_status_next_action_findings_and_stop_reason_are_tied_together(
        payload, why):
    """Without these, "changes_requested carrying no findings" and
    "blocked with no reason" both validated -- a refusal nobody can act
    on and a stop nobody can explain."""
    schema = (schemas.CODE_AUDIT_RESULT_SCHEMA
              if payload["role"] == contract.Role.EVALUATOR
              else schemas.IMPLEMENTER_RESULT_SCHEMA)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(payload)


# =====================================================================
# THE PRIVACY BOUNDARY -- enforced by TYPE, not by redaction
# =====================================================================

def test_a_locked_finding_has_no_free_text_field_to_hide_a_passage_in():
    """A runner cannot look at free text and decide whether it is a code
    observation or a sentence lifted out of a locked document, and a
    redaction heuristic that tried would be a guess. So the boundary is
    the TYPE: the locked schema simply has nowhere to put prose."""
    properties = schemas.LOCKED_AUDIT_RESULT_SCHEMA["properties"]
    finding = properties["findings"]["items"]
    assert finding["additionalProperties"] is False
    assert set(finding["properties"]) == set(
        contract.LOCKED_FINDING_TRANSFERABLE_FIELDS)
    for text_field in ("claim", "required_action", "file", "line"):
        assert text_field not in finding["properties"]


def test_a_locked_finding_carrying_a_passage_is_rejected():
    payload = _code_audit_reply(
        audit_kind=contract.AuditKind.LOCKED,
        status=contract.Status.CHANGES_REQUESTED, next_action="await_repair",
        findings=[dict(_locked_finding(), claim="GIZLI_HOLDOUT_PASAJI")])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.LOCKED_AUDIT_RESULT_SCHEMA).validate(
            payload)


def test_a_healthy_locked_finding_still_validates():
    """The boundary must not also block the legitimate signal: an opaque
    id, an error class and a count are exactly what may cross."""
    Draft202012Validator(schemas.LOCKED_AUDIT_RESULT_SCHEMA).validate(
        _locked_audit_reply(status=contract.Status.CHANGES_REQUESTED,
                            next_action="await_repair",
                            findings=[_locked_finding()]))


@pytest.mark.parametrize(
    "payload",
    [{"summary": "GIZLI_HOLDOUT_PASAJI"},
     {"stop_reason": "GIZLI_HOLDOUT_PASAJI", "status": "blocked",
      "next_action": "stop"},
     {"run_id": "gizli-belge-adi"}],
    ids=["ozet-serbest-metin", "durma-sebebi-serbest-metin", "kimlik-slug"])
def test_the_locked_envelope_carries_no_free_text_either(payload):
    """The findings were made textless and the ENVELOPE was not: a
    required 2000-character `summary` sat right beside them, `stop_reason`
    was a free string, and the ids were model-chosen slugs -- a slug is
    free text on a short leash, and "gizli-belge-adi" is a valid one."""
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.LOCKED_AUDIT_RESULT_SCHEMA).validate(
            _locked_audit_reply(**payload))


def test_locked_identifiers_are_opaque_and_fixed_length():
    """Minted by the runner before the call and validated against the
    allowlist it handed out, so an id the runner did not issue is a
    schema violation rather than a new case."""
    assert re.fullmatch(contract.OPAQUE_ID_PATTERN.strip("^$"), "0" * 32)
    finding = schemas.LOCKED_AUDIT_RESULT_SCHEMA["properties"]["findings"][
        "items"]["properties"]
    for field in ("finding_id", "mechanism_id"):
        assert finding[field]["pattern"] == contract.OPAQUE_ID_PATTERN


def test_the_two_audit_kinds_are_not_interchangeable():
    """A code finding may not arrive under the locked schema, and the
    reverse, or the type boundary means nothing."""
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.LOCKED_AUDIT_RESULT_SCHEMA).validate(
            _code_audit_reply(audit_kind=contract.AuditKind.LOCKED,
                              status=contract.Status.CHANGES_REQUESTED,
                              next_action="await_repair",
                              findings=[_code_finding()]))
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.CODE_AUDIT_RESULT_SCHEMA).validate(
            _code_audit_reply(status=contract.Status.CHANGES_REQUESTED,
                              next_action="await_repair",
                              findings=[_locked_finding()]))


def test_holdout_content_is_not_a_transferable_field():
    crossing = set(contract.FINDING_TRANSFERABLE_FIELDS) | set(
        contract.LOCKED_FINDING_TRANSFERABLE_FIELDS)
    assert not crossing & set(contract.NEVER_TRANSFERABLE)


# =====================================================================
# THE STATE MACHINE AND THE STRUCTURAL RULES
# =====================================================================

@pytest.mark.parametrize(
    ("patch", "why"),
    [({"state": "uydurma_basarili"}, "uydurma-durum"),
     ({"stop_reason": "uydurma_sebep"}, "uydurma-sebep"),
     ({"rounds": {"implementation": 999, "repair": 999, "evaluator": 999}},
      "sinirsiz-tur")],
    ids=["uydurma-durum", "uydurma-sebep", "sinirsiz-tur"])
def test_the_state_file_cannot_invent_a_status_or_a_round_count(patch, why):
    """`state: "uydurma_basarili"` validated cleanly before -- a run
    that can invent its own successful ending. So did 999 rounds, which
    is the round cap the whole design rests on."""
    state = {"protocol_version": contract.PROTOCOL_VERSION,
             "run_id": "kurgu-run-1", "state": contract.State.APPROVED,
             "started_at": "t", "updated_at": "t",
             "rounds": {"implementation": 1, "repair": 1, "evaluator": 2},
             "budget": {"max_usd": 1.0, "spent_usd": 0.5}}
    state.update(patch)
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.STATE_SCHEMA).validate(state)


def test_the_budget_invariant_is_declared_even_though_json_schema_cannot(
        ):
    """`spent_usd <= max_usd` relates two fields, which this schema
    cannot express -- so it is a RUNNER invariant, written down rather
    than assumed, and the gap is visible instead of silent."""
    assert contract.BUDGET_INVARIANT == "spent_usd <= max_usd"
    budget = schemas.STATE_SCHEMA["properties"]["budget"]["properties"]
    assert budget["spent_usd"]["minimum"] == 0


def test_an_event_cannot_carry_free_text(workspace):
    """`detail` was a 500-character free string in the file written on
    every single step -- a document-sized hole in the most frequently
    touched artefact."""
    assert "detail" not in schemas.EVENT_SCHEMA["properties"]
    assert schemas.EVENT_SCHEMA["properties"]["event"]["enum"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.EVENT_SCHEMA).validate(
            {"ts": "t", "run_id": "kurgu-run-1",
             "event": contract.EventCode.RUN_FINISHED,
             "detail": "GIZLI_HOLDOUT_PASAJI"})
    Draft202012Validator(schemas.EVENT_SCHEMA).validate(
        {"ts": "t", "run_id": "kurgu-run-1",
         "event": contract.EventCode.OUTPUT_TRUNCATED, "bytes_truncated": 42})


@pytest.mark.parametrize(
    ("state", "extra", "ok"),
    [("approved", {"stop_reason": "completed"}, True),
     ("approved", {}, False),
     ("approved", {"stop_reason": "timeout"}, False),
     ("blocked", {}, False),
     ("blocked", {"stop_reason": "budget_exhausted"}, True),
     ("implementing", {}, True),
     ("implementing", {"stop_reason": "completed"}, False)],
    ids=["onay-tam", "onay-sebepsiz", "onay-yanlis-sebep", "blok-sebepsiz",
         "blok-tam", "yurutuluyor-temiz", "yurutuluyor-sebepli"])
def test_a_terminal_state_carries_exactly_one_reason(state, extra, ok):
    """`approved` with no stop_reason validated before, which
    contradicts the contract the state file is supposed to record. A
    running state carrying `completed` is the same contradiction the
    other way round."""
    payload = {"protocol_version": contract.PROTOCOL_VERSION,
               "run_id": "kurgu-run-1", "state": state,
               "started_at": "t", "updated_at": "t",
               "rounds": {"implementation": 1, "repair": 0, "evaluator": 1},
               "budget": {"max_usd": 1.0, "spent_usd": 0.1}, **extra}
    validator = Draft202012Validator(schemas.STATE_SCHEMA)
    if ok:
        validator.validate(payload)
    else:
        with pytest.raises(ValidationError):
            validator.validate(payload)


def test_a_locked_reply_is_bound_to_the_ids_the_runner_issued():
    """The static pattern proves SHAPE only: any 32 hex characters
    satisfy it, including ids the runner never minted. The binding is an
    allowlist -- `run_id` pinned with const, the id fields an enum of
    exactly what was handed out -- so an unissued id is a schema
    violation rather than a new case to act on."""
    issued_finding, issued_mechanism, run_id = "f" * 32, "e" * 32, "a" * 32
    schema = schemas.locked_audit_schema(
        run_id=run_id, issued_finding_ids=[issued_finding],
        issued_mechanism_ids=[issued_mechanism])
    validator = Draft202012Validator(schema)

    def reply(finding_patch=None, **overrides):
        finding = dict(_locked_finding(), finding_id=issued_finding,
                       mechanism_id=issued_mechanism)
        finding.update(finding_patch or {})
        fields = {"run_id": run_id,
                  "status": contract.Status.CHANGES_REQUESTED,
                  "summary_code": contract.SummaryCode.REGRESSION_DETECTED,
                  "next_action": "await_repair", "findings": [finding]}
        fields.update(overrides)          # a caller may override run_id
        return _locked_audit_reply(**fields)

    validator.validate(reply())
    for bad in ({"finding_patch": {"finding_id": "b" * 32}},
                {"finding_patch": {"mechanism_id": "c" * 32}},
                {"run_id": "d" * 32}):
        with pytest.raises(ValidationError):
            validator.validate(reply(**bad))
    assert contract.OPAQUE_ID_BITS == 128
    assert contract.LOCKED_IDS_ARE_ALLOWLISTED is True


def test_the_state_machine_has_no_third_patch_edge():
    """THE point of the loop. FINAL_AUDITING may end approved or blocked
    -- never back to REPAIRING, because that edge IS the third patch."""
    assert contract.State.REPAIRING not in contract.ALLOWED_TRANSITIONS[
        contract.State.FINAL_AUDITING]
    for terminal in contract.TERMINAL_STATES:
        assert contract.ALLOWED_TRANSITIONS[terminal] == ()


def test_the_loop_never_runs_an_arbitrary_shell():
    """THE RULE, stated precisely, because the loose version reads as a
    contradiction: `p0_gate` runs `bash scripts/p0_gate.sh`. What is
    forbidden is an ARBITRARY or INLINE shell -- `shell=True`,
    `bash -c`, `powershell -Command`, any string a model could
    influence. Invoking a FIXED, TRACKED, REVIEWED script file is not
    the hazard; an unreviewed command string is.

    Read from the AST, not from the text: the previous version searched
    the source for "shell=True" and matched the COMMENT that forbids
    it -- the same trap the `$schema` check fell into."""
    offenders = []
    for module in (REPO / "tools" / "agent_loop").rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell" and                             getattr(keyword.value, "value", False) is True:
                        offenders.append(f"{module.name}:shell=True")
                target = ast.unparse(node.func)
                if target in ("os.system", "os.popen", "subprocess.getoutput"):
                    offenders.append(f"{module.name}:{target}")
    assert not offenders, f"keyfi kabuk: {offenders}"


def test_no_registry_command_is_an_inline_shell():
    """A registry entry may name a tracked script; it may never carry an
    inline command string."""
    offenders = []
    for command_id, entry in contract.COMMAND_REGISTRY.items():
        argv = list(entry["argv"])
        for index, token in enumerate(argv):
            if token in ("-c", "-Command", "/c", "/C") and index:
                offenders.append(f"{command_id}:inline")
        for token in argv[1:]:
            if token.endswith((".sh", ".ps1", ".bat", ".cmd")):
                script = REPO / token
                assert script.exists(), f"{command_id}: script yok {token}"
                tracked = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", token],
                    cwd=REPO, capture_output=True).returncode == 0
                assert tracked, f"{command_id}: script izlenmiyor {token}"
    assert not offenders, f"satir-ici kabuk: {offenders}"


def test_the_state_directory_is_ignored_by_git():
    probe = f"{contract.STATE_DIR_NAME}/state.json"
    ignored = subprocess.run(["git", "check-ignore", "-q", probe],
                             cwd=REPO).returncode == 0
    assert ignored, f"{probe} gitignore altinda degil"


def test_defaults_allow_exactly_one_implementation_and_one_repair():
    assert contract.DEFAULTS["max_implementation_rounds"] == 1
    assert contract.DEFAULTS["max_repair_rounds"] == 1
    assert contract.DEFAULTS["max_evaluator_rounds"] == 2
    bounds = schemas.TASK_SCHEMA["properties"]
    assert bounds["max_implementation_rounds"]["maximum"] == 1
    assert bounds["max_repair_rounds"]["maximum"] == 1


def test_a_task_with_no_acceptance_command_is_rejected(workspace):
    task = dict(workspace["task"], acceptance_commands=[])
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas.TASK_SCHEMA).validate(task)


# =====================================================================
# RED UNTIL PHASE B -- every one needs the runner, and every one passes
# it the FAKE binaries
# =====================================================================

def _run(runner, workspace, **kwargs):
    return runner.run(workspace["task_path"], repo=workspace["repo"],
                      binaries=workspace["binaries"], **kwargs)


def test_a_dirty_worktree_stops_before_any_model_is_called(workspace):
    runner = _runner()
    (workspace["repo"] / "pipeline" / "kurgu.py").write_text(
        "VALUE = 2\n", encoding="utf-8")
    result = runner.preflight(workspace["task_path"], repo=workspace["repo"],
                              binaries=workspace["binaries"])
    assert result.stop_reason == contract.StopReason.DIRTY_WORKTREE
    assert not workspace["record"].exists(), "kirli agacta model cagrildi"


def test_staged_changes_stop_the_run(workspace):
    runner = _runner()
    (workspace["repo"] / "pipeline" / "yeni.py").write_text(
        "x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "pipeline/yeni.py"],
                   cwd=workspace["repo"],
                   check=True)
    result = runner.preflight(workspace["task_path"], repo=workspace["repo"],
                              binaries=workspace["binaries"])
    assert result.stop_reason == contract.StopReason.STAGED_CHANGES


def test_a_baseline_sha_mismatch_stops_the_run(workspace):
    runner = _runner()
    task = dict(workspace["task"], baseline_sha="0" * 40)
    workspace["task_path"].write_text(json.dumps(task), encoding="utf-8")
    result = runner.preflight(workspace["task_path"], repo=workspace["repo"],
                              binaries=workspace["binaries"])
    assert result.stop_reason == contract.StopReason.BASELINE_MISMATCH


def test_the_runner_refuses_to_run_without_explicit_binaries(workspace):
    """No discovery, no default: the seam is mandatory or the promise
    that no real model is called is unenforceable."""
    runner = _runner()
    with pytest.raises(TypeError):
        runner.run(workspace["task_path"], repo=workspace["repo"])


def test_an_implementer_edit_outside_allowed_paths_is_caught(workspace):
    """Checked against the DIFF, not against what the model claims."""
    runner = _runner()
    result = _run(runner, workspace,
                  _test_edit={"contracts/gizli.md": "kurgu"})
    assert result.stop_reason == contract.StopReason.PATH_NOT_ALLOWED


def test_an_evaluator_that_modifies_the_workspace_halts_the_loop(workspace):
    """Proven by hashes before and after, not by trusting a flag. No
    automatic rollback: a reviewer that wrote is a broken protocol, and
    restoring the tree would restore the files but not the trust."""
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        "open('sizdi.txt','w').write('kurgu')\n"
        + _emits(_code_audit_reply()))
    result = _run(runner, workspace)
    assert result.state == contract.State.FAILED
    assert result.stop_reason == \
        contract.StopReason.EVALUATOR_MODIFIED_WORKSPACE


def test_the_evaluator_is_never_invoked_without_its_read_only_flags(workspace):
    runner = _runner()
    _run(runner, workspace)
    calls = [json.loads(line) for line in
             workspace["record"].read_text(encoding="utf-8").splitlines()]
    evaluator = [c for c in calls if "exec" in c["argv"]]
    assert evaluator, "evaluator hic cagrilmadi"
    for call in evaluator:
        assert contract.CODEX_SANDBOX_READ_ONLY in call["argv"]
        assert contract.CODEX_APPROVAL_OVERRIDE[1] in call["argv"]


@pytest.mark.parametrize(
    "bad", ['{"not": "the schema"}', "duz metin, JSON degil",
            '{"role": "implementer"}'],
    ids=["yanlis-alanlar", "json-degil", "eksik-alanlar"])
def test_a_reply_that_fails_the_schema_is_a_failure_not_a_repair(workspace,
                                                                 bad):
    runner = _runner()
    workspace["record_fake"](workspace["binaries"]["implementer"],
                             workspace["record"], f"print({bad!r})")
    result = _run(runner, workspace)
    assert result.stop_reason == contract.StopReason.SCHEMA_VIOLATION


def test_a_model_timeout_kills_the_whole_process_tree(workspace):
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["implementer"], workspace["record"],
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])\n"
        "time.sleep(300)\n")
    task = dict(workspace["task"], model_call_timeout_seconds=30)
    workspace["task_path"].write_text(json.dumps(task), encoding="utf-8")
    result = _run(runner, workspace)
    assert result.stop_reason == contract.StopReason.TIMEOUT
    assert result.surviving_children == 0


def test_the_same_mechanism_failing_twice_blocks_instead_of_patching(
        workspace):
    """The rule the whole loop exists for."""
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        _emits(_code_audit_reply(
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_code_finding(mechanism_id="kurgu-mekanizma-a")],
            next_action="await_repair")))
    result = _run(runner, workspace)
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == \
        contract.StopReason.REPEATED_MECHANISM_FAILURE
    assert result.repair_rounds <= workspace["task"]["max_repair_rounds"]


def test_exceeding_max_repair_rounds_blocks(workspace):
    """A DIFFERENT mechanism each round still stops at the budget of
    one repair -- the round cap and the second-patch rule are two
    separate limits, and either alone is escapable."""
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        "import os\n"
        "n = int(os.environ.get('AGENT_LOOP_ROUND', '0'))\n"
        "reply = " + repr(_code_audit_reply(
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_code_finding()], next_action="await_repair")) + "\n"
        "reply['findings'][0]['mechanism_id'] = 'kurgu-mekanizma-%d' % n\n"
        "emit(reply)\n")
    result = _run(runner, workspace)
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.REPAIR_ROUNDS_EXHAUSTED
    assert result.repair_rounds == workspace["task"]["max_repair_rounds"]


def test_an_exhausted_budget_prevents_the_next_model_call(workspace):
    """Checked BEFORE the call. A budget enforced afterwards has already
    been spent."""
    runner = _runner()
    task = dict(workspace["task"], max_budget_usd=0.0)
    workspace["task_path"].write_text(json.dumps(task), encoding="utf-8")
    result = _run(runner, workspace)
    assert result.stop_reason == contract.StopReason.BUDGET_EXHAUSTED
    assert not workspace["record"].exists(), "butce bittigi halde cagrildi"


def test_an_out_of_scope_finding_blocks_rather_than_widening(workspace):
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        _emits(_code_audit_reply(
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_code_finding(file="pipeline/index/db.py",
                                    mechanism_id="kapsam-disi")],
            next_action="await_repair")))
    result = _run(runner, workspace)
    assert result.stop_reason == contract.StopReason.OUT_OF_SCOPE_FINDING


@pytest.mark.parametrize("gate", ["git_commit", "git_push", "git_add"])
def test_a_gated_action_stops_and_asks_instead_of_proceeding(workspace, gate):
    runner = _runner()
    result = _run(runner, workspace, _test_request_gate=gate)
    assert result.state == contract.State.BLOCKED
    assert result.stop_reason == contract.StopReason.USER_APPROVAL_REQUIRED
    assert gate in result.pending_approval


def test_the_loop_never_stages_or_commits_anything(workspace):
    runner = _runner()
    before = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=workspace["repo"], capture_output=True,
                            text=True).stdout.strip()
    _run(runner, workspace)
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace["repo"],
                           capture_output=True, text=True).stdout.strip()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                            cwd=workspace["repo"], capture_output=True,
                            text=True).stdout.strip()
    assert before == after and staged == ""


def test_locked_findings_reach_the_implementer_as_counters_only(workspace):
    """The type boundary, checked over what the run actually produced:
    the implementer's prompt and every artefact carry the class and the
    count, never a case.

    THE REPLY IS BUILT INSIDE THE FAKE, from the ids the runner minted
    and exported for this call. It has to be: a locked envelope is
    TEXTLESS, so its `run_id` and both id fields are opaque, and the
    per-call schema pins them with `const` and `enum` to exactly what
    was issued. A recording carrying `kurgu-run-1` and `ffff...` names
    ids the runner never handed out, which is a schema violation rather
    than a locked audit -- and the test would then be green about
    nothing."""
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        "reply = " + repr(_locked_audit_reply(
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_locked_finding()], next_action="await_repair")) + "\n"
        "reply['run_id'] = os.environ['AGENT_LOOP_LOCKED_RUN_ID']\n"
        "reply['findings'][0]['finding_id'] = "
        "issued('AGENT_LOOP_FINDING_IDS')[0]\n"
        "reply['findings'][0]['mechanism_id'] = "
        "issued('AGENT_LOOP_MECHANISM_IDS')[0]\n"
        "emit(reply)\n")
    _run(runner, workspace, audit_kind=contract.AuditKind.LOCKED)
    state_dir = workspace["repo"] / contract.STATE_DIR_NAME
    written = " ".join(p.read_text(encoding="utf-8", errors="ignore")
                       for p in state_dir.rglob("*") if p.is_file())
    assert contract.LockedFindingClass.WRONG_ROW in written
    for banned in contract.NEVER_TRANSFERABLE:
        assert banned not in written


def test_a_half_written_state_file_resumes_from_the_last_good_one(workspace):
    runner = _runner()
    _run(runner, workspace)
    state_path = workspace["repo"] / contract.STATE_DIR_NAME / "state.json"
    state_path.write_text('{"protocol_version": "1.0", "run_',
                          encoding="utf-8")
    resumed = runner.resume(repo=workspace["repo"],
                            binaries=workspace["binaries"])
    assert resumed.recovered_from_backup is True


def test_a_second_instance_cannot_take_the_lock(workspace):
    runner = _runner()
    with runner.single_instance_lock(workspace["repo"]):
        with pytest.raises(runner.LockHeld):
            with runner.single_instance_lock(workspace["repo"]):
                pass


def test_an_interrupt_leaves_a_resumable_state(workspace):
    runner = _runner()
    result = _run(runner, workspace,
                  _test_interrupt_after=contract.State.IMPLEMENTING)
    assert result.stop_reason == contract.StopReason.INTERRUPTED
    assert (workspace["repo"] / contract.STATE_DIR_NAME
            / "state.json").exists()


def test_oversized_model_output_is_truncated_and_the_event_recorded(
        workspace):
    """Silent truncation is a lie about what was read."""
    runner = _runner()
    workspace["record_fake"](workspace["binaries"]["implementer"],
                             workspace["record"], "print('k' * 5_000_000)")
    task = dict(workspace["task"], max_output_bytes=4096)
    workspace["task_path"].write_text(json.dumps(task), encoding="utf-8")
    _run(runner, workspace)
    events = workspace["repo"] / contract.STATE_DIR_NAME / "events.jsonl"
    kinds = [json.loads(line)["event"]
             for line in events.read_text(encoding="utf-8").splitlines()]
    assert "output_truncated" in kinds


def test_a_clean_run_walks_implement_audit_fix_audit_and_approves(workspace):
    """The happy path as a SEQUENCE: an approval that skipped the audit
    is not an approval."""
    runner = _runner()
    replies = [_code_audit_reply(status=contract.Status.CHANGES_REQUESTED,
                                 findings=[_code_finding()],
                                 next_action="await_repair"),
               _code_audit_reply()]
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        "import os\n"
        "n = int(os.environ.get('AGENT_LOOP_ROUND', '0'))\n"
        f"emit({replies!r}[min(n, 1)])\n")
    result = _run(runner, workspace)
    assert result.state == contract.State.APPROVED
    assert result.visited == [
        contract.State.PREFLIGHT, contract.State.IMPLEMENTING,
        contract.State.ACCEPTANCE, contract.State.AUDITING,
        contract.State.REPAIRING, contract.State.ACCEPTANCE_2,
        contract.State.FINAL_AUDITING, contract.State.APPROVED]


def test_a_final_audit_asking_for_more_changes_blocks(workspace):
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        _emits(_code_audit_reply(
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_code_finding(mechanism_id="kurgu-mekanizma-c")],
            next_action="await_repair")))
    result = _run(runner, workspace)
    assert result.state == contract.State.BLOCKED
    assert result.state != contract.State.APPROVED


# =====================================================================
# B4-R2 -- TWO SCHEMA AUTHORITIES: transport constrains generation,
# the authoritative schema decides acceptance
# =====================================================================
#
# MEASURED, not assumed. Two authorized diagnostics against the real
# CLI (2.1.220, authenticated) both ended in a 4xx client error with
# `terminal_reason: api_error`, and an offline inventory of the frozen
# schema found the keyword occurrences the published structured-output
# subset does not accept: `pattern` (5), `if`/`then` (3 each),
# `minLength`/`maxLength` (5 each), `minimum` (6), `maximum` (1),
# `maxItems` (3).
#
# The OLD invariant was "the schema on the argv and the schema that
# validates the reply are the same bytes". That invariant assumed the
# API would accept the same rich schema the acceptance gate needs, and
# the API refused it. The new invariant is stated rather than hidden:
#
#     argv schema constrains generation;
#     authoritative local schema decides acceptance;
#     both exact canonical bindings are independently hash-pinned.
#
# The transport schema is NOT an acceptance authority. A reply that
# satisfies it and fails the authoritative schema is refused.

_TRANSPORT_UNSUPPORTED = ("pattern", "if", "then", "minLength", "maxLength",
                          "minimum", "maximum", "maxItems")

# The vocabulary this test file knows how to recognise. Property NAMES
# are arbitrary strings and must not be mistaken for keywords, so the
# subset check is an intersection against this set rather than a claim
# about every key in the document.
_KNOWN_KEYWORDS = frozenset({
    "type", "properties", "items", "required", "additionalProperties",
    "enum", "const", "allOf", "anyOf", "oneOf", "if", "then", "else",
    "pattern", "minLength", "maxLength", "minimum", "maximum",
    "minItems", "maxItems", "uniqueItems", "format", "default",
    "description", "title", "$comment", "$ref", "$defs", "definitions",
})


def _keywords(node, seen=None):
    """Every mapping key anywhere in a schema document, with counts."""
    seen = {} if seen is None else seen
    if isinstance(node, dict):
        for key, value in node.items():
            seen[key] = seen.get(key, 0) + 1
            _keywords(value, seen)
    elif isinstance(node, list):
        for item in node:
            _keywords(item, seen)
    return seen


def test_the_authoritative_schema_keeps_every_strict_keyword():
    """The acceptance authority is UNCHANGED by the transport work.

    Its hash is pinned: a transport derivation that reached back and
    relaxed the real schema would be the whole defect this split exists
    to avoid."""
    counts = _keywords(schemas.AUTHORITATIVE_RESULT_SCHEMA)
    assert counts["pattern"] == 5
    assert counts["allOf"] == 1
    assert counts["if"] == 3 and counts["then"] == 3
    assert counts["const"] == 8 and counts["enum"] == 6
    assert counts["minLength"] == 5 and counts["maxLength"] == 5
    assert counts["minimum"] == 6 and counts["maximum"] == 1
    assert schemas.AUTHORITATIVE_RESULT_SCHEMA is \
        schemas.IMPLEMENTER_RESULT_SCHEMA
    assert schemas.IMPLEMENTER_SCHEMA_BINDING.sha256 == \
        "4241b1100e9a03f706d4d7ede872fe1e7d0522f28a0c8596afc6aea814f0a86d"


def test_the_transport_schema_carries_only_the_supported_subset():
    """Every keyword the published subset refuses is GONE -- proven by
    walking the whole document, not by checking the top level."""
    counts = _keywords(schemas.CLAUDE_TRANSPORT_SCHEMA)
    for keyword in _TRANSPORT_UNSUPPORTED:
        assert counts.get(keyword, 0) == 0, f"tasima semasi {keyword} tasiyor"
    used = set(counts) & _KNOWN_KEYWORDS
    assert used <= set(schemas.CLAUDE_TRANSPORT_KEYWORDS), sorted(
        used - set(schemas.CLAUDE_TRANSPORT_KEYWORDS))


def test_the_transport_schema_keeps_every_closed_field_it_may_keep():
    """Dropping the unsupported constraints must not quietly open the
    object: `additionalProperties: false`, the required lists and the
    closed vocabularies are what still bound the model's generation."""
    transport = schemas.CLAUDE_TRANSPORT_SCHEMA
    counts = _keywords(transport)
    # Every closed object stays closed: 3 of 3 survive.
    assert counts["additionalProperties"] == 3
    # 8 -> 3. The five that leave are the ones INSIDE the conditionals:
    # each of the 3 `if` branches carries `required: [status]`, and 2 of
    # the 3 `then` branches carry `required: [stop_reason]`. They go
    # with `allOf`, and the authoritative schema still enforces them.
    assert counts["required"] == 3
    # 8 -> 2, for the same reason: 6 of the 8 consts are the status and
    # next_action pins inside the conditionals. What remains is
    # `protocol_version` and `role`.
    assert counts["const"] == 2
    # every closed vocabulary survives -- none of them lived in a
    # conditional
    assert counts["enum"] == 6
    assert transport["additionalProperties"] is False
    assert transport["type"] == "object"
    assert set(transport["required"]) == set(
        schemas.AUTHORITATIVE_RESULT_SCHEMA["required"])
    assert transport["properties"]["role"] == {"const": "implementer"}
    assert transport["properties"]["status"]["enum"] == \
        schemas.AUTHORITATIVE_RESULT_SCHEMA["properties"]["status"]["enum"]


def test_a_transport_valid_reply_still_faces_the_authoritative_schema():
    """THE POINT OF THE SPLIT. `run_id` loses its pattern on the way to
    the API, so the API can no longer refuse a malformed one -- and the
    acceptance gate still does."""
    reply = {"protocol_version": "1.0", "run_id": "BUYUK HARF VE BOSLUK",
             "role": "implementer", "status": "blocked",
             "summary": "kurgu", "next_action": "stop",
             "stop_reason": "interrupted"}
    transport = Draft202012Validator(schemas.CLAUDE_TRANSPORT_SCHEMA)
    assert transport.is_valid(reply), "tasima semasi bunu zaten reddediyor"
    with pytest.raises(ValidationError):
        schemas.IMPLEMENTER_SCHEMA_BINDING.validate(reply)


def test_the_transport_schema_cannot_relax_the_status_conditionals():
    """`if/then` is gone from the transport copy, so a `blocked` reply
    with no `stop_reason` passes it -- and must still be refused by the
    authority."""
    reply = {"protocol_version": "1.0", "run_id": "kosu-abc",
             "role": "implementer", "status": "blocked",
             "summary": "kurgu", "next_action": "stop"}
    assert Draft202012Validator(schemas.CLAUDE_TRANSPORT_SCHEMA).is_valid(reply)
    with pytest.raises(ValidationError):
        schemas.IMPLEMENTER_SCHEMA_BINDING.validate(reply)


def test_the_two_bindings_are_independently_hash_pinned():
    import hashlib

    authoritative = schemas.IMPLEMENTER_SCHEMA_BINDING
    transport = schemas.IMPLEMENTER_TRANSPORT_BINDING
    assert transport.sha256 != authoritative.sha256
    assert transport.canonical_json != authoritative.canonical_json
    for binding in (authoritative, transport):
        assert re.fullmatch(r"[0-9a-f]{64}", binding.sha256)
        assert hashlib.sha256(
            binding.canonical_json.encode("utf-8")).hexdigest() == \
            binding.sha256


def test_the_transport_derivation_is_deterministic():
    """Derived twice from the same source, byte-identical -- and the
    module-level binding is that same value."""
    again = schemas.claude_transport_schema(
        schemas.AUTHORITATIVE_RESULT_SCHEMA)
    assert schemas.canonical_json(again) == \
        schemas.IMPLEMENTER_TRANSPORT_BINDING.canonical_json


def test_the_derivation_refuses_a_keyword_nobody_classified():
    """NOT a silent stripper. A keyword that is neither supported nor on
    the declared drop list is a decision nobody has made, and it raises
    instead of vanishing -- otherwise a constraint added tomorrow would
    be dropped in silence and the reply would be judged by less than the
    author wrote."""
    with pytest.raises(schemas.TransportSchemaError):
        schemas.claude_transport_schema(
            {"type": "object", "properties": {"a": {"type": "string"}},
             "uniqueItems": True})


def test_the_transport_schema_carries_no_constraint_prose():
    """The SDKs move dropped constraints into `description` text. This
    one does not: a description is free text on a boundary that exists
    to keep free text out."""
    counts = _keywords(schemas.CLAUDE_TRANSPORT_SCHEMA)
    assert counts.get("description", 0) == 0
    assert counts.get("title", 0) == 0
    assert counts.get("$comment", 0) == 0


def test_the_argv_carries_the_transport_schema_and_not_the_authority():
    """What the CLI receives is the schema the API can compile. The
    acceptance authority never travels."""
    argv = cli.build_implementer_argv("/kurgu/claude", budget_usd=1.0)
    token = argv[argv.index("--json-schema") + 1]
    assert token == schemas.IMPLEMENTER_TRANSPORT_BINDING.canonical_json
    assert token != schemas.IMPLEMENTER_SCHEMA_BINDING.canonical_json
    for keyword in _TRANSPORT_UNSUPPORTED:
        assert f'"{keyword}"' not in token


# =====================================================================
# B4-R3 -- THE EVALUATOR RUNS IN A BOUND FLAT WORKSPACE
# =====================================================================
#
# MEASURED against codex-cli 0.147.0-alpha.6.6, launched with the exact
# production argv: exit 1 after 341 ms, 0 bytes on stdout, 105 on
# stderr, and no last-message file. The whole message was
#
#   "Reading prompt from stdin...
#    Not inside a trusted directory and --skip-git-repo-check was not
#    specified."
#
# The working directory is the flat workspace's IMPLEMENTER ROOT, which
# carries no `.git` by design -- it is a copy of tracked files, not a
# clone. So the CLI's own git-repository check is asking a question this
# loop already answers, and answers with something stronger: the
# directory is not a caller's path but one `flat_workspace.assert_binding`
# derives from the recorded run, and the sandbox stays `read-only`.
#
# WHAT THIS IS NOT. It is not the schema split the implementer road
# needed. The measured text names no schema and no keyword, and the
# process died before the schema file was ever read -- an earlier
# hypothesis built on the audit schema's keyword inventory was wrong and
# the measurement retired it.

SKIP_GIT_CHECK = "--skip-git-repo-check"


def _evaluator_argv(tmp_path, **overrides):
    call = {"repo": tmp_path, "schema_path": tmp_path / "s.json",
            "last_message_path": tmp_path / "o.txt"}
    call.update(overrides)
    return cli.build_evaluator_argv(tmp_path / "sahte_codex.py", **call)


def test_the_evaluator_argv_skips_the_git_repo_check_exactly_once(tmp_path):
    """EXACTLY ONCE: a repeated flag has no agreed meaning here, and a
    missing one is the measured failure."""
    argv = _evaluator_argv(tmp_path)
    assert argv.count(SKIP_GIT_CHECK) == 1
    # it belongs to `exec`, so it comes after the subcommand
    assert argv.index(SKIP_GIT_CHECK) > argv.index("exec")


def test_the_skip_flag_is_fixed_and_not_a_caller_parameter(tmp_path):
    """A SECURITY-SHAPED FLAG IS NOT CONFIGURATION. There is no keyword
    through which a caller -- or a task, or a model -- can remove it,
    add a second one, or flip it: the builder simply does not take one,
    so an attempt is a `TypeError` rather than a quiet override."""
    import inspect

    parameters = inspect.signature(cli.build_evaluator_argv).parameters
    assert not any("git" in name or "skip" in name for name in parameters)
    for attempt in ({"skip_git_repo_check": False},
                    {"skip_git_repo_check": True},
                    {"git_check": True}):
        with pytest.raises(TypeError):
            _evaluator_argv(tmp_path, **attempt)
    # and nothing a task or a model can say reaches it: the only inputs
    # are the four paths and the model name
    assert _evaluator_argv(tmp_path).count(SKIP_GIT_CHECK) == 1


def test_the_implementer_argv_never_carries_the_skip_flag(tmp_path):
    """The flag is the CODEX road's answer to the CODEX binary's check.
    Claude has no such check and no such flag."""
    argv = cli.build_implementer_argv(tmp_path / "sahte_claude.py",
                                      budget_usd=1.0)
    assert SKIP_GIT_CHECK not in argv


def test_the_evaluator_security_invariants_survive_the_skip_flag(tmp_path):
    """Adding one flag must not have loosened any of the others: this
    re-asserts the whole evaluator contract beside the new token."""
    argv = _evaluator_argv(tmp_path)
    for flag in contract.CODEX_REQUIRED_FLAGS:
        assert flag in argv, f"eksik bayrak: {flag}"
    assert argv[1] == "exec"
    assert argv[argv.index("--sandbox") + 1] == \
        contract.CODEX_SANDBOX_READ_ONLY
    flag, value = contract.CODEX_APPROVAL_OVERRIDE
    assert value in argv and argv[argv.index(value) - 1] == flag
    assert "--ask-for-approval" not in argv and "-a" not in argv
    assert argv[argv.index("--cd") + 1] == str(tmp_path)
    assert argv[argv.index("--output-schema") + 1] == str(tmp_path / "s.json")
    assert argv[argv.index("--output-last-message") + 1] == \
        str(tmp_path / "o.txt")
    # the prompt is nowhere on the command line -- it goes over stdin
    assert not any("Adayi" in token or "incele" in token for token in argv)
    # and the flag did not smuggle in anything the forbidden list bars
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "--full-auto" not in argv
