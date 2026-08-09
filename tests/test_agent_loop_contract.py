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
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import cli, contract, schemas

REPO = Path(__file__).resolve().parent.parent
BACKSLASH = chr(92)


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
with open(RECORD, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
try:
    sys.stdin.read()
except Exception:
    pass
BODY
'''


def _write_fake(path: Path, record: Path, body: str) -> Path:
    path.write_text(
        _FAKE.replace("RECORD", repr(str(record))).replace("BODY", body),
        encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return path


def _prints(payload: dict) -> str:
    return f"print(json.dumps({payload!r}))"


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
                 ["config", "user.name", "Kurgu"]):
        subprocess.run(["git", *argv], cwd=repo, check=True)
    (repo / "pipeline" / "kurgu.py").write_text("VALUE = 1\n",
                                                encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "kurgu"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()

    bin_dir = tmp_path / "sahte-bin"
    bin_dir.mkdir()
    record = tmp_path / "cagrilar.jsonl"
    binaries = {
        "implementer": _write_fake(bin_dir / "sahte_claude.py", record,
                                   _prints(_implementer_reply())),
        "evaluator": _write_fake(bin_dir / "sahte_codex.py", record,
                                 _prints(_code_audit_reply())),
    }
    task = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "objective": "kurgu hedef",
        "baseline_sha": head,
        "allowed_paths": ["pipeline/"],
        "forbidden_paths": ["contracts/", "data/"],
        "acceptance_commands": [{"command_id": "pytest_full"}],
        "acceptance_criteria": ["kurgu olcut"],
        "max_implementation_rounds": 1,
        "max_repair_rounds": 1,
        "max_wall_clock_minutes": 5,
        "max_budget_usd": 1.0,
        "max_output_bytes": 65536,
        "leak_policy": {"command_id": "leak_scan", "max_hard_findings": 0},
        "dirty_tree_allowlist": [],
    }
    task_path = repo / "kurgu-task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return {"repo": repo, "head": head, "task": task, "task_path": task_path,
            "binaries": binaries, "record": record, "record_fake": _write_fake}


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
        tmp_path / "sahte_claude.py", schema_path=tmp_path / "s.json",
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
                                   schema_path=tmp_path / "s.json",
                                   budget_usd=1.0, allowed_tools=[])


def test_a_fake_binary_really_runs_through_the_built_argv(workspace, tmp_path):
    """THE SEAM, end to end and without a runner. The argv the builder
    produces is executed, the fake answers, and the recording proves the
    call reached the fake rather than anything on PATH."""
    argv = cli.build_evaluator_argv(
        workspace["binaries"]["evaluator"], repo=workspace["repo"],
        schema_path=tmp_path / "s.json", last_message_path=tmp_path / "o.txt")
    done = subprocess.run([sys.executable, *argv], input="", text=True,
                          capture_output=True)
    assert done.returncode == 0, done.stderr
    reply = json.loads(done.stdout)
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
    schema_file = tmp_path / "sema.json"
    schema_file.write_text(json.dumps(schemas.IMPLEMENTER_RESULT_SCHEMA),
                           encoding="utf-8")
    argv = cli.build_implementer_argv(
        workspace["binaries"]["implementer"], schema_path=schema_file,
        budget_usd=0.25, allowed_tools=["Edit", "Read"])
    done = subprocess.run([sys.executable, *argv], input="kurgu istem",
                          text=True, capture_output=True)
    assert done.returncode == 0, done.stderr
    Draft202012Validator(schemas.IMPLEMENTER_RESULT_SCHEMA).validate(
        json.loads(done.stdout))

    recorded = [json.loads(line) for line in
                workspace["record"].read_text(encoding="utf-8").splitlines()]
    assert recorded, "sahte implementer cagrilmadi"
    seen = recorded[-1]["argv"]
    assert "--print" in seen
    assert seen[seen.index("--json-schema") + 1] == str(schema_file)
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
                                   schema_path="/kurgu/s.json",
                                   budget_usd=1.0, allowed_tools=tools)


def test_the_implementer_tool_list_is_a_contract_constant_not_an_argument():
    assert contract.IMPLEMENTER_ALLOWED_TOOLS == (
        "Read", "Glob", "Grep", "Edit", "Write")
    for banned in ("Bash", "Agent", "WebFetch", "WebSearch"):
        assert banned in contract.IMPLEMENTER_FORBIDDEN_TOOLS
        assert banned not in contract.IMPLEMENTER_ALLOWED_TOOLS
    argv = cli.build_implementer_argv("/kurgu/claude",
                                      schema_path="/kurgu/s.json",
                                      budget_usd=1.0)
    tail = argv[argv.index("--allowedTools") + 1:]
    assert tail[:len(contract.IMPLEMENTER_ALLOWED_TOOLS)] == list(
        contract.IMPLEMENTER_ALLOWED_TOOLS)


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


def test_phase_b_is_frozen_to_worktree_isolation():
    """Hashing after the fact detects damage that has already happened
    and been left in the operator's tree. The implementer runs in a
    disposable worktree built from the baseline; only a verified
    allowed-path patch comes back."""
    assert contract.IMPLEMENTER_RUNS_IN_DISPOSABLE_WORKTREE is True


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
        + _prints(_code_audit_reply()))
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
        _prints(_code_audit_reply(
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
        "print(json.dumps(reply))\n")
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
        _prints(_code_audit_reply(
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
    count, never a case."""
    runner = _runner()
    workspace["record_fake"](
        workspace["binaries"]["evaluator"], workspace["record"],
        _prints(_code_audit_reply(
            audit_kind=contract.AuditKind.LOCKED,
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_locked_finding()], next_action="await_repair")))
    _run(runner, workspace)
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
        f"print(json.dumps({replies!r}[min(n, 1)]))\n")
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
        _prints(_code_audit_reply(
            status=contract.Status.CHANGES_REQUESTED,
            findings=[_code_finding(mechanism_id="kurgu-mekanizma-c")],
            next_action="await_repair")))
    result = _run(runner, workspace)
    assert result.state == contract.State.BLOCKED
    assert result.state != contract.State.APPROVED
