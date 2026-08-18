"""PACKAGE B2B-A -- the verified implementer change set.

MIGRATED IN B2B-B2A2. The change set is derived from a D3A FLAT
WORKSPACE and its filesystem evidence, not from a disposable git
worktree, so everything this file used to assert about a worktree's
index, its HEAD and its `.git` link has moved or disappeared:

  * the index-flag P0, the staged-change gate, the moved HEAD and the
    replaced `.git` link were all statements about an authority the
    workspace no longer has -- the flat evidence never asks git
    anything, which is pinned in `test_agent_loop_b2_changes_flat.py`;
  * the porcelain-record unit tests went with `_classify`;
  * the file-type, dirty-tree and classification scenarios moved to the
    flat file, where they are filesystem facts rather than git records.

WHAT STAYS HERE. The gates that did NOT move: authorization and scope,
the task manifest binding, the MAIN CHECKOUT guard -- which is still
git-based on purpose until B2B-B2B -- privacy, and the structural pins.

NO REAL MODEL IS CALLED. Every binary is a stub written into `tmp_path`,
and an autouse guard records each launch and fails the test if anything
outside `tmp_path` was ever executed as a model. Git is real, on
throwaway repositories created per test; the project's own checkout and
document tree are never touched.

EVERY REJECTION TEST PROVES ITS SETUP. A test that died at an earlier
gate is red for the wrong reason, and this project has produced that
exact false evidence before -- so each scenario asserts the world it
meant to build actually exists before it claims the refusal.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tools.agent_loop import (changes, contract, execution, flat_workspace,
                              state)
from tools.agent_loop import process as process_module

RUN = "kurgu-run-1"
SENTINEL = "KURGU-GIZLI-NOBETCI-" + "q" * 8
# where `_stub` puts its shims; the only programs that count as "a model
# ran". `flat_workspace.create()` runs git through the same contained
# launch seam, so a recorder that counted every process would never be
# empty and every "nothing started" assertion would measure the fixture.
STUB_HOLDER = "sahte-bin"

BASE_TASK = {
    "protocol_version": contract.PROTOCOL_VERSION,
    "objective": "kurgu hedef",
    "allowed_paths": ["pipeline/"],
    "forbidden_paths": ["pipeline/gizli/"],
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

# The stub runs INSIDE the implementer root, so every relative path it
# touches is a change this gate has to find and judge. An ABSOLUTE path
# in an op is how a scenario reaches the main checkout or the reference
# tree on purpose.
_HELPER = '''\
import json, os, shutil, subprocess, sys
from pathlib import Path


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if cfg.get("cwd_record"):
        Path(cfg["cwd_record"]).write_text(os.getcwd(), encoding="utf-8")
    for op in cfg.get("ops", []):
        kind = op["kind"]
        target = Path(op["path"]) if op.get("path") else None
        if kind == "write_keep_mtime":
            # SAME SIZE, and every metadata field put back exactly: an
            # in-place rewrite keeps the file identity, and `utime`
            # restores the timestamp the walker recorded. This is the
            # shape the metadata-only class is known NOT to see.
            info = os.stat(str(target))
            with open(target, "r+", encoding="utf-8") as handle:
                handle.seek(0)
                handle.write(op["text"])
                handle.truncate()
            os.utime(str(target), ns=(info.st_atime_ns, info.st_mtime_ns))
        elif kind == "copytree":
            shutil.copytree(op["src"], str(target))
        elif kind == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # IN PLACE. A file may be read-only or hidden, and
                # opening such a file with "w" (CREATE_ALWAYS) fails on
                # Windows -- so the scenario would die on the attribute
                # instead of reaching what it is about.
                try:
                    os.chmod(str(target), 0o666)
                except OSError:
                    pass
                with open(target, "r+", encoding="utf-8") as handle:
                    handle.seek(0)
                    handle.write(op["text"])
                    handle.truncate()
            else:
                target.write_text(op["text"], encoding="utf-8")
        elif kind == "delete":
            target.unlink()
        elif kind == "mkdir":
            target.mkdir(parents=True, exist_ok=True)
        elif kind == "rmdir":
            target.rmdir()
        elif kind == "symlink":
            os.symlink(op["target"], str(target), target_is_directory=False)
        elif kind == "junction":
            import _winapi
            _winapi.CreateJunction(op["target"], str(target))
        elif kind == "chmod":
            os.chmod(str(target), int(op["mode"], 8))
        elif kind == "mkfifo":
            os.mkfifo(str(target))
        elif kind == "rename":
            target.parent.mkdir(parents=True, exist_ok=True)
            Path(op["src"]).rename(target)
        elif kind == "git":
            subprocess.run(["git", *op["argv"]], capture_output=True)
    sys.stdin.read()
    if cfg.get("stdout_hex"):
        sys.stdout.buffer.write(bytes.fromhex(cfg["stdout_hex"]))
        sys.stdout.buffer.flush()
    sys.exit(cfg.get("code", 0))


main()
'''


def _git(repo, *args):
    """The REAL subprocess module, for fixture setup only."""
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _reply(**overrides):
    """One implementer reply, in the protocol the TRANSPORT asks for.

    NO `next_action` (B7-R1). The field left the transport schema because
    its conditional `const` rules cannot survive the provider subset, and
    the adapter derives it from `status` before the authority judges the
    reply. A stub that still sent it would be speaking a protocol no
    schema asks for -- and would be REFUSED, which is the point."""
    payload = {"protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
               "role": "implementer", "status": "implemented",
               "summary": "kurgu ozet"}
    payload.update(overrides)
    return payload


def _envelope(payload):
    """The RESULT ENVELOPE the real CLI was measured to return (B4-R3).

    `claude --print --output-format json` never answers with a bare
    implementer payload: it answers with `type`/`subtype`/`is_error`,
    the payload under `structured_output`, the same payload rendered as
    text under `result`, and identifiers and usage the adapter must
    refuse to carry anywhere. A fake still printing the bare payload
    would be exercising a protocol no binary speaks."""
    return {
        "type": "result", "subtype": "success", "is_error": False,
        "terminal_reason": "completed", "stop_reason": "tool_use",
        "num_turns": 2, "duration_ms": 7523, "duration_api_ms": 5011,
        "total_cost_usd": 0.056757, "permission_denials": [],
        "session_id": "00000000-0000-4000-8000-000000000000",
        "uuid": "00000000-0000-4000-8000-000000000001",
        "usage": {"input_tokens": 4, "output_tokens": 5},
        "modelUsage": {}, "result": json.dumps(payload),
        "structured_output": payload,
    }


def _stub(tmp_path, name="sahte_claude", ops=(), reply=None, code=0,
          cwd_record=None):
    holder = tmp_path / STUB_HOLDER
    holder.mkdir(exist_ok=True)
    helper = holder / "yardimci.py"
    helper.write_text(_HELPER, encoding="utf-8")
    config = {"ops": list(ops), "code": code}
    if reply is not None:
        config["stdout_hex"] = json.dumps(
            _envelope(reply)).encode("utf-8").hex()
    if cwd_record is not None:
        config["cwd_record"] = str(cwd_record)
    settings = holder / f"{name}.json"
    settings.write_text(json.dumps(config), encoding="utf-8")
    if os.name == "nt":
        shim = holder / f"{name}.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{helper}" "{settings}" %*\r\n',
            encoding="ascii")
    else:
        shim = holder / f"{name}.sh"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{helper}" "{settings}" "$@"\n',
            encoding="ascii")
        shim.chmod(0o755)
    return shim


# ---------------------------------------------------------------------
# THE SHARED WORLD. Both change-set suites build it from here, so the
# two files cannot drift apart in what "a bound run" means.
# ---------------------------------------------------------------------

def build_repo(tmp_path, index=0):
    """A throwaway repository with a stand-in control plane, committed --
    so every protected file really exists inside the workspace."""
    repo = tmp_path / f"kurgu-depo-{index}"
    (repo / "pipeline").mkdir(parents=True)
    for argv in (["init", "-q"],
                 ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(repo, *argv)
    (repo / "pipeline" / "kurgu.py").write_text("VALUE = 1\n",
                                                encoding="utf-8")
    (repo / "pipeline" / "silinecek.py").write_text("VALUE = 2\n",
                                                    encoding="utf-8")
    (repo / "pipeline" / "gizli").mkdir(parents=True, exist_ok=True)
    (repo / "pipeline" / "gizli" / "sir.py").write_text("VALUE = 3\n",
                                                        encoding="utf-8")
    for control in contract.CONTROL_PLANE_PATHS:
        target = repo / control.rstrip("/")
        if control.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            target = target / "contract.py"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("FROZEN = True\n", encoding="utf-8")
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / "tests" / "test_agent_loop_b1.py").write_text(
        "def test_kurgu():\n    assert True\n", encoding="utf-8")
    (repo / ".gitignore").write_text("data/\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "kurgu")
    return repo, _git(repo, "rev-parse", "HEAD")


def build_gate(tmp_path, index=0, task_outside=False, gitignore_task=False,
               **task_overrides):
    """A REAL repository, a REAL D3A flat workspace at its baseline, a
    task manifest bound by digest, and the identity tuple the seam takes.

    Every gate gets its OWN state directory: one binding document names
    one workspace, so two gates sharing a directory would leave the
    second one's binding answering for the first."""
    repo, baseline = build_repo(tmp_path, index)
    if gitignore_task:
        (repo / ".gitignore").write_text("data/\nkurgu-task.json\n",
                                         encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "kurgu-yoksay")
        baseline = _git(repo, "rev-parse", "HEAD")

    # UNTRACKED, exactly like the Phase A run: the main-checkout snapshot
    # has to tolerate its presence and still notice it changing. A
    # manifest OUTSIDE the repository is the case the snapshot cannot see
    # at all, which is where the digest re-check is the only defence.
    task_file = ((tmp_path / f"disarida-{index}.json") if task_outside
                 else repo / "kurgu-task.json")
    payload = dict(BASE_TASK, baseline_sha=baseline)
    payload.update(task_overrides)         # an override may name baseline_sha
    task_file.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(task_file.read_bytes()).hexdigest()

    state_dir = tmp_path / f"durum-{index}"
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace = flat_workspace.create(repo, state_dir=state_dir, run_id=RUN,
                                      baseline_sha=baseline)
    state.write_binding(state_dir, {
        "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
        "repo_id": state.repo_identity(repo), "baseline_sha": baseline,
        "manifest_digest": digest, "workspace_id": workspace.workspace_id})
    return types.SimpleNamespace(
        repo=repo, workspace=workspace, tree=workspace.implementer_root,
        reference=workspace.reference_root, state_dir=state_dir,
        task=task_file, digest=digest, baseline=baseline,
        workspace_id=workspace.workspace_id,
        identity={"repo": repo, "state_dir": state_dir,
                  "task_path": task_file, "manifest_digest": digest,
                  "run_id": RUN, "workspace_id": workspace.workspace_id,
                  "baseline_sha": baseline})


@pytest.fixture(autouse=True)
def private_runner_root(tmp_path, monkeypatch):
    """Every test gets its OWN runner root, so nothing here can create,
    list or delete a holder a real agent loop is using."""
    private = tmp_path / "runner-koku"
    private.mkdir()
    monkeypatch.setattr(flat_workspace, "runner_temp_root", lambda: private)
    yield private
    shutil.rmtree(private, ignore_errors=True)


@pytest.fixture(autouse=True)
def only_fake_models_may_run(tmp_path, monkeypatch):
    """THE claim this file rests on, enforced rather than repeated.

    Only stub launches are RECORDED -- `flat_workspace.create()` runs git
    through this same seam -- but every launch is still JUDGED: anything
    that is neither a stub under `tmp_path` nor the fixture's own git is
    a failure."""
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
    izinli = ("git", "git.exe", "taskkill", "taskkill.exe")
    strayed = [argv[0] for argv in hepsi
               if not str(argv[0]).casefold().startswith(root)
               and Path(argv[0]).name.casefold() not in izinli]
    assert strayed == [], f"tmp_path disinda bir program calistirildi: {strayed}"
    assert [p.pid for p in started if p.poll() is None] == []


@pytest.fixture
def make_gate(tmp_path):
    made = []

    def build(**overrides):
        made.append(None)
        return build_gate(tmp_path, index=len(made) - 1, **overrides)
    return build


@pytest.fixture
def gate(make_gate):
    return make_gate()


def _run(binary, gate_obj, **overrides):
    settings = {"prompt": "kurgu istem", "budget_usd": 1.0,
                "timeout_seconds": 60, "max_output_bytes": 65536}
    settings.update(overrides)
    return changes.run_verified_implementation(binary, **gate_obj.identity,
                                               **settings)


def _write(path, text=SENTINEL):
    return {"kind": "write", "path": str(path), "text": text + "\n"}


# =====================================================================
# A. POSITIVE CONTROL -- an implementation that rejects everything
#    cannot pass this file
# =====================================================================

def test_an_allowed_edit_is_verified(tmp_path, gate,
                                     only_fake_models_may_run):
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    verified = _run(binary, gate)
    assert verified.changed_files == ("pipeline/kurgu.py",)
    assert (verified.added, verified.modified, verified.deleted) == (0, 1, 0)
    assert len(verified.fingerprint) == 64
    assert verified.status == "implemented"
    assert verified.run_id == RUN and verified.baseline_sha == gate.baseline
    assert only_fake_models_may_run, "senaryo kurulmadi: model hic calismadi"


# =====================================================================
# B. THE EVIDENCE IS THE AUTHORITY, THE DECLARATION IS A CLAIM
# =====================================================================

def test_a_declared_file_nothing_changed_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(changed_files=["pipeline/hayal.py"]))
    with pytest.raises(changes.DeclarationMismatch) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.SCHEMA_VIOLATION


def test_a_change_the_model_omits_is_refused(tmp_path, gate):
    """Omission is the direction a subset comparison cannot see."""
    binary = _stub(tmp_path,
                   ops=[_write("pipeline/bir.py"), _write("pipeline/iki.py")],
                   reply=_reply(changed_files=["pipeline/bir.py"]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


def test_a_fictitious_extra_declaration_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/bir.py")],
                   reply=_reply(changed_files=["pipeline/bir.py",
                                               "pipeline/yok.py"]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


def test_a_duplicate_declaration_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/bir.py")],
                   reply=_reply(changed_files=["pipeline/bir.py",
                                               "pipeline/bir.py"]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


def test_a_reply_naming_another_run_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(run_id="baska-kosu-1",
                                          changed_files=[]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


def test_a_blocked_reply_that_edited_files_yields_no_verified_set(
        tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(status="blocked",
                                stop_reason="interrupted",
                                changed_files=["pipeline/kurgu.py"]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


def test_the_fingerprint_covers_content_not_only_paths(tmp_path, make_gate):
    """Two runs, same path, same status, DIFFERENT bytes. A path-only
    digest calls these identical, and a later transfer step would have
    no way to tell which change it was carrying."""
    seen = []
    for text in ("BIRINCI-ICERIK", "IKINCI-ICERIK"):
        current = make_gate()
        binary = _stub(tmp_path, name=f"sahte-{text}",
                       ops=[_write("pipeline/kurgu.py", text)],
                       reply=_reply(changed_files=["pipeline/kurgu.py"]))
        verified = _run(binary, current)
        assert verified.changed_files == ("pipeline/kurgu.py",)
        seen.append(verified.fingerprint)
    assert seen[0] != seen[1], "icerik parmak izine girmiyor"


# =====================================================================
# C. SCOPE BOUNDARIES
# =====================================================================

def test_a_prefix_sibling_directory_is_not_covered(tmp_path, make_gate):
    """`pipeline/` permits `pipeline/a.py` and must not permit
    `pipeline_private/a.py` -- the whole-segment lesson, paid for once
    already in preflight."""
    current = make_gate()
    binary = _stub(tmp_path, ops=[_write("pipeline_private/a.py")],
                   reply=_reply(changed_files=["pipeline_private/a.py"]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert (current.tree / "pipeline_private" / "a.py").exists(), \
        "senaryo kurulmadi: dosya hic yazilmadi"


def test_forbidden_beats_allowed(tmp_path, gate):
    """`pipeline/gizli/` sits inside the allowed `pipeline/`."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py")],
                   reply=_reply(changed_files=["pipeline/gizli/a.py"]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


def test_an_exact_allowed_file_does_not_cover_a_prefix_sibling(
        tmp_path, make_gate):
    current = make_gate(allowed_paths=["pipeline/kurgu.py"])
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py.ozel")],
                   reply=_reply(changed_files=["pipeline/kurgu.py.ozel"]))
    with pytest.raises(changes.UnsafeChange):
        _run(binary, current)


def test_the_task_manifest_path_is_immutable_even_inside_allowed_paths(
        tmp_path, make_gate):
    """The manifest is what grants the permissions; a run that may
    rewrite it may widen its own scope mid-call."""
    current = make_gate(allowed_paths=["pipeline/", "kurgu-task.json"])
    binary = _stub(tmp_path, ops=[_write("kurgu-task.json", "{}")],
                   reply=_reply(changed_files=["kurgu-task.json"]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert (current.tree / "kurgu-task.json").exists(), \
        "senaryo kurulmadi: gorev dosyasi calisma alaninda yazilmadi"


@pytest.mark.parametrize(
    "target",
    ["tools/agent_loop/contract.py", "tests/test_agent_loop_b1.py",
     "eval/tools/leak_scan.py", "scripts/p0_gate.sh"])
def test_a_control_plane_edit_is_terminal(tmp_path, make_gate, target):
    current = make_gate(allowed_paths=["pipeline/"])
    binary = _stub(tmp_path, ops=[_write(target)],
                   reply=_reply(changed_files=[target]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.CONTROL_PLANE_MODIFIED
    assert (current.tree / target).read_text(encoding="utf-8").strip() \
        == SENTINEL, "senaryo kurulmadi: kontrol duzlemi yazilmadi"


def test_a_test_file_invented_tomorrow_is_caught_by_the_glob(tmp_path, gate):
    """`tests/test_agent_loop_future.py` is in no list today. The frozen
    FAMILY pattern is what protects it."""
    target = "tests/test_agent_loop_future.py"
    binary = _stub(tmp_path, ops=[_write(target)],
                   reply=_reply(changed_files=[target]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.CONTROL_PLANE_MODIFIED


def test_a_task_cannot_allowlist_the_control_plane(tmp_path, make_gate):
    """Even a manifest that names it: authority order puts the control
    plane first, and no permission a task writes may reach it."""
    current = make_gate()
    with pytest.raises(Exception):
        make_gate(allowed_paths=["tools/agent_loop/"]) and \
            changes._authorize("tools/agent_loop/contract.py",
                               allowed=("tools/agent_loop/",), forbidden=(),
                               task_relative=None)
    assert current.tree.exists()


@pytest.mark.parametrize(
    "spelling", ["PIPELINE/GIZLI/", "./pipeline/gizli/", "pipeline//gizli/"])
def test_a_forbidden_entry_cannot_be_escaped_by_spelling(
        tmp_path, make_gate, spelling):
    """All three walked straight past the forbidden list: the
    comparisons were raw string operations and the manifest side was
    never normalised at all.

    A BACKSLASH spelling is deliberately absent -- the frozen task
    schema refuses such a manifest outright, so it never reaches this
    module, and putting it here would test Phase A's gate rather than
    this one."""
    if spelling.startswith("PIPELINE") and os.name != "nt":
        pytest.skip("harf katlama yalnizca Windows kuralinda gecerli")
    current = make_gate(forbidden_paths=[spelling])
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py")],
                   reply=_reply(changed_files=["pipeline/gizli/a.py"]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


# =====================================================================
# D. THE MAIN CHECKOUT -- STILL GUARDED THROUGH GIT, ON PURPOSE
#
# B2B-B2B replaces this instrument with filesystem evidence. Until then
# the guard keeps working exactly as it did, and these are the tests
# that say so.
# =====================================================================

def test_an_edit_to_the_operator_checkout_is_caught(tmp_path, gate):
    """The flat workspace exists so the main tree is never the thing at
    risk; a model that reached it anyway must not pass. A TRACKED file,
    so this scenario does not also depend on the untracked inventory."""
    hedef = gate.repo / "pipeline" / "kurgu.py"
    binary = _stub(tmp_path, ops=[_write(hedef)],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert SENTINEL in hedef.read_text(encoding="utf-8"), \
        "senaryo kurulmadi: ana agaca yazilmadi"


def test_no_error_path_carries_the_content_of_a_refused_file(
        tmp_path, gate):
    """The CONTENT of a refused file is a channel privacy had not been
    tested on: it travels into whatever a report writes if a message
    ever repeats it.

    The git-stderr half of this test went with the git guard -- there is
    no stderr to leak now -- and its successor is the scan-failure
    privacy test in `test_agent_loop_b2_main_guard.py`."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py", SENTINEL)],
                   reply=_reply(changed_files=["pipeline/gizli/a.py"]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    assert SENTINEL not in str(refusal.value) + repr(refusal.value)


def test_a_healthy_call_leaves_the_operator_checkout_untouched(
        tmp_path, gate):
    before = (_git(gate.repo, "rev-parse", "HEAD"),
              _git(gate.repo, "status", "--porcelain"),
              gate.task.read_bytes())
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    _run(binary, gate)
    assert (_git(gate.repo, "rev-parse", "HEAD"),
            _git(gate.repo, "status", "--porcelain"),
            gate.task.read_bytes()) == before


# =====================================================================
# E. THE TASK MANIFEST BINDING
# =====================================================================

def test_a_rewritten_task_manifest_is_caught(tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write(gate.task, "{}")],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange):
        _run(binary, gate)
    assert gate.task.read_text(encoding="utf-8").strip() != "", \
        "senaryo kurulmadi"


def test_a_manifest_outside_the_repository_is_refused(tmp_path, make_gate):
    """A manifest the repository has no relationship with must not be
    able to grant this run its permissions."""
    current = make_gate(task_outside=True)
    assert current.repo not in current.task.parents, "senaryo kurulmadi"
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.EvidenceUnavailable):
        _run(binary, current)


def test_an_ignored_manifest_inside_the_repository_is_still_re_verified(
        tmp_path, make_gate):
    """The case the main-checkout snapshot CANNOT see: a manifest that
    lives in the repository but is GITIGNORED, so git never lists it and
    the snapshot never hashes it. Here the digest re-check is the only
    layer standing."""
    current = make_gate(gitignore_task=True)
    assert _git(current.repo, "status", "--porcelain") == "", \
        "senaryo kurulmadi: gorev dosyasi git tarafindan hala goruluyor"
    binary = _stub(tmp_path, ops=[_write(current.task, "{}")],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    # THE SENTENCE, not only the reason. B2B-B2B made the main-checkout
    # guard a filesystem walk, and a filesystem walk DOES see a gitignored
    # file -- so the manifest gate and the main-tree gate now both catch
    # this scenario and both close to `path_not_allowed`. Asserting the
    # reason alone therefore stopped saying anything about the gate this
    # test is named for: with the digest re-check deleted, the run stayed
    # green here and the mutation that removes it survived unnoticed.
    assert str(refusal.value) == "gorev dosyasi degistirildi", \
        "ret manifest kapisindan gelmedi"
    assert current.task.read_text(encoding="utf-8").strip() == "{}", \
        "senaryo kurulmadi: gorev dosyasi hic degismedi"


def test_a_manifest_naming_another_baseline_is_refused(tmp_path, make_gate):
    """The manifest's own `baseline_sha` was never compared with the
    baseline every other binding names."""
    current = make_gate(baseline_sha="0" * 40)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.BASELINE_MISMATCH


# =====================================================================
# F. THE CALL BOUNDARY AND `finally`
# =====================================================================

def test_a_failing_call_that_changed_nothing_keeps_its_own_error(
        tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(changed_files=[]), code=7)
    with pytest.raises(execution.ProcessFailed) as refusal:
        _run(binary, gate)
    assert refusal.value.exit_code == 7


def test_a_forbidden_edit_outranks_the_call_failure_and_chains_it(
        tmp_path, gate):
    """The safety violation is the headline; the failure that hid it is
    chained rather than erased."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py")],
                   reply=_reply(changed_files=[]), code=5)
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert isinstance(refusal.value.__cause__, execution.ProcessFailed), \
        "asil hata zincirlenmedi"


def test_a_control_plane_edit_outranks_the_call_failure(tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("tools/agent_loop/contract.py")],
                   reply=_reply(changed_files=[]), code=5)
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.CONTROL_PLANE_MODIFIED
    assert refusal.value.__cause__ is not None


def test_the_post_check_runs_even_when_the_reply_is_unparsable(
        tmp_path, gate):
    """Not only on the happy path: a schema violation is still an exit,
    and files may already have moved."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py")],
                   reply={"bu": "sema disi"})
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


# =====================================================================
# G. PRIVACY, STRUCTURE AND PLATFORM
# =====================================================================

def test_no_refusal_echoes_an_attacker_controlled_path(tmp_path, gate):
    hedef = f"pipeline_private/{SENTINEL}.py"
    binary = _stub(tmp_path, ops=[_write(hedef)],
                   reply=_reply(changed_files=[hedef]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    metin = str(refusal.value) + repr(refusal.value)
    assert SENTINEL not in metin, "ret metni saldirgan yolunu tasiyor"
    assert "/" not in metin and "\\" not in metin


def test_no_result_or_error_carries_file_bytes(tmp_path, gate):
    import dataclasses

    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    verified = _run(binary, gate)
    values = [getattr(verified, field.name)
              for field in dataclasses.fields(verified)]
    carried = " ".join(str(value) for value in values)
    assert SENTINEL not in carried, "sonuc dosya icerigi tasiyor"
    assert not any(isinstance(value, (bytes, bytearray)) for value in values)


def test_the_module_asks_no_program_and_writes_no_file():
    """Structural: a write path or a launched program that appears only
    on a rare branch would pass every functional test in this file.

    The git assertions here are not about style. Every authority this
    module used to take from git -- the inventory, the index, HEAD --
    was answered through configuration the model can reach, which is
    what made `skip-worktree` a P0 twice. There is nothing left to
    configure."""
    import ast

    source = Path(changes.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports, calls, writers = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        if isinstance(node, ast.Call):
            called = ast.unparse(node.func)
            calls.add(called)
            if called.endswith((".write_text", ".write_bytes", ".unlink",
                                ".rename", ".mkdir", ".rmtree", ".chmod")):
                writers.add(called)
            for keyword in node.keywords:
                assert not (keyword.arg == "shell"
                            and getattr(keyword.value, "value", False)), \
                    "kabuk kullanimi"
    assert "subprocess" not in imports, "subprocess yeniden ice aktarildi"
    assert not [call for call in calls
                if call.startswith("subprocess.")
                or call in ("os.system", "os.popen", "os.execv")], \
        "modul yeniden program calistiriyor"
    assert writers == set(), f"dosya sistemine yazma: {sorted(writers)}"
    # the positive control: an assertion battery that would also pass on
    # an empty file is documentation, not a test
    assert "fs_evidence.scan" in calls, \
        "senaryo kurulmadi: dosya sistemi taramasi hic cagrilmiyor"


def test_scope_matching_follows_the_declared_platform_rule(tmp_path):
    """Case-sensitive on POSIX, Windows's standard fold on Windows --
    the repository's already-declared rule, not a new claim."""
    assert changes._covered("pipeline/a.py", ("pipeline/",))
    assert changes._covered("pipeline/a.py", ("./pipeline//",))
    assert not changes._covered("pipeline_private/a.py", ("pipeline/",))
    assert changes._fold("./a//b/") == changes._fold("a/b")
    assert changes._same_place("/a/b", "/a/b")
    if os.name == "nt":
        assert changes._same_place("C:/A", "c:/a")
    else:
        assert not changes._same_place("/a/B", "/a/b")
