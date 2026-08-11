"""PACKAGE B2B-A -- the verified implementer change set.

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
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from tools.agent_loop import (changes, contract, execution, schemas,
                              state, worktree)
from tools.agent_loop import process as process_module

RUN = "kurgu-run-1"
SENTINEL = "KURGU-GIZLI-NOBETCI-" + "q" * 8

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

# The stub runs INSIDE the disposable worktree, so every relative path
# it touches is a change this gate has to find and judge.
_HELPER = '''\
import json, os, subprocess, sys
from pathlib import Path


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for op in cfg.get("ops", []):
        kind = op["kind"]
        target = Path(op["path"]) if op.get("path") else None
        if kind == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # IN PLACE. A worktree's `.git` file is HIDDEN on
                # Windows, and opening a hidden file with "w"
                # (CREATE_ALWAYS) fails with access denied -- so the
                # scenario would have died on the attribute instead of
                # reaching what it is about. A real process rewrites it
                # the same way.
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
    """The REAL subprocess module: the recorder below only replaces what
    the code under test launches models with."""
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _reply(**overrides):
    payload = {"protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
               "role": "implementer", "status": "implemented",
               "summary": "kurgu ozet", "next_action": "await_acceptance"}
    payload.update(overrides)
    return payload


def _stub(tmp_path, name="sahte_claude", ops=(), reply=None, code=0):
    holder = tmp_path / "sahte-bin"
    holder.mkdir(exist_ok=True)
    helper = holder / "yardimci.py"
    helper.write_text(_HELPER, encoding="utf-8")
    config = {"ops": list(ops), "code": code}
    if reply is not None:
        config["stdout_hex"] = json.dumps(reply).encode("utf-8").hex()
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


@pytest.fixture(autouse=True)
def private_worktree_root(tmp_path):
    """Every test gets its OWN runner temp root, so nothing here can
    create, list or delete a holder a real agent loop is using."""
    private = tmp_path / "runner-temp"
    private.mkdir()
    isolation = pytest.MonkeyPatch()
    isolation.setattr(tempfile, "tempdir", str(private))
    for variable in ("TMPDIR", "TEMP", "TMP"):
        isolation.setenv(variable, str(private))
    root = worktree.runner_temp_root()
    assert root.parent.resolve() == private.resolve()
    yield root
    isolation.undo()


@pytest.fixture(autouse=True)
def only_fake_models_may_run(tmp_path, monkeypatch):
    """THE claim this file rests on, enforced rather than repeated."""
    launched, started = [], []
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
            launched.append(list(argv))
            return subprocess.run(argv, **kwargs)

        @staticmethod
        def Popen(argv, **kwargs):                 # noqa: N802 -- stdlib
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
    strayed = [argv[0] for argv in launched
               if not str(argv[0]).casefold().startswith(root)
               and Path(argv[0]).name.casefold() not in ("taskkill.exe",
                                                         "taskkill")]
    assert strayed == [], f"tmp_path disinda bir model calistirildi: {strayed}"
    assert [p.pid for p in started if p.poll() is None] == []


@pytest.fixture
def make_gate(tmp_path):
    """A REAL repository, a REAL B1 worktree at its baseline, a task
    manifest bound by digest, and the identity tuple the seam takes."""
    made = []

    def build(task_outside=False, gitignore_task=False, **task_overrides):
        # a fresh repository per call: two gates in one test must not
        # share a checkout, or the second build fails on the first's
        # directories and the test dies before its scenario exists
        repo = tmp_path / f"kurgu-depo-{len(made)}"
        made.append(repo)
        (repo / "pipeline").mkdir(parents=True)
        for argv in (["init", "-q"],
                     ["config", "user.email", "k@example.invalid"],
                     ["config", "user.name", "Kurgu"]):
            _git(repo, *argv)
        (repo / "pipeline" / "kurgu.py").write_text("VALUE = 1\n",
                                                    encoding="utf-8")
        (repo / "pipeline" / "silinecek.py").write_text("VALUE = 2\n",
                                                        encoding="utf-8")
        # a stand-in control plane, tracked, so an edit to it inside the
        # worktree is a real change this gate has to call terminal
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
        (repo / ".gitignore").write_text(
            "data/\n" + ("kurgu-task.json\n" if gitignore_task else ""),
            encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "kurgu")
        baseline = _git(repo, "rev-parse", "HEAD")

        # UNTRACKED, exactly like the Phase A run: the snapshot has to
        # tolerate its presence and still notice it changing. A manifest
        # OUTSIDE the repository is the case the snapshot cannot see at
        # all, which is where the digest re-check is the only defence.
        task_file = ((tmp_path / f"disarida-{len(made)}.json") if task_outside
                     else repo / "kurgu-task.json")
        payload = dict(BASE_TASK, baseline_sha=baseline)
        payload.update(task_overrides)     # an override may name baseline_sha
        task_file.write_text(json.dumps(payload), encoding="utf-8")
        digest = hashlib.sha256(task_file.read_bytes()).hexdigest()

        state_dir = tmp_path / "durum"
        path, worktree_id = worktree.create(repo, state_dir=state_dir,
                                            run_id=RUN,
                                            baseline_sha=baseline)
        state.write_binding(state_dir, {
            "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
            "repo_id": state.repo_identity(repo), "baseline_sha": baseline,
            "manifest_digest": digest, "worktree_id": worktree_id})
        return types.SimpleNamespace(
            repo=repo, tree=path, state_dir=state_dir, task=task_file,
            digest=digest, worktree_id=worktree_id, baseline=baseline,
            identity={"repo": repo, "state_dir": state_dir,
                      "task_path": task_file, "manifest_digest": digest,
                      "run_id": RUN, "worktree_id": worktree_id,
                      "baseline_sha": baseline})
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
    return {"kind": "write", "path": path, "text": text + "\n"}


# =====================================================================
# A. POSITIVE CONTROLS -- an implementation that rejects everything
#    cannot pass this file
# =====================================================================

def test_an_allowed_tracked_edit_is_verified(tmp_path, gate,
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


def test_an_allowed_untracked_file_is_verified(tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/yeni.py")],
                   reply=_reply(changed_files=["pipeline/yeni.py"]))
    verified = _run(binary, gate)
    assert verified.changed_files == ("pipeline/yeni.py",)
    assert (verified.added, verified.modified, verified.deleted) == (1, 0, 0)


def test_an_allowed_deletion_is_verified(tmp_path, gate):
    binary = _stub(tmp_path,
                   ops=[{"kind": "delete", "path": "pipeline/silinecek.py"}],
                   reply=_reply(changed_files=["pipeline/silinecek.py"]))
    verified = _run(binary, gate)
    assert verified.changed_files == ("pipeline/silinecek.py",)
    assert (verified.added, verified.modified, verified.deleted) == (0, 0, 1)


def test_a_filename_with_spaces_survives_the_inventory(tmp_path, gate):
    """NUL-delimited plumbing, not line splitting: an ordinary space is
    the cheapest way to prove the parser is not reading columns."""
    name = "pipeline/iki kelimeli dosya.py"
    binary = _stub(tmp_path, ops=[_write(name)], reply=_reply(
        changed_files=[name]))
    assert _run(binary, gate).changed_files == (name,)


def test_a_clean_no_op_is_verified_as_empty(tmp_path, gate):
    """The contract has no "must edit something" rule, so inventing one
    here would refuse a legitimate implementation."""
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    verified = _run(binary, gate)
    assert verified.changed_files == ()
    assert verified.fingerprint == hashlib.sha256(b"").hexdigest()


def test_declaration_order_does_not_matter(tmp_path, gate):
    binary = _stub(tmp_path,
                   ops=[_write("pipeline/bir.py"), _write("pipeline/iki.py")],
                   reply=_reply(changed_files=["pipeline/iki.py",
                                               "pipeline/bir.py"]))
    assert _run(binary, gate).changed_files == ("pipeline/bir.py",
                                                "pipeline/iki.py")


# =====================================================================
# B. GIT IS THE AUTHORITY, THE DECLARATION IS A CLAIM
# =====================================================================

def test_a_declared_file_git_never_saw_is_refused(tmp_path, gate):
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


def test_the_same_content_gives_the_same_fingerprint(tmp_path, make_gate):
    seen = []
    for index in range(2):
        current = make_gate()
        binary = _stub(tmp_path, name=f"kararli-{index}",
                       ops=[_write("pipeline/kurgu.py", "AYNI-ICERIK")],
                       reply=_reply(changed_files=["pipeline/kurgu.py"]))
        seen.append(_run(binary, current).fingerprint)
    assert seen[0] == seen[1], "parmak izi belirlenimci degil"


def test_an_untracked_file_that_git_diff_would_omit_is_still_seen(
        tmp_path, gate):
    """The measurement, not the claim: `git diff` really is blind to a
    new file, and the gate really is not."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gorunmez.py")],
                   reply=_reply(changed_files=["pipeline/gorunmez.py"]))
    verified = _run(binary, gate)
    assert _git(gate.tree, "diff", "--name-only") == "", \
        "senaryo kurulmadi: diff bu dosyayi zaten goruyor"
    assert verified.changed_files == ("pipeline/gorunmez.py",)


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
    # the frozen schema refuses such a manifest outright, so the check
    # is asserted directly on the authorization rule as well
    with pytest.raises(Exception):
        make_gate(allowed_paths=["tools/agent_loop/"]) and \
            changes._authorize("tools/agent_loop/contract.py",
                               allowed=("tools/agent_loop/",), forbidden=(),
                               task_relative=None)
    assert current.tree.exists()


# =====================================================================
# D. GIT STATE INTEGRITY
# =====================================================================

def test_a_dirty_worktree_is_refused_before_the_model_starts(
        tmp_path, gate, only_fake_models_may_run):
    """Attribution is impossible in a tree that was already dirty, so
    the refusal has to come before a process exists."""
    (gate.tree / "pipeline" / "onceden.py").write_text("X\n",
                                                       encoding="utf-8")
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.DIRTY_WORKTREE
    assert only_fake_models_may_run == [], "kirli agacta model calistirildi"


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_an_index_flag_cannot_blind_the_gate(tmp_path, gate, flag):
    """THE P0. `git status` is not the whole truth: an entry marked
    `skip-worktree` or `assume-unchanged` is one git has been TOLD to
    stop looking at. A probe used exactly that to rewrite a
    control-plane file while `status` stayed empty and this gate
    returned an empty change set -- a terminal violation reported as
    "nothing happened"."""
    target = "tools/agent_loop/contract.py"
    binary = _stub(tmp_path, ops=[
        {"kind": "git", "argv": ["update-index", flag, target]},
        _write(target, "SAHIPLENILDI")],
        reply=_reply(changed_files=[]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    # the setup is only meaningful if git really was blinded
    assert _git(gate.tree, "status", "--porcelain") == "", \
        "senaryo kurulmadi: git degisikligi hala goruyor"
    assert "SAHIPLENILDI" in (gate.tree / target).read_text(
        encoding="utf-8"), "senaryo kurulmadi: dosya hic degismedi"
    assert refusal.value.reason in (contract.StopReason.PREFLIGHT_FAILED,
                                    contract.StopReason.STAGED_CHANGES)


def test_a_pre_existing_index_flag_is_refused_before_the_model_starts(
        tmp_path, gate, only_fake_models_may_run):
    _git(gate.tree, "update-index", "--skip-worktree", "pipeline/kurgu.py")
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange):
        _run(binary, gate)
    assert only_fake_models_may_run == [], \
        "gizlenmis indekste model calistirildi"


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


def test_a_symlink_deletion_is_refused_and_modes_reach_the_fingerprint():
    """Every deletion used to be recorded as `000000`, which accepted
    the removal of a SYMLINK as an ordinary file and erased the field
    that tells two otherwise identical deletions apart."""
    with pytest.raises(changes.UnsafeChange):
        changes._classify(
            b"1 .D N... 120000 120000 000000 aaa bbb pipeline/link.py",
            None, Path("."))
    modes = []
    for mode in ("100644", "100755"):
        record = (f"1 .D N... {mode} {mode} 000000 aaa bbb "
                  f"pipeline/ayni.py").encode("ascii")
        change = changes._classify(record, None, Path("."))
        assert change.mode == mode
        modes.append(changes._fingerprint((change,)))
    assert modes[0] != modes[1], \
        "ayni yolun iki silme modu ayni parmak izini veriyor"


def test_a_staged_change_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, ops=[
        _write("pipeline/kurgu.py"),
        {"kind": "git", "argv": ["add", "pipeline/kurgu.py"]}],
        reply=_reply(changed_files=["pipeline/kurgu.py"]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.STAGED_CHANGES


def test_a_moved_head_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, ops=[
        {"kind": "git", "argv": ["commit", "--allow-empty", "-qm", "kurgu2"]}],
        reply=_reply(changed_files=[]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    assert refusal.value.reason in (contract.StopReason.BASELINE_MISMATCH,
                                    contract.StopReason.PREFLIGHT_FAILED)
    assert _git(gate.tree, "rev-parse", "HEAD") != gate.baseline, \
        "senaryo kurulmadi: HEAD hic oynamadi"


def test_a_replaced_git_link_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write(".git", "gitdir: /kurgu/yok")],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    assert refusal.value.reason in (contract.StopReason.PREFLIGHT_FAILED,
                                    contract.StopReason.PATH_NOT_ALLOWED)


def test_a_failing_git_query_is_unverifiable_not_clean(tmp_path, gate,
                                                       monkeypatch):
    """An empty inventory from a FAILED command reads as "no changes",
    which is how an unverifiable state becomes a verified one."""
    real_run = changes.subprocess.run
    seen = {"count": 0}

    def failing(argv, **kwargs):
        if "status" in argv:
            seen["count"] += 1
            if seen["count"] > 1:                  # let preflight pass
                return subprocess.CompletedProcess(argv, 1, b"", b"kurgu")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(changes.subprocess, "run", failing)
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    with pytest.raises(changes.EvidenceUnavailable):
        _run(binary, gate)
    assert seen["count"] > 1, "senaryo kurulmadi: enjeksiyon hic tetiklenmedi"


def test_a_git_timeout_is_unverifiable(tmp_path, gate, monkeypatch):
    real_run = changes.subprocess.run
    seen = {"count": 0}

    def timing_out(argv, **kwargs):
        if "status" in argv:
            seen["count"] += 1
            if seen["count"] > 1:
                raise subprocess.TimeoutExpired(argv, 1)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(changes.subprocess, "run", timing_out)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.EvidenceUnavailable):
        _run(binary, gate)
    assert seen["count"] > 1, "senaryo kurulmadi"


def test_an_unknown_porcelain_record_is_refused():
    """Guessing how to follow an unfamiliar record is how an unreviewed
    edit reaches the main checkout."""
    with pytest.raises(changes.UnsafeChange):
        changes._classify(b"X . N... 100644 100644 100644 aaa bbb kurgu.py",
                          None, Path("."))


def test_a_rename_record_is_refused_rather_than_reinterpreted():
    with pytest.raises(changes.UnsafeChange):
        changes._classify(b"2 .R N... 100644 100644 100644 aaa bbb R100 y.py",
                          b"x.py", Path("."))


def test_an_unmerged_record_is_refused():
    with pytest.raises(changes.UnsafeChange) as refusal:
        changes._classify(b"u UU N... 100644 100644 100644 100644 a b c k.py",
                          None, Path("."))
    assert refusal.value.reason == contract.StopReason.STAGED_CHANGES


def test_rename_detection_is_disabled_so_one_edit_is_one_record(gate):
    """Asserted on the RECORDS, not on a refusal: the refusal happened
    either way -- rename-detected records are also refused -- so the
    old shape passed with `--no-renames` removed and proved nothing
    about the flag. What the flag buys is that BOTH paths appear as
    ordinary records this gate can judge separately."""
    (gate.tree / "pipeline" / "silinecek.py").rename(
        gate.tree / "pipeline" / "tasindi.py")
    kinds = [record[:1] for record, _ in changes._status(gate.tree)]
    assert b"2" not in kinds, "yeniden adlandirma kaydi hala uretiliyor"
    assert sorted(kinds) == [b"1", b"?"], f"beklenmeyen kayitlar: {kinds}"


def test_a_renamed_destination_outside_scope_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, ops=[{"kind": "rename",
                                   "src": "pipeline/silinecek.py",
                                   "path": "pipeline_private/tasindi.py"}],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


# =====================================================================
# E. FILE TYPES
# =====================================================================

def test_a_symlink_is_refused(tmp_path, gate):
    probe = tmp_path / "sonda-baglanti"
    try:
        os.symlink(str(tmp_path), str(probe), target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    binary = _stub(tmp_path, ops=[{"kind": "symlink",
                                   "path": "pipeline/baglanti.py",
                                   "target": "kurgu.py"}],
                   reply=_reply(changed_files=["pipeline/baglanti.py"]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


def test_a_junction_is_refused(tmp_path, gate):
    if os.name != "nt":
        pytest.skip("kavsak noktasi Windows'a ozgu")
    hedef = tmp_path / "kavsak-hedefi"
    hedef.mkdir()
    binary = _stub(tmp_path, ops=[{"kind": "junction",
                                   "path": "pipeline/kavsak",
                                   "target": str(hedef)}],
                   reply=_reply(changed_files=["pipeline/kavsak"]))
    with pytest.raises(changes.ChangeSetError):
        _run(binary, gate)


def test_a_gitlink_record_is_refused():
    with pytest.raises(changes.UnsafeChange):
        changes._classify(b"1 .M S..U 160000 160000 160000 aaa bbb altmodul",
                          None, Path("."))


def test_a_mode_only_change_is_refused():
    with pytest.raises(changes.UnsafeChange):
        changes._classify(b"1 .M N... 100644 100644 100755 aaa bbb kurgu.py",
                          None, Path("."))


def test_a_special_file_is_invisible_to_git_and_cannot_be_declared(
        tmp_path, gate):
    """MEASURED on Linux, where this actually runs: git does not report
    a FIFO at all, even with `--untracked-files=all`. So a special file
    can never reach the classifier through the inventory -- the earlier
    version of this test asserted a type refusal that the gate was
    never in a position to make, and only the Linux run exposed it.

    Two true things are pinned instead: the declaration for something
    git never saw is refused, and the type gate itself does refuse a
    special file when it is handed one."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("bu platformda FIFO yaratilamiyor")
    binary = _stub(tmp_path, ops=[{"kind": "mkfifo", "path": "pipeline/boru"}],
                   reply=_reply(changed_files=["pipeline/boru"]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)
    fifo = gate.tree / "pipeline" / "boru"
    assert fifo.exists() and not fifo.is_file(), \
        "senaryo kurulmadi: FIFO yaratilmadi"
    assert changes._status(gate.tree) == [], \
        "git FIFO'yu raporluyor; bu testin dayandigi olcum degismis"
    with pytest.raises(changes.UnsafeChange) as refusal:
        changes._plain_file_digest(fifo)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


# =====================================================================
# F. THE CALL BOUNDARY AND `finally`
# =====================================================================

def test_a_failing_call_that_changed_nothing_keeps_its_own_error(
        tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(changed_files=[]), code=7)
    with pytest.raises(execution.ProcessFailed) as refusal:
        _run(binary, gate)
    assert refusal.value.exit_code == 7


def test_a_failing_call_that_edited_an_allowed_file_yields_no_result(
        tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]), code=9)
    with pytest.raises(execution.ProcessFailed):
        _run(binary, gate)
    assert (gate.tree / "pipeline" / "kurgu.py").read_text(
        encoding="utf-8").strip() == SENTINEL, "senaryo kurulmadi"


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


def test_an_edit_to_the_operator_checkout_is_caught(tmp_path, gate):
    """The disposable worktree exists so the main tree is never the
    thing at risk; a model that reached it anyway must not pass."""
    hedef = gate.repo / "pipeline" / "ana-agacta.py"
    binary = _stub(tmp_path, ops=[{"kind": "write", "path": str(hedef),
                                   "text": SENTINEL}],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert hedef.exists(), "senaryo kurulmadi: ana agaca yazilmadi"


def test_a_rewritten_task_manifest_is_caught(tmp_path, gate):
    binary = _stub(tmp_path, ops=[{"kind": "write", "path": str(gate.task),
                                   "text": "{}"}],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange):
        _run(binary, gate)
    assert gate.task.read_text(encoding="utf-8").strip() != "", \
        "senaryo kurulmadi"


def test_a_manifest_outside_the_repository_is_refused(tmp_path, make_gate):
    """CORRECTED: the previous version pinned acceptance as if it were
    the contract. A manifest the repository has no relationship with
    must not be able to grant this run its permissions."""
    current = make_gate(task_outside=True)
    assert current.repo not in current.task.parents, "senaryo kurulmadi"
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.EvidenceUnavailable):
        _run(binary, current)


def test_an_ignored_manifest_inside_the_repository_is_still_re_verified(
        tmp_path, make_gate):
    """The case the main-checkout snapshot CANNOT see: a manifest that
    lives in the repository but is GITIGNORED, so git never lists it
    and the snapshot never hashes it. Here the digest re-check is the
    only layer standing."""
    current = make_gate(gitignore_task=True)
    assert _git(current.repo, "status", "--porcelain") == "", \
        "senaryo kurulmadi: gorev dosyasi git tarafindan hala goruluyor"
    binary = _stub(tmp_path, ops=[{"kind": "write", "path": str(current.task),
                                   "text": "{}"}],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, current)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
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


def test_the_post_check_runs_even_when_the_reply_is_unparsable(
        tmp_path, gate):
    """Not only on the happy path: a schema violation is still an exit,
    and files may already have moved."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py")],
                   reply={"bu": "sema disi"})
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


def test_a_blocked_reply_that_edited_files_yields_no_verified_set(
        tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(status="blocked", next_action="stop",
                                stop_reason="interrupted",
                                changed_files=["pipeline/kurgu.py"]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


def test_a_reply_naming_another_run_is_refused(tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(run_id="baska-kosu-1",
                                          changed_files=[]))
    with pytest.raises(changes.DeclarationMismatch):
        _run(binary, gate)


# =====================================================================
# G. PRIVACY AND NON-MUTATION
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
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    import dataclasses

    verified = _run(binary, gate)
    values = [getattr(verified, field.name)
              for field in dataclasses.fields(verified)]
    carried = " ".join(str(value) for value in values)
    assert SENTINEL not in carried, "sonuc dosya icerigi tasiyor"
    assert not any(isinstance(value, (bytes, bytearray)) for value in values)


def test_an_unreadable_snapshot_file_is_typed_and_silent(tmp_path, gate,
                                                         monkeypatch):
    """A file git listed but nobody can read is an unanswered
    question. It used to escape as a raw `OSError` carrying the
    operating system's own message -- text this module may not
    repeat."""
    # an untracked file the SNAPSHOT reads and nothing else does. The
    # first version injected on the task manifest, which `_bind_task`
    # opens first -- so the test was green on that handler and never
    # reached the snapshot's, which a mutation run then exposed.
    stray = gate.repo / "izlenmeyen-not.txt"
    stray.write_text("K", encoding="utf-8")
    assert "izlenmeyen-not.txt" in _git(gate.repo, "status", "--porcelain"), \
        "senaryo kurulmadi: anlik goruntu bu dosyayi gormuyor"
    real_read = Path.read_bytes
    seen = {"count": 0}

    def unreadable(self, *args, **kwargs):
        if self.name == "izlenmeyen-not.txt":
            seen["count"] += 1
            raise OSError(SENTINEL)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.EvidenceUnavailable) as refusal:
        _run(binary, gate)
    assert seen["count"] > 0, "senaryo kurulmadi: enjeksiyon tetiklenmedi"
    assert SENTINEL not in str(refusal.value) + repr(refusal.value)


def test_no_error_path_carries_file_bytes_or_git_stderr(tmp_path, make_gate,
                                                        monkeypatch):
    """Two channels privacy had not been tested on: the CONTENT of a
    refused file, and git's own stderr. Both travel into whatever a
    report writes if a message ever repeats them."""
    first = make_gate()
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py", SENTINEL)],
                   reply=_reply(changed_files=["pipeline/gizli/a.py"]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, first)
    assert SENTINEL not in str(refusal.value) + repr(refusal.value)
    # a FRESH tree: the first call left the previous one dirty, and a
    # second run there would die at the pristine gate instead of at the
    # injection this half is about
    gate = make_gate()

    real_run = changes.subprocess.run
    seen = {"count": 0}

    def loud_failure(argv, **kwargs):
        if "status" in argv:
            seen["count"] += 1
            if seen["count"] > 1:
                return subprocess.CompletedProcess(
                    argv, 128, b"", SENTINEL.encode("ascii"))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(changes.subprocess, "run", loud_failure)
    clean = _stub(tmp_path, name="temiz", reply=_reply(changed_files=[]))
    with pytest.raises(changes.EvidenceUnavailable) as unavailable:
        _run(clean, gate)
    assert seen["count"] > 1, "senaryo kurulmadi: enjeksiyon tetiklenmedi"
    assert SENTINEL not in str(unavailable.value) + repr(unavailable.value), \
        "ret metni git stderr'ini tasiyor"


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


def test_the_module_never_stages_commits_or_applies_anything():
    """Structural: a write path that appears only on a rare branch
    would pass every functional test in this file."""
    import ast

    source = Path(changes.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # the git SUBCOMMANDS this module can actually reach, read off the
    # call sites -- scanning raw text matched the word "checkout" in a
    # comment about the operator's checkout, which is prose, not a
    # command, and a test that fires on prose stops meaning anything
    subcommands, writers = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = ast.unparse(node.func)
            if called.endswith("_git"):
                subcommands.update(
                    argument.value for argument in node.args[1:]
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str))
            if called.endswith((".write_text", ".write_bytes", ".unlink",
                                ".rename", ".mkdir", ".rmtree", ".chmod")):
                writers.add(called)
            for keyword in node.keywords:
                assert not (keyword.arg == "shell"
                            and getattr(keyword.value, "value", False)), \
                    "kabuk kullanimi"
            assert ast.unparse(node.func) not in ("os.system", "os.popen")
    assert subcommands, "senaryo kurulmadi: hic git cagrisi bulunamadi"
    forbidden = {"add", "commit", "push", "reset", "checkout", "clean",
                 "apply", "stash", "restore", "rm", "mv", "worktree"}
    assert not subcommands & forbidden, \
        f"yazma komutu: {sorted(subcommands & forbidden)}"
    assert writers == set(), f"dosya sistemine yazma: {sorted(writers)}"


def test_the_public_api_offers_no_path_or_scope_override():
    import inspect

    parameters = inspect.signature(
        changes.run_verified_implementation).parameters
    assert set(parameters) == {
        "binary", "repo", "state_dir", "task_path", "manifest_digest",
        "run_id", "worktree_id", "baseline_sha", "prompt", "budget_usd",
        "timeout_seconds", "max_output_bytes", "model"}
    for escape in ("worktree", "cwd", "path", "allowed_paths",
                   "forbidden_paths", "changed_files", "diff", "patch",
                   "status", "workdir"):
        assert escape not in parameters, f"kacis parametresi: {escape}"


def test_the_result_is_frozen_and_slotted(tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    verified = _run(binary, gate)
    with pytest.raises(Exception):
        verified.changed_files = ("kurgu",)


# =====================================================================
# H. PLATFORM
# =====================================================================

def test_scope_matching_follows_the_declared_platform_rule(tmp_path):
    """Case-sensitive on POSIX, Windows's standard fold on Windows --
    the repository's already-declared rule, not a new claim."""
    # the old form ended in `or True`, which made every claim on that
    # line vacuous -- an assertion that cannot fail is documentation
    assert changes._covered("pipeline/a.py", ("pipeline/",))
    assert changes._covered("pipeline/a.py", ("./pipeline//",))
    assert not changes._covered("pipeline_private/a.py", ("pipeline/",))
    assert changes._fold("./a//b/") == changes._fold("a/b")
    assert not changes._covered("pipeline_private/a.py", ("pipeline/",))
    assert changes._same_place("/a/b", "/a/b")
    if os.name == "nt":
        assert changes._same_place("C:/A", "c:/a")
    else:
        assert not changes._same_place("/a/B", "/a/b")
