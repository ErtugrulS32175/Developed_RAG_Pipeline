"""PACKAGE B1 -- preflight, state, lock and disposable worktree.

Every adversarial test here proves its own SETUP was reached before it
claims the refusal. A test that fails for an unrelated reason is red for
the wrong reason, and this session already produced one battery where
the count was right and the cause was not.

NO MODEL IS CALLED anywhere in this file: B1 does not invoke Claude or
Codex, and the binaries it checks for are ordinary files created in a
temporary directory.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools.agent_loop import (contract, locking, preflight, schemas,
                              state, worktree)

BASE_TASK = {
    "protocol_version": contract.PROTOCOL_VERSION,
    "objective": "kurgu hedef",
    "allowed_paths": ["pipeline/"],
    "forbidden_paths": ["contracts/"],
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


RUN = "kurgu-run-1"


def _git(repo, *args, check=True):
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    if check:
        assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture(autouse=True)
def private_worktree_root(tmp_path):
    """Every test gets its OWN runner temp root.

    `runner_temp_root()` derives from the system temp directory, which
    is shared by every process on the machine -- so this battery was
    creating, listing and deleting directories in the same place a real
    agent loop uses. An audit caught the consequence: a foreign holder
    created while the battery ran was swept away by a test.

    Redirecting `tempfile.tempdir` moves the whole root under this
    test's `tmp_path`, and the environment variables carry the same
    redirect into child processes. After this, "delete what is in the
    root" cannot reach anybody else's work, because nobody else's work
    is in this root.

    It uses its OWN `MonkeyPatch`, not the test's. The `monkeypatch`
    fixture is a single shared instance per test, so a test calling
    `monkeypatch.undo()` for its own patch would silently undo this
    redirect as well -- and every later call would resolve a DIFFERENT
    root mid-test. That is not hypothetical; it broke a test the first
    time this fixture landed."""
    private = tmp_path / "runner-temp"
    private.mkdir()
    isolation = pytest.MonkeyPatch()
    isolation.setattr(tempfile, "tempdir", str(private))
    for variable in ("TMPDIR", "TEMP", "TMP"):
        isolation.setenv(variable, str(private))
    root = worktree.runner_temp_root()
    assert root.parent.resolve() == private.resolve(), \
        "gecici kok bu teste ozel degil"
    yield root
    isolation.undo()


@pytest.fixture
def repo(tmp_path):
    """A throwaway repository with private directories that must NOT
    reach the disposable worktree."""
    root = tmp_path / "kurgu-depo"
    (root / "pipeline").mkdir(parents=True)
    for argv in (["init", "-q"], ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(root, *argv)
    (root / "pipeline" / "kurgu.py").write_text("VALUE = 1\n", encoding="utf-8")
    # a stand-in control plane, TRACKED and committed, so the gate that
    # compares it against HEAD has something real to compare
    for control in contract.CONTROL_PLANE_PATHS:
        target = root / control.rstrip("/")
        if control.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            target = target / "contract.py"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("FROZEN = True\n", encoding="utf-8")
    # a SECOND member of the agent-loop test family. The contract names
    # the Phase A file explicitly, so only a file it does NOT name can
    # show that the family PATTERN is what protects the rest.
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_agent_loop_b1.py").write_text(
        "def test_kurgu():\n    assert True\n", encoding="utf-8")
    (root / ".gitignore").write_text("data/\noutput/\nlogs/\nuploads/\n",
                                     encoding="utf-8")
    for private in ("data", "output", "logs", "uploads"):
        directory = root / private
        directory.mkdir()
        (directory / "ozel.txt").write_text("KURGU_OZEL_ICERIK\n",
                                            encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "kurgu")
    return root


@pytest.fixture
def binaries(tmp_path):
    holder = tmp_path / "sahte-bin"
    holder.mkdir()
    made = {}
    for role in ("implementer", "evaluator"):
        # a stub that GENUINELY LAUNCHES on this platform. It used to be
        # a `.py` file, which passed the old gate because that gate
        # accepted any regular file on Windows -- and `.PY` is not in
        # PATHEXT, so the file could never actually have been started.
        # A fixture that cannot do the thing under test makes every
        # assertion around it meaningless.
        if os.name == "nt":
            path = holder / f"sahte_{role}.cmd"
            path.write_text("@echo off\r\necho kurgu\r\n", encoding="ascii")
        else:
            path = holder / f"sahte_{role}.sh"
            path.write_text("#!/bin/sh\necho kurgu\n", encoding="ascii")
            path.chmod(0o755)
        assert subprocess.run([str(path)], capture_output=True,
                              timeout=60).returncode == 0, \
            f"sahte ikili gercekten calismiyor: {path.name}"
        made[role] = path
    return made


@pytest.fixture
def task(repo, tmp_path):
    payload = dict(BASE_TASK, baseline_sha=_git(repo, "rev-parse", "HEAD"))
    path = tmp_path / "kurgu-task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "durum"


def _valid_state(**overrides):
    payload = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "kurgu-run-1",
        "state": contract.State.PREFLIGHT,
        "started_at": "t0", "updated_at": "t0",
        "rounds": {"implementation": 0, "repair": 0, "evaluator": 0},
        "budget": {"max_usd": 1.0, "spent_usd": 0.0},
    }
    payload.update(overrides)
    return payload


def _binding(repo, task_path, **overrides):
    snapshot = preflight.snapshot_manifest(task_path)
    payload = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "kurgu-run-1",
        "repo_id": state.repo_identity(repo),
        "baseline_sha": _git(repo, "rev-parse", "HEAD"),
        "manifest_digest": snapshot.digest,
        "worktree_id": "b" * 32,
    }
    payload.update(overrides)
    return payload


def _pair(repo, task_path, state_dir, **state_overrides):
    """A COMPLETE state directory: the two documents only mean anything
    together, so the fixtures never write one without the other."""
    binding = _binding(repo, task_path)
    state.write_binding(state_dir, binding)
    state.write_state(state_dir, _valid_state(run_id=binding["run_id"],
                                              **state_overrides))
    return binding


# =====================================================================
# PREFLIGHT -- a failure must cost nothing
# =====================================================================

def _worktrees(repo):
    listing = _git(repo, "worktree", "list", "--porcelain", check=False)
    return [l for l in listing.splitlines() if l.startswith("worktree ")]


def _is_registered(repo, path):
    """Git reports forward slashes; `str(Path)` on Windows reports
    backslashes. Comparing them raw made the setup guard in the
    two-repository test always true, which is a guard that guards
    nothing -- normalise both sides."""
    wanted = str(Path(path).resolve()).replace(chr(92), "/").casefold()
    for line in _worktrees(repo):
        entry = str(Path(line.split(" ", 1)[1]).resolve())
        if entry.replace(chr(92), "/").casefold() == wanted:
            return True
    return False


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [("manifest", contract.StopReason.PREFLIGHT_FAILED),
     ("dirty", contract.StopReason.DIRTY_WORKTREE),
     ("staged", contract.StopReason.STAGED_CHANGES),
     ("protected", contract.StopReason.PATH_NOT_ALLOWED),
     ("baseline", contract.StopReason.BASELINE_MISMATCH),
     ("binary", contract.StopReason.PREFLIGHT_FAILED)],
    ids=["gecersiz-manifest", "kirli-agac", "staged", "korunan-yol",
         "baseline", "eksik-ikili"])
def test_a_failing_precondition_creates_no_worktree(repo, task, binaries,
                                                    break_it, expected):
    """Each refusal is asserted TWICE: the stop reason, and the fact that
    the filesystem was never touched. A preflight that returned the right
    verdict after creating a worktree has still failed."""
    before = _worktrees(repo)
    if break_it == "manifest":
        task.write_text("{bu JSON degil", encoding="utf-8")
    elif break_it == "dirty":
        (repo / "pipeline" / "kurgu.py").write_text("VALUE = 2\n",
                                                    encoding="utf-8")
    elif break_it == "staged":
        (repo / "pipeline" / "yeni.py").write_text("x = 1\n", encoding="utf-8")
        _git(repo, "add", "pipeline/yeni.py")
    elif break_it == "protected":
        payload = json.loads(task.read_text(encoding="utf-8"))
        payload["allowed_paths"] = ["tools/"]
        task.write_text(json.dumps(payload), encoding="utf-8")
    elif break_it == "baseline":
        payload = json.loads(task.read_text(encoding="utf-8"))
        payload["baseline_sha"] = "0" * 40
        task.write_text(json.dumps(payload), encoding="utf-8")
    elif break_it == "binary":
        binaries["evaluator"].unlink()

    result = preflight.run_preflight(task, repo=repo, binaries=binaries)

    assert result.ok is False
    assert result.stop_reason == expected
    assert _worktrees(repo) == before, "reddedilen preflight worktree kurdu"


def test_a_healthy_preflight_passes_and_reports_no_stop_reason(repo, task,
                                                               binaries):
    """The other direction: the fixture must be able to SUCCEED, or every
    refusal above proves nothing."""
    result = preflight.run_preflight(task, repo=repo, binaries=binaries)
    assert result.ok is True
    assert result.stop_reason is None
    assert result.baseline_sha == _git(repo, "rev-parse", "HEAD")
    assert result.manifest is not None


def test_a_preflight_failure_never_becomes_a_successful_state(repo, task,
                                                              binaries,
                                                              state_dir):
    """A refusal writes no state at all -- not a failed one, not a
    partial one. An accepted state file is a run that exists."""
    task.write_text("{bozuk", encoding="utf-8")
    result = preflight.run_preflight(task, repo=repo, binaries=binaries,
                                     state_dir=state_dir)
    assert result.ok is False
    assert not (state_dir / state.STATE_FILENAME).exists()
    assert not (state_dir / state.BINDING_FILENAME).exists()


def test_a_preflight_report_carries_no_path_and_no_manifest_content(
        repo, task, binaries):
    """A stop reason is a vocabulary word and a short fixed sentence.
    An absolute path or a copied exception message is how a private
    location reaches a report that gets forwarded."""
    payload = json.loads(task.read_text(encoding="utf-8"))
    payload["objective"] = "GIZLI_GOREV_METNI"
    task.write_text(json.dumps(payload) + "   ", encoding="utf-8")
    payload["allowed_paths"] = ["tools/"]
    task.write_text(json.dumps(payload), encoding="utf-8")
    result = preflight.run_preflight(task, repo=repo, binaries=binaries)
    rendered = f"{result.stop_reason} {result.detail}"
    assert "GIZLI_GOREV_METNI" not in rendered
    assert str(repo) not in rendered
    assert str(task) not in rendered
    assert result.stop_reason in contract.ALL_STOP_REASONS


# =====================================================================
# STATE
# =====================================================================

def test_state_survives_an_atomic_round_trip(state_dir):
    state.write_state(state_dir, _valid_state())
    assert state.read_state(state_dir)["state"] == contract.State.PREFLIGHT


def test_a_truncated_state_file_is_refused_not_repaired(state_dir):
    state.write_state(state_dir, _valid_state())
    (state_dir / state.STATE_FILENAME).write_text('{"protocol_versio',
                                                  encoding="utf-8")
    with pytest.raises(state.CorruptState):
        state.read_state(state_dir)


def test_an_unknown_field_fails_closed(state_dir):
    with pytest.raises(state.CorruptState):
        state.write_state(state_dir, _valid_state(kurgu_alan="x"))


def test_a_counter_beyond_the_contract_bound_is_refused(state_dir):
    with pytest.raises(state.CorruptState):
        state.write_state(state_dir, _valid_state(
            rounds={"implementation": 999, "repair": 999, "evaluator": 999}))


@pytest.mark.parametrize(
    ("current", "target"),
    [(contract.State.PREFLIGHT, contract.State.AUDITING),
     (contract.State.FINAL_AUDITING, contract.State.REPAIRING),
     (contract.State.APPROVED, contract.State.IMPLEMENTING)],
    ids=["atlama", "ucuncu-yama", "terminalden-cikis"])
def test_an_illegal_transition_is_refused(current, target):
    with pytest.raises(state.IllegalTransition):
        state.assert_transition(current, target)


def test_a_terminal_state_cannot_resume(state_dir):
    state.write_state(state_dir, _valid_state(
        state=contract.State.APPROVED, stop_reason="completed"))
    with pytest.raises(state.IllegalTransition):
        state.advance(state_dir, contract.State.IMPLEMENTING)


def test_a_failure_during_replacement_preserves_the_last_valid_state(
        state_dir, monkeypatch):
    """The injected failure is asserted to have been REACHED -- otherwise
    a green here would only mean the write happened to succeed."""
    state.write_state(state_dir, _valid_state())
    reached = []

    def exploding_replace(source, target):
        reached.append(target)
        raise OSError("kurgu disk hatasi")

    monkeypatch.setattr(state.os, "replace", exploding_replace)
    with pytest.raises(OSError):
        state.write_state(state_dir, _valid_state(state=contract.State.AUDITING))
    assert reached, "enjekte edilen hataya hic ulasilmadi"
    monkeypatch.undo()
    assert state.read_state(state_dir)["state"] == contract.State.PREFLIGHT
    leftovers = [p.name for p in state_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"gecici dosya kaldi: {leftovers}"


@pytest.mark.parametrize("wrong", ["repo_id", "baseline_sha",
                                   "manifest_digest"])
def test_state_from_another_run_is_refused(repo, task, state_dir, wrong):
    """Three ways a state directory can belong elsewhere -- a copied
    checkout, a moved baseline, an edited task -- and each is told
    apart."""
    binding = _binding(repo, task)
    state.write_binding(state_dir, binding)
    expected = dict(binding)
    expected[wrong] = ("f" * 32 if wrong == "repo_id"
                       else "f" * (40 if wrong == "baseline_sha" else 64))
    with pytest.raises(state.IncompatibleState):
        state.assert_binding(state_dir, repo_id=expected["repo_id"],
                             baseline_sha=expected["baseline_sha"],
                             manifest_digest=expected["manifest_digest"])


def test_the_binding_records_identity_and_never_a_path(repo, task, state_dir):
    state.write_binding(state_dir, _binding(repo, task))
    raw = (state_dir / state.BINDING_FILENAME).read_text(encoding="utf-8")
    assert str(repo) not in raw and str(task) not in raw
    assert state.repo_identity(repo) in raw


# =====================================================================
# LOCK
# =====================================================================

def test_the_first_owner_takes_the_lock_and_a_second_is_refused(state_dir):
    with locking.single_instance_lock(state_dir):
        with pytest.raises(locking.LockHeld):
            with locking.single_instance_lock(state_dir):
                pass


def test_no_module_path_can_delete_a_lock_file(state_dir):
    """THE fix for the audited race, stated as an absence.

    The old release read the owner record, compared the token, and then
    unlinked the path -- three steps with the lock changing hands in
    between, so a departing owner could delete the INCOMING owner's
    lock and two runners would proceed. Narrowing that window would
    leave the same shape with a smaller target, so the deletion is gone
    instead: ownership is an open handle the kernel arbitrates, and no
    code path in this module removes the file at all.

    Checked structurally rather than by trying to provoke the race,
    because a race test that happens to pass proves nothing about the
    interleaving it did not hit."""
    import ast

    source = (Path(locking.__file__)).read_text(encoding="utf-8")
    removals = [ast.unparse(node.func)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and ast.unparse(node.func).split(".")[-1]
                in ("unlink", "remove", "rmtree", "rmdir")]
    assert removals == [], f"kilit dosyasini silen yol var: {removals}"


def test_the_lock_survives_a_release_and_is_reusable(state_dir):
    lock = locking.acquire(state_dir)
    assert locking.is_held(state_dir) is True
    locking.release(lock)
    # the FILE is still there and that is correct: existence is not
    # ownership any more, so a leftover file cannot wedge the next run
    assert locking.lock_path(state_dir).exists()
    assert locking.is_held(state_dir) is False
    again = locking.acquire(state_dir)
    assert again.token != lock.token
    locking.release(again)


def test_releasing_twice_is_refused(state_dir):
    """A second release is a caller that thinks it still owns something
    it gave back. Under the old design that call DELETED whatever lock
    was there by then."""
    lock = locking.acquire(state_dir)
    locking.release(lock)
    with pytest.raises(locking.LockNotOwned):
        locking.release(lock)


def test_a_stranger_cannot_release_someone_elses_lock(state_dir):
    """Ownership is the handle, so there is no token to guess: a
    fabricated lock object does not name the held descriptor."""
    lock = locking.acquire(state_dir)
    try:
        with pytest.raises(locking.LockNotOwned):
            locking.release("kurgu-baska-jeton")
        assert locking.is_held(state_dir) is True
    finally:
        locking.release(lock)


def test_a_leftover_lock_file_from_a_crash_does_not_block_the_next_run(
        state_dir):
    """A crashed runner's handle is closed by the operating system, so
    the lock is gone even though the file is not. This replaces the
    stale-lock rule, which had to guess from a pid and answered "assume
    alive" whenever it could not tell -- and carried the same
    read-then-delete race as the release path.

    Refusing to RESUME a crashed run is a separate question, answered
    from the state directory where there is actual evidence."""
    locking.lock_path(state_dir).parent.mkdir(parents=True, exist_ok=True)
    locking.lock_path(state_dir).write_text(
        json.dumps({"token": "eski", "pid": 999999, "created_at": 0.0}),
        encoding="utf-8")
    assert locking.is_held(state_dir) is False
    lock = locking.acquire(state_dir)
    locking.release(lock)


def test_an_unreadable_lock_record_does_not_confuse_ownership(state_dir):
    lock = locking.acquire(state_dir)
    try:
        report = locking.inspect(state_dir)
        assert report["held"] is True and report["readable"] is True
    finally:
        locking.release(lock)
    locking.lock_path(state_dir).write_text("bu JSON degil", encoding="utf-8")
    report = locking.inspect(state_dir)
    assert report["readable"] is False
    assert report["held"] is False, "okunamayan kayit sahiplik degildir"


@pytest.mark.parametrize(
    "record",
    [{"token": "abc", "pid": 7, "created_at": "dun"},
     {"token": "abc", "pid": "yedi", "created_at": 0.0},
     {"token": "abc", "pid": 7},
     {"token": "abc", "pid": True, "created_at": 0.0},
     {"token": "abc", "pid": 7, "created_at": float("inf")},
     ["bu", "bir", "liste"]],
    ids=["created_at-metin", "pid-metin", "created_at-yok", "pid-bool",
         "created_at-sonsuz", "nesne-degil"])
def test_a_well_formed_but_wrong_typed_lock_record_does_not_crash_inspect(
        state_dir, record):
    """Valid JSON is not a valid record. `created_at` as a string parsed
    fine and then killed `inspect` on an unguarded `float()` -- and a
    diagnostic path that raises is worse than useless, because it is
    reached exactly when something is already wrong.

    Fail-closed and STRUCTURED: unreadable, carrying no values, rather
    than half a record."""
    locking.lock_path(state_dir).parent.mkdir(parents=True, exist_ok=True)
    locking.lock_path(state_dir).write_text(json.dumps(record),
                                            encoding="utf-8")
    report = locking.inspect(state_dir)
    assert report["readable"] is False
    assert report["pid"] is None and report["age_seconds"] is None
    assert report["held"] is False


def test_a_cleanup_failure_does_not_replace_the_primary_error(state_dir):
    """The body's exception is what the caller needs to see; a release
    that then fails must not overwrite it."""
    with pytest.raises(ValueError, match="kurgu birincil"):
        with locking.single_instance_lock(state_dir) as lock:
            locking.release(lock)          # break the context's release
            raise ValueError("kurgu birincil hata")


# =====================================================================
# DISPOSABLE WORKTREE
# =====================================================================

def test_the_worktree_is_the_exact_baseline_and_leaves_the_main_tree_alone(
        repo, state_dir):
    baseline = _git(repo, "rev-parse", "HEAD")
    before = sorted((p.relative_to(repo).as_posix(), p.stat().st_size)
                    for p in repo.rglob("*") if p.is_file()
                    and ".git" not in p.parts)
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    try:
        assert _git(path, "rev-parse", "HEAD") == baseline
        after = sorted((p.relative_to(repo).as_posix(), p.stat().st_size)
                       for p in repo.rglob("*") if p.is_file()
                       and ".git" not in p.parts)
        assert after == before, "ana calisma agaci degisti"
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


@pytest.mark.parametrize("private", ["data", "output", "logs", "uploads"])
def test_private_directories_never_reach_the_worktree(repo, state_dir,
                                                      private):
    """`git worktree add` materialises TRACKED content only, and these
    are gitignored -- so the document tree is absent by construction
    rather than by a filter somebody has to maintain."""
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    try:
        assert (repo / private / "ozel.txt").exists(), "senaryo kurulmadi"
        assert not (path / private).exists(), f"{private}/ worktree'ye sizdi"
        leaked = [p for p in path.rglob("*") if p.is_file()
                  and "KURGU_OZEL_ICERIK" in p.read_text(encoding="utf-8",
                                                         errors="ignore")]
        assert not leaked
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


def test_the_worktree_is_created_under_the_runner_owned_temp_root(repo,
                                                                  state_dir):
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    try:
        assert worktree.runner_temp_root() in path.resolve().parents
        assert repo.resolve() not in path.resolve().parents
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


def test_the_worktree_carries_no_file_of_ours(repo, state_dir):
    """The tree the implementer works in is the baseline and nothing
    else. An earlier design wrote an ownership marker INSIDE it, which
    put an untracked file of ours into the diff the implementer's work
    would later be read out of."""
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    try:
        assert _git(path, "status", "--porcelain") == "",             "worktree bizim dosyamizla kirli basliyor"
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


# ---------------------------------------------------------------------
# CLEANUP AUTHORITY. Third design; the first two were checks layered
# over a path the caller supplied, and each was broken by a different
# argument. There is no path parameter now.
# ---------------------------------------------------------------------

def test_remove_accepts_no_path_at_all(repo):
    """`remove` derives its target from an id, so a caller cannot point
    it at an arbitrary directory. NECESSARY BUT NOT SUFFICIENT: the id
    names a directory, and directories are shared by every repository on
    the machine -- which is exactly how an audit made one repository
    delete another's worktree. Authority is the record; see below."""
    import inspect as inspect_module

    assert set(inspect_module.signature(worktree.remove).parameters) == {
        "repo", "state_dir", "worktree_id"}
    assert set(inspect_module.signature(worktree.create).parameters) == {
        "repo", "state_dir", "run_id", "baseline_sha"}


def test_one_repository_cannot_remove_another_repositorys_worktree(
        repo, tmp_path, state_dir):
    """THE critical finding, and the reason ownership stopped being a
    name. The id was minted by repository B; repository A asked to
    remove it. A's git removal failed because A had never registered the
    tree, this module ignored that failure, deleted B's directory
    anyway, and left stale entries behind in B's registry."""
    other = tmp_path / "ikinci-depo"
    (other / "pipeline").mkdir(parents=True)
    for argv in (["init", "-q"], ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(other, *argv)
    (other / "pipeline" / "k.py").write_text("V = 1\n", encoding="utf-8")
    _git(other, "add", "-A")
    _git(other, "commit", "-qm", "kurgu")

    path_b, id_b = worktree.create(other, state_dir=state_dir, run_id=RUN,
                                   baseline_sha=_git(other, "rev-parse",
                                                     "HEAD"))
    (path_b / "ikinci_deponun_isi.txt").write_text("kurgu", encoding="utf-8")
    try:
        assert _is_registered(other, path_b), "senaryo kurulmadi"
        assert not _is_registered(repo, path_b),             "senaryo kurulmadi: ilk depo bu agaci tanimamali"

        with pytest.raises(worktree.WorktreeError):
            worktree.remove(repo, state_dir=state_dir, worktree_id=id_b)

        assert (path_b / "ikinci_deponun_isi.txt").exists(),             "baska bir deponun calisma agaci silindi"
        assert _is_registered(other, path_b), "ikinci deponun kaydi bozuldu"
    finally:
        worktree.remove(other, state_dir=state_dir, worktree_id=id_b)


def test_a_record_from_another_state_directory_is_not_authority(repo,
                                                                state_dir,
                                                                tmp_path):
    """No record, no removal. An id on its own says nothing."""
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=_git(repo, "rev-parse",
                                                          "HEAD"))
    try:
        elsewhere = tmp_path / "baska-durum"
        elsewhere.mkdir()
        with pytest.raises(worktree.WorktreeError):
            worktree.remove(repo, state_dir=elsewhere, worktree_id=worktree_id)
        assert path.exists()
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


def test_a_failed_git_removal_is_not_stepped_over(repo, state_dir,
                                                  monkeypatch):
    """`git worktree remove` failing used to be ignored, and an
    unconditional recursive delete ran anyway. The tree stays put.

    Patched at the SUBPROCESS boundary, not at `_git`. Replacing `_git`
    made the fake raise on its own, so `check=True` was never what
    refused -- a mutation flipping it to `check=False` left this test
    green. Now git really returns non-zero and the `check` argument is
    the only thing that can turn that into a refusal."""
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=_git(repo, "rev-parse",
                                                          "HEAD"))
    real_run = worktree.subprocess.run

    class _Shim:
        @staticmethod
        def run(argv, **kwargs):
            if len(argv) > 4 and argv[3:5] == ["worktree", "remove"]:
                return subprocess.CompletedProcess(args=argv, returncode=128,
                                                   stdout="", stderr="")
            return real_run(argv, **kwargs)

    monkeypatch.setattr(worktree, "subprocess", _Shim)
    with pytest.raises(worktree.WorktreeError):
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    monkeypatch.undo()
    assert path.exists(), "git reddetti ama dizin yine de silindi"
    assert worktree.read_record(state_dir, worktree_id) is not None, \
        "git reddetti ama kayit silindi"
    worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


def test_an_unanswerable_git_query_deletes_nothing(repo, state_dir,
                                                   monkeypatch):
    """A question we could not ask has no answer, and no answer is not
    `False`.

    `git worktree list` ran with `check=False`, so a failure returned an
    empty listing -- and empty read as "not registered in this
    repository", which is the branch that goes on to delete. An
    unverifiable git state became a verified absence."""
    record = worktree.register(state_dir, repo=repo, run_id=RUN,
                               baseline_sha=_git(repo, "rev-parse", "HEAD"))
    holder = worktree.holder_for(record["worktree_id"])
    holder.mkdir()
    (holder / "onemli.txt").write_text("kurgu", encoding="utf-8")
    real_run = worktree.subprocess.run

    class _Shim:
        @staticmethod
        def run(argv, **kwargs):
            if len(argv) > 4 and argv[3:5] == ["worktree", "list"]:
                return subprocess.CompletedProcess(args=argv, returncode=128,
                                                   stdout="", stderr="")
            return real_run(argv, **kwargs)

    monkeypatch.setattr(worktree, "subprocess", _Shim)
    with pytest.raises(worktree.WorktreeError):
        worktree.remove(repo, state_dir=state_dir,
                        worktree_id=record["worktree_id"])
    monkeypatch.undo()
    assert (holder / "onemli.txt").exists(), "dogrulanamayan durumda silindi"
    assert worktree.read_record(state_dir, record["worktree_id"]) is not None
    worktree.remove(repo, state_dir=state_dir,
                    worktree_id=record["worktree_id"])


def test_the_record_outlives_a_holder_that_could_not_be_removed(repo,
                                                                state_dir,
                                                                monkeypatch):
    """The ledger entry is the only thing that makes residue findable,
    so it goes last and only once the directory is provably gone.

    `ignore_errors=True` plus an unconditional record deletion was a way
    to LOSE a directory: on Windows an open handle makes removal fail
    silently, the holder stayed, and the record naming it was dropped --
    after which `find_orphans` could never see it again."""
    record = worktree.register(state_dir, repo=repo, run_id=RUN,
                               baseline_sha=_git(repo, "rev-parse", "HEAD"))
    worktree_id = record["worktree_id"]
    holder = worktree.holder_for(worktree_id)
    holder.mkdir()
    (holder / "kalici.txt").write_text("kurgu", encoding="utf-8")

    class _ShutilShim:
        @staticmethod
        def rmtree(path, **kwargs):
            return None                       # silently does nothing

    monkeypatch.setattr(worktree, "shutil", _ShutilShim)
    with pytest.raises(worktree.WorktreeError):
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    monkeypatch.undo()

    assert holder.exists(), "senaryo kurulmadi: holder duruyor olmali"
    assert worktree.read_record(state_dir, worktree_id) is not None, \
        "silinemeyen dizinin kaydi dusuruldu"
    assert [r["worktree_id"] for r in worktree.find_orphans(
        repo, state_dir=state_dir)] == [worktree_id], \
        "kayit dustugu icin yetim artik bulunamiyor"
    worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    assert not holder.exists()


def test_a_brand_new_directory_chain_is_made_durable_link_by_link(tmp_path,
                                                                  monkeypatch):
    """The FIRST record creates `worktrees/` on the way in, and flushing
    only the directory the file lands in leaves that new directory's own
    entry -- in its parent -- unflushed."""
    flushed = []
    real = state.fsync_directory
    monkeypatch.setattr(state, "fsync_directory",
                        lambda d: (flushed.append(Path(d).resolve()),
                                   real(d))[1])
    deep = tmp_path / "yepyeni" / "durum" / worktree.REGISTRY_DIRNAME
    assert not deep.exists(), "senaryo kurulmadi"
    state.write_json_atomically(deep / "k.json", _valid_state(),
                                schemas.STATE_SCHEMA, "durum")

    for level in (tmp_path / "yepyeni", tmp_path / "yepyeni" / "durum", deep):
        assert level.parent.resolve() in flushed, \
            f"yeni dizinin ust girdisi flush edilmedi: {level.name}"
    assert deep.resolve() in flushed, "hedef dizin flush edilmedi"


def test_the_repository_check_refuses_even_when_git_would_agree(repo,
                                                                state_dir):
    """The repository check, ISOLATED.

    In the two-repository test git does not recognise the tree either,
    so the `_registered_here` guard refuses first and the repository
    check never runs -- a mutation deleting it changed nothing. Here git
    DOES list the worktree as this repository's, and only the record's
    `repo_id` disagrees, so nothing else can say no."""
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=_git(repo, "rev-parse",
                                                          "HEAD"))
    record_file = (state_dir / worktree.REGISTRY_DIRNAME
                   / f"{worktree_id}.json")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    assert record["repo_id"] == state.repo_identity(repo), "senaryo kurulmadi"
    record["repo_id"] = "c" * 32
    record_file.write_text(json.dumps(record), encoding="utf-8")
    assert _is_registered(repo, path), \
        "senaryo kurulmadi: git bu agaci bu depoya ait gostermeli"

    with pytest.raises(worktree.WorktreeError):
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    assert path.exists(), "kayit baska depoya aitken silindi"

    record["repo_id"] = state.repo_identity(repo)
    record_file.write_text(json.dumps(record), encoding="utf-8")
    worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


@pytest.mark.parametrize(
    "hostile",
    ["../../../../etc", "..", "0" * 31, "0" * 33, "G" * 32, "", "0" * 16,
     "../" + "a" * 29])
def test_an_id_that_could_name_another_directory_is_refused(repo, state_dir,
                                                            hostile):
    """Containment stops being a check and becomes a property of the
    vocabulary: an id is 32 hex characters, so no id denotes a path
    outside the runner-owned root. `..` is not an id."""
    with pytest.raises(worktree.WorktreeError):
        worktree.holder_for(hostile)
    with pytest.raises(worktree.WorktreeError):
        worktree.remove(repo, state_dir=state_dir, worktree_id=hostile)


def test_a_directory_under_the_temp_root_that_is_not_ours_is_untouchable(
        repo, state_dir):
    """Another run's holder, with real content, next to ours."""
    other = worktree.runner_temp_root() / f"{worktree.TEMP_PREFIX}{'a' * 32}"
    other.mkdir(parents=True, exist_ok=True)
    (other / "baskasinin_isi.txt").write_text("kurgu", encoding="utf-8")
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    try:
        assert worktree.owns(other, worktree_id=worktree_id) is False
        assert worktree.owns(path, worktree_id=worktree_id) is True
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
        assert (other / "baskasinin_isi.txt").exists(),             "baska bir kosunun dizini silindi"
        assert not path.exists()
    finally:
        shutil.rmtree(other, ignore_errors=True)


def test_a_holder_that_appears_mid_test_is_never_swept(repo, state_dir,
                                                       private_worktree_root,
                                                       monkeypatch):
    """A foreign holder created DURING a test must survive it.

    The battery used to run against the shared system-wide root and one
    test removed "everything that was not here when I started" -- so a
    real agent loop that happened to start in another repository while
    this suite ran could have its holder deleted. An audit measured
    exactly that, and a race probe reproduced it.

    Two things stop it now, and this asserts both: the root is private
    to this test, and cleanup names the holders its OWN ledger records
    rather than sweeping whatever is new."""
    def refusing_remove(*args, **kwargs):
        raise worktree.WorktreeError("kurgu temizlik hatasi")

    intruder = private_worktree_root / f"{worktree.TEMP_PREFIX}{'d' * 32}"

    with monkeypatch.context() as sabotage:
        sabotage.setattr(worktree, "remove", refusing_remove)
        with pytest.raises(worktree.WorktreeError):
            worktree.create(repo, state_dir=state_dir, run_id=RUN,
                            baseline_sha="0" * 40)
        # another repository's loop starts right here
        intruder.mkdir()
        (intruder / "baska_deponun_isi.txt").write_text("ONEMLI",
                                                        encoding="utf-8")

    ours = worktree.find_orphans(repo, state_dir=state_dir)
    assert len(ours) == 1, "senaryo kurulmadi: kendi yetimimiz olmali"
    for record in ours:
        worktree.remove(repo, state_dir=state_dir,
                        worktree_id=record["worktree_id"])

    assert (intruder / "baska_deponun_isi.txt").exists(), \
        "baska bir kosunun dizini temizlik sirasinda silindi"
    assert worktree.find_orphans(repo, state_dir=state_dir) == []
    shutil.rmtree(intruder, ignore_errors=True)


def test_an_existing_holder_is_never_reused(repo, state_dir, monkeypatch):
    """`mkdir` is EXCLUSIVE, and that has to be tested by forcing the
    collision: at 128 bits it never happens by chance, so a mutation
    turning it into `exist_ok=True` broke no test at all.

    Reuse would mean adopting a directory whose contents were never
    verified -- and, worse, the run that already owns that id would have
    its worktree deleted out from under it by the newcomer's cleanup."""
    collision = "e" * 32
    monkeypatch.setattr(worktree.secrets, "token_hex", lambda n: collision)
    squatter = worktree.holder_for(collision)
    squatter.mkdir(parents=True, exist_ok=True)
    (squatter / "onceki_kosunun_isi.txt").write_text("kurgu", encoding="utf-8")
    try:
        with pytest.raises(worktree.WorktreeError):
            worktree.create(repo, state_dir=state_dir, run_id=RUN,
                            baseline_sha=_git(repo, "rev-parse", "HEAD"))
        assert (squatter / "onceki_kosunun_isi.txt").exists(), \
            "var olan kap yeniden kullanildi ve icerigi kayboldu"
    finally:
        shutil.rmtree(squatter, ignore_errors=True)


def test_ownership_needs_no_file_to_be_readable(repo):
    """The failure the audit injected: the marker write raised, the
    holder survived unmarked, and the ownership-checked `remove` then
    refused to clean up a directory this module had just created.

    There is no marker to fail to write. `mkdir` is atomic, so the
    directory and its ownership come into existence together -- proven
    here by making EVERY file write fail and showing creation still
    leaves nothing behind."""
    root = worktree.runner_temp_root()
    before = {p.name for p in root.iterdir()}
    real_write = Path.write_text

    def refusing_write(self, *args, **kwargs):
        raise OSError("kurgu: hicbir dosya yazilamiyor")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "write_text", refusing_write)
    try:
        with pytest.raises(BaseException):
            worktree.create(repo, state_dir=state_dir, run_id=RUN,
                        baseline_sha="0" * 40)
    finally:
        monkeypatch.undo()
    assert real_write is Path.write_text, "yama geri alinmadi"
    assert {p.name for p in root.iterdir()} == before,         "isaretsiz kap geride kaldi"


def test_a_crashed_run_can_find_its_own_worktree_again(repo, state_dir):
    """Recovery WITHOUT knowing the id.

    The previous design minted the id inside `create` and returned it
    only on success, and recovery required it as an argument -- so a
    process that died between `mkdir` and its first write left an orphan
    whose id existed nowhere but the dead process's memory. The record
    is written BEFORE anything is created, so the registry is the thing
    that survives."""
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    del worktree_id                       # the crash: nobody recorded it
    try:
        found = worktree.find_orphans(repo, state_dir=state_dir)
        assert [r["worktree_id"] for r in found] == [path.parent.name.replace(
            worktree.TEMP_PREFIX, "")]
        assert found[0]["run_id"] == RUN
        assert worktree.find_orphans(repo, state_dir=state_dir,
                                     run_id="baska-kosu") == []
    finally:
        for record in worktree.find_orphans(repo, state_dir=state_dir):
            worktree.remove(repo, state_dir=state_dir,
                            worktree_id=record["worktree_id"])
    assert worktree.find_orphans(repo, state_dir=state_dir) == []


def test_recovery_does_not_report_another_repositorys_worktrees(repo, tmp_path,
                                                                state_dir):
    """The registry is shared; the answer must not be."""
    other = tmp_path / "ucuncu-depo"
    (other / "pipeline").mkdir(parents=True)
    for argv in (["init", "-q"], ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(other, *argv)
    (other / "pipeline" / "k.py").write_text("V = 1\n", encoding="utf-8")
    _git(other, "add", "-A")
    _git(other, "commit", "-qm", "kurgu")
    _, id_other = worktree.create(other, state_dir=state_dir, run_id=RUN,
                                  baseline_sha=_git(other, "rev-parse", "HEAD"))
    try:
        assert worktree.find_orphans(repo, state_dir=state_dir) == []
        mine = worktree.find_orphans(other, state_dir=state_dir)
        assert [r["worktree_id"] for r in mine] == [id_other]
    finally:
        worktree.remove(other, state_dir=state_dir, worktree_id=id_other)


def _crash_at(stage, repo, state_dir, private_temp):
    """Start the helper, wait for it to reach `stage`, then KILL it.

    `kill()` is SIGKILL on POSIX and TerminateProcess on Windows: no
    handler, no `finally`, no cleanup. That is the point -- an exception
    inside `create` is caught and tidied up, so it tests the opposite of
    what a crash does."""
    helper = Path(__file__).resolve().parent / "crash_helper_b1.py"
    victim = subprocess.Popen(
        [sys.executable, str(helper), stage, str(repo), str(state_dir),
         _git(repo, "rev-parse", "HEAD"), str(Path(private_temp).parent)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = victim.stdout.readline()
        assert line.startswith("READY "), \
            f"kurban asamaya ulasmadi ({stage}): {line!r} {victim.stderr.read()}"
        return line.split()[1]
    finally:
        victim.kill()
        victim.wait(timeout=30)


@pytest.mark.parametrize(
    "stage", ["record_only", "holder_made", "git_added", "ready"])
def test_a_killed_process_always_leaves_recoverable_residue(
        repo, state_dir, private_worktree_root, stage):
    """The four windows a crash can land in, each with a real killed
    process rather than a raised exception.

    In every one the ledger is what survives and what makes the residue
    findable: recovery never needs the id, because the id was persisted
    before the first directory existed."""
    worktree_id = _crash_at(stage, repo, state_dir,
                            private_worktree_root)

    orphans = worktree.find_orphans(repo, state_dir=state_dir)
    assert [r["worktree_id"] for r in orphans] == [worktree_id], \
        f"{stage}: kayit kurtarilamadi"
    assert orphans[0]["status"] == (worktree.STATUS_READY if stage == "ready"
                                    else worktree.STATUS_PLANNED)

    holder = worktree.holder_for(worktree_id)
    assert holder.exists() is (stage != "record_only")
    assert _is_registered(repo, holder / worktree.WORKTREE_DIRNAME) is (
        stage in ("git_added", "ready"))

    worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    assert worktree.find_orphans(repo, state_dir=state_dir) == []
    assert not holder.exists()
    assert not _is_registered(repo, holder / worktree.WORKTREE_DIRNAME)
    assert _worktrees(repo) == [f"worktree {repo.as_posix()}"] or len(
        _worktrees(repo)) == 1, "git kaydinda bayat girdi kaldi"


def test_the_atomic_write_reports_its_real_durability(state_dir):
    """`fsync` on the file is not `fsync` on the directory, and the two
    guarantee different things. Whatever this platform can actually do
    is what gets reported -- an unsupported directory flush is not
    swallowed and then described as durable."""
    state_dir.mkdir(parents=True, exist_ok=True)
    level = state.durability_of(state_dir)
    assert level in ("power-loss", "process-crash")
    if os.name == "nt":
        assert level == "process-crash", \
            "Windows'ta guc-kaybi dayanikliligi iddia edilemez"
    else:
        assert level == "power-loss"
    assert state.fsync_directory(state_dir) is (level == "power-loss")


def test_the_record_is_written_before_anything_exists_on_disk(repo, state_dir):
    """Write-ahead, asserted directly: after `register` there is a record
    and NO directory."""
    record = worktree.register(state_dir, repo=repo, run_id=RUN,
                               baseline_sha=_git(repo, "rev-parse", "HEAD"))
    holder = worktree.holder_for(record["worktree_id"])
    assert not holder.exists(), "kayittan once dizin olusmus"
    assert worktree.read_record(state_dir, record["worktree_id"]) == record
    assert record["repo_id"] == state.repo_identity(repo)
    assert str(repo) not in json.dumps(record), "kayit mutlak yol tasiyor"
    assert worktree.find_orphans(repo, state_dir=state_dir) == [record]
    worktree.remove(repo, state_dir=state_dir,
                    worktree_id=record["worktree_id"])


def test_a_creation_failure_leaves_no_worktree_no_state_and_no_residue(
        repo, state_dir):
    """A baseline git cannot check out: nothing accepted, nothing left.

    "Nothing left" includes the EMPTY HOLDER. The holder is a real
    directory before git is asked for anything, and the first version
    returned the error without removing it -- ten empty directories
    accumulated in the runner-owned root during a single session, which
    is both litter and a broken invariant: if a failed creation can
    leave a directory, "no residue" is no longer something a test can
    check."""
    root = worktree.runner_temp_root()
    before = sorted(p.name for p in root.iterdir())

    with pytest.raises(worktree.WorktreeError):
        worktree.create(repo, state_dir=state_dir, run_id=RUN,
                        baseline_sha="0" * 40)

    assert not (state_dir / state.STATE_FILENAME).exists()
    assert sorted(p.name for p in root.iterdir()) == before,         "basarisiz kurulum gecici kokta dizin birakti"


def test_a_successful_create_and_remove_also_leaves_no_residue(repo,
                                                               state_dir):
    """The residue invariant on the ORDINARY path, not just the failing
    one. Only the failure path was asserted, so a holder surviving a
    normal cleanup would not have been noticed -- and two empty holders
    did survive a session, which is what prompted looking."""
    root = worktree.runner_temp_root()
    before = sorted(p.name for p in root.iterdir())
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    assert sorted(p.name for p in root.iterdir()) != before,         "senaryo kurulmadi: kurulum gecici kokte iz birakmali"
    worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    assert sorted(p.name for p in root.iterdir()) == before
    assert worktree.find_orphans(repo, state_dir=state_dir) == []


def test_a_cleanup_failure_does_not_replace_the_creation_failure(
        repo, state_dir, monkeypatch):
    """`create` cleans up on the way out, and that cleanup used to be
    able to raise over the top of the real error -- leaving the caller
    holding a complaint about removal instead of the reason the
    worktree was refused."""
    def refusing_remove(*args, **kwargs):
        raise worktree.WorktreeError("kurgu temizlik hatasi")

    root = worktree.runner_temp_root()
    before = {p.name for p in root.iterdir()}
    monkeypatch.setattr(worktree, "remove", refusing_remove)
    try:
        with pytest.raises(worktree.WorktreeError) as refusal:
            worktree.create(repo, state_dir=state_dir, run_id=RUN,
                        baseline_sha="0" * 40)
        assert "temizlik" not in str(refusal.value), str(refusal.value)
    finally:
        # this test SABOTAGES cleanup, so it owns the litter it caused.
        # It removes exactly the holders ITS OWN ledger records name --
        # never "everything that appeared since I started", which is how
        # a foreign holder created by another repository during the test
        # window was destroyed.
        monkeypatch.undo()
        for record in worktree.find_orphans(repo, state_dir=state_dir):
            worktree.remove(repo, state_dir=state_dir,
                            worktree_id=record["worktree_id"])


def test_the_worktree_is_created_without_a_shell_or_a_prompt():
    """argv lists only, and git's interactive prompts disabled -- an
    unattended run must never park waiting for credentials."""
    env = worktree._no_prompt_env()             # noqa: SLF001 -- contract
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "never"


# =====================================================================
# MANIFEST IMMUTABILITY
# =====================================================================

def test_the_digest_and_the_parse_come_from_one_read(task):
    snapshot = preflight.snapshot_manifest(task)
    import hashlib

    assert snapshot.digest == hashlib.sha256(task.read_bytes()).hexdigest()
    assert snapshot.task["objective"] == BASE_TASK["objective"]
    assert snapshot.size == len(task.read_bytes())


def test_a_byte_change_after_preflight_is_detected(task):
    snapshot = preflight.snapshot_manifest(task)
    assert preflight.manifest_changed(task, snapshot) is False
    task.write_bytes(task.read_bytes() + b" ")
    assert preflight.manifest_changed(task, snapshot) is True


def test_the_same_json_reserialised_still_counts_as_a_change(task):
    """Byte comparison, not semantic: identical JSON written with
    different whitespace is a file somebody edited, and the run was
    bound to what it actually read."""
    snapshot = preflight.snapshot_manifest(task)
    reserialised = json.dumps(json.loads(task.read_text(encoding="utf-8")),
                              indent=2)
    task.write_text(reserialised, encoding="utf-8")
    assert json.loads(task.read_text(encoding="utf-8")) == snapshot.task
    assert preflight.manifest_changed(task, snapshot) is True


def test_a_missing_manifest_counts_as_changed(task):
    snapshot = preflight.snapshot_manifest(task)
    task.unlink()
    assert preflight.manifest_changed(task, snapshot) is True


# =====================================================================
# IDENTITY -- what the audit found unbound
# =====================================================================

def test_a_real_worktree_id_can_actually_be_written_to_the_binding(repo, task,
                                                                   state_dir):
    """The field was optional AND unsatisfiable: `create` derived the id
    from a temp-directory suffix, which is not 32 hex, so the binding
    refused it -- and nothing noticed, because the schema did not
    require the field. "Bound to a worktree" was documentation only.

    Both halves are checked here: the id `create` really produces is
    accepted, and a binding WITHOUT one is refused.

    The id now carries a second job -- it is also the capability that
    authorises removing the worktree -- so the two vocabularies have to
    stay the same one."""
    baseline = _git(repo, "rev-parse", "HEAD")
    path, produced = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    try:
        state.write_binding(state_dir, _binding(repo, task,
                                                worktree_id=produced))
        assert state.read_binding(state_dir)["worktree_id"] == produced
        assert worktree.WORKTREE_ID.match(produced)

        incomplete = _binding(repo, task)
        incomplete.pop("worktree_id")
        with pytest.raises(state.CorruptState):
            state.write_binding(state_dir, incomplete)
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=produced)


def test_a_worktree_id_that_is_not_ours_is_refused(repo, task, state_dir):
    state.write_binding(state_dir, _binding(repo, task, worktree_id="c" * 32))
    with pytest.raises(state.IncompatibleState):
        state.assert_binding(state_dir, repo_id=state.repo_identity(repo),
                             baseline_sha=_git(repo, "rev-parse", "HEAD"),
                             manifest_digest=preflight.snapshot_manifest(
                                 task).digest,
                             worktree_id="d" * 32)


@pytest.mark.parametrize("keep", ["state", "binding"])
def test_half_a_state_directory_is_refused(repo, task, binaries, state_dir,
                                           keep):
    """A `state.json` with no `binding.json` used to pass preflight
    unexamined: the binding was only validated IF it was present, so
    deleting one file of the two turned a foreign run's leftovers into
    an acceptable starting point."""
    _pair(repo, task, state_dir)
    (state_dir / (state.BINDING_FILENAME if keep == "state"
                  else state.STATE_FILENAME)).unlink()

    with pytest.raises(state.CorruptState) as refusal:
        state.assert_state_directory(
            state_dir, repo_id=state.repo_identity(repo),
            baseline_sha=_git(repo, "rev-parse", "HEAD"),
            manifest_digest=preflight.snapshot_manifest(task).digest)
    # the refusal has to name the HALF PAIR. Reading either document
    # would fail on its own with "file missing", which is true and
    # useless: it sends an operator off to recreate the file that is
    # gone, when the actual verdict is that this directory is not a
    # state directory any more. A mutation run caught exactly this --
    # deleting the pair check left the test green on the downstream
    # error.
    assert "yarim" in str(refusal.value), str(refusal.value)

    result = preflight.run_preflight(task, repo=repo, binaries=binaries,
                                     state_dir=state_dir)
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED


def test_the_two_documents_must_describe_the_same_run(repo, task, binaries,
                                                      state_dir):
    """Nothing compared their `run_id` values, so a state directory
    could hold two documents about two different runs and validate.

    The state written here is TERMINAL on purpose. With a running one
    the unfinished-run gate refuses first and this test passes without
    the run_id comparison ever happening -- which is how it was written,
    and a mutation run caught it: deleting the comparison changed
    nothing."""
    _pair(repo, task, state_dir)
    state.write_state(state_dir, _valid_state(run_id="bambaska-kosu",
                                              state=contract.State.APPROVED,
                                              stop_reason="completed"))
    with pytest.raises(state.IncompatibleState):
        state.assert_state_directory(
            state_dir, repo_id=state.repo_identity(repo),
            baseline_sha=_git(repo, "rev-parse", "HEAD"),
            manifest_digest=preflight.snapshot_manifest(task).digest)
    assert preflight.run_preflight(task, repo=repo, binaries=binaries,
                                   state_dir=state_dir).ok is False


def test_an_unfinished_previous_run_is_not_silently_restarted(repo, task,
                                                              binaries,
                                                              state_dir):
    """This is the guarantee the stale-lock rule used to reach for by
    guessing at a pid. Resumability has evidence on disk; liveness does
    not, and the two are now answered separately."""
    _pair(repo, task, state_dir, state=contract.State.IMPLEMENTING)
    assert preflight.run_preflight(task, repo=repo, binaries=binaries,
                                   state_dir=state_dir).ok is False
    assert preflight.run_preflight(task, repo=repo, binaries=binaries,
                                   state_dir=state_dir,
                                   allow_resume=True).ok is True


def test_a_finished_run_leaves_a_resumable_directory_alone(repo, task,
                                                           binaries,
                                                           state_dir):
    """The boundary in the other direction: a terminal state is not an
    obstacle, or the refusal above would just be "never start twice"."""
    _pair(repo, task, state_dir, state=contract.State.APPROVED,
          stop_reason="completed")
    assert preflight.run_preflight(task, repo=repo, binaries=binaries,
                                   state_dir=state_dir).ok is True


def test_an_empty_state_directory_is_a_fresh_run(repo, task, binaries,
                                                 state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    assert state.assert_state_directory(
        state_dir, repo_id=state.repo_identity(repo),
        baseline_sha=_git(repo, "rev-parse", "HEAD"),
        manifest_digest=preflight.snapshot_manifest(task).digest) is None
    assert preflight.run_preflight(task, repo=repo, binaries=binaries,
                                   state_dir=state_dir).ok is True


# =====================================================================
# STATE INVARIANTS -- what a legal transition may not do
# =====================================================================

@pytest.mark.parametrize(
    ("field", "value"),
    [("run_id", "bambaska-kosu"), ("baseline_sha", "b" * 40),
     ("started_at", "t99"), ("protocol_version", "0.0")])
def test_a_legal_transition_cannot_rewrite_the_runs_identity(state_dir, field,
                                                             value):
    """`advance` merged every keyword straight over the document, so a
    caller could change WHICH run this is in the middle of an otherwise
    valid move -- and the result still validated, because the schema
    constrains shapes, not continuity."""
    state.write_state(state_dir, _valid_state(baseline_sha="a" * 40))
    with pytest.raises(state.IncompatibleState):
        state.advance(state_dir, contract.State.IMPLEMENTING, **{field: value})
    unchanged = state.read_state(state_dir)
    assert unchanged["state"] == contract.State.PREFLIGHT, \
        "reddedilen gecis yine de yazilmis"
    assert unchanged["run_id"] == "kurgu-run-1"


def test_passing_an_identity_field_unchanged_is_still_allowed(state_dir):
    """The boundary: the rule is immutability, not a ban on naming the
    field, or every caller that echoes its own state would be blocked."""
    state.write_state(state_dir, _valid_state(baseline_sha="a" * 40))
    moved = state.advance(state_dir, contract.State.IMPLEMENTING,
                          run_id="kurgu-run-1", baseline_sha="a" * 40)
    assert moved["state"] == contract.State.IMPLEMENTING


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_never_reaches_disk(state_dir, bad):
    """JSON Schema does not catch these. `{"type": "number", "minimum":
    0}` accepts NaN, because `minimum` asks whether the value is BELOW
    the bound and every comparison with NaN is false. Python then
    writes the literal `NaN`, which is not JSON, and reads it back --
    so a budget could be spent-unknown and survive the round trip."""
    with pytest.raises(state.CorruptState):
        state.write_state(state_dir,
                          _valid_state(budget={"max_usd": 1.0,
                                               "spent_usd": bad}))
    assert not (state_dir / state.STATE_FILENAME).exists()


def test_a_hand_edited_non_finite_literal_is_refused_on_read(state_dir):
    state.write_state(state_dir, _valid_state())
    raw = (state_dir / state.STATE_FILENAME).read_text(encoding="utf-8")
    (state_dir / state.STATE_FILENAME).write_text(
        raw.replace('"spent_usd": 0.0', '"spent_usd": NaN'), encoding="utf-8")
    with pytest.raises(state.CorruptState):
        state.read_state(state_dir)


def test_the_budget_invariant_is_enforced_and_not_merely_recorded(state_dir):
    """`contract.BUDGET_INVARIANT` is documented next to the frozen
    schemas as a RUNNER invariant because it relates two fields.
    Recorded and unenforced is the weaker half of that."""
    over = _valid_state(budget={"max_usd": 1.0, "spent_usd": 1.5})
    with pytest.raises(state.IncompatibleState):
        state.write_state(state_dir, over)
    assert not (state_dir / state.STATE_FILENAME).exists()

    state.write_state(state_dir, _valid_state())
    with pytest.raises(state.IncompatibleState):
        state.advance(state_dir, contract.State.IMPLEMENTING,
                      budget={"max_usd": 1.0, "spent_usd": 1.5})
    assert state.read_state(state_dir)["state"] == contract.State.PREFLIGHT


def test_spending_exactly_the_ceiling_is_allowed(state_dir):
    """The boundary, so the gate is `>` and not `>=`."""
    state.write_state(state_dir, _valid_state(
        budget={"max_usd": 1.0, "spent_usd": 1.0}))
    assert state.read_state(state_dir)["budget"]["spent_usd"] == 1.0


# =====================================================================
# PREFLIGHT GATES -- prefix accidents and things that are not binaries
# =====================================================================

def test_the_dirty_allowlist_stops_at_a_path_segment(repo, task, binaries):
    """One permitted file covered any sibling sharing its opening
    characters, so an uncommitted file nobody approved passed the
    dirty-tree gate by prefix accident."""
    (repo / "notlar.md").write_text("izinli\n", encoding="utf-8")
    (repo / "notlar.md.ozel").write_text("KARDES\n", encoding="utf-8")
    result = preflight.run_preflight(task, repo=repo, binaries=binaries,
                                     dirty_allowlist=["notlar.md"])
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.DIRTY_WORKTREE


def test_the_allowlist_still_covers_what_it_should(repo, task, binaries):
    """The boundary: exact matches and real subtrees must keep passing,
    or the fix above is just "allow nothing"."""
    (repo / "notlar.md").write_text("izinli\n", encoding="utf-8")
    (repo / "gecici").mkdir()
    (repo / "gecici" / "ic.txt").write_text("izinli\n", encoding="utf-8")
    result = preflight.run_preflight(task, repo=repo, binaries=binaries,
                                     dirty_allowlist=["notlar.md", "gecici"])
    assert result.ok is True, result.detail


@pytest.mark.parametrize("role", ["implementer", "evaluator"])
def test_a_directory_is_not_a_model_binary(repo, task, binaries, tmp_path,
                                           role):
    """`Path.exists()` was the whole check, and a directory exists --
    so preflight passed and the failure moved to the first model call,
    after a worktree had been built and the run had started costing
    something."""
    not_a_binary = tmp_path / "dizin-ikili"
    not_a_binary.mkdir()
    broken = dict(binaries, **{role: not_a_binary})
    result = preflight.run_preflight(task, repo=repo, binaries=broken)
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED
    assert role in result.detail


# =====================================================================
# CONTROL PLANE -- the loop's own code, verified against git alone
# =====================================================================

def _tamper_with_the_control_plane(repo):
    """Modify a tracked file under a control-plane path, in the throwaway
    repo. Returns the file so the caller can put it back."""
    victim = repo / "tools" / "agent_loop" / "contract.py"
    victim.write_text("FROZEN = False   # KURGU KURCALAMA\n", encoding="utf-8")
    return victim


def test_a_modified_control_plane_stops_the_run(repo, task, binaries):
    _tamper_with_the_control_plane(repo)
    result = preflight.run_preflight(task, repo=repo, binaries=binaries)
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.CONTROL_PLANE_MODIFIED


def test_the_manifest_cannot_excuse_a_modified_control_plane(repo, binaries,
                                                             tmp_path):
    """THE finding. `dirty_tree_allowlist` was only checked for relative
    path SHAPE, and the dirty-tree gate honoured it with no control-plane
    exception -- so a manifest could name `tools/agent_loop` and walk a
    tampered loop past preflight. Taking a hash of the control plane
    afterwards would not have helped: the run's reference would already
    be the tampered code.

    The gate that refuses this asks git and reads nothing from the task,
    so no field a task can write participates in the answer."""
    _tamper_with_the_control_plane(repo)
    laundering = tmp_path / "aklayan-task.json"
    laundering.write_text(json.dumps(dict(
        BASE_TASK, baseline_sha=_git(repo, "rev-parse", "HEAD"),
        dirty_tree_allowlist=["tools/agent_loop"])), encoding="utf-8")

    result = preflight.run_preflight(laundering, repo=repo, binaries=binaries)
    assert result.ok is False
    assert result.stop_reason in (contract.StopReason.CONTROL_PLANE_MODIFIED,
                                  contract.StopReason.PATH_NOT_ALLOWED)


@pytest.mark.parametrize(
    "entry", ["tools/agent_loop", "tools/agent_loop/", "tools/agent_loop/cli.py",
              "tools", "tests/test_agent_loop_contract.py",
              "eval/tools/leak_scan.py", "scripts/p0_gate.sh",
              # NOT named in the contract: only the family pattern can
              # refuse these, and a mutation deleting it went unnoticed
              # while every parameter here was a name the contract listed
              "tests/test_agent_loop_b1.py", "tests/test_agent_loop_b2.py"])
def test_no_manifest_path_list_may_name_the_control_plane(repo, binaries,
                                                          tmp_path, entry):
    """One rule, applied to every path list the manifest carries.
    `allowed_paths` had it; `dirty_tree_allowlist` did not."""
    hostile = tmp_path / "hasmane-task.json"
    hostile.write_text(json.dumps(dict(
        BASE_TASK, baseline_sha=_git(repo, "rev-parse", "HEAD"),
        dirty_tree_allowlist=[entry])), encoding="utf-8")
    result = preflight.run_preflight(hostile, repo=repo, binaries=binaries)
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.PATH_NOT_ALLOWED


@pytest.mark.parametrize(
    "entry", ["tools/agent_loop", "tools/agent_loop/", "tools/agent_loop/cli.py",
              "tools", "tests/test_agent_loop_contract.py",
              "eval/tools/leak_scan.py", "scripts/p0_gate.sh",
              # NOT named in the contract: only the family pattern can
              # refuse these, and a mutation deleting it went unnoticed
              # while every parameter here was a name the contract listed
              "tests/test_agent_loop_b1.py", "tests/test_agent_loop_b2.py"])
def test_the_caller_cannot_allowlist_the_control_plane_either(repo, task,
                                                              binaries, entry):
    """The same rule on the OTHER entry point: `dirty_allowlist` handed
    straight to `run_preflight`, bypassing the manifest.

    The manifest here is CLEAN. An earlier version of this reused the
    hostile manifest from the test above, so the manifest check refused
    first and the direct check never ran -- a mutation deleting it
    changed nothing."""
    assert preflight.snapshot_manifest(task).task["dirty_tree_allowlist"] \
        == [], "senaryo kurulmadi: manifest temiz olmali"
    result = preflight.run_preflight(task, repo=repo, binaries=binaries,
                                     dirty_allowlist=[entry])
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.PATH_NOT_ALLOWED


def test_the_whole_agent_loop_test_family_is_control_plane(repo, task,
                                                          binaries):
    """The loop's OWN SAFETY TESTS are control plane too.

    `contract.CONTROL_PLANE_PATHS` names the Phase A test file one by
    one, so `test_agent_loop_b1.py` fell outside it the moment it was
    written -- this battery could be edited and excused through the
    dirty-tree allowlist, and every future B2/B3 battery would have
    inherited the same gap. The inventory is a pattern over the family
    now, and this test uses a member the contract does NOT name, so the
    pattern is the only thing that can refuse it."""
    victim = repo / "tests" / "test_agent_loop_b1.py"
    assert victim.is_file(), "senaryo kurulmadi"
    assert "tests/test_agent_loop_b1.py" not in contract.CONTROL_PLANE_PATHS,         "senaryo kurulmadi: bu dosya sozlesmede tek tek listelenmemeli"
    victim.write_text("# KURCALANDI\n", encoding="utf-8")

    result = preflight.run_preflight(task, repo=repo, binaries=binaries)
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.CONTROL_PLANE_MODIFIED
    # and it cannot be excused, either
    assert preflight.run_preflight(
        task, repo=repo, binaries=binaries,
        dirty_allowlist=["tests/test_agent_loop_b1.py"]).ok is False


def test_the_atomic_write_flushes_the_directory_entry_too(state_dir,
                                                          monkeypatch):
    """The CALL SITE, not just the helper. A mutation removing the
    directory flush from `write_json_atomically` broke nothing, because
    the durability test called the helper directly -- so the write path
    was never shown to use it."""
    seen = []
    real = state.fsync_directory
    monkeypatch.setattr(state, "fsync_directory",
                        lambda directory: (seen.append(Path(directory)),
                                           real(directory))[1])
    state.write_state(state_dir, _valid_state())
    # the chain adds flushes of its own for directories created on the
    # way in; what this pins is that the directory the file LANDED in is
    # flushed, and flushed LAST -- after the replace, not before it
    assert seen[-1] == Path(state_dir), \
        f"replace sonrasi dizin girdisi flush edilmedi: {seen}"


def test_an_ordinary_dirty_file_is_still_allowlistable(repo, task, binaries):
    """The boundary: the legitimate use of the allowlist must survive,
    or the fix above is just "allow nothing"."""
    (repo / "pipeline" / "gecici.txt").write_text("izinli\n", encoding="utf-8")
    assert preflight.run_preflight(task, repo=repo, binaries=binaries,
                                   dirty_allowlist=["pipeline/gecici.txt"]
                                   ).ok is True


# =====================================================================
# EVIDENCE -- a gate whose check did not run has not passed
# =====================================================================

def _break_git(monkeypatch, subcommand, stderr=""):
    """Make one git SUBCOMMAND fail the way a broken index or a
    contended lock does: non-zero, empty stdout.

    Patched at the subprocess boundary on purpose. An earlier version of
    this test replaced `preflight._git` -- the very function whose job
    is to notice the exit code -- so the fake answered instead of the
    code under test and the test passed against a still-broken gate."""
    real_run = preflight.subprocess.run

    class _Shim:
        CompletedProcess = subprocess.CompletedProcess

        @staticmethod
        def run(argv, **kwargs):
            if len(argv) > 3 and argv[0] == "git" and argv[3] == subcommand:
                return subprocess.CompletedProcess(args=argv, returncode=128,
                                                   stdout="", stderr=stderr)
            return real_run(argv, **kwargs)

    monkeypatch.setattr(preflight, "subprocess", _Shim)

@pytest.mark.parametrize("broken", ["status", "diff", "rev-parse"])
def test_a_git_command_that_fails_is_not_a_clean_tree(repo, task, binaries,
                                                      monkeypatch, broken):
    """The safety gates read `git diff --cached` and `git status` and
    trusted the text without looking at the exit code. Feeding both a
    failure and an empty stdout -- a broken index, a contended lock --
    made preflight report "nothing staged, tree clean" and pass."""
    _break_git(monkeypatch, broken)
    result = preflight.run_preflight(task, repo=repo, binaries=binaries)
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED


def test_the_failure_detail_never_carries_git_stderr(repo, task, binaries,
                                                     monkeypatch):
    """A refusal may name the command and the code. Not the message --
    git's stderr can carry a path, a remote or a credential helper's
    complaint, and this text ends up in a report that gets forwarded."""
    # a sentinel with no shape of its own: the earlier version looked
    # like a Windows path carrying a key, which made the leak scanner
    # stop and ask a human about a value this file invented
    sentinel = "KURGU-STDERR-NOBETCI-" + "z" * 8
    _break_git(monkeypatch, "status", stderr=sentinel)
    result = preflight.run_preflight(task, repo=repo, binaries=binaries)
    assert result.ok is False
    assert sentinel not in result.detail
    assert "NOBETCI" not in result.detail


# =====================================================================
# RUNNABILITY -- asked once, of the thing that will do the launching
# =====================================================================

def test_a_file_that_cannot_be_launched_is_not_a_binary(repo, task, binaries,
                                                        tmp_path):
    """Third attempt at this gate. `exists()` accepted a directory;
    adding `is_file()` left a Windows branch that accepted ANY regular
    file, so a plain `.txt` passed preflight and then failed with
    WinError 193 at the first real call.

    The test proves the gate against REALITY rather than against the
    rule: whatever `_is_runnable` accepts must actually launch."""
    plain = tmp_path / "hic-de-ikili-degil.txt"
    plain.write_text("bu sadece metin\n", encoding="utf-8")
    plain.chmod(0o644)

    assert preflight._is_runnable(plain) is False    # noqa: SLF001
    try:
        subprocess.run([str(plain)], capture_output=True, timeout=15)
        launched = True
    except OSError:
        launched = False
    assert launched is False, "senaryo kurulmadi: dosya gercekten calisiyor"

    result = preflight.run_preflight(task, repo=repo,
                                     binaries=dict(binaries,
                                                   implementer=plain))
    assert result.ok is False
    assert result.stop_reason == contract.StopReason.PREFLIGHT_FAILED


def _script(directory, name, windows_body, posix_body):
    if os.name == "nt":
        path = directory / f"{name}.cmd"
        path.write_text(windows_body, encoding="ascii")
    else:
        path = directory / f"{name}.sh"
        path.write_text(posix_body, encoding="ascii")
        path.chmod(0o755)
    return path


def test_a_binary_that_never_returns_is_refused(tmp_path, monkeypatch):
    """Fail-closed on a hang. A handshake with no deadline would park
    preflight forever on a binary that reads stdin or spins."""
    sleeper = _script(tmp_path, "uyuyan",
                      "@echo off\r\nping -n 30 127.0.0.1 >nul\r\n",
                      "#!/bin/sh\nsleep 30\n")
    monkeypatch.setattr(preflight, "HANDSHAKE_TIMEOUT_SECONDS", 1)
    assert preflight._is_runnable(sleeper) is False      # noqa: SLF001


def test_a_binary_that_exits_non_zero_still_counts_as_runnable(tmp_path):
    """The DELIBERATE boundary, stated so it can be argued with: the
    exit code is not consulted. A CLI may exit non-zero on a flag it
    does not recognise, and that is still a program the operating system
    started. Requiring zero would reject a working binary, which is the
    expensive direction of this error."""
    grumpy = _script(tmp_path, "kizgin", "@echo off\r\nexit /b 3\r\n",
                     "#!/bin/sh\nexit 3\n")
    assert subprocess.run([str(grumpy)], capture_output=True,
                          timeout=30).returncode == 3, "senaryo kurulmadi"
    assert preflight._is_runnable(grumpy) is True        # noqa: SLF001


def test_the_handshake_launches_only_the_given_path_with_a_fixed_argument(
        binaries, monkeypatch):
    """Two claims at once: nothing is discovered, and the argument is
    not something a task can choose. An argv a manifest could supply is
    an argv a manifest could turn into a real, billable invocation."""
    seen = []
    real_run = preflight.subprocess.run

    class _Shim:
        # the module under test also reads exception classes off this
        # name, so the stand-in has to carry them
        DEVNULL = subprocess.DEVNULL
        SubprocessError = subprocess.SubprocessError

        @staticmethod
        def run(argv, **kwargs):
            seen.append((list(argv), kwargs.get("timeout")))
            return real_run(argv, **kwargs)

    monkeypatch.setattr(preflight, "subprocess", _Shim)
    assert preflight._is_runnable(binaries["implementer"]) is True
    monkeypatch.undo()

    argv, timeout = seen[0]
    assert argv[0] == str(binaries["implementer"])
    assert argv[1:] == list(preflight.HANDSHAKE_ARGV) == ["--version"]
    assert timeout == preflight.HANDSHAKE_TIMEOUT_SECONDS


def test_a_bare_command_name_is_never_resolved_through_the_path():
    """`git` exists on this machine's PATH. It is still not a binary
    this gate accepts, because nothing here searches for anything."""
    assert preflight._is_runnable("git") is False        # noqa: SLF001
    assert preflight._is_runnable("cmd") is False        # noqa: SLF001


def test_something_genuinely_executable_is_accepted(repo, task, binaries):
    """The boundary, against reality again: the interpreter running this
    test really does launch, so the gate must accept it. Without this
    the fix above could be "reject everything"."""
    real_program = Path(sys.executable)
    assert preflight._is_runnable(real_program) is True   # noqa: SLF001
    assert subprocess.run([str(real_program), "-c", "pass"],
                          capture_output=True, timeout=60).returncode == 0
    assert preflight.run_preflight(
        task, repo=repo,
        binaries={"implementer": real_program,
                  "evaluator": real_program}).ok is True


def _mutation_tool():
    import importlib.util

    path = (Path(__file__).resolve().parent.parent / "eval" / "tools"
            / "mutate_agent_loop.py")
    spec = importlib.util.spec_from_file_location("mutate_agent_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_mutation_still_applies_to_the_current_source():
    """The mutation manifest is only evidence while it still describes
    this code.

    A "32/32 caught" claim that cannot be re-run is a story about a
    session, so the harness lives in the repository -- and a pattern
    that has drifted out of the source would silently become
    NOT-APPLIED, turning a missing guard into a quiet pass. That failure
    is loud here instead."""
    tool = _mutation_tool()
    # PINNED at the current census, not a stale floor: `>= 32` stayed
    # green with every R2A mutation deleted, because 35 B1 entries
    # already cleared it. The R2A labels are pinned BY NAME too, so a
    # deletion has to show up as a missing name rather than as a
    # count that still happens to pass.
    assert len(tool.MUTATIONS) >= 72
    labels = {label for label, *_ in tool.MUTATIONS}
    expected_r2a = {
        "r2a-hazir-kontrolu", "r2a-kayit-depo", "r2a-kayit-kosu",
        "r2a-kayit-taban", "r2a-gomulu-kimlik", "r2a-bag-varligi",
        "r2a-bag-agac", "r2a-bag-depo", "r2a-bag-kosu", "r2a-bag-taban",
        "r2a-bas-kontrolu", "r2a-git-kutugu", "r2a-ana-agac",
        "r2a-kap-sinirlama", "r2a-bag-cagrisi-yok", "r2a-cagiran-yolu"}
    assert expected_r2a <= labels, \
        f"eksik r2a mutasyonu: {sorted(expected_r2a - labels)}"
    expected_r2b = {
        "r2b-yol-degeri", "r2b-kanonik-sirasiz", "r2b-ham-sozluk",
        "r2b-esitlik-kontrolu", "r2b-yanlis-sha",
        "r2br1-dondurma", "r2br1-tip-kontrolu", "r2br1-kodlama-sarici",
        "r2br11-alt-sinif"}
    assert expected_r2b <= labels, \
        f"eksik r2b mutasyonu: {sorted(expected_r2b - labels)}"
    expected_b2a = {
        "b2a-arac-tipi", "b2a-ikili-donusu", "b2a-butce-tavani",
        "b2a-butce-tipi", "b2a-sure-tipi", "b2a-istem-tipi",
        "b2a-argv-token-tipi", "b2a-model-tipi", "b2a-kimlik-tipi",
        "b2ar1-builder-butce", "b2ar1-tipli-ret", "b2ar1-arac-yansimasi"}
    assert expected_b2a <= labels, \
        f"eksik b2a mutasyonu: {sorted(expected_b2a - labels)}"
    root = Path(__file__).resolve().parent.parent
    stale = [label for label, module, old, _, _ in tool.MUTATIONS
             if old not in (root / "tools" / "agent_loop"
                            / f"{module}.py").read_text(encoding="utf-8")]
    assert stale == [], f"kaynakta artik bulunmayan mutasyon deseni: {stale}"


def test_every_mutation_names_a_test_that_exists():
    """An expected-target that matches nothing would make every verdict
    MISDIRECTED, which reads like a finding and is a typo. R2A grew the
    manifest targets into the B2 execution battery, so both agent-loop
    test files are the namespace now."""
    tool = _mutation_tool()
    here = Path(__file__).resolve().parent
    body = "".join(
        (here / name).read_text(encoding="utf-8")
        for name in ("test_agent_loop_b1.py",
                     "test_agent_loop_b2_execution.py",
                     "test_agent_loop_contract.py"))
    missing = sorted({expected for *_, expected in tool.MUTATIONS
                      if expected not in body})
    assert missing == [], f"var olmayan hedef test adi: {missing}"


def test_the_mutation_harness_exit_code_is_fail_closed():
    """The harness printed its verdicts and exited 0 REGARDLESS -- a
    red baseline, a missed mutation, a misdirected kill and a modified
    live tree all looked like success to automation. The exit code is
    the verdict now, and each failure class is pinned here cheaply,
    without running the battery."""
    tool = _mutation_tool()
    caught = {"mutasyon": "kurgu", "hukum": "YAKALANDI"}
    assert tool.verdict_exit_code(1, [], True) != 0, "kirmizi taban 0 cikti"
    assert tool.verdict_exit_code(0, [], True) != 0, "hic hukum yokken 0"
    for bad in ("KACIRILDI", "YANLIS-HEDEF", "UYGULANAMADI", "GECERSIZ"):
        verdicts = [caught, {"mutasyon": "kurgu2", "hukum": bad}]
        assert tool.verdict_exit_code(0, verdicts, True) != 0, \
            f"{bad} hukmu 0 cikti"
    assert tool.verdict_exit_code(0, [caught], False) != 0, \
        "degisen ana agac 0 cikti"
    assert tool.verdict_exit_code(0, [caught], True) == 0, \
        "temiz kosu 0 donmuyor"


def test_the_mutation_harness_main_is_wired_to_the_exit_code():
    """`main` computing a code that `__main__` throws away would be the
    same hole with extra steps."""
    body = (Path(__file__).resolve().parent.parent / "eval" / "tools"
            / "mutate_agent_loop.py").read_text(encoding="utf-8")
    assert "sys.exit(main())" in body, "cikis kodu surece baglanmamis"
    assert "return verdict_exit_code(" in body, \
        "main hukum kodunu hesaplamiyor"


def test_no_b1_module_runs_a_shell():
    import ast

    offenders = []
    for name in ("state", "locking", "worktree", "preflight"):
        module = Path(__file__).resolve().parent.parent / "tools" \
            / "agent_loop" / f"{name}.py"
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell" and \
                            getattr(keyword.value, "value", False) is True:
                        offenders.append(f"{name}:shell=True")
                if ast.unparse(node.func) in ("os.system", "os.popen"):
                    offenders.append(f"{name}:{ast.unparse(node.func)}")
    assert not offenders, f"keyfi kabuk: {offenders}"


# =====================================================================
# R2A -- the execution binding seam: identities in, ONE derived path out
# =====================================================================

def test_the_execution_binding_returns_only_the_derived_path(repo, state_dir):
    """The positive face of the seam: a worktree B1 itself created and
    marked READY binds, and what comes back is the holder-derived path
    -- the same one `create` returned."""
    baseline = _git(repo, "rev-parse", "HEAD")
    path, worktree_id = worktree.create(repo, state_dir=state_dir, run_id=RUN,
                                        baseline_sha=baseline)
    state.write_binding(state_dir, {
        "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
        "repo_id": state.repo_identity(repo), "baseline_sha": baseline,
        "manifest_digest": "0" * 64, "worktree_id": worktree_id})
    try:
        derived = worktree.assert_execution_binding(
            repo, state_dir=state_dir, run_id=RUN, worktree_id=worktree_id,
            baseline_sha=baseline)
        assert derived == worktree.holder_for(worktree_id) \
            / worktree.WORKTREE_DIRNAME
        assert derived.resolve() == path.resolve()
    finally:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)


def test_the_execution_binding_takes_identities_never_a_path():
    """No parameter carries a location. The path COMES OUT of the seam;
    a caller holding a directory has nowhere to put it."""
    import inspect as inspect_module

    parameters = inspect_module.signature(
        worktree.assert_execution_binding).parameters
    assert set(parameters) == {"repo", "state_dir", "run_id",
                               "worktree_id", "baseline_sha"}
    for escape in ("path", "cwd", "directory", "worktree"):
        assert escape not in parameters


def test_the_execution_binding_error_texts_stay_abstract(repo, state_dir):
    """Every refusal is a fixed sentence -- no absolute path, no id, no
    sha, no record content. Checked on a real refusal rather than by
    reading the source."""
    baseline = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(worktree.WorktreeError) as refusal:
        worktree.assert_execution_binding(
            repo, state_dir=state_dir, run_id=RUN, worktree_id="e" * 32,
            baseline_sha=baseline)
    text = str(refusal.value) + repr(refusal.value)
    assert "/" not in text and "\\" not in text
    assert "e" * 32 not in text and baseline not in text


def test_case_twin_paths_are_distinct_on_case_sensitive_filesystems(tmp_path):
    """R2A-R1 P1: unconditional casefold gave two DISTINCT case-twin
    directories one `_comparable` spelling and two distinct case-twin
    repositories one `repo_identity` -- an identity two repositories
    share authorises either against records the other wrote. Folding
    now happens only where the filesystem folds."""
    probe = tmp_path / "kucuk-harf-sondasi.txt"
    probe.write_text("k", encoding="ascii")
    if (tmp_path / "KUCUK-HARF-SONDASI.TXT").exists():
        pytest.skip("dosya sistemi harfe duyarsiz; ikiz kurulamiyor")
    upper = tmp_path / "IKIZ"
    lower = tmp_path / "ikiz"
    upper.mkdir()
    lower.mkdir()
    assert upper.resolve() != lower.resolve(), "senaryo kurulmadi"
    assert worktree._comparable(upper) != worktree._comparable(lower), \
        "case-twin dizinler ayni sayildi"
    assert state.repo_identity(upper) != state.repo_identity(lower), \
        "case-twin depolar ayni kimligi aldi"
