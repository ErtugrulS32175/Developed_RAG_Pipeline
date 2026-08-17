"""PACKAGE B6-R2 -- archive a terminal run, then reset the repository.

WHAT THIS FILE IS ABOUT. One public call has to do what has until now
been a hand-run sequence: measure a terminal run's identities, copy its
evidence somewhere byte-exact, PROVE the copy, and only then take the
originals away. The order is the whole safety argument, so most of the
battery below is about what must NOT have happened when something goes
wrong -- not a single source byte removed, not a workspace touched, not a
task manifest deleted.

REAL TERMINAL RUNS, NOT HAND-BUILT STATE. Every world here is produced by
`runner.run()` against the fake shims the B3 battery already owns, so the
state document, the binding, the events journal, the acceptance receipt,
the ledger record and the workspace holder are the real articles. MEASURED
while writing this, and each one shaped a rule:

    approved  -- state, backup, binding, events, receipt, findings;
                 ledger EMPTY and no holder, because the runner releases
                 the workspace on success
    failed    -- the same MINUS findings.json, ledger record present,
                 holder present
    blocked   -- at preflight, ONLY run.lock: a preflight failure writes
                 no state document at all, so there is no run to finalize

No vendor process is ever started, and nothing in this file finalizes a
real run: every repository is a disposable one under `tmp_path`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import test_agent_loop_b3_runner as world_module
import test_agent_loop_contract as base
from tools.agent_loop import (acceptance_workspace, application, contract,
                              finalization, flat_workspace, runner)
from tools.agent_loop import state as state_module

SENTINEL = "KURGU-GIZLI-ARSIV-" + "w" * 8


# ---------------------------------------------------------------------
# THE FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def private_roots(tmp_path, monkeypatch):
    """Every test gets its OWN flat root and mirror root.

    NOT optional here, and MEASURED: the B3 battery does not redirect
    these, so its real runs build holders in the shared temp root -- and a
    file whose whole subject is DELETING workspaces must never be able to
    enumerate or remove one a real agent loop is using."""
    flat_root = tmp_path / "runner-koku"
    mirror_root = tmp_path / "ayna-koku"
    flat_root.mkdir()
    mirror_root.mkdir()
    monkeypatch.setattr(flat_workspace, "runner_temp_root", lambda: flat_root)
    monkeypatch.setattr(acceptance_workspace, "mirror_temp_root",
                        lambda: mirror_root)
    yield types.SimpleNamespace(flat=flat_root, mirror=mirror_root)
    shutil.rmtree(flat_root, ignore_errors=True)
    shutil.rmtree(mirror_root, ignore_errors=True)


def _run_world(tmp_path, index=0, **overrides):
    return world_module.build_world(tmp_path, index, **overrides)


def approved(tmp_path, index=0):
    """A real APPROVED run. Its workspace is already gone -- the runner
    releases it on success -- which is a case finalize has to handle
    rather than a shortcut this fixture took."""
    world = _run_world(tmp_path, index)
    result = world_module.run(world)
    assert result.state == contract.State.APPROVED, \
        f"senaryo kurulmadi: {result.state}/{result.stop_reason}"
    return world, result


def failed(tmp_path, index=0):
    """A real FAILED run: the evaluator answers for another run, so the
    workspace is PRESERVED and there is no findings document."""
    world = _run_world(tmp_path, index)
    world_module.evaluator(world, base._emits(json.dumps({
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "kosu-000000000000000000000000",
        "role": "evaluator", "status": "failed",
        "summary": "kurgu", "findings": []})))
    result = world_module.run(world)
    assert result.state == contract.State.FAILED, \
        f"senaryo kurulmadi: {result.state}/{result.stop_reason}"
    return world, result


def blocked(tmp_path, index=0):
    """A real BLOCKED run that GOT PAST preflight, so a state document
    exists. The same mechanism twice is what stops it."""
    world = _run_world(tmp_path, index)
    world_module.evaluator(world, base._emits(base._code_audit_reply(
        status=contract.Status.CHANGES_REQUESTED,
        findings=[base._code_finding(mechanism_id="kurgu-mekanizma-a")],
        next_action="await_repair")))
    result = world_module.run(world)
    assert result.state == contract.State.BLOCKED, \
        f"senaryo kurulmadi: {result.state}/{result.stop_reason}"
    return world, result


def archive_root(tmp_path, name="arsiv-koku"):
    """A destination OUTSIDE the repository, the state directory, the
    runner temp roots and every holder."""
    root = tmp_path / name
    root.mkdir()
    return root


def head_of(repo):
    return world_module.legacy._git(repo, "rev-parse", "HEAD")


def commit_everything(world, subject="kurgu-gonderi"):
    """Commit the applied candidate, so an APPROVED run can be finalized
    as `shipped`. `origin/main` is made to point at the same commit,
    because that is what the shipment gate compares.

    ONLY the candidate's own tree. `git add -A` would take the task
    manifest in with it, and a tracked manifest is refused -- correctly,
    because deleting a file git knows about is a commit nobody asked
    for."""
    world_module.legacy._git(world.repo, "add", "--", "pipeline")
    world_module.legacy._git(
        world.repo, "-c", "user.email=k@example.invalid",
        "-c", "user.name=Kurgu", "commit", "-qm", subject)
    head = head_of(world.repo)
    world_module.legacy._git(world.repo, "update-ref", "refs/remotes/origin/main",
                             head)
    return head


def finalize(world, root, **overrides):
    """The run id is read from the state document only when the caller did
    not supply one -- after a finalize there IS no state document, and
    computing it eagerly would raise before the call under test ran."""
    settings = {"repo": world.repo, "archive_root": root}
    if "expected_run_id" not in overrides:
        settings["expected_run_id"] = run_id_of(world)
    settings.update(overrides)
    return runner.finalize(**settings)


def run_id_of(world):
    return state_module.read_state(world.state_dir)["run_id"]


def state_files(world):
    return sorted(item.name for item in world.state_dir.iterdir()
                  if item.is_file())


def ledger_records(world):
    ledger = world.state_dir / finalization.LEDGER_DIRNAME
    return sorted(p.name for p in ledger.iterdir()) if ledger.exists() else []


def holder_of(world):
    records = ledger_records(world)
    if not records:
        return None
    return flat_workspace.holder_for(records[0].removesuffix(".json"))


def archive_dir(root, result):
    return root / result.archive_name


def manifest_of(root, result):
    return json.loads((archive_dir(root, result)
                       / finalization.MANIFEST_NAME).read_text(
                           encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(directory, skip=("run.lock",)):
    """Every file under a directory, by relative name and bytes.

    `run.lock` IS EXCLUDED, and measured rather than assumed: acquiring
    the single-instance lock rewrites the record inside it -- a fresh
    token and a fresh timestamp -- so its bytes differ across every call
    that takes the lock, finalize included. It is the locking authority's
    own bookkeeping, not this run's evidence, and comparing it would make
    every "nothing changed" assertion in this file fail for a reason that
    has nothing to do with finalize."""
    root = Path(directory)
    if not root.exists():
        return {}
    return {item.relative_to(root).as_posix(): item.read_bytes()
            for item in sorted(root.rglob("*"))
            if item.is_file() and item.name not in skip}


# ---------------------------------------------------------------------
# 1-3  THE THREE TERMINAL STATES
# ---------------------------------------------------------------------

def test_an_approved_run_with_a_shipped_commit_is_finalized(tmp_path):
    """The daily case. The workspace is already gone, the candidate is in
    the checkout and committed, and what is left is state evidence."""
    world, result = approved(tmp_path)
    shipped = commit_everything(world)
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path,
                       shipped_commit=shipped, ci_run_id="32047794474")

    assert outcome.status == finalization.FinalizeStatus.FINALIZED
    assert outcome.terminal_state == contract.State.APPROVED
    assert outcome.shipment == finalization.Shipment.SHIPPED
    assert outcome.run_id == run_id_of_archive(root, outcome)
    assert outcome.removed_workspace is False, \
        "approved kosuda kaldirilacak workspace yoktu"
    assert outcome.removed_task_manifest is True
    assert outcome.ready_for_new_task is True
    assert outcome.pending_applications == ()
    assert outcome.recovered_partial_archive is False
    # the state documents are gone, and the two keepers are not
    assert state_files(world) == ["run.lock"]
    assert (world.state_dir / "flat-workspaces").is_dir()
    assert not world.task_path.exists()


def run_id_of_archive(root, outcome):
    return json.loads((root / outcome.archive_name
                       / finalization.MANIFEST_NAME).read_text(
                           encoding="utf-8"))["run_id"]


def test_a_failed_run_is_archived_and_its_workspace_released(tmp_path):
    """A failed run holds the ONLY copy of the candidate, so the archive
    has to carry it before the holder goes."""
    world, result = failed(tmp_path)
    root = archive_root(tmp_path)
    holder = holder_of(world)
    assert holder is not None and holder.exists(), \
        "senaryo kurulmadi: workspace korunmadi"

    outcome = finalize(world, root, task_path=world.task_path)

    assert outcome.status == finalization.FinalizeStatus.FINALIZED
    assert outcome.terminal_state == contract.State.FAILED
    assert outcome.shipment == finalization.Shipment.NOT_APPLICABLE
    assert outcome.removed_workspace is True
    assert not holder.exists(), "holder kaldirilmadi"
    assert ledger_records(world) == [], "ledger kaydi kaldirilmadi"
    assert state_files(world) == ["run.lock"]
    assert outcome.ready_for_new_task is True


def test_a_blocked_run_is_archived_and_reset(tmp_path):
    world, result = blocked(tmp_path)
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path)

    assert outcome.status == finalization.FinalizeStatus.FINALIZED
    assert outcome.terminal_state == contract.State.BLOCKED
    assert outcome.removed_workspace is True
    assert state_files(world) == ["run.lock"]
    assert outcome.ready_for_new_task is True


def test_an_approved_run_that_was_never_committed_is_unshipped(tmp_path):
    """Approved is not the same as shipped. Recording an uncommitted run
    as shipped would put a claim in the archive that the repository does
    not support."""
    world, _result = approved(tmp_path)
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path)

    assert outcome.shipment == finalization.Shipment.UNSHIPPED
    assert manifest_of(root, outcome)["shipment"] == \
        finalization.Shipment.UNSHIPPED
    assert "shipped_commit" not in manifest_of(root, outcome)


# ---------------------------------------------------------------------
# 4-7  WHAT MAY NOT BE FINALIZED
# ---------------------------------------------------------------------

def test_a_non_terminal_run_is_refused_and_nothing_is_written(tmp_path):
    """The strongest refusal in the file. A run still in flight has a
    workspace somebody is using and a state document somebody is
    advancing."""
    world, _result = failed(tmp_path)
    # Rewind to a non-terminal state, the way an interrupted run looks.
    # `stop_reason` goes with it: the state schema will not accept a run
    # that is still going and has already stopped.
    payload = {name: value
               for name, value in state_module.read_state(
                   world.state_dir).items() if name != "stop_reason"}
    payload["state"] = contract.State.IMPLEMENTING
    state_module.write_state(world.state_dir, payload)
    root = archive_root(tmp_path)
    before_state = snapshot(world.state_dir)
    holder = holder_of(world)
    before_holder = snapshot(holder)

    with pytest.raises(finalization.RunNotTerminal):
        finalize(world, root, task_path=world.task_path)

    assert list(root.iterdir()) == [], "reddedilen cagri arsive yazdi"
    assert snapshot(world.state_dir) == before_state
    assert snapshot(holder) == before_holder
    assert world.task_path.exists()


def test_a_wrong_expected_run_id_is_refused(tmp_path):
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    before = snapshot(world.state_dir)

    with pytest.raises(finalization.StateBindingFailed):
        finalize(world, root, task_path=world.task_path,
                 expected_run_id="kosu-ffffffffffffffffffffffff")

    assert list(root.iterdir()) == []
    assert snapshot(world.state_dir) == before


def test_a_pending_application_refuses_the_whole_operation(tmp_path):
    """An unfinished application means the checkout is in a state nobody
    has described. Archiving over it would file the run as closed."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    pending = _plant_pending_application(world)
    assert application.find_pending_applications(world.repo) == (pending,), \
        "senaryo kurulmadi: bekleyen uygulama yok"
    before = snapshot(world.state_dir)

    with pytest.raises(finalization.PendingApplication):
        finalize(world, root, task_path=world.task_path)

    assert list(root.iterdir()) == []
    assert snapshot(world.state_dir) == before
    assert application.find_pending_applications(world.repo) == (pending,), \
        "bekleyen uygulama sessizce silindi"


def _plant_pending_application(world):
    """A REAL journal in the real apply root, written the way the
    application layer writes one."""
    application_id = "b6" + "0" * 30
    holder = application.holder_for(world.repo, application_id)
    application.apply_root_for(world.repo).mkdir(parents=True, exist_ok=True)
    holder.mkdir()
    payload = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "journal_version": application.JOURNAL_VERSION,
        "application_id": application_id,
        "repo_id": state_module.repo_identity(world.repo),
        "run_id": run_id_of(world),
        "workspace_id": state_module.read_binding(
            world.state_dir)["workspace_id"],
        "baseline_sha": world.baseline,
        "manifest_digest": world.digest,
        "candidate_fingerprint": "0" * 64,
        "state": application.JournalState.REGISTERED,
        "created_at": "2026-01-01T00:00:00Z",
        "moves": [], "directories": [], "applied_index": 0}
    state_module.write_json_atomically(
        holder / application.JOURNAL_NAME, payload,
        application.JOURNAL_SCHEMA, "uygulama gunlugu")
    return application_id


@pytest.mark.parametrize("damage", ["state", "binding", "ledger", "owner",
                                    "other-run"])
def test_a_broken_binding_between_the_documents_is_refused(tmp_path, damage):
    """Four documents describe one run and they have to agree. Any one of
    them naming a different run is a refusal, not something to reconcile.

    A MISSING state document is its own answer -- there is no run to
    finalize -- so it is `RunNotTerminal` rather than a binding failure."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    holder = holder_of(world)
    records = ledger_records(world)
    run_id = run_id_of(world)

    if damage == "state":
        (world.state_dir / "state.json").unlink()
        expected = finalization.RunNotTerminal
    elif damage == "binding":
        (world.state_dir / "binding.json").unlink()
        expected = finalization.StateBindingFailed
    elif damage == "ledger":
        (world.state_dir / "flat-workspaces" / records[0]).unlink()
        expected = finalization.StateBindingFailed
    elif damage == "owner":
        (holder / "workspace-owner.json").unlink()
        expected = finalization.StateBindingFailed
    else:
        payload = dict(state_module.read_binding(world.state_dir),
                       run_id="kosu-aaaaaaaaaaaaaaaaaaaaaaaa")
        state_module.write_binding(world.state_dir, payload)
        expected = finalization.StateBindingFailed

    with pytest.raises(expected):
        finalize(world, root, task_path=world.task_path,
                 expected_run_id=run_id)

    assert list(root.iterdir()) == [], "bozuk baglamaya ragmen arsive yazdi"
    assert holder.exists(), "bozuk baglamada workspace silindi"


# ---------------------------------------------------------------------
# 8-11  THE TRANSACTION
# ---------------------------------------------------------------------

def test_a_source_that_drifts_during_archiving_stops_everything(tmp_path):
    """Copying many files takes time, and a file rewritten while the loop
    was elsewhere leaves an archive that is a snapshot of no single moment.

    The file rewritten here is `state.json`, which is copied FIRST -- so by
    the time the third copy runs it is already in the archive and the
    rewrite really is behind the loop. Rewriting a file that had not been
    reached yet would prove nothing: the copy would simply pick up the new
    bytes and agree with them."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    state_json = world.state_dir / "state.json"
    original = state_json.read_bytes()
    real = finalization._copy_verified
    calls = []

    def drifting(*args, **kwargs):
        calls.append(1)
        outcome = real(*args, **kwargs)
        if len(calls) == 3:
            state_json.write_bytes(original + b"\n")
        return outcome

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_copy_verified", drifting)
        with pytest.raises(finalization.ArchiveVerificationFailed):
            finalize(world, root, task_path=world.task_path)

    assert len(calls) >= 3, "senaryo kurulmadi: ucuncu kopyaya ulasilmadi"
    assert state_json.exists(), "kaynak silindi"
    assert holder_of(world).exists(), "workspace silindi"
    assert ledger_records(world) != [], "ledger kaydi silindi"


def test_a_corrupted_read_back_is_refused_before_any_cleanup(tmp_path):
    """The read-back is the proof, so a read-back that disagrees has to
    stop the operation with every source still in place."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    real = finalization._read_back

    def corrupting(*args, **kwargs):
        data = real(*args, **kwargs)
        return data + b"BOZULMA"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_read_back", corrupting)
        with pytest.raises(finalization.ArchiveVerificationFailed):
            finalize(world, root, task_path=world.task_path)

    assert (world.state_dir / "state.json").exists()
    assert holder_of(world).exists()
    assert ledger_records(world) != []


def test_a_half_written_archive_is_never_counted_as_success(tmp_path):
    """A crash between the first copy and the completion record leaves a
    directory that LOOKS like an archive. A second call must refuse it
    rather than clean up against it."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    real = finalization._write_completion

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_write_completion",
                      lambda *a, **k: (_ for _ in ()).throw(
                          KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            finalize(world, root, task_path=world.task_path)

    # the archive directory exists, without its completion record
    names = sorted(p.name for p in root.iterdir())
    assert names, "senaryo kurulmadi: arsiv dizini yaratilmadi"
    partial = root / names[0]
    assert not (partial / finalization.COMPLETE_NAME).exists()
    assert (world.state_dir / "state.json").exists(), "yarim arsivde silindi"

    with pytest.raises(finalization.PartialArchivePresent):
        finalize(world, root, task_path=world.task_path)
    assert (world.state_dir / "state.json").exists()
    assert holder_of(world).exists()
    assert real is not None


def test_a_completed_archive_lets_a_second_call_finish_the_cleanup(tmp_path):
    """The crash window AFTER the archive is proven. The sources are still
    there, the archive is complete, and the honest thing is to verify it
    and carry on -- reported as a recovery, not as a fresh archive."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_cleanup",
                      lambda *a, **k: (_ for _ in ()).throw(
                          KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            finalize(world, root, task_path=world.task_path)

    names = sorted(p.name for p in root.iterdir())
    assert len(names) == 1
    assert (root / names[0] / finalization.COMPLETE_NAME).exists(), \
        "senaryo kurulmadi: arsiv tamamlanmadi"
    assert (world.state_dir / "state.json").exists()

    outcome = finalize(world, root, task_path=world.task_path)

    assert outcome.recovered_partial_archive is True
    assert outcome.status == finalization.FinalizeStatus.FINALIZED
    assert state_files(world) == ["run.lock"]
    assert outcome.ready_for_new_task is True


def test_a_second_call_after_a_complete_finalize_is_deterministic(tmp_path):
    """Idempotency, and it must not invent a success: the run is gone, so
    the answer is a closed `already_finalized` rather than a fresh
    archive or a crash."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    first = finalize(world, root, task_path=world.task_path)
    before = snapshot(archive_dir(root, first))

    second = finalize(world, root, expected_run_id=first.run_id)

    assert second.status == finalization.FinalizeStatus.ALREADY_FINALIZED
    assert second.archive_name == first.archive_name
    assert second.run_id == first.run_id
    assert second.removed_state_files == ()
    assert second.ready_for_new_task is True
    assert snapshot(archive_dir(root, first)) == before, "arsiv degisti"


def test_an_occupied_archive_name_is_never_written_into(tmp_path):
    """The destination name is derived, not chosen, so a collision means
    something is already there -- and whatever it is, it is not this
    call's to write into or delete."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    # exactly the name this run will derive, occupied in advance by
    # something with no completion record
    name = finalization.archive_name_for(run_id_of(world),
                                        contract.State.FAILED)
    squatter = root / name
    squatter.mkdir()
    (squatter / "ELDE-VAR").write_bytes(b"kurgu\n")
    before = snapshot(squatter)

    with pytest.raises(finalization.PartialArchivePresent):
        finalize(world, root, task_path=world.task_path)

    assert snapshot(squatter) == before, "var olan dizin degistirildi"
    assert (world.state_dir / "state.json").exists()
    assert holder_of(world).exists()


# ---------------------------------------------------------------------
# 12-13  THE ARCHIVE DESTINATION
# ---------------------------------------------------------------------

@pytest.mark.parametrize("where", ["repo", "state", "parent", "flat",
                                    "holder"])
def test_an_archive_root_that_overlaps_the_run_is_refused(tmp_path, where,
                                                          private_roots):
    """Writing the evidence INTO the thing being cleaned up is how an
    archive deletes itself. Checked in both directions: a root under one
    of these, and a root ABOVE it."""
    world, _result = failed(tmp_path)
    if where == "repo":
        root = world.repo / "arsiv"
    elif where == "state":
        root = world.state_dir / "arsiv"
    elif where == "parent":
        root = world.repo.parent            # an ANCESTOR of the repository
    elif where == "flat":
        root = private_roots.flat / "arsiv"
    else:
        root = holder_of(world) / "arsiv"
    root.mkdir(parents=True, exist_ok=True)
    before = snapshot(world.state_dir)

    with pytest.raises(finalization.ArchiveContainmentFailed):
        finalize(world, root, task_path=world.task_path)

    assert snapshot(world.state_dir) == before
    assert holder_of(world).exists()


def test_a_reparse_point_as_the_archive_root_is_refused(tmp_path):
    """A REAL junction on Windows and a REAL symlink on POSIX. The
    destination is where the operator's evidence lands; a link there sends
    it somewhere they did not name."""
    world, _result = failed(tmp_path)
    outside = tmp_path / "disarisi"
    outside.mkdir()
    trap = tmp_path / "tuzak-koku"
    if os.name == "nt":
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(trap),
                               str(outside)], capture_output=True)
        if done.returncode != 0:
            pytest.skip("bu makinede baglanti olusturulamiyor")
    else:
        os.symlink(outside, trap)
    before = snapshot(world.state_dir)

    with pytest.raises(finalization.ArchiveContainmentFailed):
        finalize(world, trap, task_path=world.task_path)

    assert snapshot(world.state_dir) == before
    assert list(outside.iterdir()) == [], "baglanti uzerinden yazildi"


# ---------------------------------------------------------------------
# 14-17  THE TASK MANIFEST AND THE SHIPMENT
# ---------------------------------------------------------------------

def test_a_task_manifest_with_another_digest_is_never_removed(tmp_path):
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    world.task_path.write_text('{"kurgu": "sonradan-degistirildi"}',
                               encoding="utf-8")

    with pytest.raises(finalization.TaskManifestMismatch):
        finalize(world, root, task_path=world.task_path)

    assert world.task_path.exists(), "digest uyusmazliginda silindi"
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("how", ["tracked", "staged"])
def test_a_tracked_or_staged_task_manifest_is_never_removed(tmp_path, how):
    """The manifest is the operator's input. Once it is in git it is
    THEIR file with THEIR history, and deleting it is a commit they did
    not ask for."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    world_module.legacy._git(world.repo, "add", "--", world.task_path.name)
    if how == "tracked":
        world_module.legacy._git(
            world.repo, "-c", "user.email=k@example.invalid",
            "-c", "user.name=Kurgu", "commit", "-qm", "kurgu-gorev")

    with pytest.raises(finalization.TaskManifestMismatch):
        finalize(world, root, task_path=world.task_path)

    assert world.task_path.exists()
    assert list(root.iterdir()) == []


def test_a_dirty_file_other_than_the_task_manifest_refuses_a_shipment(
        tmp_path):
    """Only for a shipment claim. A repository with uncommitted work
    cannot be described as having shipped the run."""
    world, _result = approved(tmp_path)
    shipped = commit_everything(world)
    (world.repo / "pipeline" / "kurgu.py").write_text("VALUE = 99\n",
                                                      encoding="utf-8")
    root = archive_root(tmp_path)

    with pytest.raises(finalization.ShipmentMismatch):
        finalize(world, root, task_path=world.task_path,
                 shipped_commit=shipped)

    assert list(root.iterdir()) == []


@pytest.mark.parametrize("bad", ["kisa", "Z" * 40, "0" * 39, "0" * 41])
def test_a_shipped_commit_that_is_not_a_full_sha_is_refused(tmp_path, bad):
    world, _result = approved(tmp_path)
    commit_everything(world)
    root = archive_root(tmp_path)

    with pytest.raises(finalization.ShipmentMismatch):
        finalize(world, root, task_path=world.task_path, shipped_commit=bad)

    assert list(root.iterdir()) == []


def test_a_shipped_commit_that_is_not_head_is_refused(tmp_path):
    world, _result = approved(tmp_path)
    commit_everything(world)
    root = archive_root(tmp_path)
    other = "1" * 40

    with pytest.raises(finalization.ShipmentMismatch):
        finalize(world, root, task_path=world.task_path,
                 shipped_commit=other)

    assert list(root.iterdir()) == []


def test_a_shipment_is_refused_when_origin_disagrees_with_head(tmp_path):
    """HEAD alone is a local claim. `shipped` means the commit is on the
    branch everybody else reads."""
    world, _result = approved(tmp_path)
    shipped = commit_everything(world)
    # move origin/main back a commit: HEAD is ahead, so nothing shipped
    world_module.legacy._git(world.repo, "update-ref",
                             "refs/remotes/origin/main", world.baseline)
    root = archive_root(tmp_path)

    with pytest.raises(finalization.ShipmentMismatch):
        finalize(world, root, task_path=world.task_path,
                 shipped_commit=shipped)

    assert list(root.iterdir()) == []


def test_a_failed_run_needs_no_shipped_commit(tmp_path):
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    outcome = finalize(world, root, task_path=world.task_path)
    assert outcome.shipment == finalization.Shipment.NOT_APPLICABLE


def test_the_ci_run_id_is_recorded_and_never_looked_up(tmp_path):
    """A closed record field. This operation makes no network call, so it
    cannot and must not invent a CI verdict."""
    import inspect

    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    outcome = finalize(world, root, task_path=world.task_path,
                       ci_run_id="32047794474")
    assert manifest_of(root, outcome)["ci_run_id"] == "32047794474"
    assert "ci_conclusion" not in manifest_of(root, outcome)
    source = inspect.getsource(finalization)
    for forbidden in ("urllib", "requests", "http", "socket", "urlopen"):
        assert forbidden not in source, f"ag cagrisi yolu: {forbidden}"


# ---------------------------------------------------------------------
# 18-21  WHAT THE ARCHIVE CONTAINS
# ---------------------------------------------------------------------

def test_absent_optional_documents_are_recorded_rather_than_invented(
        tmp_path):
    """MEASURED: a failed run has no `findings.json`. Its absence is a
    closed fact about the run, not a gap in the archive."""
    world, _result = failed(tmp_path)
    assert not (world.state_dir / "findings.json").exists(), \
        "senaryo kurulmadi: findings zaten yok olmali"
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path)
    manifest = manifest_of(root, outcome)

    assert "findings.json" in manifest["absent"]
    assert all(entry["name"] != "findings.json"
               for entry in manifest["files"])


def test_every_archived_file_is_byte_exact_and_digest_recorded(tmp_path):
    world, _result = failed(tmp_path)
    sources = {name: digest(world.state_dir / name)
               for name in state_files(world)}
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path)
    manifest = manifest_of(root, outcome)
    directory = archive_dir(root, outcome)

    assert outcome.archived_file_count == len(manifest["files"])
    for entry in manifest["files"]:
        copy = directory / entry["name"]
        assert copy.is_file(), f"arsivde yok: {entry['name']}"
        assert copy.stat().st_size == entry["size"]
        assert digest(copy) == entry["sha256"]
    # and the state documents really are the ones that were on disk.
    # `run.lock` is excluded on purpose: acquiring the lock rewrites its
    # record, so what the archive holds is the snapshot taken DURING this
    # call -- correct, but not comparable to a reading from before it.
    for name, want in sources.items():
        if name == "run.lock":
            continue
        entry = [e for e in manifest["files"]
                 if e["name"] == f"state/{name}"]
        assert entry and entry[0]["sha256"] == want, f"digest farkli: {name}"


def test_the_candidate_evidence_comes_from_the_change_authority(tmp_path):
    """Not a blind copy of the workspace. Only the files the change
    authority says the candidate changed, and nothing outside the
    manifest's allowed paths."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path)
    manifest = manifest_of(root, outcome)
    candidate = [e["name"] for e in manifest["files"]
                 if e["name"].startswith("candidate/")]

    assert candidate, "aday kaniti arsivlenmedi"
    for name in candidate:
        relative = name[len("candidate/"):]
        assert relative.startswith("pipeline/"), \
            f"izinli yol disinda aday dosyasi: {relative}"


def test_a_candidate_change_outside_the_allowed_paths_stops_everything(
        tmp_path):
    """SCOPE WIDENING. Something edited the workspace after the run ended,
    outside what the manifest allowed. The archive must not quietly carry
    it -- and the cleanup must not run, because a holder containing
    changes this call cannot account for is the last copy of them."""
    world, _result = failed(tmp_path)
    holder = holder_of(world)
    smuggled = holder / flat_workspace.IMPLEMENTER_DIRNAME / "contracts"
    smuggled.mkdir(parents=True, exist_ok=True)
    (smuggled / "kacak.py").write_text(f"# {SENTINEL}\n", encoding="utf-8")
    root = archive_root(tmp_path)

    with pytest.raises(finalization.CandidateScopeRefused):
        finalize(world, root, task_path=world.task_path)

    assert list(root.iterdir()) == [], "kapsam disi degisiklik arsivlendi"
    assert holder.exists(), "kapsam disi degisiklikle workspace silindi"
    assert (smuggled / "kacak.py").exists(), "kullanicinin bayti silindi"
    assert (world.state_dir / "state.json").exists()


def test_a_live_workspace_without_a_task_path_is_refused(tmp_path):
    """The change authority needs the exact manifest bytes to say which
    files are the candidate. Without them the answer is unknown, and
    unknown is a refusal rather than a blind copy of the holder."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)

    with pytest.raises(finalization.CandidateScopeRefused):
        finalize(world, root)

    assert list(root.iterdir()) == []
    assert holder_of(world).exists()


def test_a_state_document_swapped_for_a_link_is_refused(tmp_path):
    """A REAL link where a source document should be. The archive must
    copy the file, not whatever something else points it at."""
    world, _result = failed(tmp_path)
    outside = tmp_path / "disarisi"
    outside.mkdir()
    canary = outside / "kanarya.json"
    canary.write_text(f'{{"gizli": "{SENTINEL}"}}', encoding="utf-8")
    events = world.state_dir / "events.jsonl"
    events.unlink()
    if os.name == "nt":
        done = subprocess.run(["cmd", "/c", "mklink", str(events),
                               str(canary)], capture_output=True)
        if done.returncode != 0:
            pytest.skip("bu makinede sembolik baglanti olusturulamiyor")
    else:
        os.symlink(canary, events)
    root = archive_root(tmp_path)

    with pytest.raises(finalization.ArchiveContainmentFailed):
        finalize(world, root, task_path=world.task_path)

    assert canary.read_text(encoding="utf-8").find(SENTINEL) >= 0
    assert holder_of(world).exists()


def test_the_manifest_carries_closed_facts_and_no_prose(tmp_path):
    """The archive outlives the run, so what it records is what a future
    reader learns. Prompts, model text, absolute paths, usernames and
    environment values are not in that set."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)

    outcome = finalize(world, root, task_path=world.task_path)
    directory = archive_dir(root, outcome)
    text = (directory / finalization.MANIFEST_NAME).read_text(
        encoding="utf-8")
    manifest = json.loads(text)

    assert set(manifest) <= set(finalization.MANIFEST_FIELDS), \
        "beklenmeyen alan: " \
        f"{sorted(set(manifest) - set(finalization.MANIFEST_FIELDS))}"
    leaks = [str(tmp_path), str(world.repo), "KOSU:", "HEDEF:", "prompt",
             "stdout", "stderr", "objective", "kurgu hedef"]
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user:
        leaks.append(user)
    for forbidden in leaks:
        assert forbidden not in text, f"manifest sizdirdi: {forbidden!r}"
    assert ":\\" not in text and "/tmp/" not in text
    finalization.assert_manifest(manifest)


def test_the_manifest_is_validated_against_its_closed_schema(tmp_path):
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    outcome = finalize(world, root, task_path=world.task_path)
    manifest = manifest_of(root, outcome)
    finalization.assert_manifest(manifest)
    with pytest.raises(finalization.ArchiveVerificationFailed):
        finalization.assert_manifest({**manifest, "kurgu_ekstra": 1})
    with pytest.raises(finalization.ArchiveVerificationFailed):
        finalization.assert_manifest({k: v for k, v in manifest.items()
                                      if k != "run_id"})


# ---------------------------------------------------------------------
# 22-25  CLEANUP ORDER, READINESS, AND NOT LOSING BYTES
# ---------------------------------------------------------------------

def test_no_source_is_removed_before_the_archive_is_complete(tmp_path):
    """The order IS the safety argument, so it is measured rather than
    asserted: at the moment the first source is unlinked, the completion
    record must already exist."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    observed = []
    real = finalization._unlink_verified

    def watching(*args, **kwargs):
        names = sorted(p.name for p in root.iterdir())
        complete = [(root / n / finalization.COMPLETE_NAME).exists()
                    for n in names]
        observed.append(all(complete) and bool(complete))
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_unlink_verified", watching)
        finalize(world, root, task_path=world.task_path)

    assert observed, "senaryo kurulmadi: hic dosya kaldirilmadi"
    assert all(observed), "arsiv tamamlanmadan kaynak silindi"


def test_a_source_that_drifts_before_its_removal_stops_the_cleanup(tmp_path):
    """Between the archive and the unlink there is a window. A file that
    changed in it is not the file that was proven, so it stays."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    events = world.state_dir / "events.jsonl"
    real = finalization._unlink_verified
    fired = []

    def drifting(directory, name, expected, **kwargs):
        if name == "events.jsonl" and not fired:
            fired.append(1)
            events.write_bytes(b'{"kurgu": "silmeden-once-degisti"}\n')
        return real(directory, name, expected, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_unlink_verified", drifting)
        with pytest.raises(finalization.StateCleanupFailed):
            finalize(world, root, task_path=world.task_path)

    assert fired, "senaryo kurulmadi: sapma tetiklenmedi"
    assert events.exists(), "sapmis dosya silindi"


def test_the_workspace_goes_only_through_the_public_remove_seam(tmp_path):
    """No second delete authority. `flat_workspace.remove` is the only
    thing that may take a holder, because it is the only thing that
    checks the record authorises it."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    calls = []
    real = flat_workspace.remove

    def counting(repo, *, state_dir, workspace_id):
        calls.append(workspace_id)
        return real(repo, state_dir=state_dir, workspace_id=workspace_id)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization.flat_workspace, "remove", counting)
        outcome = finalize(world, root, task_path=world.task_path)

    assert len(calls) == 1, f"public seam bir kez cagrilmadi: {calls}"
    assert outcome.removed_workspace is True


def test_finalization_never_reaches_the_model_adapters():
    """The same invariant the runner carries, extended to the module it
    now delegates to. `execution` and `audit` are where money is spent and
    where model output arrives; a cleanup seam that could reach either is
    a cleanup seam that could start a run."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(finalization))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[-1] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {alias.name for alias in node.names}
    for forbidden in ("execution", "audit", "plan_auth", "acceptance"):
        assert forbidden not in imported, \
            f"finalize model yoluna ulasiyor: {forbidden}"


def test_the_public_seam_is_reachable_from_the_runner(tmp_path):
    """One call, from the module an operator already imports -- and it
    takes no `binaries`, because it starts no process of any kind."""
    import inspect

    signature = inspect.signature(runner.finalize)
    assert set(signature.parameters) == {
        "repo", "archive_root", "expected_run_id", "task_path",
        "shipped_commit", "ci_run_id"}
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for parameter in signature.parameters.values()), \
        "her parametre keyword-only olmali"
    assert "binaries" not in signature.parameters
    assert runner.FinalizeResult is finalization.FinalizeResult
    assert runner.FinalizeRefused is finalization.FinalizeRefused


def test_the_result_is_frozen_and_closed(tmp_path):
    """A caller must not be able to edit the record of what happened, and
    the record must not carry an absolute path."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    outcome = finalize(world, root, task_path=world.task_path)

    with pytest.raises(Exception):
        outcome.status = "kurgu"
    assert not hasattr(outcome, "__dict__"), "slotted degil"
    for value in vars(type(outcome)).get("__slots__", ()):
        text = str(getattr(outcome, value))
        assert str(tmp_path) not in text, f"mutlak yol tasindi: {value}"


def test_no_broad_delete_primitive_appears_in_the_module():
    """A pin on the source. `rmtree`, a glob delete or a recursive walk
    would each be a new delete authority, and the whole design rests on
    there being exactly one."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(finalization))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    for forbidden in ("rmtree", "glob", "rglob", "iglob", "removedirs",
                      "walk", "shutil"):
        assert forbidden not in names, f"genis silme yolu: {forbidden}"


def test_a_real_preflight_completes_after_a_finalize(tmp_path):
    """The point of the whole operation. A terminal run's leftovers used
    to make the next `preflight` refuse, and this is the assertion that
    the repository is genuinely ready for a new task."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    finalize(world, root, task_path=world.task_path)

    # The NEXT job, in the SAME checkout -- which is the situation a daily
    # user is in. The manifest is rebuilt for this repository's current
    # HEAD and given the name the dirty-tree allowlist already knows.
    payload = dict(world.task, baseline_sha=head_of(world.repo),
                   objective="kurgu ikinci hedef")
    nxt = world.repo / "kurgu-task.json"
    nxt.write_text(json.dumps(payload), encoding="utf-8")

    outcome = runner.preflight(nxt, repo=world.repo,
                              binaries=world.binaries)

    assert outcome.stop_reason == contract.StopReason.COMPLETED, \
        f"finalize sonrasi preflight engellendi: {outcome.stop_reason}"
    assert outcome.state == contract.State.PREFLIGHT


def test_a_failed_cleanup_never_loses_the_operators_bytes(tmp_path):
    """The worst case: cleanup breaks halfway. Whatever is still on disk
    plus what is in the archive must together account for every byte that
    existed when the call started."""
    world, _result = failed(tmp_path)
    root = archive_root(tmp_path)
    before = snapshot(world.state_dir)
    real = finalization._unlink_verified
    calls = []

    def breaking(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("kurgu silme arizasi")
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(finalization, "_unlink_verified", breaking)
        with pytest.raises(finalization.StateCleanupFailed):
            finalize(world, root, task_path=world.task_path)

    archived = snapshot(archive_dir(root, types.SimpleNamespace(
        archive_name=sorted(p.name for p in root.iterdir())[0])))
    still = snapshot(world.state_dir)
    for name, data in before.items():
        in_archive = any(value == data for value in archived.values())
        assert name in still or in_archive, f"bayt kayboldu: {name}"
