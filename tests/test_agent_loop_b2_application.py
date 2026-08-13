"""PACKAGE B2B-C2 -- transactional application to the main checkout.

ONE question: does a candidate that was ALREADY verified and ALREADY
accepted reach the operator's checkout as exactly itself and nothing
else, with a write-ahead journal in front of every repository write and a
PROVEN rollback behind every failure.

WHAT IS DELIBERATELY NOT HERE. No evaluator, no approval decision, no
state-machine move, no `git add`, no commit, no push, no runner. This
package moves bytes into a working tree and can prove afterwards which
bytes moved.

THE WORLD IS REAL AND THROWAWAY. A real repository, a real D3A flat
workspace, a real sibling apply root, real junctions and real symlinks --
and never the project's own checkout. `apply_accepted_candidate` is never
called against anything outside `tmp_path`.

EVERY REFUSAL TEST PROVES ITS SETUP. A test that died at an earlier gate
is red for the wrong reason, and a "nothing was written" assertion is
worthless unless the attack it names was actually built -- so each
scenario asserts the junction exists, the drift is real or the collision
is present before it claims the refusal. The counter is MECHANICAL:
`rename_child` is the only primitive that mutates the main checkout, so
counting it counts main-tree writes exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import test_agent_loop_b2_acceptance as world_module
import test_agent_loop_b2_changes as legacy
from tools.agent_loop import (acceptance, acceptance_workspace, application,
                              application_transport, changes, contract,
                              flat_workspace, fs_evidence, state)
from tools.agent_loop import process as process_module

RUN = legacy.RUN
SENTINEL = "GIZLI-ICERIK-9182-OMEGA"


# ---------------------------------------------------------------------
# THE WORLD
# ---------------------------------------------------------------------

def build_world(tmp_path, index=0, **overrides):
    """A real repository, a real flat workspace at its baseline, a bound
    task manifest and the identity tuple the apply seam takes.

    Reused from the acceptance battery rather than rebuilt: two fixtures
    that build "the same" world drift, and the drift shows up as a test
    passing for a reason the other file already disproved."""
    world = world_module.build_world(tmp_path, index, **overrides)
    _match_checkout_to_its_baseline(world)
    return world


def _match_checkout_to_its_baseline(world):
    """Give the operator's checkout the line endings its own baseline has.

    A REAL checkout agrees with its blobs. THIS FIXTURE'S DID NOT, and
    for a reason that has nothing to do with the code under test: the
    shared builder writes through `Path.write_text`, which turns every
    `\\n` into `\\r\\n` on Windows, while `git add` normalises it straight
    back out again under this machine's system-level `core.autocrlf`. So
    every tracked file differed from the baseline it was made from, the
    drift precondition correctly refused all of them, and every scenario
    below would have been a test of the fixture.

    The refusal itself is NOT worked around anywhere -- it is exactly
    what `test_a_target_that_drifted_from_the_baseline_is_never_touched`
    pins, and a checkout whose bytes really do differ from its baseline
    is still refused. This only removes a difference the fixture
    invented."""
    for target in world.repo.rglob("*"):
        if not target.is_file() or ".git" in target.parts:
            continue
        if target == world.task:
            continue                  # its digest is already bound
        data = target.read_bytes()
        if b"\r\n" in data:
            target.write_bytes(data.replace(b"\r\n", b"\n"))
    assert (world.repo / "pipeline" / "kurgu.py").read_bytes() == \
        (world.reference / "pipeline" / "kurgu.py").read_bytes(), \
        "senaryo kurulmadi: checkout tabanla ayni degil"


def edit(world, relative, text):
    target = world.tree / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def drop(world, relative):
    (world.tree / relative).unlink()


def verified_for(world, **overrides):
    return world_module.verified_for(world, **overrides)


def accept(world, **overrides):
    """THE REAL GATE. Commands actually run, in a real disposable mirror,
    and the persisted receipt is a side effect of that having happened.

    NO TEST IN THIS FILE FABRICATES A PASSING RECEIPT. The previous
    revision had a helper that built an `AcceptanceReport` with the
    public constructor and handed it to the apply seam, and every success
    path here used it -- so the whole battery was green against a
    candidate no gate had ever seen. That is the defect this revision
    closes, and a helper that could still spell a receipt would quietly
    reopen it, which is why there is no such helper and no production
    seam that writes one."""
    settings = {"timeout_seconds": 300, "max_output_bytes": 65536}
    settings.update(overrides)
    return acceptance.run_acceptance(
        **world.identity, verified_changes=verified_for(world), **settings)


def forged_report(real, **overrides):
    """A report built with the EXACT public constructor.

    Only ever used on refusal paths, and only to make one field wrong at
    a time: the point of each such test is that the object is exactly the
    right type and still is not authority."""
    fields = {field: getattr(real, field)
              for field in acceptance.AcceptanceReport.__slots__}
    fields.update(overrides)
    return acceptance.AcceptanceReport(**fields)


def apply_it(world, **overrides):
    settings = {}
    if "verified_changes" not in overrides:
        settings["verified_changes"] = verified_for(world)
    if "acceptance_report" not in overrides:
        settings["acceptance_report"] = accept(world)
    settings.update(overrides)
    return application.apply_accepted_candidate(**world.identity, **settings)


def receipt_of(world):
    return acceptance.read_receipt(world.state_dir)


def overwrite_receipt(world, payload):
    """Tamper with the persisted receipt, as an attacker would: raw bytes
    straight onto the file, never through a production seam."""
    acceptance.receipt_path(world.state_dir).write_bytes(
        json.dumps(payload).encode("utf-8") if payload is not None
        else b"{ bozuk")


def main_view(repo):
    """The operator's checkout as SEMANTIC content, through the module
    that owns that question."""
    key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    return changes.main_projection(repo, key=key,
                                   policy=changes.freeze_main_policy(repo))


def difference(before, after):
    return tuple((item.path, item.kind)
                 for item in changes.main_difference(before, after))


def read(path):
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# THE FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def private_roots(tmp_path, monkeypatch):
    """Every test gets its OWN flat root and mirror root, so nothing here
    can list or delete a directory a real agent loop is using, and every
    residue assertion measures this test only.

    The APPLY root is deliberately NOT redirected: it is a sibling of the
    repository by contract, the repository lives in `tmp_path`, and a
    redirected root would stop the containment rule from being tested."""
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


@pytest.fixture
def writes(monkeypatch):
    """The MECHANICAL main-write counter.

    `rename_child` is the only primitive that moves an object into or out
    of the operator's checkout -- staging writes land in the holder
    through `create_child_file` -- so its call count IS the number of
    main-tree mutations attempted. Counting an outcome instead would let
    a refusal that wrote and undid look identical to one that never
    wrote."""
    seen = []
    real = application_transport.rename_child

    def recorder(source, source_name, target, target_name):
        seen.append(target_name)
        return real(source, source_name, target, target_name)

    monkeypatch.setattr(application.transport, "rename_child", recorder)
    return seen


@pytest.fixture
def gate_runs(private_roots, monkeypatch):
    """Counts processes launched INSIDE the acceptance mirror.

    Narrowed by working directory on purpose: `flat_workspace.create()`
    runs git through the same seam with the repository as its cwd, so a
    recorder that counted every launch would make "the gate never ran"
    measure the fixture instead of the claim."""
    seen = []
    real_popen = process_module.subprocess.Popen
    mirror_root = str(private_roots.mirror).casefold()

    def recorder(argv, **kwargs):
        if mirror_root in str(kwargs.get("cwd", "")).casefold():
            seen.append(argv[0])
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(process_module.subprocess, "Popen", recorder)
    return seen


@pytest.fixture(autouse=True)
def no_apply_residue(tmp_path):
    """Nothing this turn created under the sibling apply root may survive
    it -- on success, on refusal, on rollback or after recovery."""
    yield
    for repo in tmp_path.glob("kurgu-depo-*"):
        root = application.apply_root_for(repo)
        if not root.exists():
            continue
        leftover = sorted(item.name for item in root.iterdir()
                          if item.name.startswith(application.HOLDER_PREFIX))
        assert leftover == [], f"uygulama artigi kaldi: {leftover}"


# ---------------------------------------------------------------------
# 1-4  THE RECEIPT AND THE BINDINGS
# ---------------------------------------------------------------------

def test_the_acceptance_report_binds_the_candidate_it_tested(tmp_path):
    """A report that names a run but not a CANDIDATE lets an edit made
    after the gate ran travel on the gate's own receipt.

    Run for real, once, because these three fields have to be filled by
    `run_acceptance` itself -- every other test in this file constructs a
    report, and a constructed one proves nothing about the producer."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    candidate = changes.derive_candidate_changes(**world.identity)
    report = accept(world)
    assert report.passed is True, "kabul kosusu gecmeliydi"
    assert report.manifest_digest == world.digest
    assert report.candidate_fingerprint == candidate.fingerprint
    assert report.command_plan_digest == acceptance.command_plan_digest(
        candidate.acceptance_commands)
    # the plan digest is a digest of the PLAN, so a different plan is a
    # different digest -- otherwise it is a constant wearing a hash
    assert acceptance.command_plan_digest(
        (("pytest_full", ()),)) != report.command_plan_digest


def test_a_candidate_edited_after_acceptance_never_reaches_the_checkout(
        tmp_path, writes):
    """The receipt is not a filesystem authority. A candidate changed
    between the gate and the apply is a DIFFERENT candidate, and the
    fresh derivation is what says so.

    EVERY OTHER GATE IS DELIBERATELY SATISFIED. The realistic shape --
    a stale change set AND a stale receipt AND a stale report -- would be
    a USELESS test: three gates would answer, the earliest would win, and
    deleting this one would leave it green. So acceptance is run TWICE,
    the second time on the candidate actually on disk, and only the
    report's own fingerprint is rolled back to the first run's. The
    sentence is asserted, because several gates raise this type."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    stale = accept(world)
    # the edit the first receipt cannot possibly have covered, followed
    # by a REAL second gate so the persisted receipt is current
    edit(world, "pipeline/yeni.py", f"VALUE = 8  # {SENTINEL}\n")
    current = accept(world)
    fresh = changes.derive_candidate_changes(**world.identity)
    assert stale.candidate_fingerprint != fresh.fingerprint, \
        "senaryo kurulmadi: aday degismemis"
    assert receipt_of(world)["candidate_fingerprint"] == fresh.fingerprint, \
        "senaryo kurulmadi: makbuz da bayat"
    report = forged_report(
        current, candidate_fingerprint=stale.candidate_fingerprint)

    before = main_view(world.repo)
    with pytest.raises(application.CandidateNotAccepted) as raised:
        apply_it(world, acceptance_report=report)
    assert str(raised.value) == "kabul raporu baska bir adayi adliyor"
    assert writes == [], "ret oncesi ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()
    assert not (world.repo / "pipeline" / "yeni.py").exists()


@pytest.mark.parametrize("break_it", ["altsinif", "gecmedi", "eksik-sonuc",
                                      "yabanci-kosu", "plan-ozeti",
                                      "gorev-ozeti", "sonuc-turu"])
def test_a_receipt_that_is_not_an_acceptance_report_is_refused(
        tmp_path, writes, break_it):
    """Seven separate ways a receipt can be the wrong one, and none of
    them may reach a repository write.

    EXACT TYPE included: a subclass answers every comparison below while
    being free to lie about any of them, and the report is the whole
    authority for "this candidate was tested"."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    # a REAL gate first, so the persisted receipt is valid and each case
    # below is refused by the report check rather than by a missing one
    real = accept(world)
    passing = real.command_results[0]

    if break_it == "altsinif":
        class Sahte(acceptance.AcceptanceReport):
            __slots__ = ()
        report = Sahte(**{field: getattr(real, field)
                          for field in acceptance.AcceptanceReport.__slots__})
        assert isinstance(report, acceptance.AcceptanceReport), \
            "senaryo kurulmadi: altsinif degil"
    elif break_it == "gecmedi":
        report = forged_report(real, passed=False)
    elif break_it == "eksik-sonuc":
        report = forged_report(real, command_results=())
    elif break_it == "yabanci-kosu":
        report = forged_report(real, run_id="baska-kosu-1")
    elif break_it == "plan-ozeti":
        report = forged_report(real, command_plan_digest="0" * 64)
    elif break_it == "gorev-ozeti":
        report = forged_report(real, manifest_digest="0" * 64)
    else:
        forged = types.SimpleNamespace(
            command_id=passing.command_id, passed=True, exit_code=0,
            duration_ms=1, stdout_bytes=0, stderr_bytes=0,
            event=passing.event)
        report = forged_report(real, command_results=(forged,))

    before = main_view(world.repo)
    with pytest.raises((application.CandidateNotAccepted,
                        application.ApplicationRefused)):
        apply_it(world, acceptance_report=report)
    assert writes == [], "ret oncesi ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()


@pytest.mark.parametrize("break_it", ["gorev-degisti", "yabanci-taban",
                                      "yabanci-workspace", "durum-yok",
                                      "gorev-disarida"])
def test_the_exact_task_state_and_workspace_bindings_are_required(
        tmp_path, writes, break_it):
    """Everything the apply is bound to is re-asserted here, not carried
    in from whoever called. A caller that could substitute one of these
    could apply a verified candidate to a repository it was never
    verified against."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    # the gate runs FIRST and honestly; every break below happens after
    # it, so each case is refused by the binding it names
    verified, report = verified_for(world), accept(world)
    identity = dict(world.identity)

    if break_it == "gorev-degisti":
        world.task.write_text(world.task.read_text(encoding="utf-8") + " ",
                              encoding="utf-8")
        assert hashlib.sha256(world.task.read_bytes()).hexdigest() \
            != world.digest, "senaryo kurulmadi: gorev ayni"
    elif break_it == "yabanci-taban":
        identity["baseline_sha"] = "b" * 40
    elif break_it == "yabanci-workspace":
        identity["workspace_id"] = secrets.token_hex(16)
    elif break_it == "durum-yok":
        (world.state_dir / state.BINDING_FILENAME).unlink()
        assert not (world.state_dir / state.BINDING_FILENAME).exists(), \
            "senaryo kurulmadi: baglama duruyor"
    else:
        outside = tmp_path / "disarida-task.json"
        outside.write_bytes(world.task.read_bytes())
        identity["task_path"] = outside

    before = main_view(world.repo)
    with pytest.raises(application.ApplicationError):
        application.apply_accepted_candidate(
            **identity, verified_changes=verified, acceptance_report=report)
    assert writes == [], "ret oncesi ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()


# ---------------------------------------------------------------------
# THE PERSISTED RECEIPT IS THE AUTHORITY (B2B-C2-R1)
# ---------------------------------------------------------------------

def test_a_report_built_by_hand_without_a_gate_is_refused(tmp_path, writes,
                                                          gate_runs):
    """THE P0 THIS REVISION EXISTS FOR, measured on the commit before it.

    `AcceptanceReport` is a public dataclass, so its constructor is
    public. On the previous commit a report built by hand -- exact type,
    every digest honestly derived from fresh evidence -- applied a
    candidate to the operator's checkout with ZERO acceptance commands
    launched and ZERO mirrors created. `type(report) is AcceptanceReport`
    proves the CLASS; it says nothing about whether the gate ran.

    Nothing here is stubbed to make the forgery work: it is the real
    class, filled with the truth, and the only thing missing is a run."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", f"VALUE = 7  # {SENTINEL}\n")
    candidate = changes.derive_candidate_changes(**world.identity)
    report = acceptance.AcceptanceReport(
        run_id=candidate.run_id, workspace_id=candidate.workspace_id,
        baseline_sha=candidate.baseline_sha, passed=True,
        command_results=tuple(
            acceptance.AcceptanceCommandResult(
                command_id=command_id, passed=True, exit_code=0,
                duration_ms=1, stdout_bytes=0, stderr_bytes=0,
                event=contract.EventCode.ACCEPTANCE_FINISHED)
            for command_id, _ in candidate.acceptance_commands),
        total_duration_ms=5, event=contract.EventCode.ACCEPTANCE_FINISHED,
        manifest_digest=world.digest,
        candidate_fingerprint=candidate.fingerprint,
        command_plan_digest=acceptance.command_plan_digest(
            candidate.acceptance_commands),
        receipt_id="a" * 32)
    assert type(report) is acceptance.AcceptanceReport, \
        "senaryo kurulmadi: exact constructor kullanilmadi"
    assert gate_runs == [], "senaryo kurulmadi: kabul komutu calismis"
    assert not acceptance.receipt_path(world.state_dir).exists(), \
        "senaryo kurulmadi: makbuz zaten var"

    before = main_view(world.repo)
    with pytest.raises(application.CandidateNotAccepted):
        apply_it(world, acceptance_report=report)

    assert gate_runs == [], "ret sirasinda kabul komutu calisti"
    assert writes == [], "makbuzsuz rapor ana agaca yazdi"
    assert difference(before, main_view(world.repo)) == ()
    assert not (world.repo / "pipeline" / "yeni.py").exists()
    assert application.find_pending_applications(world.repo) == (), \
        "ret oncesi journal olustu"


def test_a_real_gate_writes_pending_then_passed_and_the_apply_follows(
        tmp_path, gate_runs):
    """The lifecycle, in order, with the ordering itself measured.

    `pending` has to be on disk BEFORE the first acceptance process
    starts -- that is what makes a crash fail closed instead of leaving
    an older green result standing -- and `passed` only after every
    command has finished."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    seen = []
    real_popen = process_module.subprocess.Popen
    mirror_root = str(tmp_path / "ayna-koku").casefold()

    def observer(argv, **kwargs):
        if mirror_root in str(kwargs.get("cwd", "")).casefold():
            try:
                seen.append(receipt_of(world)["status"])
            except acceptance.AcceptanceError:
                seen.append(None)
        return real_popen(argv, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(process_module.subprocess, "Popen", observer)
        report = accept(world)

    assert seen, "senaryo kurulmadi: hic kabul sureci calismadi"
    assert set(seen) == {acceptance.STATUS_PENDING}, \
        f"ilk surecten once pending yazilmamis: {seen}"
    persisted = receipt_of(world)
    assert persisted["status"] == acceptance.STATUS_PASSED
    assert persisted["receipt_id"] == report.receipt_id
    assert persisted["command_count"] == len(report.command_results)

    outcome = apply_it(world, acceptance_report=report)
    assert outcome.applied_files == ("pipeline/yeni.py",)
    assert read(world.repo / "pipeline" / "yeni.py") == "VALUE = 7\n"


@pytest.mark.parametrize("state_of", ["yok", "pending", "failed", "bilinmeyen"])
def test_a_receipt_that_does_not_say_passed_is_refused(tmp_path, writes,
                                                       state_of):
    """Only `passed` is permission. `pending` is a run that started and
    never finished -- exactly what a crash leaves -- and treating it as
    anything but a refusal is how an interrupted gate becomes a green
    one."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    report = accept(world)
    persisted = receipt_of(world)
    assert persisted["status"] == acceptance.STATUS_PASSED, "senaryo kurulmadi"

    if state_of == "yok":
        acceptance.receipt_path(world.state_dir).unlink()
    else:
        overwrite_receipt(world, {**persisted, "status": state_of})

    before = main_view(world.repo)
    with pytest.raises(application.CandidateNotAccepted):
        apply_it(world, acceptance_report=report)
    assert writes == [], "gecmemis makbuzla ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()
    assert application.find_pending_applications(world.repo) == ()


@pytest.mark.parametrize("field", ["repo_id", "run_id", "workspace_id",
                                   "baseline_sha", "manifest_digest",
                                   "candidate_fingerprint",
                                   "command_plan_digest", "command_count",
                                   "receipt_id"])
def test_a_receipt_that_names_something_else_is_refused(tmp_path, writes,
                                                        field):
    """Every identity the receipt carries is compared against evidence
    this call derived for itself. A receipt that is valid, `passed` and
    about a DIFFERENT run is the most plausible forgery there is -- it
    can simply be copied from another state directory."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    report = accept(world)
    persisted = receipt_of(world)
    replacement = {"command_count": persisted["command_count"] + 1,
                   "run_id": "baska-kosu-1"}.get(
        field, "b" * len(str(persisted[field])))
    assert replacement != persisted[field], "senaryo kurulmadi: alan ayni"
    overwrite_receipt(world, {**persisted, field: replacement})

    before = main_view(world.repo)
    with pytest.raises(application.CandidateNotAccepted):
        apply_it(world, acceptance_report=report)
    assert writes == [], "yabanci makbuzla ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()


@pytest.mark.parametrize("shape", ["bozuk-json", "sema-disi", "baglanti"])
def test_a_receipt_that_is_corrupt_or_a_link_is_refused(tmp_path, writes,
                                                        shape):
    """A receipt is read through a HANDLE, not resolved as a path.

    The link case is a real filesystem object pointing at a real file
    outside the state directory: without the no-follow open, the gate
    would happily read whatever it addresses."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    report = accept(world)
    persisted = receipt_of(world)
    target = acceptance.receipt_path(world.state_dir)
    outside = tmp_path / "disarisi"
    outside.mkdir()

    if shape == "bozuk-json":
        overwrite_receipt(world, None)
    elif shape == "sema-disi":
        overwrite_receipt(world, {**persisted, "kurgu_alani": 1})
    else:
        planted = outside / "kanarya-makbuz.json"
        planted.write_bytes(json.dumps(persisted).encode("utf-8"))
        target.unlink()
        if os.name == "nt":
            done = subprocess.run(["cmd", "/c", "mklink", str(target),
                                   str(planted)], capture_output=True)
            if done.returncode != 0:
                pytest.skip("bu makinede sembolik baglanti olusturulamiyor")
            planted_ok = bool(os.lstat(target).st_file_attributes & 0x400)
        else:
            os.symlink(planted, target)
            planted_ok = os.path.islink(target)
        assert planted_ok, "senaryo kurulmadi: makbuz bir baglanti degil"
        assert json.loads(target.read_text(encoding="utf-8")) == persisted, \
            "senaryo kurulmadi: baglanti gecerli bir makbuzu gostermiyor"

    before = main_view(world.repo)
    with pytest.raises(application.CandidateNotAccepted):
        apply_it(world, acceptance_report=report)
    assert writes == [], "bozuk makbuzla ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()
    if shape == "baglanti":
        assert json.loads(
            (outside / "kanarya-makbuz.json").read_text(encoding="utf-8")) \
            == persisted, "disaridaki kanarya degisti"


def test_a_failed_second_gate_invalidates_the_earlier_passed_receipt(tmp_path,
                                                                     writes):
    """A green result may never be inherited.

    The second run's commands fail, so the receipt that was `passed`
    becomes `failed` -- and the FIRST run's report, which is still a
    perfectly valid object, stops being usable with it. Keeping the old
    receipt around and only refusing the new report would leave exactly
    the hole this closes."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    first = accept(world)
    assert receipt_of(world)["status"] == acceptance.STATUS_PASSED

    # the same candidate path, now carrying a test that fails
    edit(world, "pipeline/test_gecer.py",
         "def test_kurgu():\n    assert False\n")
    second = accept(world)
    assert second.passed is False, "senaryo kurulmadi: ikinci kapi gecti"
    assert receipt_of(world)["status"] == acceptance.STATUS_FAILED

    before = main_view(world.repo)
    for report in (first, second):
        with pytest.raises(application.CandidateNotAccepted):
            apply_it(world, acceptance_report=report)
    assert writes == [], "basarisiz ikinci kapidan sonra yazildi"
    assert difference(before, main_view(world.repo)) == ()


def test_an_interrupt_during_the_gate_leaves_no_passed_receipt(tmp_path,
                                                               writes):
    """An operator's Ctrl-C during acceptance is not a pass.

    The receipt is already `pending` when the interrupt lands, so the
    guard holds even if nothing further can be written -- and the
    interrupt itself travels out exactly as it arrived."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    good = accept(world)
    assert receipt_of(world)["status"] == acceptance.STATUS_PASSED, \
        "senaryo kurulmadi: once gecen bir makbuz gerekiyor"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(acceptance, "_measure",
                      lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            accept(world)

    assert receipt_of(world)["status"] != acceptance.STATUS_PASSED, \
        "kesme sonrasi makbuz hala gecmis gorunuyor"
    before = main_view(world.repo)
    with pytest.raises(application.CandidateNotAccepted):
        apply_it(world, acceptance_report=good)
    assert writes == [], "kesilmis kapidan sonra ana agaca yazildi"
    assert difference(before, main_view(world.repo)) == ()


def test_a_receipt_that_cannot_be_persisted_blocks_the_report(tmp_path,
                                                              gate_runs):
    """A status this lifecycle cannot demonstrate on disk is a status
    nothing downstream can verify, so no report is built at all.

    Two moments matter and both are covered: if the PENDING write fails
    no acceptance process may start, and if the final write fails the
    run may not hand back a passing report."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    real_write = state.write_json_atomically

    def refuse_first(path, payload, schema, what):
        if str(path).endswith(acceptance.RECEIPT_NAME):
            raise OSError("kurgu makbuz yazma arizasi")
        return real_write(path, payload, schema, what)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(acceptance.state_module, "write_json_atomically",
                      refuse_first)
        with pytest.raises(acceptance.AcceptanceError):
            accept(world)
    assert gate_runs == [], "makbuz yazilamadan kabul sureci baslatildi"
    assert not acceptance.receipt_path(world.state_dir).exists()

    calls = []

    def refuse_final(path, payload, schema, what):
        if str(path).endswith(acceptance.RECEIPT_NAME):
            calls.append(payload["status"])
            if payload["status"] != acceptance.STATUS_PENDING:
                raise OSError("kurgu makbuz guncelleme arizasi")
        return real_write(path, payload, schema, what)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(acceptance.state_module, "write_json_atomically",
                      refuse_final)
        with pytest.raises(acceptance.AcceptanceError):
            accept(world)
    assert acceptance.STATUS_PASSED in calls, \
        "senaryo kurulmadi: son guncelleme hic denenmedi"
    assert receipt_of(world)["status"] == acceptance.STATUS_PENDING, \
        "guncellenemeyen makbuz gecmis olarak kaldi"


# ---------------------------------------------------------------------
# 5-8  THE THREE KINDS, AND ALL OF THEM AT ONCE
# ---------------------------------------------------------------------

def test_an_added_file_is_applied(tmp_path):
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", f"VALUE = 7  # {SENTINEL}\n")
    before = main_view(world.repo)

    report = apply_it(world)

    target = world.repo / "pipeline" / "yeni.py"
    assert target.is_file()
    assert read(target) == f"VALUE = 7  # {SENTINEL}\n"
    assert report.applied_files == ("pipeline/yeni.py",)
    assert (report.added, report.modified, report.deleted) == (1, 0, 0)
    assert report.rollback_performed is False
    assert difference(before, main_view(world.repo)) == (
        ("pipeline/yeni.py", changes.ADDED),)


def test_a_modified_file_is_applied_with_its_mode(tmp_path):
    """The bytes AND the recorded permission bits. A mode dropped on the
    way in is a script that arrives unrunnable, and the change set has
    carried the field since B2B-A precisely so it can be restored."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", f"VALUE = 9  # {SENTINEL}\n")
    candidate = changes.derive_candidate_changes(**world.identity)
    assert [item.kind for item in candidate.changes] == [changes.MODIFIED], \
        "senaryo kurulmadi: degisiklik MODIFIED degil"
    before = main_view(world.repo)

    report = apply_it(world)

    target = world.repo / "pipeline" / "kurgu.py"
    assert read(target) == f"VALUE = 9  # {SENTINEL}\n"
    assert (report.added, report.modified, report.deleted) == (0, 1, 0)
    assert difference(before, main_view(world.repo)) == (
        ("pipeline/kurgu.py", changes.MODIFIED),)
    if os.name != "nt":
        expected = int(candidate.changes[0].mode, 8) & 0o7777
        assert target.stat().st_mode & 0o7777 == expected


def test_a_deleted_file_is_removed(tmp_path):
    world = build_world(tmp_path)
    drop(world, "pipeline/silinecek.py")
    target = world.repo / "pipeline" / "silinecek.py"
    assert target.is_file(), "senaryo kurulmadi: hedef zaten yok"
    before = main_view(world.repo)

    report = apply_it(world)

    assert not target.exists()
    assert (report.added, report.modified, report.deleted) == (0, 0, 1)
    assert difference(before, main_view(world.repo)) == (
        ("pipeline/silinecek.py", changes.DELETED),)


def test_a_mixed_change_set_lands_exactly_and_nothing_else_moves(tmp_path):
    """The whole point, stated as one assertion: the semantic difference
    the operator's checkout shows across the call IS the candidate."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    edit(world, "pipeline/alt/derin.py", "VALUE = 8\n")   # a NEW parent
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    drop(world, "pipeline/silinecek.py")
    before = main_view(world.repo)

    report = apply_it(world)

    assert report.applied_files == ("pipeline/alt/derin.py",
                                    "pipeline/kurgu.py",
                                    "pipeline/silinecek.py",
                                    "pipeline/yeni.py")
    assert (report.added, report.modified, report.deleted) == (2, 1, 1)
    assert difference(before, main_view(world.repo)) == (
        ("pipeline/alt/derin.py", changes.ADDED),
        ("pipeline/kurgu.py", changes.MODIFIED),
        ("pipeline/silinecek.py", changes.DELETED),
        ("pipeline/yeni.py", changes.ADDED))
    assert read(world.repo / "pipeline" / "alt" / "derin.py") == "VALUE = 8\n"
    # the untouched sibling is the control: a copy-the-tree implementation
    # would pass every assertion above and still rewrite this file
    assert read(world.repo / "pipeline" / "gizli" / "sir.py") == "VALUE = 3\n"


# ---------------------------------------------------------------------
# 9-13  THE MAIN CHECKOUT IS NOT A BLANK PAGE
# ---------------------------------------------------------------------

def test_an_added_target_that_already_exists_is_never_overwritten(tmp_path,
                                                                  writes):
    """ADDED means the operator does not have this file. If they do, the
    file is theirs -- an untracked note, a local scratch file -- and the
    only safe answer is to refuse the whole application."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    victim = world.repo / "pipeline" / "yeni.py"
    victim.write_text(f"KULLANICI DOSYASI {SENTINEL}\n", encoding="utf-8")
    assert victim.is_file(), "senaryo kurulmadi: hedef yok"
    before = main_view(world.repo)

    with pytest.raises(application.MainCheckoutMismatch):
        apply_it(world)

    assert read(victim) == f"KULLANICI DOSYASI {SENTINEL}\n"
    assert difference(before, main_view(world.repo)) == ()
    assert writes == [], "carpismaya ragmen ana agaca yazildi"


@pytest.mark.parametrize("kind", ["modified", "deleted"])
def test_a_target_that_drifted_from_the_baseline_is_never_touched(tmp_path,
                                                                  writes, kind):
    """A MODIFIED or DELETED record describes a transformation OF THE
    BASELINE. If the operator's copy is no longer the baseline, applying
    it silently discards whatever they did instead."""
    world = build_world(tmp_path)
    if kind == "modified":
        edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
        target = world.repo / "pipeline" / "kurgu.py"
    else:
        drop(world, "pipeline/silinecek.py")
        target = world.repo / "pipeline" / "silinecek.py"
    target.write_text(f"YEREL CALISMA {SENTINEL}\n", encoding="utf-8")
    assert SENTINEL in read(target), "senaryo kurulmadi: sapma yok"
    before = main_view(world.repo)

    with pytest.raises(application.MainCheckoutMismatch):
        apply_it(world)

    assert read(target) == f"YEREL CALISMA {SENTINEL}\n"
    assert difference(before, main_view(world.repo)) == ()
    assert writes == [], "sapmaya ragmen ana agaca yazildi"


def test_a_reparse_parent_cannot_take_a_write_outside_the_repository(tmp_path,
                                                                     writes):
    """A REAL junction on Windows and a REAL symlink on POSIX, planted as
    a parent directory inside the repository and pointing at a tree with a
    canary in it.

    Not a mocked refusal: the whole design rests on the parent being
    OPENED rather than spelled, and only a real reparse point can show
    that a spelled path would have followed it."""
    world = build_world(tmp_path)
    edit(world, "pipeline/alt/derin.py", "VALUE = 8\n")
    outside = tmp_path / "disarisi"
    outside.mkdir()
    (outside / "kanarya.txt").write_text(SENTINEL, encoding="utf-8")

    trap = world.repo / "pipeline" / "alt"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(trap), str(outside)],
                       capture_output=True, check=True)
        planted = os.path.isdir(trap) and bool(
            os.lstat(trap).st_file_attributes & 0x400)
    else:
        os.symlink(outside, trap)
        planted = os.path.islink(trap)
    assert planted, "senaryo kurulmadi: yeniden ayrisma noktasi yok"
    assert (trap / "kanarya.txt").exists(), \
        "senaryo kurulmadi: tuzak disariyi gostermiyor"

    with pytest.raises((application.ApplicationContainment,
                        application.MainCheckoutMismatch)):
        apply_it(world)

    assert writes == [], "ayrisma noktasi uzerinden yazildi"
    assert sorted(item.name for item in outside.iterdir()) == ["kanarya.txt"]
    assert read(outside / "kanarya.txt") == SENTINEL


def test_a_final_component_swapped_for_a_link_is_refused(tmp_path, writes):
    """The LAST component, not a parent. A modified target replaced by a
    link is the classic write-through: the bytes land wherever the link
    points, and every path-based check upstream still saw a file."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    outside = tmp_path / "disarisi"
    outside.mkdir()
    canary = outside / "kanarya.py"
    canary.write_text(SENTINEL, encoding="utf-8")

    target = world.repo / "pipeline" / "kurgu.py"
    target.unlink()
    if os.name == "nt":
        done = subprocess.run(["cmd", "/c", "mklink", str(target),
                               str(canary)], capture_output=True)
        if done.returncode != 0:
            pytest.skip("bu makinede sembolik baglanti olusturulamiyor")
        planted = bool(os.lstat(target).st_file_attributes & 0x400)
    else:
        os.symlink(canary, target)
        planted = os.path.islink(target)
    assert planted, "senaryo kurulmadi: son bilesen baglanti degil"

    with pytest.raises((application.ApplicationContainment,
                        application.MainCheckoutMismatch)):
        apply_it(world)

    assert writes == [], "baglanti uzerinden yazildi"
    assert read(canary) == SENTINEL, "kanarya ezildi"


def test_a_parent_directory_swapped_mid_apply_is_refused(tmp_path):
    """The race the handle model exists for: the parent is replaced
    AFTER every precondition has passed and BEFORE the operation runs.

    A path-based apply re-resolves and writes into the new object. A
    handle-bound one is still holding the old one, so the swap can only
    make the operation fail -- never succeed somewhere else."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    outside = tmp_path / "disarisi"
    outside.mkdir()
    (outside / "kanarya.txt").write_text(SENTINEL, encoding="utf-8")
    victim = world.repo / "pipeline"
    swapped = []

    real = application_transport.rename_child

    def swap_then_rename(source, source_name, target, target_name):
        if not swapped:
            swapped.append(target_name)
            shutil.move(str(victim), str(world.repo / "pipeline-tasindi"))
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(victim),
                                str(outside)], capture_output=True, check=True)
            else:
                os.symlink(outside, victim)
        return real(source, source_name, target, target_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", swap_then_rename)
        with pytest.raises(application.ApplicationError) as raised:
            apply_it(world)

    assert swapped, "senaryo kurulmadi: takas hic denenmedi"
    assert not isinstance(raised.value, application.RollbackFailed)
    assert sorted(item.name for item in outside.iterdir()) == ["kanarya.txt"], \
        "takas edilen ust dizin uzerinden disariya yazildi"


# ---------------------------------------------------------------------
# 14-18  THE JOURNAL, AND WHAT HAPPENS WHEN A STEP FAILS
# ---------------------------------------------------------------------

def test_the_journal_reaches_prepared_before_the_first_repository_write(
        tmp_path):
    """Write-ahead means the record of the intent is durable BEFORE the
    intent is acted on. A journal written after the first move describes
    a repository that has already been changed."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    seen = []
    real = application_transport.rename_child

    def observer(source, source_name, target, target_name):
        seen.append(sorted(application.find_pending_applications(world.repo)))
        seen.append(application.journal_state_of(world.repo, seen[-1][0])
                    if seen[-1] else None)
        return real(source, source_name, target, target_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", observer)
        apply_it(world)

    assert seen, "senaryo kurulmadi: hic tasima olmadi"
    assert len(seen[0]) == 1, "ilk yazidan once journal yoktu"
    assert seen[1] in (application.JournalState.PREPARED,
                       application.JournalState.APPLYING)


def test_every_apply_step_is_journalled_before_it_happens(tmp_path):
    """Not only the first one. A journal that stops being updated after
    the first move cannot say WHERE a crash landed, which is the one
    question recovery has to answer."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    drop(world, "pipeline/silinecek.py")
    progress = []
    real = application_transport.rename_child

    def observer(source, source_name, target, target_name):
        pending = application.find_pending_applications(world.repo)
        assert len(pending) == 1, "tam bir journal yok"
        progress.append((application.journal_state_of(world.repo, pending[0]),
                         application.journal_progress_of(world.repo,
                                                         pending[0])))
        return real(source, source_name, target, target_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", observer)
        apply_it(world)

    assert len(progress) >= 3, f"beklenenden az tasima: {len(progress)}"
    assert all(step == application.JournalState.APPLYING
               for step, _ in progress[1:])
    indexes = [index for _, index in progress]
    assert indexes == sorted(indexes) and indexes[0] < indexes[-1], \
        f"sira bilgisi ilerlemiyor: {indexes}"


@pytest.mark.parametrize("fail_at", [0, 1, 3])
def test_a_failure_at_any_operation_rolls_back_exactly(tmp_path, fail_at):
    """First, middle and last. A rollback that only works from the last
    step is a rollback nobody has run, and the interesting failures are
    the ones that leave half a change set behind."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    edit(world, "pipeline/alt/derin.py", "VALUE = 8\n")
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    drop(world, "pipeline/silinecek.py")
    before = main_view(world.repo)
    calls, fired = [], []
    real = application_transport.rename_child

    def breaker(source, source_name, target, target_name):
        calls.append(target_name)
        if len(calls) == fail_at + 1:
            # ONE injection, at the chosen operation. The counter keeps
            # rising afterwards because the ROLLBACK moves objects too,
            # so "how many moves happened" is not the setup assertion --
            # "did the intended one fail" is.
            fired.append(fail_at)
            raise OSError("kurgu tasima arizasi")
        return real(source, source_name, target, target_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", breaker)
        with pytest.raises(application.ApplicationError) as raised:
            apply_it(world)

    assert fired == [fail_at], "senaryo kurulmadi: ariza tetiklenmedi"
    # The rollback's WORK is proven by the state below, not by a call
    # count: when the FIRST operation is the one that fails there is
    # nothing to move back at all, and the only thing to undo is the
    # directory this call created.
    assert len(calls) >= fail_at + 1
    assert not isinstance(raised.value, application.RollbackFailed)
    assert difference(before, main_view(world.repo)) == (), \
        "geri alma tam degil"
    assert read(world.repo / "pipeline" / "kurgu.py") == "VALUE = 1\n"
    assert (world.repo / "pipeline" / "silinecek.py").is_file()
    assert not (world.repo / "pipeline" / "yeni.py").exists()
    assert not (world.repo / "pipeline" / "alt").exists(), \
        "bu cagrinin yarattigi dizin kaldi"


@pytest.mark.parametrize("raiser", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_rolls_back_and_travels_out_unchanged(tmp_path, raiser):
    """An operator's Ctrl-C is a decision, not a finding. It still may not
    leave half a change set on disk, so the rollback runs and THEN the
    interrupt continues out exactly as it arrived."""
    world = build_world(tmp_path)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    before = main_view(world.repo)
    calls, fired = [], []
    real = application_transport.rename_child

    def breaker(source, source_name, target, target_name):
        calls.append(target_name)
        if len(calls) == 2:
            fired.append(raiser)
            raise raiser()
        return real(source, source_name, target, target_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", breaker)
        with pytest.raises(raiser):
            apply_it(world)

    assert fired == [raiser], "senaryo kurulmadi: kesme tetiklenmedi"
    # the rollback ran AFTER the interrupt and before it travelled out,
    # which is the whole claim -- so more moves happened than the two the
    # interrupt saw
    assert len(calls) > 2, "kesme sonrasi geri alma calismadi"
    assert difference(before, main_view(world.repo)) == ()
    assert read(world.repo / "pipeline" / "kurgu.py") == "VALUE = 1\n"


def test_a_rollback_that_cannot_be_proven_is_typed_and_stays_findable(
        tmp_path):
    """The one failure that outranks every other. A verified rollback is
    an ordinary red; an UNVERIFIED one means the machine is in a state
    nobody has described, so it gets its own type, the journal stays on
    disk as FAILED, and the backups are not deleted."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    calls = []
    real = application_transport.rename_child

    def breaker(source, source_name, target, target_name):
        calls.append(target_name)
        if len(calls) == 1:
            return real(source, source_name, target, target_name)
        raise OSError("kurgu geri alma arizasi")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", breaker)
        with pytest.raises(application.RollbackFailed):
            apply_it(world)

    assert len(calls) >= 2, "senaryo kurulmadi: geri alma denenmedi"
    pending = application.find_pending_applications(world.repo)
    assert len(pending) == 1, "basarisiz journal bulunamiyor"
    assert application.journal_state_of(world.repo, pending[0]) == \
        application.JournalState.FAILED
    # the operator's bytes still exist SOMEWHERE, which is the whole
    # reason a failed rollback refuses to clean up after itself
    holder = application.holder_for(world.repo, pending[0])
    saved = [item for item in holder.rglob("*") if item.is_file()]
    assert saved, "yedek silindi"
    application.recover_application(world.repo, application_id=pending[0])


# ---------------------------------------------------------------------
# 19-21  OTHER WRITERS, LATE MOVEMENT, AND WHAT MAY LEAVE
# ---------------------------------------------------------------------

def test_a_concurrent_write_is_neither_overwritten_nor_deleted(tmp_path):
    """Somebody else edits a target between the precondition and the
    move. Rollback may put back what it took, and may NOT put back over
    something it did not take."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    rival = world.repo / "pipeline" / "yeni.py"
    calls = []
    real = application_transport.rename_child

    def interloper(source, source_name, target, target_name):
        calls.append(target_name)
        if len(calls) == 1:
            rival.write_text(f"BASKA YAZAR {SENTINEL}\n", encoding="utf-8")
        return real(source, source_name, target, target_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", interloper)
        with pytest.raises(application.ApplicationError) as raised:
            apply_it(world)

    assert calls, "senaryo kurulmadi: hic tasima olmadi"
    assert not isinstance(raised.value, application.RollbackFailed)
    assert rival.is_file(), "baska yazarin dosyasi silindi"
    assert read(rival) == f"BASKA YAZAR {SENTINEL}\n", "baska yazar ezildi"
    assert read(world.repo / "pipeline" / "kurgu.py") == "VALUE = 1\n"


@pytest.mark.parametrize("moved", ["gorev", "baska-dosya"])
def test_a_late_change_anywhere_is_not_a_success(tmp_path, moved):
    """The application is not finished when the last byte lands.

    TWO INDEPENDENT GATES, and each is given a case only IT can answer.
    `gorev` moves the manifest: whatever chose the change set has to
    still be what it was, or the thing that just happened was authorised
    by a document that no longer exists. `baska-dosya` moves an unrelated
    tracked file instead -- the manifest is untouched, the workspace is
    untouched, and the ONLY thing that can notice is the exact
    post-verification of the checkout's semantic difference. Without a
    case of its own that check could be deleted and this test would stay
    green on the other one."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    bystander = world.repo / "pipeline" / "gizli" / "sir.py"
    before = main_view(world.repo)
    calls = []
    real = application_transport.rename_child

    def mover(source, source_name, target, target_name):
        calls.append(target_name)
        outcome = real(source, source_name, target, target_name)
        if moved == "gorev":
            world.task.write_text(world.task.read_text(encoding="utf-8") + " ",
                                  encoding="utf-8")
        else:
            bystander.write_bytes(f"VALUE = 3  # {SENTINEL}\n".encode())
        return outcome

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", mover)
        with pytest.raises(application.ApplicationError) as raised:
            apply_it(world)

    assert calls, "senaryo kurulmadi: hic tasima olmadi"
    assert not isinstance(raised.value, application.RollbackFailed)
    # whichever file the test moved is the ONE expected difference, and
    # it is not one of the candidate's: every candidate path has to be
    # back exactly as it was found
    touched = ("kurgu-task.json" if moved == "gorev"
               else "pipeline/gizli/sir.py")
    assert difference(before, main_view(world.repo)) == (
        (touched, changes.MODIFIED),), \
        "gec degisiklige ragmen uygulama birakildi"
    assert read(world.repo / "pipeline" / "kurgu.py") == "VALUE = 1\n"


def test_no_report_or_error_carries_a_path_bytes_or_a_cause(tmp_path):
    """Everything this package hands upward, swept for the four things it
    may never carry: an absolute path, a candidate's bytes, the operating
    system's own message, and a chained exception whose text has both."""
    world = build_world(tmp_path, index=1)
    edit(world, "pipeline/yeni.py", f"VALUE = 7  # {SENTINEL}\n")
    report = apply_it(world)

    forbidden = (SENTINEL, str(tmp_path), str(world.repo), world.repo.name,
                 "Traceback", "Errno")
    for field in application.ApplicationReport.__slots__:
        value = getattr(report, field)
        items = value if type(value) is tuple else (value,)
        for item in items:
            text = str(item)
            for needle in forbidden:
                assert needle not in text, f"{field} sizinti tasiyor: {needle}"
            # NOT a ban on the separator: a repo-relative path is allowed
            # to contain one and on POSIX that IS `os.sep`, so banning it
            # would have made this assertion platform noise. What may
            # never appear is a LOCATION -- an absolute path, or the
            # backslash spelling that only a raw OS path carries.
            assert not os.path.isabs(text), f"{field} mutlak yol tasiyor"
            assert "\\" not in text, f"{field} ham OS yolu tasiyor"
    assert report.applied_files == ("pipeline/yeni.py",)

    world_two = build_world(tmp_path, index=2)
    edit(world_two, "pipeline/yeni.py", f"VALUE = 7  # {SENTINEL}\n")
    (world_two.repo / "pipeline" / "yeni.py").write_text(
        f"KULLANICI {SENTINEL}\n", encoding="utf-8")
    with pytest.raises(application.ApplicationError) as raised:
        apply_it(world_two)

    hata = raised.value
    assert hata.__cause__ is None and hata.__context__ is None, \
        "ham hata zinciri disari sizdi"
    body = " ".join([str(hata), *getattr(hata, "__notes__", ())])
    for needle in (SENTINEL, str(tmp_path), world_two.repo.name, "Errno"):
        assert needle not in body, f"hata metni sizinti tasiyor: {needle}"
    assert hata.reason in contract.ALL_STOP_REASONS


# ---------------------------------------------------------------------
# 22-24  CRASH, RECOVERY, RESIDUE
# ---------------------------------------------------------------------

def test_a_crashed_application_is_found_and_recovered(tmp_path):
    """A process that dies mid-apply leaves a journal and a half-applied
    tree. The recovery seam finds it by enumeration -- never by being
    handed a path -- and finishes it in the ROLLBACK direction."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    before = main_view(world.repo)
    calls = []
    real = application_transport.rename_child

    class Crash(BaseException):
        """A death this package must not catch and tidy away."""

    def crasher(source, source_name, target, target_name):
        calls.append(target_name)
        outcome = real(source, source_name, target, target_name)
        if len(calls) == 1:
            raise Crash()
        return outcome

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(application.transport, "rename_child", crasher)
        # the rollback itself needs the primitive back, so the crash is
        # made unrecoverable by leaving the journal mid-flight instead
        patch.setattr(application, "_rollback",
                      lambda *args, **kwargs: (_ for _ in ()).throw(Crash()))
        with pytest.raises(Crash):
            apply_it(world)

    assert calls, "senaryo kurulmadi: hic tasima olmadi"
    assert difference(before, main_view(world.repo)) != (), \
        "senaryo kurulmadi: kaza yarim uygulama birakmadi"

    pending = application.find_pending_applications(world.repo)
    assert len(pending) == 1, f"kaza kaydi bulunamadi: {pending}"
    assert application.journal_state_of(world.repo, pending[0]) in (
        application.JournalState.APPLYING, application.JournalState.PREPARED)

    outcome = application.recover_application(world.repo,
                                              application_id=pending[0])
    assert outcome.rolled_back is True
    assert outcome.state == application.JournalState.ROLLED_BACK
    assert difference(before, main_view(world.repo)) == (), \
        "kurtarma agaci taban duruma dondurmedi"
    assert application.find_pending_applications(world.repo) == ()


def test_recovery_touches_only_its_own_holder(tmp_path):
    """A recovery pass runs at startup, when nobody knows what else is on
    the machine. A holder without a marker, one belonging to another
    repository and one that is a link are all somebody else's -- and this
    function is not allowed to have an opinion about any of them."""
    world = build_world(tmp_path)
    root = application.apply_root_for(world.repo)
    root.mkdir(parents=True, exist_ok=True)

    stranger = root / f"{application.HOLDER_PREFIX}{'c' * 32}"
    stranger.mkdir()
    (stranger / "kanarya.txt").write_text(SENTINEL, encoding="utf-8")
    unmarked = root / f"{application.HOLDER_PREFIX}{'d' * 32}"
    unmarked.mkdir()
    (unmarked / "kanarya.txt").write_text(SENTINEL, encoding="utf-8")
    assert stranger.is_dir() and unmarked.is_dir(), "senaryo kurulmadi"

    assert application.find_pending_applications(world.repo) == (), \
        "isaretsiz dizin bekleyen uygulama sayildi"
    for stray in ("e" * 32, application.HOLDER_PREFIX, "../kacis"):
        with pytest.raises(application.ApplicationError):
            application.recover_application(world.repo, application_id=stray)

    assert read(stranger / "kanarya.txt") == SENTINEL, "yabanci holder silindi"
    assert read(unmarked / "kanarya.txt") == SENTINEL, "isaretsiz holder silindi"
    shutil.rmtree(stranger)
    shutil.rmtree(unmarked)


def test_nothing_of_this_turn_survives_success_or_refusal(tmp_path):
    """Success, refusal and rollback all end with the sibling apply root
    exactly as empty as this turn found it, and with no handle of ours
    still open on the repository.

    The autouse residue guard asserts the same thing for every other test
    in this file; this one states it as the claim rather than the
    background."""
    world = build_world(tmp_path)
    root = application.apply_root_for(world.repo)
    edit(world, "pipeline/yeni.py", "VALUE = 7\n")
    apply_it(world)
    assert not root.exists() or list(root.iterdir()) == [], \
        "basaridan sonra artik kaldi"

    world_two = build_world(tmp_path, index=2)
    edit(world_two, "pipeline/yeni.py", "VALUE = 7\n")
    (world_two.repo / "pipeline" / "yeni.py").write_text("KULLANICI\n",
                                                         encoding="utf-8")
    with pytest.raises(application.ApplicationError):
        apply_it(world_two)
    root_two = application.apply_root_for(world_two.repo)
    assert not root_two.exists() or list(root_two.iterdir()) == [], \
        "retten sonra artik kaldi"

    # the applied file is still deletable and the repository is still
    # removable, which is what an unreleased handle would prevent.
    # `onexc` is for git's read-only object files, not for anything this
    # package left behind -- a handle still open would raise WinError 32
    # (sharing violation), which this handler does not touch.
    (world.repo / "pipeline" / "yeni.py").unlink()
    shutil.rmtree(world.repo, onexc=_force_writable)
    assert not world.repo.exists()


def _force_writable(function, path, error):
    os.chmod(path, 0o700)
    function(path)
