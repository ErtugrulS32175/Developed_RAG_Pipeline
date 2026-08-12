"""PACKAGE B2B-C1 -- frozen acceptance commands in a disposable mirror.

ONE question: are the acceptance commands a task NAMES resolved from the
frozen registry and run against a THROWAWAY copy of the verified
candidate, with the operator's checkout and both flat roots untouched and
nothing of the run left behind.

WHAT IS DELIBERATELY NOT HERE. No repair loop, no state machine, no
patch applied to the operator's checkout, no runner. This package runs
commands and reports; everything it learns is a number or a closed code.

THE WORLD IS REAL AND THROWAWAY. A real repository, a real D3A flat
workspace, a real disposable mirror, real `python`/`git` processes -- and
never the project's own checkout, never the operator's document tree,
never a model. The one thing that is stubbed is the implementer: this
package is given a `VerifiedChangeSet` and re-derives it from the
filesystem, so the model call belongs to the package next door.

EVERY REFUSAL TEST PROVES ITS SETUP, and every "nothing started"
assertion is NARROWED by working directory: `flat_workspace.create()`
runs git through the same contained launch seam, so a recorder that
counted every process would be measuring the fixture.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time
import types
from pathlib import Path

import pytest

import test_agent_loop_b2_changes as legacy
from tools.agent_loop import (acceptance, acceptance_workspace, changes,
                              contract, flat_workspace, state)
from tools.agent_loop import process as process_module

RUN = legacy.RUN
ROOT = Path(__file__).resolve().parent.parent

# A window wide enough for the scanner's 14-character default and loud
# enough to carry digits, which is what makes a document hit HARD.
CORPUS_LINE = "SENTETIK-KAYIT 4711 ALFA-BETA-GAMA-DELTA-9182 OMEGA"

SELECTED = "pytest_selected"
GECER = "pipeline/test_gecer.py"

_PASSING_TEST = "def test_kurgu():\n    assert True\n"


def _commands(*references):
    return [dict(reference) for reference in references]


def _selected(*paths):
    return {"command_id": SELECTED, "paths": list(paths)}


# ---------------------------------------------------------------------
# THE WORLD
# ---------------------------------------------------------------------

def build_world(tmp_path, index=0, seed=(), corpus=(), saved=(),
                **task_overrides):
    """A real repository whose BASELINE already carries `seed`, a real
    flat workspace at that baseline, a bound task manifest, and the
    identity tuple `run_acceptance` takes.

    `corpus` and `saved` are written into the operator's checkout under
    `data/` and `output/` -- both git-ignored, exactly like the real
    tree -- so the corpus-snapshot path has something synthetic to copy.
    No document of the user's is ever involved."""
    repo, _ = legacy.build_repo(tmp_path, index)
    (repo / ".gitignore").write_text("data/\noutput/\n", encoding="utf-8")
    (repo / "pipeline" / "test_gecer.py").write_text(_PASSING_TEST,
                                                     encoding="utf-8")
    for relative, text in seed:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    legacy._git(repo, "add", "-A")
    legacy._git(repo, "commit", "-qm", "kurgu-taban")
    baseline = legacy._git(repo, "rev-parse", "HEAD")

    for relative, text in corpus:
        target = repo / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for relative, text in saved:
        target = repo / "output" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    payload = dict(legacy.BASE_TASK, baseline_sha=baseline)
    payload["acceptance_commands"] = _commands(_selected(GECER))
    payload.update(task_overrides)
    task_file = repo / "kurgu-task.json"
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
        repo=repo, tree=workspace.implementer_root,
        reference=workspace.reference_root, state_dir=state_dir,
        task=task_file, digest=digest, baseline=baseline,
        workspace_id=workspace.workspace_id,
        identity={"repo": repo, "state_dir": state_dir,
                  "task_path": task_file, "manifest_digest": digest,
                  "run_id": RUN, "workspace_id": workspace.workspace_id,
                  "baseline_sha": baseline})


def edit(world, relative, text):
    target = world.tree / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def verified_for(world, **overrides):
    """The `VerifiedChangeSet` an implementer call WOULD have produced,
    derived from the very evidence this package re-derives.

    Built rather than obtained from a model run on purpose: the point of
    the fresh re-verification gate is that these two derivations are
    compared, so a test that wants a STALE claim has to be able to spell
    one -- `overrides` is how."""
    candidate = changes.derive_candidate_changes(**world.identity)
    fields = {"run_id": candidate.run_id,
              "workspace_id": candidate.workspace_id,
              "baseline_sha": candidate.baseline_sha,
              "status": contract.Status.IMPLEMENTED,
              "changed_files": candidate.changed_files,
              "added": candidate.added, "modified": candidate.modified,
              "deleted": candidate.deleted,
              "fingerprint": candidate.fingerprint,
              "exit_code": 0, "duration_ms": 1, "stdout_bytes": 2,
              "stderr_bytes": 0, "schema_sha256": "a" * 64,
              "event": contract.EventCode.MODEL_CALL_FINISHED}
    fields.update(overrides)
    return changes.VerifiedChangeSet(**fields)


def run(world, verified=None, **overrides):
    settings = {"timeout_seconds": 300, "max_output_bytes": 65536}
    settings.update(overrides)
    return acceptance.run_acceptance(
        **world.identity,
        verified_changes=verified if verified is not None
        else verified_for(world), **settings)


# ---------------------------------------------------------------------
# THE FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def private_roots(tmp_path, monkeypatch):
    """Every test gets its OWN runner root AND its own mirror root, so
    nothing here can create, list or delete a directory a real agent loop
    is using -- and so residue assertions measure this test only."""
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


@pytest.fixture(autouse=True)
def acceptance_processes(private_roots, monkeypatch):
    """Only launches whose WORKING DIRECTORY is inside the mirror count.

    `flat_workspace.create()` runs git through this same seam with the
    repository as its cwd, so a recorder that counted every process would
    make every "nothing started" assertion measure the fixture instead of
    the thing under test."""
    inside, started = [], []
    real_popen = process_module.subprocess.Popen
    mirror_root = str(private_roots.mirror).casefold()

    class _Recorder:
        DEVNULL = subprocess.DEVNULL
        PIPE = subprocess.PIPE
        TimeoutExpired = subprocess.TimeoutExpired
        SubprocessError = subprocess.SubprocessError
        CREATE_NEW_PROCESS_GROUP = getattr(subprocess,
                                           "CREATE_NEW_PROCESS_GROUP", 0)

        @staticmethod
        def run(argv, **kwargs):
            return subprocess.run(argv, **kwargs)

        @staticmethod
        def Popen(argv, **kwargs):                 # noqa: N802 -- stdlib
            cwd = str(kwargs.get("cwd", "")).casefold()
            if cwd.startswith(mirror_root):
                inside.append({"argv": list(argv), "cwd": kwargs.get("cwd"),
                               "env": kwargs.get("env"),
                               "shell": kwargs.get("shell")})
            process = real_popen(argv, **kwargs)
            started.append(process)
            return process

    monkeypatch.setattr(process_module, "subprocess", _Recorder)
    yield inside
    for process in started:
        if process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=15)
            except Exception:                      # noqa: BLE001
                pass


def residue(private_roots):
    return sorted(entry.name for entry in private_roots.mirror.iterdir())


# =====================================================================
# 1. THE SEAM, THE RESULT AND THE BINDING ORDER
# =====================================================================

def test_the_public_seam_is_keyword_only_and_offers_no_escape():
    """Mechanism 12. A task names a command; it may not choose an argv,
    a shell, a working directory, an environment or the mirror."""
    parameters = inspect.signature(acceptance.run_acceptance).parameters
    assert set(parameters) == {
        "repo", "state_dir", "task_path", "manifest_digest", "run_id",
        "workspace_id", "baseline_sha", "verified_changes",
        "timeout_seconds", "max_output_bytes"}
    for parameter in parameters.values():
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    for escape in ("argv", "command", "commands", "shell", "cwd", "env",
                   "environment", "mirror", "mirror_root", "git_dir",
                   "binary", "executable", "paths"):
        assert escape not in parameters, f"kacis parametresi: {escape}"


def test_the_report_and_command_results_are_frozen_and_textless(tmp_path):
    """Mechanisms 5 and 19. The shapes are pinned here so a field that
    could carry output, a path or a sentence cannot be added quietly."""
    world = build_world(tmp_path)
    report = run(world)
    assert [field for field in report.__slots__] == [
        "run_id", "workspace_id", "baseline_sha", "passed", "command_results",
        "total_duration_ms", "event"]
    (result,) = report.command_results
    assert [field for field in result.__slots__] == [
        "command_id", "passed", "exit_code", "duration_ms", "stdout_bytes",
        "stderr_bytes", "event"]
    for frozen in (report, result):
        with pytest.raises(Exception):
            frozen.passed = False
        with pytest.raises(Exception):
            frozen.kurgu_alani = 1                 # slotted, not just frozen
    assert report.event == contract.EventCode.ACCEPTANCE_FINISHED
    assert result.event in contract.ALL_EVENT_CODES


@pytest.mark.parametrize(
    "senaryo", ["gorev-baytlari", "gorev-ozeti", "sema-disi",
                "durum-baglamasi",
                "duz-baglama", "yabanci-kosu", "yabanci-calisma-alani",
                "yabanci-taban", "uygulanamaz-sonuc", "sure-siniri",
                "cikti-siniri"])
def test_every_binding_is_asserted_before_a_mirror_or_a_process_exists(
        tmp_path, private_roots, acceptance_processes, senaryo):
    """Mechanisms 1-4 and the ordering rule that makes them worth having:
    if one of these falls, the mirror counter and the process counter are
    both still zero."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 7\n")
    verified = verified_for(world)
    settings = {"timeout_seconds": 300, "max_output_bytes": 65536}
    identity = dict(world.identity)

    if senaryo == "gorev-baytlari":
        world.task.write_text(json.dumps({"bozuk": True}), encoding="utf-8")
    elif senaryo == "gorev-ozeti":
        identity["manifest_digest"] = "b" * 64
    elif senaryo == "sema-disi":
        # the digest is re-issued for the tampered bytes, so the ONLY
        # gate left is the schema
        world.task.write_text(json.dumps({"protocol_version": "1.0"}),
                              encoding="utf-8")
        identity["manifest_digest"] = hashlib.sha256(
            world.task.read_bytes()).hexdigest()
    elif senaryo == "durum-baglamasi":
        (world.state_dir / state.BINDING_FILENAME).unlink()
    elif senaryo == "duz-baglama":
        holder = flat_workspace.holder_for(world.workspace_id)
        (holder / flat_workspace.MARKER_NAME).unlink()
    elif senaryo == "yabanci-kosu":
        verified = verified_for(world, run_id="baska-kosu")
    elif senaryo == "yabanci-calisma-alani":
        verified = verified_for(world, workspace_id="c" * 32)
    elif senaryo == "yabanci-taban":
        verified = verified_for(world, baseline_sha="d" * 40)
    elif senaryo == "uygulanamaz-sonuc":
        verified = verified_for(world, status=contract.Status.FAILED)
    elif senaryo == "sure-siniri":
        settings["timeout_seconds"] = 0
    else:
        settings["max_output_bytes"] = 8

    with pytest.raises(acceptance.AcceptanceError) as refusal:
        acceptance.run_acceptance(**identity, verified_changes=verified,
                                  **settings)

    assert residue(private_roots) == [], "ret sonrasi ayna kaldi"
    assert acceptance_processes == [], "ret oncesinde kabul sureci basladi"
    metin = str(refusal.value) + repr(refusal.value)
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"


def test_a_stale_verified_change_set_is_refused_against_fresh_evidence(
        tmp_path, private_roots, acceptance_processes):
    """Mechanism 5, and the reason this package exists at all: the claim
    is re-derived, not believed. The candidate is edited AFTER the change
    set was built, so the two derivations disagree by exactly one file."""
    world = build_world(tmp_path)
    edit(world, "pipeline/kurgu.py", "VALUE = 7\n")
    stale = verified_for(world)
    edit(world, "pipeline/kurgu.py", "VALUE = 8\n")
    taze = changes.derive_candidate_changes(**world.identity)
    assert taze.fingerprint != stale.fingerprint, \
        "senaryo kurulmadi: iki turetme zaten ayni"

    with pytest.raises(acceptance.CandidateMismatch) as refusal:
        run(world, verified=stale)
    assert str(refusal.value) == "aday degisiklikler taze kanitla ayni degil"
    assert residue(private_roots) == [] and acceptance_processes == []


# =====================================================================
# 2. THE MIRROR
# =====================================================================

def test_a_healthy_run_walks_the_registry_in_order_and_repeats_a_command(
        tmp_path, private_roots, acceptance_processes):
    """Mechanisms 6, 7 and 10. The commands come from the task's list IN
    ORDER, a repeated command runs twice, and every process ran inside a
    mirror that is outside the repository and both flat roots."""
    world = build_world(tmp_path, acceptance_commands=_commands(
        _selected(GECER), _selected(GECER)))
    edit(world, "pipeline/kurgu.py", "VALUE = 7\n")
    report = run(world)

    assert report.passed is True
    assert [result.command_id for result in report.command_results] == \
        [SELECTED, SELECTED]
    assert all(result.exit_code == 0 for result in report.command_results)
    assert report.run_id == RUN and report.workspace_id == world.workspace_id
    assert report.total_duration_ms >= 0

    kabul = [entry for entry in acceptance_processes
             if entry["argv"][0] != "git"]
    assert len(kabul) == 2, "kayitli komut iki kez calismadi"
    for entry in kabul:
        cwd = os.path.realpath(entry["cwd"])
        assert cwd.startswith(os.path.realpath(private_roots.mirror))
        for yasak in (world.repo, world.tree, world.reference, ROOT):
            assert not cwd.startswith(os.path.realpath(yasak)), \
                f"kabul komutu {yasak} icinde calisti"
        assert entry["shell"] in (None, False), "kabuk uzerinden calisti"
    assert residue(private_roots) == [], "basarili kosudan sonra ayna kaldi"


@pytest.mark.parametrize("senaryo", ["ekleme", "degisiklik", "silme",
                                     "calistirilabilir"])
def test_the_mirror_carries_exactly_the_verified_candidate(
        tmp_path, monkeypatch, senaryo):
    """Mechanisms 7 and 8. The mirror is materialised from the baseline's
    RAW GIT OBJECTS and then patched with the fresh change set -- never
    copied blindly from the tree the model wrote in -- and the semantic
    projection of the result has to equal the candidate's before a single
    command may start."""
    world = build_world(tmp_path)
    if senaryo == "ekleme":
        edit(world, "pipeline/yeni.py", "VALUE = 9\n")
        beklenen = "pipeline/yeni.py"
    elif senaryo == "degisiklik":
        edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
        beklenen = "pipeline/kurgu.py"
    elif senaryo == "silme":
        (world.tree / "pipeline" / "silinecek.py").unlink()
        beklenen = "pipeline/silinecek.py"
    else:
        if os.name == "nt":
            pytest.skip("POSIX mod bitleri Windows'ta tasinmiyor")
        hedef = edit(world, "pipeline/kurgu.py", "VALUE = 9\n")
        os.chmod(hedef, 0o755)
        beklenen = "pipeline/kurgu.py"

    gorulen = {}
    gercek = acceptance_workspace.create

    def izleyen(**kwargs):
        mirror = gercek(**kwargs)
        gorulen["tree"] = mirror.tree
        gorulen["var"] = (mirror.tree / beklenen).exists()
        if gorulen["var"]:
            gorulen["bayt"] = (mirror.tree / beklenen).read_bytes()
            gorulen["mode"] = os.stat(mirror.tree / beklenen).st_mode & 0o777
        return mirror

    monkeypatch.setattr(acceptance_workspace, "create", izleyen)
    report = run(world)
    assert report.passed is True

    if senaryo == "silme":
        assert gorulen["var"] is False, "silinen dosya aynada duruyor"
    else:
        assert gorulen["var"] is True, "aday dosya aynaya tasinmadi"
        assert gorulen["bayt"] == (world.tree / beklenen).read_bytes()
    if senaryo == "calistirilabilir":
        assert gorulen["mode"] & 0o111, "calistirilabilir mod tasinmadi"


@pytest.mark.parametrize("tur", ["symlink", "junction", "fifo"])
def test_a_candidate_entry_this_mirror_cannot_represent_is_refused(
        tmp_path, private_roots, acceptance_processes, tur):
    """Mechanism 9. A symlink, a junction and a special file each mean
    something different, and inventing a representation for one is how an
    unreviewed object reaches a command's working directory.

    THE REFUSAL IS ASSERTED BY SENTENCE, not only by type: these all
    close on the same reason, and a reason-only assertion stays green
    when a different gate answers a question it was never asked."""
    beklenen = {"symlink": "calisma alaninda temsil edilemeyen giris",
                "junction": "calisma alaninda temsil edilemeyen giris",
                "fifo": "siradan olmayan dosya turu"}[tur]
    world = build_world(tmp_path)
    # the change set is built BEFORE the object exists, so the refusal
    # under test comes from the fresh re-derivation inside the seam
    # rather than from the helper that made the claim
    verified = verified_for(world)
    target = world.tree / "pipeline" / "girdi"
    if tur == "symlink":
        try:
            os.symlink("kurgu.py", str(target))
        except OSError:
            pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    elif tur == "junction":
        if os.name != "nt":
            pytest.skip("kavsak noktasi Windows'a ozgu")
        import _winapi
        disarisi = tmp_path / "kavsak-hedefi"
        disarisi.mkdir()
        _winapi.CreateJunction(str(disarisi), str(target))
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("bu platformda FIFO yaratilamiyor")
        os.mkfifo(str(target))
    assert os.path.lexists(target), "senaryo kurulmadi: nesne yaratilmadi"

    with pytest.raises(acceptance.AcceptanceError) as refusal:
        run(world, verified=verified)
    assert str(refusal.value) == beklenen, "beklenen kapi reddetmedi"
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert residue(private_roots) == [] and acceptance_processes == []


def test_the_corpus_snapshot_is_independent_and_leaves_the_originals_alone(
        tmp_path, private_roots, monkeypatch):
    """Section 8. The mirror gets REAL, independent files -- never a hard
    link, a symlink or a junction back into the operator's tree -- and
    the originals are proven byte- and metadata-identical afterwards."""
    world = build_world(tmp_path, seed=_leak_seed(),
                        corpus=[("belge.txt", CORPUS_LINE + "\n")],
                        acceptance_commands=_commands({"command_id":
                                                       "leak_scan"}))
    kaynak = world.repo / "data" / "belge.txt"
    onceki = kaynak.stat()
    gorulen = {}
    gercek = acceptance_workspace.create

    def izleyen(**kwargs):
        mirror = gercek(**kwargs)
        kopya = mirror.tree / "data" / "belge.txt"
        gorulen["var"] = kopya.is_file()
        gorulen["bayt"] = kopya.read_bytes() if kopya.is_file() else b""
        gorulen["ayri"] = (os.stat(kopya).st_ino != os.stat(kaynak).st_ino
                           if os.name != "nt" else True)
        gorulen["baglanti"] = os.path.islink(kopya)
        return mirror

    monkeypatch.setattr(acceptance_workspace, "create", izleyen)
    run(world)

    assert gorulen["var"] is True, "korpus aynaya kopyalanmadi"
    assert gorulen["bayt"] == kaynak.read_bytes()
    assert gorulen["ayri"] is True, "kopya kaynakla ayni dosya nesnesi"
    assert gorulen["baglanti"] is False, "kopya bir baglanti"
    sonraki = kaynak.stat()
    assert (sonraki.st_size, sonraki.st_mtime_ns) == \
        (onceki.st_size, onceki.st_mtime_ns), "korpus kaynagi degisti"
    assert residue(private_roots) == []


# =====================================================================
# 3. THE COMMANDS
# =====================================================================

@pytest.mark.parametrize("senaryo", ["yol-almaz", "ust-dizin", "mutlak",
                                     "bayrak", "ters-bolu"])
def test_only_the_registry_decides_what_a_path_argument_may_be(
        tmp_path, private_roots, acceptance_processes, senaryo):
    """Mechanisms 11 and 12. `pytest_selected` takes repo-relative test
    paths and nothing else; `pytest_full` takes none at all. The refusal
    comes from `cli.resolve_registry_command`, before any mirror."""
    referans = {"yol-almaz": {"command_id": "pytest_full",
                              "paths": [GECER]},
                "ust-dizin": _selected("pipeline/../../disari.py"),
                "mutlak": _selected("/etc/passwd"),
                "bayrak": _selected("--co"),
                "ters-bolu": _selected("pipeline\\a.py")}[senaryo]
    if senaryo in ("ust-dizin", "mutlak", "ters-bolu"):
        # the frozen task schema already refuses these spellings, so the
        # scenario is built by handing the resolver the same reference
        # the task would have carried
        from tools.agent_loop import cli
        with pytest.raises(cli.UnsafeInvocation):
            cli.resolve_registry_command(referans["command_id"],
                                         contract.COMMAND_REGISTRY,
                                         paths=referans["paths"])
        return
    world = build_world(tmp_path, acceptance_commands=_commands(referans))
    with pytest.raises(acceptance.AcceptanceError) as refusal:
        run(world)
    assert str(refusal.value) == "kabul komutu cozumlenemedi"
    assert residue(private_roots) == [] and acceptance_processes == []


def test_the_first_failing_command_stops_the_rest(
        tmp_path, private_roots, acceptance_processes):
    """Mechanism 13. A run that kept going after a failure would report
    the LAST answer instead of the first, which is how a red gate becomes
    a green one."""
    world = build_world(tmp_path, acceptance_commands=_commands(
        _selected("pipeline/test_yok.py"), _selected(GECER)))
    report = run(world)

    assert report.passed is False
    assert len(report.command_results) == 1, "basarisizliktan sonra devam etti"
    assert report.command_results[0].passed is False
    assert report.command_results[0].exit_code not in (0, None)
    kabul = [entry for entry in acceptance_processes
             if entry["argv"][0] != "git"]
    assert len(kabul) == 1, "ikinci komut yine de baslatildi"
    assert residue(private_roots) == []


@pytest.mark.parametrize("senaryo", ["temiz", "tasma", "sure", "okuma"])
def test_output_overflow_timeout_and_a_failed_reader_are_all_refusals(
        tmp_path, private_roots, monkeypatch, senaryo):
    """Mechanism 14. `capture_output=True` was measured taking 50,331,648
    bytes with no ceiling applied, so every stream is bounded WHILE it is
    read -- and a reader that FAILED is refused even when the command
    exited zero."""
    if senaryo == "tasma":
        # the assertion has to FAIL: pytest captures a passing test's
        # output and never prints it, so a green noisy test produces a
        # hundred bytes and the ceiling is never approached
        seed = [("pipeline/test_gurultu.py",
                 "def test_g():\n    print('A' * 200000)\n    assert False\n")]
        commands = _commands(_selected("pipeline/test_gurultu.py"))
        limits = {"max_output_bytes": 1024}
    elif senaryo == "sure":
        seed = [("pipeline/test_uyku.py",
                 "import time\n\n\ndef test_u():\n    time.sleep(60)\n")]
        commands = _commands(_selected("pipeline/test_uyku.py"))
        limits = {"timeout_seconds": 2}
    else:
        seed = []
        commands = _commands(_selected(GECER))
        limits = {}

    world = build_world(tmp_path, seed=seed, acceptance_commands=commands)
    if senaryo == "okuma":
        gercek = process_module.BoundedStream

        class _Kirik(gercek):
            def run(self):
                super().run()
                self.outcome = process_module.READ_FAILED

        # NARROWED to the acceptance command itself. The stream class is
        # shared with the git transport that materialises the baseline,
        # so replacing it for the whole call broke the fixture instead of
        # the mechanism -- measured, and it looked exactly like a kill.
        gercek_run = acceptance._run_command

        def izleyen(argv, **kwargs):
            if argv[0] == "git":
                return gercek_run(argv, **kwargs)
            monkeypatch.setattr(process_module, "BoundedStream", _Kirik)
            try:
                return gercek_run(argv, **kwargs)
            finally:
                monkeypatch.setattr(process_module, "BoundedStream", gercek)

        monkeypatch.setattr(acceptance, "_run_command", izleyen)

    report = run(world, **limits)
    (result,) = report.command_results
    if senaryo == "temiz":
        assert report.passed is True and result.exit_code == 0
        assert result.event == contract.EventCode.ACCEPTANCE_FINISHED
    else:
        assert report.passed is False, f"{senaryo} basari sayildi"
        assert result.passed is False
    if senaryo == "tasma":
        assert result.event == contract.EventCode.OUTPUT_TRUNCATED
        assert result.stdout_bytes <= 1024 + process_module.READ_CHUNK_BYTES
    if senaryo == "sure":
        assert result.event == contract.EventCode.INTERRUPTED
    if senaryo == "okuma":
        assert result.event == contract.EventCode.INTERRUPTED, \
            "okuma basarisizligi temiz cikisa yenildi"
    assert residue(private_roots) == []


def test_a_successful_parent_may_not_leave_a_living_grandchild(
        tmp_path, private_roots):
    """Mechanism 15. The command exits zero while something it started is
    still running -- the exact shape a container exists for. The witness
    is the grandchild's OWN pid, read from disk after the call."""
    izci = tmp_path / "torun-pid.txt"
    kaynak = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def test_torun():\n"
        "    child = subprocess.Popen([sys.executable, '-c',\n"
        "                              'import time; time.sleep(120)'])\n"
        f"    Path(r'{izci}').write_text(str(child.pid))\n"
        "    time.sleep(0.5)\n"
        "    assert True\n")
    world = build_world(
        tmp_path, seed=[("pipeline/test_torun.py", kaynak)],
        acceptance_commands=_commands(_selected("pipeline/test_torun.py")))
    report = run(world)

    assert izci.is_file(), "senaryo kurulmadi: torun hic baslatilmadi"
    pid = int(izci.read_text().strip())
    assert report.passed is True
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.1)
    assert not _alive(pid), "basarili ebeveynin torunu hala yasiyor"
    assert residue(private_roots) == []


def _alive(pid):
    if os.name == "nt":
        done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                              capture_output=True, text=True)
        return str(pid) in done.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# =====================================================================
# 4. DISPOSABLE GIT, THE CORPUS AND THE LEAK SCANNER
# =====================================================================

def _leak_seed():
    """The real scanner, its package and a candidate file, as BASELINE
    content -- so the mirror materialises them from git objects like
    anything else."""
    return [("eval/tools/leak_scan.py",
             (ROOT / "eval" / "tools" / "leak_scan.py").read_text(
                 encoding="utf-8")),
            ("eval/tools/__init__.py", ""),
            ("pipeline/temiz.py", "VALUE = 1\n")]


def test_the_disposable_git_never_reaches_the_main_checkout_or_flat_roots(
        tmp_path, private_roots, acceptance_processes):
    """Mechanism 16. The mirror gets git metadata OF ITS OWN; the
    operator's `.git` is neither linked, named nor changed, and neither
    flat root grows one."""
    world = build_world(tmp_path, seed=_leak_seed(),
                        corpus=[("belge.txt", CORPUS_LINE + "\n")],
                        acceptance_commands=_commands({"command_id":
                                                       "leak_scan"}))
    ana = world.repo / ".git"
    onceki = sorted((yol.relative_to(ana).as_posix(), yol.stat().st_size)
                    for yol in ana.rglob("*") if yol.is_file())
    run(world)

    assert not (world.tree / ".git").exists(), "duz koke git yazildi"
    assert not (world.reference / ".git").exists(), "referans koke git yazildi"
    sonraki = sorted((yol.relative_to(ana).as_posix(), yol.stat().st_size)
                     for yol in ana.rglob("*") if yol.is_file())
    assert sonraki == onceki, "ana checkout .git degisti"
    for entry in acceptance_processes:
        birlesik = " ".join(entry["argv"])
        assert str(world.repo) not in birlesik, "argv ana depoyu adliyor"
        env = entry["env"] or {}
        assert "GIT_DIR" not in env and "GIT_WORK_TREE" not in env
        assert env.get("GIT_CONFIG_NOSYSTEM") == "1"
        assert env.get("GIT_CONFIG_COUNT"), \
            "cagriya ait git yapilandirmasi yok"
    assert residue(private_roots) == []


def test_the_leak_scanner_really_reads_the_snapshotted_corpus(
        tmp_path, private_roots):
    """Mechanism 17, proved by a PAIR rather than by one exit code.

    A missing corpus makes the scanner exit non-zero too, so a single run
    cannot tell "found the planted fragment" from "had nothing to scan".
    Two candidates that differ only in whether a repo file repeats the
    synthetic document therefore have to produce DIFFERENT exit codes."""
    temiz = build_world(tmp_path, index=0, seed=_leak_seed(),
                        corpus=[("belge.txt", CORPUS_LINE + "\n")],
                        acceptance_commands=_commands({"command_id":
                                                       "leak_scan"}))
    temiz_rapor = run(temiz)

    sizan = build_world(tmp_path, index=1, seed=_leak_seed(),
                        corpus=[("belge.txt", CORPUS_LINE + "\n")],
                        acceptance_commands=_commands({"command_id":
                                                       "leak_scan"}))
    edit(sizan, "pipeline/temiz.py", f"NOT = {CORPUS_LINE!r}\n")
    sizan_rapor = run(sizan)

    assert sizan_rapor.command_results[0].exit_code == 1, \
        "planted fragment sert bulgu uretmedi"
    assert sizan_rapor.passed is False
    assert temiz_rapor.command_results[0].exit_code != 1, \
        "temiz aday da sert bulgu verdi: korpus hic taranmamis olabilir"
    assert residue(private_roots) == []


def test_a_leak_scan_triage_exit_is_never_a_pass(tmp_path, private_roots):
    """Section 8's last rule. Exit 2 means "a human still has to look",
    and a package that treated it as success would turn the one code that
    exists to stop a run into a green light."""
    world = build_world(
        tmp_path, seed=_leak_seed(),
        corpus=[("belge.txt", CORPUS_LINE + "\n")],
        saved=[("artik.bin", "0")],
        acceptance_commands=_commands({"command_id": "leak_scan"},
                                      _selected(GECER)))
    report = run(world)
    assert report.command_results[0].exit_code == 2, \
        "senaryo kurulmadi: triyaj cikisi uretilmedi"
    assert report.passed is False, "triyaj cikisi basari sayildi"
    assert len(report.command_results) == 1, "triyajdan sonra devam edildi"
    assert residue(private_roots) == []


# =====================================================================
# 5. THE MANIFEST, PRIVACY AND WHAT IS LEFT BEHIND
# =====================================================================

def test_a_task_manifest_edited_between_commands_is_terminal(
        tmp_path, private_roots, monkeypatch):
    """Mechanism 18. The manifest chooses the commands, so a manifest
    that moves mid-run means the remaining commands were chosen by
    something nobody verified."""
    world = build_world(tmp_path, acceptance_commands=_commands(
        _selected(GECER), _selected(GECER)))
    gercek = acceptance._run_command
    sayac = []

    def izleyen(argv, **kwargs):
        # the git preparation runs through this same seam, so the edit is
        # placed after the first ACCEPTANCE command rather than after the
        # first process -- otherwise the scenario never reaches the gate
        # it is about
        if argv[0] != "git":
            sayac.append(1)
        sonuc = gercek(argv, **kwargs)
        if len(sayac) == 1 and argv[0] != "git":
            world.task.write_text(json.dumps({"protocol_version": "1.0"}),
                                  encoding="utf-8")
        return sonuc

    monkeypatch.setattr(acceptance, "_run_command", izleyen)
    with pytest.raises(acceptance.AcceptanceError) as refusal:
        run(world)
    assert str(refusal.value) == "gorev dosyasi degistirildi"
    assert len(sayac) == 1, "gorev degistikten sonra ikinci komut calisti"
    assert residue(private_roots) == []


def test_nothing_that_leaves_carries_output_paths_or_corpus(
        tmp_path, private_roots):
    """Mechanism 19. Everything a report carries is a number or a closed
    contract code -- never a byte the command printed, never the mirror,
    never a corpus fragment."""
    import dataclasses

    world = build_world(
        tmp_path, seed=_leak_seed() + [
            ("pipeline/test_konusan.py",
             "def test_k():\n    print('AYNA-NOBETCISI-" + "z" * 8
             + "')\n    assert True\n")],
        corpus=[("belge.txt", CORPUS_LINE + "\n")],
        acceptance_commands=_commands(_selected("pipeline/test_konusan.py")))
    report = run(world)

    parcalar = [str(getattr(report, alan.name))
                for alan in dataclasses.fields(report)]
    for result in report.command_results:
        parcalar += [str(getattr(result, alan.name))
                     for alan in dataclasses.fields(result)]
    tasinan = " ".join(parcalar)
    for yasak in ("AYNA-NOBETCISI", CORPUS_LINE, str(tmp_path),
                  str(private_roots.mirror), str(world.repo), "belge.txt"):
        assert yasak not in tasinan, f"sonuc {yasak!r} tasiyor"
    assert not any(isinstance(getattr(report, alan.name), (bytes, bytearray))
                   for alan in dataclasses.fields(report))


@pytest.mark.parametrize("senaryo", ["basari", "basarisizlik", "sure",
                                     "kesinti", "temizlik"])
def test_every_exit_clears_this_runs_mirror_and_spares_a_foreign_one(
        tmp_path, private_roots, monkeypatch, senaryo):
    """Mechanism 20. Success, failure, timeout, interruption and a
    cleanup that cannot finish: in the first four the mirror is gone, in
    the last the failure is STRONGLY TYPED rather than an ordinary
    acceptance failure -- and in every one of them a foreign directory
    sitting in the same root is untouched."""
    yabanci = private_roots.mirror / (acceptance_workspace.TEMP_PREFIX
                                      + "0" * 32)
    yabanci.mkdir()
    (yabanci / "kanarya.txt").write_text("YABANCI", encoding="utf-8")

    seed, commands, limits = [], _commands(_selected(GECER)), {}
    if senaryo == "basarisizlik":
        commands = _commands(_selected("pipeline/test_yok.py"))
    elif senaryo == "sure":
        seed = [("pipeline/test_uyku.py",
                 "import time\n\n\ndef test_u():\n    time.sleep(60)\n")]
        commands = _commands(_selected("pipeline/test_uyku.py"))
        limits = {"timeout_seconds": 2}
    world = build_world(tmp_path, seed=seed, acceptance_commands=commands)

    if senaryo == "kesinti":
        def kesen(*args, **kwargs):
            raise KeyboardInterrupt
        monkeypatch.setattr(acceptance, "_run_command", kesen)
        with pytest.raises(KeyboardInterrupt):
            run(world, **limits)
    elif senaryo == "temizlik":
        monkeypatch.setattr(acceptance_workspace, "_remove_tree",
                            lambda holder: False)
        with pytest.raises(acceptance_workspace.MirrorCleanupFailed):
            run(world, **limits)
    else:
        report = run(world, **limits)
        assert report.passed is (senaryo == "basari")

    kalan = residue(private_roots)
    if senaryo == "temizlik":
        assert len(kalan) == 2, "temizlik basarisizliginda ayna yok oldu"
    else:
        assert kalan == [yabanci.name], f"bu turun artigi kaldi: {kalan}"
    kanarya = (yabanci / "kanarya.txt").read_text(encoding="utf-8")
    assert kanarya == "YABANCI", "yabanci kanarya silindi"


def test_no_acceptance_module_runs_a_shell_or_takes_a_caller_environment(
        tmp_path, private_roots, monkeypatch):
    """Mechanism 12, structurally AND behaviourally. No `shell=True`, no
    `os.system`, no inline interpreter string -- and the environment the
    child actually receives is a closed, mirror-owned map that carries
    none of the runner's own variables."""
    import ast

    offenders = []
    for name in ("acceptance", "acceptance_workspace"):
        module = ROOT / "tools" / "agent_loop" / f"{name}.py"
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell" and \
                            getattr(keyword.value, "value", False) is True:
                        offenders.append(f"{name}:shell=True")
                if ast.unparse(node.func) in ("os.system", "os.popen"):
                    offenders.append(f"{name}:{ast.unparse(node.func)}")
            if isinstance(node, ast.Constant) and \
                    isinstance(node.value, str) and \
                    node.value in ("-c", "/c", "-Command"):
                offenders.append(f"{name}:{node.value}")
    assert offenders == [], f"keyfi kabuk yuzeyi: {offenders}"

    monkeypatch.setenv("KABUL_NOBETCISI", "AYNA-SIZINTISI")
    world = build_world(tmp_path)
    gorulen = {}
    gercek = acceptance._acceptance_env

    def izleyen(mirror):
        gorulen["env"] = gercek(mirror)
        gorulen["mirror"] = mirror
        return gorulen["env"]

    monkeypatch.setattr(acceptance, "_acceptance_env", izleyen)
    run(world)

    env = gorulen["env"]
    assert "KABUL_NOBETCISI" not in env, "cagiranin ortami cocuga sizdi"
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    kok = str(gorulen["mirror"].holder)
    for anahtar in ("HOME", "TEMP", "TMPDIR"):
        assert env[anahtar].startswith(kok), f"{anahtar} aynaya ait degil"
    assert str(world.repo) not in " ".join(env.values()), \
        "kabul ortami ana depoyu adliyor"
