"""PACKAGE B2B-B2A2 -- change verification over a D3A flat workspace.

ONE question: is the change set derived from the FILESYSTEM difference
between the reference tree and the implementer tree, bound to one
workspace identity, and closed against everything this evidence model
cannot represent.

WHY A SECOND FILE. The suite next door is the authorization, manifest,
main-checkout and privacy battery, and those gates did not move. This
file is only about the migrated mechanism, so a reviewer can read the
diff of the thing that actually changed. The stub machinery and the
world builder are IMPORTED from it rather than copied, so the two files
cannot drift apart in what "a bound run" means.

GIT IS NOT ABSENT, IT IS SCOPED. The main-checkout guard still runs git
against the OPERATOR'S repository until B2B-B2B, and that is deliberate.
What this package forbids is git as the authority over the flat roots,
so the test that pins it distinguishes the calls by their `-C` TARGET
rather than asserting that no git process exists.

NO REAL MODEL IS CALLED, and every refusal test proves both that its
scenario was really built and that no model process started when the
refusal was supposed to come first.
"""
from __future__ import annotations

import inspect
import os
import shutil

import pytest

import test_agent_loop_b2_changes as legacy
from tools.agent_loop import (changes, contract, execution, flat_workspace,
                              state)

RUN = legacy.RUN
SENTINEL = legacy.SENTINEL

# reuse the world builder, the stub builder, the private runner root and
# the model-launch guard: one definition of "a fake model", one
# definition of "a bound run"
build_gate = legacy.build_gate
_stub = legacy._stub
_reply = legacy._reply
_write = legacy._write
private_runner_root = legacy.private_runner_root
only_fake_models_may_run = legacy.only_fake_models_may_run


@pytest.fixture
def gate(tmp_path):
    return build_gate(tmp_path)


def _run(binary, gate_obj, **overrides):
    settings = {"prompt": "kurgu istem", "budget_usd": 1.0,
                "timeout_seconds": 60, "max_output_bytes": 65536}
    settings.update(overrides)
    return changes.run_verified_implementation(binary, **gate_obj.identity,
                                               **settings)


def _snap(kind, mode="0o644", sha256="", extra=()):
    """One projected entry, built by hand. Used only where the scenario
    cannot be produced from a git baseline -- an empty directory, for
    instance, is something git never stores."""
    return changes._Snap(kind=kind, mode=mode, sha256=sha256,
                         compare=(kind, mode, sha256) + tuple(extra))


# =====================================================================
# 1. THE SEAM AND THE IDENTITIES
# =====================================================================

def test_the_public_seam_names_a_workspace_and_never_a_path():
    """`worktree_id` is gone from the surface, `workspace_id` is
    keyword-only, and nothing that could select a directory arrived in
    its place."""
    parameters = inspect.signature(
        changes.run_verified_implementation).parameters
    assert set(parameters) == {
        "binary", "repo", "state_dir", "task_path", "manifest_digest",
        "run_id", "workspace_id", "baseline_sha", "prompt", "budget_usd",
        "timeout_seconds", "max_output_bytes", "model"}
    assert "worktree_id" not in parameters
    for escape in ("worktree", "cwd", "path", "root", "reference_root",
                   "implementer_root", "allowed_paths", "forbidden_paths",
                   "changed_files", "diff", "patch", "status", "workdir"):
        assert escape not in parameters, f"kacis parametresi: {escape}"
    for name, parameter in parameters.items():
        if name != "binary":
            assert parameter.kind == inspect.Parameter.KEYWORD_ONLY, name


def test_the_result_carries_the_workspace_identity_and_stays_frozen(
        tmp_path, gate):
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    verified = _run(binary, gate)
    assert verified.workspace_id == gate.workspace_id
    assert not hasattr(verified, "worktree_id"), \
        "sonuc hala eski calisma agaci kimligini tasiyor"
    with pytest.raises(Exception):
        verified.changed_files = ("kurgu",)
    with pytest.raises(Exception):
        verified.kurgu_alani = 1               # slotted, not just frozen


def test_the_state_binding_is_asserted_with_the_workspace_identity(
        tmp_path, gate, monkeypatch):
    """The state document has to be asked about THIS workspace. Asking
    with nothing at all would be answered by any binding, which is the
    hole `state.assert_binding` was given two identity slots to close."""
    gercek = state.assert_binding
    gorulen = []

    def izleyen(state_dir, **kwargs):
        gorulen.append(dict(kwargs))
        return gercek(state_dir, **kwargs)

    monkeypatch.setattr(state, "assert_binding", izleyen)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    _run(binary, gate)

    assert gorulen, "senaryo kurulmadi: durum baglamasi hic sorulmadi"
    for cagri in gorulen:
        assert cagri["workspace_id"] == gate.workspace_id
        assert cagri.get("worktree_id") is None, \
            "durum baglamasi hala calisma agaci kimligi soruyor"
        assert cagri["repo_id"] == state.repo_identity(gate.repo)
        assert cagri["baseline_sha"] == gate.baseline
        assert cagri["manifest_digest"] == gate.digest
    assert len(gorulen) >= 2, "son kontrolde durum baglamasi yeniden sorulmadi"


def test_the_flat_binding_is_asked_for_exactly_the_callers_identities(
        tmp_path, gate, monkeypatch):
    gercek = flat_workspace.assert_binding
    gorulen = []

    def izleyen(repo, **kwargs):
        gorulen.append(dict(kwargs, repo=str(repo)))
        return gercek(repo, **kwargs)

    monkeypatch.setattr(flat_workspace, "assert_binding", izleyen)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    _run(binary, gate)

    assert gorulen, "senaryo kurulmadi: calisma alani baglamasi hic sorulmadi"
    ilk = gorulen[0]
    assert ilk["repo"] == str(gate.repo)
    assert str(ilk["state_dir"]) == str(gate.state_dir)
    assert ilk["run_id"] == RUN
    assert ilk["workspace_id"] == gate.workspace_id
    assert ilk["baseline_sha"] == gate.baseline


def test_the_model_runs_only_in_the_implementer_root(tmp_path, gate,
                                                     only_fake_models_may_run):
    """The witness is the CHILD's own report of its working directory --
    the caller's word for where it ran is not evidence about where it
    ran -- and the reference copy is neither reachable nor mentioned."""
    iz = tmp_path / "cocugun-cwd.txt"
    binary = _stub(tmp_path, reply=_reply(changed_files=[]), cwd_record=iz)
    verified = _run(binary, gate)

    assert only_fake_models_may_run, "senaryo kurulmadi: model hic calismadi"
    gorulen = os.path.realpath(iz.read_text(encoding="utf-8").strip())
    assert gorulen == os.path.realpath(gate.tree)
    assert gorulen != os.path.realpath(gate.reference)
    assert str(gate.reference) not in repr(verified), "referans kok sonucta"
    argv_metni = " ".join(" ".join(argv)
                          for argv in only_fake_models_may_run)
    assert str(gate.reference) not in argv_metni, "referans kok argv'de"


# =====================================================================
# 2. WHAT THE FILESYSTEM SAYS
# =====================================================================

def test_a_healthy_no_op_is_empty_although_the_copies_differ_on_disk(
        tmp_path, gate):
    """The two trees are independent copies: different file identities,
    different timestamps, the same content. A projection that compared
    either of those would refuse every healthy workspace."""
    sol = gate.reference / "pipeline" / "kurgu.py"
    sag = gate.tree / "pipeline" / "kurgu.py"
    assert os.stat(sol).st_ino != os.stat(sag).st_ino or os.name == "nt", \
        "senaryo kurulmadi: iki kopya ayni dosya nesnesi"
    assert sol.read_bytes() == sag.read_bytes(), "senaryo kurulmadi"

    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    verified = _run(binary, gate)
    assert verified.changed_files == ()
    assert (verified.added, verified.modified, verified.deleted) == (0, 0, 0)


@pytest.mark.parametrize("senaryo", ["ekleme", "ayni-boyut", "silme", "mod"])
def test_the_change_kind_comes_from_the_filesystem(tmp_path, gate, senaryo):
    """ADDED, MODIFIED and DELETED are read off the two trees. The
    same-size edit is the one a listing's own metadata was measured to
    miss 117 times in 200, which is why the content is hashed."""
    hedef = "pipeline/kurgu.py"
    if senaryo == "ekleme":
        hedef = "pipeline/yeni.py"
        ops = [_write(hedef)]
        beklenen = (1, 0, 0)
    elif senaryo == "ayni-boyut":
        # exactly as many BYTES as `VALUE = 1\n`, and no newline of its
        # own: a text-mode write turns "\n" into two bytes on Windows,
        # which would make the scenario about size after all
        eski = (gate.tree / hedef).stat().st_size
        ops = [{"kind": "write", "path": hedef, "text": "V" * eski}]
        beklenen = (0, 1, 0)
    elif senaryo == "silme":
        hedef = "pipeline/silinecek.py"
        ops = [{"kind": "delete", "path": hedef}]
        beklenen = (0, 0, 1)
    else:
        ops = [{"kind": "chmod", "path": hedef, "mode": "0444"}]
        beklenen = (0, 1, 0)

    binary = _stub(tmp_path, ops=ops, reply=_reply(changed_files=[hedef]))
    verified = _run(binary, gate)
    assert verified.changed_files == (hedef,)
    assert (verified.added, verified.modified, verified.deleted) == beklenen
    if senaryo == "ayni-boyut":
        assert (gate.tree / hedef).stat().st_size == \
            (gate.reference / hedef).stat().st_size, \
            "senaryo kurulmadi: boyut da degisti"


def test_many_changes_arrive_sorted_with_the_right_counts(tmp_path, gate):
    binary = _stub(tmp_path, ops=[
        _write("pipeline/iki.py"), _write("pipeline/bir.py"),
        _write("pipeline/kurgu.py"),
        {"kind": "delete", "path": "pipeline/silinecek.py"}],
        reply=_reply(changed_files=["pipeline/silinecek.py",
                                    "pipeline/kurgu.py", "pipeline/iki.py",
                                    "pipeline/bir.py"]))
    verified = _run(binary, gate)
    assert verified.changed_files == ("pipeline/bir.py", "pipeline/iki.py",
                                      "pipeline/kurgu.py",
                                      "pipeline/silinecek.py")
    assert (verified.added, verified.modified, verified.deleted) == (2, 1, 1)


def test_the_fingerprint_is_deterministic_over_the_same_evidence(
        tmp_path, gate):
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py", "AYNI")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    verified = _run(binary, gate)
    ikinci = build_gate(tmp_path, index=1)
    tekrar = _stub(tmp_path, name="ikinci",
                   ops=[_write("pipeline/kurgu.py", "AYNI")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    assert _run(tekrar, ikinci).fingerprint == verified.fingerprint, \
        "ayni kanit iki farkli parmak izi verdi"


def test_the_classifier_keeps_modes_and_refuses_empty_directory_changes():
    """Two cases a git baseline cannot produce, pinned on the classifier
    itself: git stores no empty directory at all, so an empty one that
    appears or disappears without a single file change is refused rather
    than guessed at -- and a deletion keeps the mode it had, which is
    the field that tells two otherwise identical deletions apart."""
    taban = {"a.py": _snap("file", mode="0o755", sha256="a" * 64),
             "dizin": _snap("dir", mode="0o755")}
    silinmis = {"dizin": _snap("dir", mode="0o755")}
    (degisiklik,) = changes._semantic_changes(taban, silinmis)
    assert degisiklik.kind == changes.DELETED
    assert degisiklik.mode == "0o755", "silme taban modunu tasimiyor"
    assert degisiklik.sha256 == ""

    with pytest.raises(changes.UnsafeChange) as bos:
        changes._semantic_changes(
            taban, dict(taban, yeni=_snap("dir", mode="0o755")))
    assert bos.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    with pytest.raises(changes.UnsafeChange):
        changes._semantic_changes(taban, {"a.py": taban["a.py"]})
    # a directory whose metadata alone moved is not a change this gate
    # knows how to carry either
    with pytest.raises(changes.UnsafeChange):
        changes._semantic_changes(
            taban, dict(taban, dizin=_snap("dir", mode="0o700")))
    # ... but a directory that appears BECAUSE a file appeared under it
    # is ordinary work
    buyuyen = dict(taban, yeni=_snap("dir", mode="0o755"))
    buyuyen["yeni/b.py"] = _snap("file", sha256="b" * 64)
    assert {c.path for c in changes._semantic_changes(taban, buyuyen)} \
        == {"yeni/b.py"}


# =====================================================================
# 3. THE DECLARATION IS COMPARED TO THE EVIDENCE
# =====================================================================

@pytest.mark.parametrize("bildirim", ["eksik", "fazla", "takma-yineleme"])
def test_a_declaration_that_does_not_match_the_evidence_is_refused(
        tmp_path, gate, bildirim):
    """A forgotten file is as much a mismatch as an invented one, and a
    duplicate cannot be smuggled past by spelling the same path two
    different ways."""
    ops = [_write("pipeline/bir.py"), _write("pipeline/iki.py")]
    if bildirim == "eksik":
        declared = ["pipeline/bir.py"]
    elif bildirim == "fazla":
        declared = ["pipeline/bir.py", "pipeline/iki.py", "pipeline/yok.py"]
    else:
        declared = ["pipeline/bir.py", "./pipeline//bir.py",
                    "pipeline/iki.py"]
    binary = _stub(tmp_path, ops=ops, reply=_reply(changed_files=declared))
    with pytest.raises(changes.DeclarationMismatch) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.SCHEMA_VIOLATION
    assert (gate.tree / "pipeline" / "iki.py").exists(), \
        "senaryo kurulmadi: dosyalar hic yazilmadi"


@pytest.mark.parametrize(
    "hedef", ["pipeline/gizli/a.py", "tools/agent_loop/contract.py",
              "kurgu-task.json"])
def test_an_unauthorised_path_is_refused_over_filesystem_evidence(
        tmp_path, gate, hedef):
    """The authority order is unchanged by the migration: control plane,
    then the task file, then forbidden, then allowed -- now applied to
    paths the filesystem reported rather than paths git did."""
    binary = _stub(tmp_path, ops=[_write(hedef)],
                   reply=_reply(changed_files=[hedef]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason in (contract.StopReason.PATH_NOT_ALLOWED,
                                    contract.StopReason.CONTROL_PLANE_MODIFIED)
    assert (gate.tree / hedef).exists(), "senaryo kurulmadi: dosya yazilmadi"


# =====================================================================
# 4. THE WORKSPACE ITSELF
# =====================================================================

def test_a_workspace_that_starts_out_of_step_is_refused_before_the_model(
        tmp_path, gate, only_fake_models_may_run):
    """Attribution is impossible when the two trees did not start equal,
    so the refusal has to come before a process exists."""
    (gate.tree / "pipeline" / "onceden.py").write_text("X\n",
                                                       encoding="utf-8")
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.DIRTY_WORKTREE
    assert only_fake_models_may_run == [], \
        "basta ayrisik calisma alaninda model calistirildi"


def test_a_reference_tree_edited_during_the_call_is_terminal(tmp_path, gate):
    """The reference copy is what the work gets compared against. If it
    moved, the comparison is against something the model chose, and no
    change set derived from it means anything."""
    hedef = gate.reference / "pipeline" / "kurgu.py"
    binary = _stub(tmp_path, ops=[_write(hedef, "REFERANS-OYNADI")],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert "REFERANS-OYNADI" in hedef.read_text(encoding="utf-8"), \
        "senaryo kurulmadi: referans agac hic degismedi"


@pytest.mark.parametrize("alan", ["repo", "run_id", "baseline_sha"])
def test_a_workspace_bound_to_something_else_is_refused_before_the_model(
        tmp_path, gate, only_fake_models_may_run, alan):
    """A workspace id on its own authorises nothing. The state binding is
    rewritten to name the foreign workspace on purpose, so this refusal
    can only come from the workspace ledger -- the state gate in front of
    it agrees with the call."""
    if alan == "repo":
        baska, baska_taban = legacy.build_repo(tmp_path, index=91)
        yabanci = flat_workspace.create(baska, state_dir=gate.state_dir,
                                        run_id=RUN, baseline_sha=baska_taban)
    elif alan == "run_id":
        yabanci = flat_workspace.create(gate.repo, state_dir=gate.state_dir,
                                        run_id="baska-kosu",
                                        baseline_sha=gate.baseline)
    else:
        (gate.repo / "pipeline" / "ikinci.py").write_text("VALUE = 9\n",
                                                          encoding="utf-8")
        legacy._git(gate.repo, "add", "-A")
        legacy._git(gate.repo, "commit", "-qm", "ikinci")
        ikinci = legacy._git(gate.repo, "rev-parse", "HEAD")
        yabanci = flat_workspace.create(gate.repo, state_dir=gate.state_dir,
                                        run_id=RUN, baseline_sha=ikinci)

    state.write_binding(gate.state_dir, {
        "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
        "repo_id": state.repo_identity(gate.repo),
        "baseline_sha": gate.baseline, "manifest_digest": gate.digest,
        "workspace_id": yabanci.workspace_id})
    kimlik = dict(gate.identity, workspace_id=yabanci.workspace_id)

    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        changes.run_verified_implementation(
            binary, **kimlik, prompt="kurgu istem", budget_usd=1.0,
            timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_models_may_run == [], \
        "yetkisiz calisma alaninda model calistirildi"
    if alan == "baseline_sha":
        assert refusal.value.reason == contract.StopReason.BASELINE_MISMATCH
    metin = str(refusal.value) + repr(refusal.value)
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"


def test_a_replaced_workspace_root_is_refused_before_the_model(
        tmp_path, gate, only_fake_models_may_run):
    """The replacement holds the same bytes under the same names; only
    the OBJECT differs. D3A recorded the root identity for exactly this,
    and this gate inherits the refusal rather than re-deriving it."""
    kok = gate.tree
    kenara = kok.parent / (kok.name + "-eski")
    kok.rename(kenara)
    shutil.copytree(kenara, kok)
    assert (kok / "pipeline" / "kurgu.py").is_file(), "senaryo kurulmadi"

    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert only_fake_models_may_run == [], \
        "degistirilmis koke ragmen model calistirildi"
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


@pytest.mark.parametrize("tur", ["symlink", "junction", "fifo", "bos-dizin",
                                 "dosyadan-dizine"])
def test_an_object_this_evidence_model_cannot_represent_is_refused(
        tmp_path, gate, tur):
    """A symlink, a junction, a special file and a bare directory each
    mean something different, and inventing a representation for one is
    how an unreviewed change reaches a patch.

    THE SENTENCE IS ASSERTED, NOT ONLY THE REASON. Every refusal here
    closes with `path_not_allowed`, so a reason-only assertion stayed
    green when the type gate was deleted: a symlink simply fell through
    to the empty-directory refusal, which is a different gate answering
    a question it was never asked. Measured -- the mutant survived until
    this line said which refusal it wanted."""
    beklenen = {"symlink": "calisma alaninda temsil edilemeyen giris",
                "junction": "calisma alaninda temsil edilemeyen giris",
                "fifo": "siradan olmayan dosya turu",
                "bos-dizin": "bos dizin degisikligi desteklenmiyor",
                "dosyadan-dizine": "yol turu degistirildi"}[tur]
    if tur == "symlink":
        sonda = tmp_path / "sonda-baglanti"
        try:
            os.symlink(str(tmp_path), str(sonda), target_is_directory=True)
        except OSError:
            pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
        ops = [{"kind": "symlink", "path": "pipeline/baglanti.py",
                "target": SENTINEL + ".py"}]
    elif tur == "junction":
        if os.name != "nt":
            pytest.skip("kavsak noktasi Windows'a ozgu")
        hedef = tmp_path / "kavsak-hedefi"
        hedef.mkdir(exist_ok=True)
        ops = [{"kind": "junction", "path": "pipeline/kavsak",
                "target": str(hedef)}]
    elif tur == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("bu platformda FIFO yaratilamiyor")
        ops = [{"kind": "mkfifo", "path": "pipeline/boru"}]
    elif tur == "bos-dizin":
        ops = [{"kind": "mkdir", "path": "pipeline/bos"}]
    else:
        ops = [{"kind": "delete", "path": "pipeline/kurgu.py"},
               {"kind": "mkdir", "path": "pipeline/kurgu.py"}]

    binary = _stub(tmp_path, ops=ops, reply=_reply(changed_files=[]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert str(refusal.value) == beklenen, "beklenen kapi reddetmedi"
    metin = str(refusal.value) + repr(refusal.value)
    assert SENTINEL not in metin, "ret metni baglanti hedefini tasiyor"
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"


# =====================================================================
# 5. THE CALL BOUNDARY, PRIVACY AND THE GIT LINE
# =====================================================================

def test_an_edit_made_by_a_failing_call_is_still_measured(tmp_path, gate):
    """A call that failed may have edited files first, so verification
    runs on every exit path -- and a safety violation outranks the
    failure that hid it."""
    binary = _stub(tmp_path, ops=[_write("pipeline/gizli/a.py")],
                   reply=_reply(changed_files=[]), code=5)
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED
    assert isinstance(refusal.value.__cause__, execution.ProcessFailed), \
        "asil hata zincirlenmedi"
    assert (gate.tree / "pipeline" / "gizli" / "a.py").exists(), \
        "senaryo kurulmadi: dosya hic yazilmadi"


def test_nothing_that_leaves_carries_bytes_or_absolute_paths(tmp_path, gate):
    import dataclasses

    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    verified = _run(binary, gate)
    degerler = [getattr(verified, alan.name)
                for alan in dataclasses.fields(verified)]
    tasinan = " ".join(str(deger) for deger in degerler)
    assert SENTINEL not in tasinan, "sonuc dosya icerigi tasiyor"
    assert str(tmp_path) not in tasinan, "sonuc mutlak yol tasiyor"
    assert str(gate.reference) not in tasinan
    assert not any(isinstance(deger, (bytes, bytearray))
                   for deger in degerler)


def test_git_never_asks_about_the_flat_roots_but_still_guards_the_checkout(
        tmp_path, gate, monkeypatch):
    """THE LINE THIS PACKAGE DRAWS, measured by CWD rather than by
    program name. The flat workspace's evidence may not come from git;
    the operator's checkout is still guarded by it until B2B-B2B."""
    gercek = changes.subprocess.run
    cagrilar = []

    def kaydeden(argv, **kwargs):
        cagrilar.append([str(parca) for parca in argv])
        return gercek(argv, **kwargs)

    monkeypatch.setattr(changes.subprocess, "run", kaydeden)
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    _run(binary, gate)

    # only the calls THIS module makes: every one of them carries the
    # switch the module pins its own git invocations with
    bizim = [argv for argv in cagrilar if "--no-optional-locks" in argv]
    assert bizim, "senaryo kurulmadi: modul hic git calistirmadi"
    holder = str(flat_workspace.holder_for(gate.workspace_id))
    for argv in bizim:
        hedef = argv[argv.index("-C") + 1]
        assert os.path.realpath(hedef) == os.path.realpath(gate.repo), \
            "git cagrisi ana checkout disinda bir dizini hedefliyor"
        assert not any(holder.casefold() in parca.casefold()
                       for parca in argv), "git argv'si calisma alanini tasiyor"
    # and the guard it serves is still live: the inventory really was
    # taken on both sides of the model call
    assert len([argv for argv in bizim if "status" in argv]) >= 2, \
        "ana checkout envanteri iki yanda alinmadi"
