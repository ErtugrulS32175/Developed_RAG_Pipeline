"""PACKAGE B2B-B2B -- the main checkout, guarded by filesystem evidence.

ONE question: does the operator's checkout get read back through the
handle-bound walker on both sides of the model call, with one key and
one policy, and is every top-level root really inside that answer.

WHY THE GIT GUARD HAD TO GO. `git status` answers about a WORKING TREE
as git has been configured to see it. An entry marked `skip-worktree` or
`assume-unchanged` is one git was TOLD to stop looking at: the bytes on
disk move and the inventory stays empty. The old guard compared that
inventory on both sides of the call, so a flag set BEFORE the call was
invisible to it in exactly the same way on both sides -- the one open
limit B2B-B2A2 closed its report with. The filesystem does not have an
opinion about which of its files git is watching.

WHAT THE POLICY IS. Every top-level root is evidence. The only reduction
is the in-repository virtual environments, which are metadata-only for
the measured cost reason D2 recorded -- and each of those keeps its
`pyvenv.cfg` as content, because that file says which interpreter runs.

A MEASURED PLATFORM FACT THIS FILE PINS. On Windows,
`DirEntry.is_dir(follow_symlinks=False)` answers True for a JUNCTION
(measured: reparse tag 0x2000000B), and False for a symlink. A policy
that only asked `is_dir` would therefore accept a junction planted as
`something_env` and hand a whole outside tree the metadata-only class.
The reparse check is not defence in depth here; it is the check.

NO REAL MODEL IS CALLED. Fixtures build throwaway repositories under
`tmp_path` only -- never the project's own `data/`, `output/` or
`contracts/` -- and each refusal test proves its scenario was really
built before it claims the refusal.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

import test_agent_loop_b2_changes as legacy
from tools.agent_loop import changes, contract, execution, fs_evidence

RUN = legacy.RUN
SENTINEL = legacy.SENTINEL

build_gate = legacy.build_gate
_stub = legacy._stub
_reply = legacy._reply
_write = legacy._write
_git = legacy._git
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


@pytest.fixture
def scans(monkeypatch):
    """Every walker call the run makes, with the arguments it made it
    with. The guard's own claim about where it looked is not evidence
    about where it looked."""
    gercek = fs_evidence.scan
    kayit = []

    def izleyen(root, **kwargs):
        kayit.append({"root": os.path.realpath(root),
                      "key": kwargs.get("key"),
                      "metadata_only": tuple(kwargs.get("metadata_only", ())),
                      "content_always": tuple(kwargs.get("content_always",
                                                         ()))})
        return gercek(root, **kwargs)

    monkeypatch.setattr(fs_evidence, "scan", izleyen)
    return kayit


def _main_scans(kayit, gate_obj):
    return [c for c in kayit
            if c["root"] == os.path.realpath(gate_obj.repo)]


def _env(repo, name="kurgu_env", cfg="home = kurgu"):
    """A stand-in virtual environment: the config file the policy keeps
    as content, and two look-alikes it must NOT."""
    kok = repo / name
    (kok / "alt").mkdir(parents=True)
    (kok / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    (kok / "pyvenv.cfg.bak").write_text(cfg, encoding="utf-8")
    (kok / "alt" / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    (kok / "kutuphane.py").write_text("VALUE = 1", encoding="utf-8")
    return kok


def _same_size_text(path: Path) -> str:
    """As many characters as the file has BYTES, and no newline of its
    own: a text-mode write turns "\\n" into two bytes on Windows, which
    would make a same-size scenario about size after all."""
    return "X" * path.stat().st_size


# =====================================================================
# 1. THE INSTRUMENT
# =====================================================================

def test_the_main_guard_reads_the_filesystem_and_never_git(
        tmp_path, gate, scans, only_fake_models_may_run):
    """POSITIVE CONTROL and structural pin in one: the walker really is
    asked about the repository root twice, and the module has no way
    left to ask git anything."""
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    _run(binary, gate)

    assert len(_main_scans(scans, gate)) == 2, \
        "ana checkout modelin iki yaninda taranmadi"
    for kaldirilan in ("_git", "_status", "_head", "_index_state",
                       "_tree_snapshot", "subprocess", "GIT_TIMEOUT_SECONDS"):
        assert not hasattr(changes, kaldirilan), \
            f"eski git mekanizmasi hala duruyor: {kaldirilan}"

    source = Path(changes.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            adlar = [alias.name for alias in node.names]
            assert "subprocess" not in adlar, "subprocess yeniden ice aktarildi"
        if isinstance(node, ast.Call):
            cagrilan = ast.unparse(node.func)
            assert not cagrilan.startswith("subprocess."), cagrilan
            assert cagrilan not in ("os.system", "os.popen"), cagrilan
            assert not cagrilan.endswith((".write_text", ".write_bytes",
                                          ".unlink", ".rename", ".mkdir",
                                          ".rmtree", ".chmod")), cagrilan
    # and nothing but the fake model ran inside the window
    assert len(only_fake_models_may_run) == 1, \
        "cagri sirasinda sahte modelden baska program calisti"


def test_before_and_after_use_one_root_one_key_and_one_policy(tmp_path, gate,
                                                              scans):
    """A second key would make two healthy manifests incomparable; a
    second policy would let a root be evidence on one side only."""
    _env(gate.repo)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    _run(binary, gate)

    ana = _main_scans(scans, gate)
    assert len(ana) == 2
    assert ana[0]["key"] == ana[1]["key"], "iki yanda ayni anahtar yok"
    assert ana[0]["metadata_only"] == ana[1]["metadata_only"]
    assert ana[0]["content_always"] == ana[1]["content_always"]
    assert ana[0]["metadata_only"] == ("kurgu_env",)
    assert ana[0]["content_always"] == ("kurgu_env/pyvenv.cfg",)
    # the flat roots keep their own call key: one leaked key is one
    # replayable key, and these are different questions
    duz = [c for c in scans if c["root"] != os.path.realpath(gate.repo)]
    assert duz, "senaryo kurulmadi: duz calisma alani hic taranmadi"
    assert all(c["key"] != ana[0]["key"] for c in duz), \
        "ana checkout ve duz kokler ayni anahtari paylasiyor"
    assert all(c["metadata_only"] == () for c in duz), \
        "duz kok politikasi degistirildi"


# =====================================================================
# 2. WHAT THE GUARD CATCHES
# =====================================================================

def test_a_replaced_repository_root_is_caught(tmp_path, gate):
    """The replacement holds the same bytes under the same names; only
    the OBJECT differs. `root_identity` is in the snapshot for this."""
    kenara = gate.repo.parent / (gate.repo.name + "-eski")
    binary = _stub(tmp_path, ops=[
        {"kind": "rename", "src": str(gate.repo), "path": str(kenara)},
        {"kind": "copytree", "src": str(kenara), "path": str(gate.repo)}],
        reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert (gate.repo / "pipeline" / "kurgu.py").is_file(), \
        "senaryo kurulmadi: kok yerine kopya konmadi"
    assert str(refusal.value) == "ana calisma agaci degistirildi"
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


def test_a_same_size_content_change_is_caught(tmp_path, gate):
    """The change a listing's own metadata was measured to miss 117
    times in 200. Content evidence is why it cannot hide here."""
    hedef = gate.repo / "pipeline" / "kurgu.py"
    onceki = hedef.read_bytes()
    binary = _stub(tmp_path, ops=[{"kind": "write", "path": str(hedef),
                                   "text": _same_size_text(hedef)}],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert hedef.stat().st_size == len(onceki), "senaryo kurulmadi: boyut oynadi"
    assert hedef.read_bytes() != onceki, "senaryo kurulmadi: icerik ayni"
    assert str(refusal.value) == "ana calisma agaci degistirildi"


@pytest.mark.parametrize("bayrak", ["--skip-worktree", "--assume-unchanged"])
def test_a_blinded_index_cannot_hide_the_edit(tmp_path, gate, bayrak):
    """THE OPEN LIMIT B2B-B2A2 CLOSED ITS REPORT WITH. The flag is set
    BEFORE the call, so the old guard read the same blind inventory on
    both sides and saw nothing at all."""
    hedef = gate.repo / "pipeline" / "kurgu.py"
    _git(gate.repo, "update-index", bayrak, "pipeline/kurgu.py")
    isaretli = [satir for satir in _git(gate.repo, "ls-files", "-v").splitlines()
                if satir.endswith("pipeline/kurgu.py")]
    assert isaretli and isaretli[0][:1] in ("S", "h"), \
        f"senaryo kurulmadi: indeks bayragi kurulmadi ({isaretli})"

    onceki = hedef.read_bytes()
    binary = _stub(tmp_path, ops=[{"kind": "write", "path": str(hedef),
                                   "text": _same_size_text(hedef)}],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)

    assert hedef.read_bytes() != onceki, "senaryo kurulmadi: dosya degismedi"
    assert "kurgu.py" not in _git(gate.repo, "status", "--porcelain"), \
        "senaryo kurulmadi: git hala goruyor, korluk kurulmadi"
    assert str(refusal.value) == "ana calisma agaci degistirildi", \
        "ret dosya sistemi guard'indan gelmedi"
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


@pytest.mark.parametrize("nasil", ["yeni", "degisen"])
def test_an_ignored_path_is_not_invisible(tmp_path, gate, nasil):
    """`data/` is gitignored in this fixture, so git never lists it.
    Being unlisted is not being unchanged."""
    (gate.repo / "data").mkdir(exist_ok=True)
    mevcut = gate.repo / "data" / "onceden.bin"
    mevcut.write_text("ILK ICERIK", encoding="utf-8")
    hedef = (gate.repo / "data" / "yeni.bin") if nasil == "yeni" else mevcut
    ops = ([_write(hedef)] if nasil == "yeni"
           else [{"kind": "write", "path": str(hedef),
                  "text": _same_size_text(hedef)}])
    binary = _stub(tmp_path, ops=ops, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert hedef.exists(), "senaryo kurulmadi: dosya yazilmadi"
    assert "data" not in _git(gate.repo, "status", "--porcelain"), \
        "senaryo kurulmadi: yol gitignore'lu degil"
    assert str(refusal.value) == "ana calisma agaci degistirildi"


@pytest.mark.parametrize("kok", [".git", "data", "output", "logs", "uploads",
                                 "contracts", "bilinmeyen_yeni_kok"])
def test_a_change_under_any_top_level_root_is_caught(tmp_path, gate, kok):
    """NO ROOT IS SKIPPED. Each of these was proposed at some point as
    "not really source", and each of them is somewhere a model could
    write while every other gate stayed quiet."""
    dizin = gate.repo / kok
    dizin.mkdir(parents=True, exist_ok=True)
    hedef = dizin / "nobetci.txt"
    binary = _stub(tmp_path, ops=[_write(hedef)],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert hedef.exists(), "senaryo kurulmadi: hedef koke yazilmadi"
    assert str(refusal.value) == "ana calisma agaci degistirildi"
    assert refusal.value.reason == contract.StopReason.PATH_NOT_ALLOWED


# =====================================================================
# 3. THE POLICY
# =====================================================================

def test_only_real_top_level_env_directories_become_metadata_only(tmp_path):
    """MEASURED, not assumed: on Windows a junction answers
    `is_dir(follow_symlinks=False)` with True. A policy that asked only
    that question would hand an outside tree the metadata-only class by
    letting somebody plant `kavsak_env` in the repository root."""
    repo = tmp_path / "policy-depo"
    (repo / "gercek_env" / "alt").mkdir(parents=True)
    (repo / "alt_dizin" / "ic_env").mkdir(parents=True)
    (repo / "notenv").mkdir()
    (repo / "dosya_env").write_text("K", encoding="utf-8")
    disarisi = tmp_path / "disarisi"
    disarisi.mkdir()

    kuruldu = []
    try:
        import _winapi
        _winapi.CreateJunction(str(disarisi), str(repo / "kavsak_env"))
        kuruldu.append("kavsak_env")
    except (ImportError, OSError):
        pass
    try:
        os.symlink(str(disarisi), str(repo / "bag_env"),
                   target_is_directory=True)
        kuruldu.append("bag_env")
    except OSError:
        pass

    metadata_only, content_always = changes._main_policy(repo)
    assert metadata_only == ("gercek_env",), \
        f"policy yanlis kok secti (kurulan baglantilar: {kuruldu})"
    assert content_always == ("gercek_env/pyvenv.cfg",)
    if os.name == "nt" and "kavsak_env" in kuruldu:
        # the measurement this test exists for, asserted rather than
        # described: the cheap check really does say yes to a junction
        with os.scandir(repo) as girdiler:
            kavsak = next(g for g in girdiler if g.name == "kavsak_env")
            assert kavsak.is_dir(follow_symlinks=False), \
                "olcum degismis: junction artik is_dir(no-follow) degil"


@pytest.mark.parametrize("hedef", ["pyvenv.cfg", "pyvenv.cfg.bak",
                                   "alt/pyvenv.cfg"])
def test_the_env_config_file_is_exact_content_always(tmp_path, hedef):
    """EXACT means exact. `pyvenv.cfg` is content because it says which
    interpreter runs; a look-alike beside it and a namesake one level
    down are ordinary environment files and stay metadata-only.

    Every case rewrites the SAME NUMBER OF BYTES and puts the timestamp
    back, so what separates them is the evidence class and nothing
    else -- which is asserted on disk before the verdict is read."""
    gate_obj = build_gate(tmp_path)
    kok = _env(gate_obj.repo)
    dosya = kok / hedef
    onceki = dosya.read_bytes()
    onceki_stat = dosya.stat()
    binary = _stub(tmp_path, ops=[{"kind": "write_keep_mtime",
                                   "path": str(dosya),
                                   "text": "home = KURGU"}],
                   reply=_reply(changed_files=[]))

    yakalandi = None
    try:
        _run(binary, gate_obj)
    except changes.UnsafeChange as refusal:
        yakalandi = refusal

    assert dosya.read_bytes() != onceki, "senaryo kurulmadi: icerik degismedi"
    assert dosya.stat().st_size == onceki_stat.st_size, \
        "senaryo kurulmadi: boyut oynadi"
    assert dosya.stat().st_mtime_ns == onceki_stat.st_mtime_ns, \
        "senaryo kurulmadi: mtime geri alinmadi"
    if hedef == "pyvenv.cfg":
        assert yakalandi is not None, "exact content-always dosyasi korunmadi"
        assert str(yakalandi) == "ana calisma agaci degistirildi"
    else:
        assert yakalandi is None, \
            "metadata-only sinir bu dosya icin genisletilmis"


@pytest.mark.parametrize("nasil", ["ekleme", "silme", "boyut", "mtime"])
def test_metadata_only_still_catches_structural_change(tmp_path, nasil):
    """What the reduced class DOES claim: addition, deletion, size,
    timestamp -- everything except bytes that moved while every metadata
    field stayed identical."""
    gate_obj = build_gate(tmp_path)
    kok = _env(gate_obj.repo)
    dosya = kok / "kutuphane.py"
    if nasil == "ekleme":
        ops = [_write(kok / "yeni_modul.py")]
    elif nasil == "silme":
        ops = [{"kind": "delete", "path": str(dosya)}]
    elif nasil == "boyut":
        ops = [{"kind": "write", "path": str(dosya), "text": "VALUE = 12345"}]
    else:
        ops = [{"kind": "write", "path": str(dosya),
                "text": _same_size_text(dosya)}]

    binary = _stub(tmp_path, ops=ops, reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate_obj)
    assert str(refusal.value) == "ana calisma agaci degistirildi"


def test_the_metadata_only_content_gap_is_a_known_open_limit(tmp_path):
    """NOT A SECURITY RESULT. This test records what the metadata-only
    class cannot see, so the limit stays measured instead of assumed: a
    rewrite that keeps the size, puts the timestamp back and leaves the
    file identity alone passes unseen.

    It is bounded by two measured facts and one contract -- the
    timestamp quantum here is about a millisecond, the implementer's
    tools are Read/Glob/Grep/Edit/Write with no shell, and hashing the
    virtual environments was measured at 1.1 minutes warm per pass
    against 16 seconds for the handle-bound metadata walk."""
    gate_obj = build_gate(tmp_path)
    kok = _env(gate_obj.repo)
    dosya = kok / "kutuphane.py"
    onceki = dosya.read_bytes()
    onceki_stat = dosya.stat()

    binary = _stub(tmp_path, ops=[{"kind": "write_keep_mtime",
                                   "path": str(dosya),
                                   "text": _same_size_text(dosya)}],
                   reply=_reply(changed_files=[]))
    verified = _run(binary, gate_obj)

    sonraki_stat = dosya.stat()
    assert dosya.read_bytes() != onceki, "senaryo kurulmadi: baytlar ayni"
    assert sonraki_stat.st_size == onceki_stat.st_size
    assert sonraki_stat.st_mtime_ns == onceki_stat.st_mtime_ns
    assert sonraki_stat.st_mode == onceki_stat.st_mode
    assert verified.changed_files == (), \
        "acik sinir kapanmis olabilir: bu testi guvenlik basarisi sayma"


def test_an_env_created_during_the_call_is_content_evidence(tmp_path, gate):
    """The policy is frozen before the model starts, so a root that did
    not exist then cannot claim the reduced class for itself."""
    yeni = gate.repo / "yeni_env"
    binary = _stub(tmp_path, ops=[_write(yeni / "kutuphane.py")],
                   reply=_reply(changed_files=[]))
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert (yeni / "kutuphane.py").exists(), "senaryo kurulmadi"
    assert str(refusal.value) == "ana calisma agaci degistirildi"


# =====================================================================
# 4. FAILURE, ORDER AND PRIVACY
# =====================================================================

def test_a_failed_scan_before_the_model_starts_nothing(
        tmp_path, gate, monkeypatch, only_fake_models_may_run):
    gercek = fs_evidence.scan
    sayac = {"n": 0}

    def bozuk(root, **kwargs):
        if os.path.realpath(root) == os.path.realpath(gate.repo):
            sayac["n"] += 1
            raise fs_evidence.EvidenceError("kurgu tarama arizasi")
        return gercek(root, **kwargs)

    monkeypatch.setattr(fs_evidence, "scan", bozuk)
    binary = _stub(tmp_path, reply=_reply(changed_files=[]))
    with pytest.raises(changes.ChangeSetError):
        _run(binary, gate)
    assert sayac["n"] == 1, "senaryo kurulmadi: hedef dikise ulasilmadi"
    assert only_fake_models_may_run == [], \
        "tarama arizasina ragmen model calistirildi"


def test_a_failed_scan_after_the_model_refuses_the_result(tmp_path, gate,
                                                          monkeypatch):
    """And the operating system's own text does not come with it."""
    gercek = fs_evidence.scan
    sayac = {"n": 0}

    def sonra_bozuk(root, **kwargs):
        if os.path.realpath(root) == os.path.realpath(gate.repo):
            sayac["n"] += 1
            if sayac["n"] > 1:
                raise OSError(SENTINEL)
        return gercek(root, **kwargs)

    monkeypatch.setattr(fs_evidence, "scan", sonra_bozuk)
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py"]))
    with pytest.raises(changes.ChangeSetError) as refusal:
        _run(binary, gate)
    assert sayac["n"] > 1, "senaryo kurulmadi: ikinci tarama hic yapilmadi"
    metin = (str(refusal.value) + repr(refusal.value)
             + " ".join(getattr(refusal.value, "__notes__", []))
             + str(refusal.value.__cause__) + str(refusal.value.__context__))
    assert SENTINEL not in metin, "ret metni isletim sistemi metnini tasiyor"
    assert str(tmp_path) not in metin, "ret metni mutlak yol tasiyor"


def test_a_main_checkout_change_outranks_a_model_failure(tmp_path, gate):
    """A call that failed may have edited the checkout first, and the
    safety violation is the headline."""
    binary = _stub(tmp_path, ops=[_write(gate.repo / "pipeline" / "kurgu.py")],
                   reply=_reply(changed_files=[]), code=5)
    with pytest.raises(changes.UnsafeChange) as refusal:
        _run(binary, gate)
    assert str(refusal.value) == "ana calisma agaci degistirildi"
    assert isinstance(refusal.value.__cause__, execution.ProcessFailed), \
        "asil hata zincirlenmedi"


def test_an_untouched_checkout_accepts_the_flat_change_set(tmp_path, gate,
                                                           scans):
    """The guard is a gate, not a filter: work done in the workspace
    comes back exactly as B2B-B2A2 measured it."""
    _env(gate.repo)
    binary = _stub(tmp_path, ops=[_write("pipeline/kurgu.py"),
                                  _write("pipeline/yeni.py")],
                   reply=_reply(changed_files=["pipeline/kurgu.py",
                                               "pipeline/yeni.py"]))
    verified = _run(binary, gate)
    assert verified.changed_files == ("pipeline/kurgu.py", "pipeline/yeni.py")
    assert (verified.added, verified.modified, verified.deleted) == (1, 1, 0)
    # nothing but the model and the evidence happened in between
    assert len(_main_scans(scans, gate)) == 2
    assert len(scans) == 6, f"beklenmeyen tarama sayisi: {len(scans)}"
