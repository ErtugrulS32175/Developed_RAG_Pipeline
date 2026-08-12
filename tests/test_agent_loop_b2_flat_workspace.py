"""PACKAGE B2B-A-D3A -- the git-less flat workspace.

WHY A FLAT TREE AND NOT A WORKTREE. Git was the evidence authority
twice and lost twice, both times to state the model could reach: the
per-repository index, and shared metadata where a clean filter declared
in `.git/config` hid a change AND executed a model-supplied command
during verification. A disposable git worktree carries a `.git` link
into the model's reach, so the workspace this package builds has no
`.git` at all -- two independent plain trees materialised from the raw
blob bytes of one commit.

WHAT "MATERIALISED FROM RAW OBJECTS" BUYS, and why `git checkout` is
not allowed to do it: checkout runs clean/smudge filters, applies EOL
conversion and fires hooks. Every one of those is a place the tree
being copied gets to influence the copy. Reading blob objects and
writing the bytes out is the only path where the repository's own
configuration has no say.

THE TESTS BELOW ARE ADVERSARIAL BY DEFAULT. Each attack test proves
three separate things: that the hostile fixture was really built, that
the attack callback was really reached, and that the refusal is typed
and closed rather than merely "an error happened". A positive control
sits next to each one, because a materializer that refuses everything
would otherwise pass the whole file.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.agent_loop import flat_workspace, git_objects, git_transport

# the repository root, for the nested probe's PYTHONPATH
_DEPO_KOKU = Path(__file__).resolve().parents[1]

SENTINEL = "KURGU-GIZLI-NOBETCI-" + "z" * 8
_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------
# disposable git repositories -- argv only, never a shell
# ---------------------------------------------------------------------

def _git(repo, *args, stdin=None, binary_out=False):
    done = subprocess.run(
        ["git", "-C", str(repo), *args], input=stdin,
        capture_output=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""})
    if done.returncode != 0:
        raise AssertionError(
            f"fikstur git komutu basarisiz: {args[0]} rc={done.returncode}")
    return done.stdout if binary_out else done.stdout.decode("utf-8",
                                                             "replace")


def _new_repo(tmp_path, name="depo") -> Path:
    repo = tmp_path / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.autocrlf", "false")
    return repo


def _blob(repo, data: bytes) -> str:
    return _git(repo, "hash-object", "-w", "-t", "blob", "--stdin",
                stdin=data).strip()


def _raw_tree(repo, girisler) -> str:
    """Write a tree object BYTE BY BYTE.

    `--literally` is the point: without it `hash-object -t tree` runs
    the very validation the materializer is being tested against, and
    refuses `.git`, `..` and duplicate names before they can ever reach
    it. A hostile repository cannot be built with porcelain that
    refuses to build it."""
    ham = b""
    for mode, ad, oid in girisler:
        ad_bayt = ad if isinstance(ad, bytes) else ad.encode("utf-8")
        ham += mode.encode("ascii") + b" " + ad_bayt + b"\0" + \
            bytes.fromhex(oid)
    return _git(repo, "hash-object", "-w", "-t", "tree", "--literally",
                "--stdin", stdin=ham).strip()


def _commit_of_tree(repo, tree: str) -> str:
    return _git(repo, "commit-tree", tree, "-m", "t").strip()


def _nested_tree(repo, dosyalar) -> str:
    """Build real subtrees. A tree object is FLAT -- a name with a
    slash in it is not a nested path, it is an invalid name."""
    dallar, yapraklar = {}, []
    for yol, deger in sorted(dosyalar.items()):
        mode, veri = deger if isinstance(deger, tuple) else ("100644", deger)
        if "/" in yol:
            bas, kalan = yol.split("/", 1)
            dallar.setdefault(bas, {})[kalan] = (mode, veri)
        else:
            yapraklar.append((mode, yol, _blob(repo, veri)))
    girisler = list(yapraklar)
    for ad, alt in sorted(dallar.items()):
        girisler.append(("40000", ad, _nested_tree(repo, alt)))
    return _raw_tree(repo, sorted(girisler, key=lambda g: g[1]))


def _commit_files(repo, dosyalar) -> str:
    """A commit built from `{path: bytes}` or `{path: (mode, bytes)}`."""
    return _commit_of_tree(repo, _nested_tree(repo, dosyalar))


@pytest.fixture(autouse=True)
def _izole_runner_koku(tmp_path, monkeypatch):
    """Give each test its OWN runner-owned root, under `tmp_path`.

    The first version snapshotted the SHARED global root and removed
    every directory that appeared during the test. That is a
    cross-run delete: a concurrent run's holder appears during the
    test through no fault of the test, and it was removed. MEASURED --
    a foreign holder's canary did not survive one teardown.

    Nothing here ever looks at the global root, so there is nothing to
    get wrong. pytest disposes of `tmp_path` itself, which is the only
    thing this fixture owns."""
    kok = tmp_path / "runner-koku"
    kok.mkdir()
    monkeypatch.setattr(flat_workspace, "runner_temp_root", lambda: kok)
    return kok


# ---------------------------------------------------------------------
# CROSS-RUN SAFETY
#
# A teardown can only be observed after it has run, and the first
# version bought that with two ORDERED tests that wrote into the
# MACHINE'S shared runner root and deleted from it -- committing the
# very act being tested against, and breaking if either test ran alone.
#
# So the scenario runs in a NESTED pytest whose TMP/TEMP/TMPDIR point
# inside this test's `tmp_path`. Its "machine-global" runner root is
# therefore disposable, the teardown is the real one, and the outer
# test reads the outcome off the filesystem after that process exits.
# One test, no ordering, nothing outside `tmp_path`.
# ---------------------------------------------------------------------

_CONFTEST_KAYNAK = "\n".join([
    '# Nested conftest for the isolation regression. Lives under a',
    '# disposable tmp_path with TMP/TEMP/TMPDIR pointed at it, so this',
    '# process\'s own "machine-global" runner root is throwaway.',
    '#',
    '# The isolated arm IMPORTS the production fixture instead of',
    '# reimplementing it: a copy staying green while the real fixture',
    '# regressed is precisely the false green this exists to prevent.',
    'import os',
    'import shutil',
    'import tempfile',
    'from pathlib import Path',
    '',
    'import pytest',
    '',
    'from tools.agent_loop import flat_workspace',
    '',
    'ESKI = os.environ.get("ESKI_DAVRANIS") == "1"',
    '',
    '# captured BEFORE any fixture runs, so it is the unpatched callable',
    '_ORIJINAL_KOK = flat_workspace.runner_temp_root',
    '',
    '_GECICI = Path(tempfile.gettempdir())',
    'ISARET = _GECICI / "teardown-calisti"',
    'KANIT = _GECICI / "kullanilan-fixture.txt"',
    '',
    '',
    'def paylasilan_kok():',
    '    """The disposable stand-in for the machine root."""',
    '    kok = _ORIJINAL_KOK()',
    '    kok.mkdir(parents=True, exist_ok=True)',
    '    return kok',
    '',
    '',
    'if ESKI:',
    '    # NEGATIVE CONTROL: the defective sweep, kept on purpose',
    '    @pytest.fixture(autouse=True)',
    '    def _temizlik(tmp_path):',
    '        kok = paylasilan_kok()',
    '        once = {p.name for p in kok.iterdir()}',
    '        yield',
    '        for p in kok.iterdir():',
    '            if p.name not in once:',
    '                shutil.rmtree(p, ignore_errors=True)',
    '        ISARET.write_bytes(b"1")',
    '',
    '    KANIT.write_text("yerel-kusurlu-fixture", encoding="utf-8")',
    'else:',
    '    # THE REAL ONE, imported rather than copied',
    '    from test_agent_loop_b2_flat_workspace import (',
    '        _izole_runner_koku as _izole_runner_koku)',
    '',
    '    KANIT.write_text(',
    '        _izole_runner_koku.__module__ + ":" + _izole_runner_koku.__name__,',
    '        encoding="utf-8")',
    '',
    '',
    'def pytest_sessionfinish(session, exitstatus):',
    '    """Runs after every fixture teardown.',
    '',
    "    For the production fixture the teardown IS monkeypatch's undo,",
    '    so the proof that it ran is that the attribute is back."""',
    '    if not ESKI:',
    '        geri = (flat_workspace.runner_temp_root is _ORIJINAL_KOK)',
    '        ISARET.write_bytes(b"1" if geri else b"0")',
])

_PROB_KAYNAK = "\n".join([
    '# The single scenario. Everything else lives in the nested conftest.',
    'from tools.agent_loop import flat_workspace',
    '',
    'from conftest import paylasilan_kok',
    '',
    '',
    'def test_bir_yabanci_holder_belirir():',
    '    """A concurrent run\'s holder appears WHILE this test runs,',
    '    which is exactly how it happens in real life."""',
    "    yabanci = paylasilan_kok() / (flat_workspace.TEMP_PREFIX + 'e' * 32)",
    '    yabanci.mkdir(parents=True, exist_ok=True)',
    "    (yabanci / 'kanarya.txt').write_bytes(b'BASKA-KOSUYA-AIT')",
    '    benim = (flat_workspace.runner_temp_root()',
    "             / (flat_workspace.TEMP_PREFIX + 'a' * 32))",
    '    benim.mkdir(parents=True, exist_ok=True)',
    "    (benim / 'kendi.txt').write_bytes(b'BENIM')",
])


@pytest.mark.parametrize("eski_davranis", [True, False],
                         ids=["eski-kuresel-tarama", "izole-kok"])
def test_the_cleanup_fixture_never_reaches_another_runs_holder(
        tmp_path, eski_davranis):
    """Both arms in one test, and the RED arm is PERMANENT.

    Under the old sweep the foreign canary must be gone. If that ever
    stops being true the test has stopped exercising the defect, and a
    green isolated arm would prove nothing."""
    sahte_tmp = tmp_path / "sahte-makine-tmp"
    sahte_tmp.mkdir()
    # `pytest_sessionfinish` is only honoured from a conftest, which
    # is also where the fixture choice belongs
    (tmp_path / "conftest.py").write_text(_CONFTEST_KAYNAK, encoding="utf-8")
    prob = tmp_path / "probe_fixture_izolasyonu.py"
    prob.write_text(_PROB_KAYNAK, encoding="utf-8")

    ortam = dict(os.environ)
    for anahtar in ("TMP", "TEMP", "TMPDIR"):
        ortam[anahtar] = str(sahte_tmp)
    ortam["PYTHONPATH"] = os.pathsep.join(
        (str(_DEPO_KOKU), str(Path(__file__).resolve().parent)))
    ortam["ESKI_DAVRANIS"] = "1" if eski_davranis else "0"
    ortam["PYTHONDONTWRITEBYTECODE"] = "1"

    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(prob), "-q",
         "-p", "no:cacheprovider", "-p", "no:randomly"],
        capture_output=True, cwd=str(tmp_path), env=ortam, timeout=300)
    cikti = done.stdout.decode("utf-8", "replace")

    kok = sahte_tmp / flat_workspace.ROOT_DIRNAME
    kanarya = (kok / (flat_workspace.TEMP_PREFIX + "e" * 32)
               / "kanarya.txt")

    assert done.returncode == 0, f"ic surec rc={done.returncode}: {cikti[-500:]}"
    assert (sahte_tmp / "teardown-calisti").exists(), \
        "senaryo kurulmadi: fixture teardown'a hic ulasilmadi"
    assert kok.is_dir(), "senaryo kurulmadi: ic surec kendi kokunu kurmadi"

    kullanilan = (sahte_tmp / "kullanilan-fixture.txt").read_text(
        encoding="utf-8")
    if eski_davranis:
        assert kullanilan == "yerel-kusurlu-fixture", \
            "senaryo kurulmadi: olumsuz kontrol fixture'i kosmadi"
        assert not kanarya.exists(), \
            "senaryo kurulmadi: eski davranis yabanci holder'i silmiyor"
    else:
        # the arm that matters ran the PRODUCTION fixture, not a copy
        assert kullanilan.endswith(":_izole_runner_koku"), \
            f"gercek fixture kosmadi: {kullanilan}"
        assert "test_agent_loop_b2_flat_workspace" in kullanilan, \
            f"baska bir moduldeki fixture kosmus: {kullanilan}"
        assert (sahte_tmp / "teardown-calisti").read_bytes() == b"1", \
            "gercek fixture'in teardown'u tamamlanmadi"
        assert kanarya.exists(), \
            "izole fixture baska kosunun holder'ini sildi"
        assert kanarya.read_bytes() == b"BASKA-KOSUYA-AIT"

    # nothing the nested run made may sit outside this test's tmp_path
    for p in kok.iterdir():
        assert str(p).startswith(str(tmp_path)), "ic surec disari yazdi"


@pytest.fixture
def repo(tmp_path):
    return _new_repo(tmp_path)


@pytest.fixture
def state_dir(tmp_path):
    yol = tmp_path / "durum"
    yol.mkdir()
    return yol


def _create(repo, state_dir, baseline, run_id="kosu-1"):
    return flat_workspace.create(repo, state_dir=state_dir, run_id=run_id,
                                 baseline_sha=baseline)


def _oku(kok: Path, yol: str) -> bytes:
    return (kok / yol).read_bytes()


# =====================================================================
# A. THE HEALTHY PATH -- without it every refusal below is meaningless
# =====================================================================

def test_raw_blob_bytes_land_identically_in_both_trees(repo, state_dir):
    icerik = b"AAAA\nBBBB\n"
    baseline = _commit_files(repo, {"pipeline/a.py": icerik, "b.txt": b"B"})
    ws = _create(repo, state_dir, baseline)

    assert _oku(ws.reference_root, "pipeline/a.py") == icerik
    assert _oku(ws.implementer_root, "pipeline/a.py") == icerik
    assert _oku(ws.reference_root, "b.txt") == b"B"
    assert ws.baseline_sha == baseline
    assert re.fullmatch(r"[0-9a-f]{32}", ws.workspace_id)


@pytest.mark.parametrize("icerik,etiket", [
    (b"satir1\r\nsatir2\r\n", "crlf"),
    (b"\xff\xfe\x00\x01ham", "utf8-degil"),
    (b"", "bos"),
    (b"\n" * 64, "yalniz-satirsonu"),
])
def test_blob_bytes_are_not_transformed(repo, state_dir, icerik, etiket):
    """No EOL conversion, no encoding guess. The bytes in the object are
    the bytes on disk, or the copy is not a copy."""
    baseline = _commit_files(repo, {"x.bin": icerik})
    ws = _create(repo, state_dir, baseline)
    assert _oku(ws.reference_root, "x.bin") == icerik
    assert _oku(ws.implementer_root, "x.bin") == icerik


def test_a_name_with_spaces_survives(repo, state_dir):
    baseline = _commit_files(repo, {"bir iki uc.txt": b"X"})
    ws = _create(repo, state_dir, baseline)
    assert _oku(ws.reference_root, "bir iki uc.txt") == b"X"


def test_the_executable_mode_is_preserved_where_the_platform_has_one(
        repo, state_dir):
    baseline = _commit_files(repo, {"run.sh": ("100755", b"#!/bin/sh\n"),
                                    "plain.txt": ("100644", b"x")})
    ws = _create(repo, state_dir, baseline)
    if _WINDOWS:
        # no POSIX permission bits here; the git mode still has to be in
        # the digest, which the determinism test pins
        assert (ws.reference_root / "run.sh").exists()
    else:
        assert os.stat(ws.reference_root / "run.sh").st_mode & 0o111
        assert not os.stat(ws.reference_root / "plain.txt").st_mode & 0o111


def test_the_two_trees_are_not_hard_links(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"AAAA\n"})
    ws = _create(repo, state_dir, baseline)
    ref = os.stat(ws.reference_root / "a.py")
    imp = os.stat(ws.implementer_root / "a.py")
    assert (ref.st_dev, ref.st_ino) != (imp.st_dev, imp.st_ino), \
        "iki agac ayni nesneyi paylasiyor"
    assert ref.st_nlink == 1 and imp.st_nlink == 1


def test_writing_in_the_implementer_tree_does_not_touch_the_reference(
        repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"AAAA\n"})
    ws = _create(repo, state_dir, baseline)
    (ws.implementer_root / "a.py").write_bytes(b"ZZZZ\n")
    assert _oku(ws.reference_root, "a.py") == b"AAAA\n"


def test_neither_tree_contains_a_git_directory(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A", "alt/b.py": b"B"})
    ws = _create(repo, state_dir, baseline)
    for kok in (ws.reference_root, ws.implementer_root):
        assert not (kok / ".git").exists()
        bulunan = [p for p in kok.rglob("*") if p.name == ".git"]
        assert not bulunan, f"{kok.name} altinda .git var"


def test_the_baseline_digest_is_deterministic_across_two_materialisations(
        repo, state_dir, tmp_path):
    baseline = _commit_files(repo, {"a.py": b"A", "alt/b.py": b"B",
                                    "c.sh": ("100755", b"#!/bin/sh\n")})
    ilk = _create(repo, state_dir, baseline)
    ikinci_state = tmp_path / "durum2"
    ikinci_state.mkdir()
    ikinci = flat_workspace.create(repo, state_dir=ikinci_state,
                                   run_id="kosu-2", baseline_sha=baseline)
    assert ilk.baseline_digest == ikinci.baseline_digest
    assert ilk.workspace_id != ikinci.workspace_id
    assert len(ilk.baseline_digest) == 64


def test_the_workspace_record_is_immutable(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    with pytest.raises(Exception):
        ws.workspace_id = "0" * 32
    with pytest.raises(Exception):
        ws.implementer_root = Path(".")


def test_the_caller_cannot_hand_in_a_path(repo, state_dir, tmp_path):
    """Containment is not a check on a path the caller supplies -- the
    caller never gets to supply one."""
    import inspect
    imzalar = {
        "create": inspect.signature(flat_workspace.create),
        "assert_binding": inspect.signature(flat_workspace.assert_binding),
        "remove": inspect.signature(flat_workspace.remove),
        "find_orphans": inspect.signature(flat_workspace.find_orphans),
    }
    for ad, imza in imzalar.items():
        for parametre in imza.parameters:
            assert parametre not in ("path", "root", "target", "directory",
                                     "workspace_root"), \
                f"{ad} disaridan yol aliyor: {parametre}"


# =====================================================================
# B. THE REPOSITORY MUST NOT GET A VOTE
# =====================================================================

def _filtre_kur(repo, sentinel_yolu: Path):
    """A clean/smudge filter that writes a sentinel when it runs.

    Declared the way the audit's attack declared it: in the repository
    config plus `.gitattributes`."""
    python = sys.executable.replace("\\", "/")
    hedef = str(sentinel_yolu).replace("\\", "/")
    komut = (f'"{python}" -c "import sys,pathlib;'
             f"pathlib.Path(r'{hedef}').write_text('calisti');"
             f'sys.stdout.buffer.write(sys.stdin.buffer.read())"')
    _git(repo, "config", "filter.kurgu.clean", komut)
    _git(repo, "config", "filter.kurgu.smudge", komut)
    return komut


def test_a_clean_smudge_filter_never_runs_during_materialisation(
        repo, state_dir, tmp_path):
    """The filter is proved to WORK first, in a separate control
    checkout. Without that, "the sentinel did not appear" would also be
    true of a filter that was simply broken."""
    kontrol_sentinel = tmp_path / "kontrol-sentinel.txt"
    _filtre_kur(repo, kontrol_sentinel)
    (repo / ".gitattributes").write_bytes(b"*.py filter=kurgu\n")
    (repo / "a.py").write_bytes(b"AAAA\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
         "commit", "-qm", "t")
    baseline = _git(repo, "rev-parse", "HEAD").strip()

    # POSITIVE CONTROL: the filter really fires on a real checkout
    kontrol = tmp_path / "kontrol"
    _git(repo, "worktree", "add", "-q", str(kontrol), baseline)
    try:
        assert kontrol_sentinel.exists(), \
            "senaryo kurulmadi: filtre normal checkout'ta bile calismadi"
    finally:
        shutil.rmtree(kontrol, ignore_errors=True)
        _git(repo, "worktree", "prune")
    kontrol_sentinel.unlink()

    ws = _create(repo, state_dir, baseline)
    assert not kontrol_sentinel.exists(), \
        "materyalizasyon sirasinda filtre calisti"
    assert _oku(ws.reference_root, "a.py") == b"AAAA\n"


def test_a_hook_never_runs_during_materialisation(repo, state_dir, tmp_path):
    sentinel = tmp_path / "hook-sentinel.txt"
    kanca = repo / ".git" / "hooks" / "post-checkout"
    kanca.parent.mkdir(parents=True, exist_ok=True)
    kanca.write_text(
        "#!/bin/sh\n" + f'echo calisti > "{sentinel.as_posix()}"\n',
        encoding="utf-8")
    kanca.chmod(0o755)
    baseline = _commit_files(repo, {"a.py": b"A"})

    _create(repo, state_dir, baseline)
    assert not sentinel.exists(), "materyalizasyon sirasinda hook calisti"


@pytest.mark.parametrize("alan", ["giris-sayisi", "tek-dosya", "toplam"])
def test_the_contract_ceilings_refuse_rather_than_truncate(
        repo, state_dir, monkeypatch, alan):
    dosyalar = {f"d{i:03d}.txt": b"x" * 32 for i in range(12)}
    baseline = _commit_files(repo, dosyalar)
    if alan == "giris-sayisi":
        limits = flat_workspace.Limits(max_entries=3)
    elif alan == "tek-dosya":
        limits = flat_workspace.Limits(max_file_bytes=8)
    else:
        limits = flat_workspace.Limits(max_total_bytes=64)
    # the seam takes no `limits` argument -- lowering a ceiling must
    # not mean widening the public API
    monkeypatch.setattr(flat_workspace, "DEFAULT_LIMITS", limits)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)


# =====================================================================
# D. THE LEDGER AND THE CRASH WINDOWS
# =====================================================================

def _kayitlar(state_dir):
    kok = Path(state_dir) / flat_workspace.REGISTRY_DIRNAME
    return sorted(kok.glob("*.json")) if kok.exists() else []


def test_the_record_is_written_before_anything_touches_the_filesystem(
        repo, state_dir, monkeypatch):
    """Write-ahead, and the ordering IS the recovery story: a process
    that dies during materialisation must still have left a record that
    names the repository and the run."""
    baseline = _commit_files(repo, {"a.py": b"A"})
    gorulen = {"kayit": False, "holder": False}
    gercek_mkdir = Path.mkdir

    def izleyen_mkdir(self, *args, **kwargs):
        if flat_workspace.TEMP_PREFIX in self.name:
            gorulen["holder"] = True
            assert gorulen["kayit"], "holder kayittan once yaratildi"
        return gercek_mkdir(self, *args, **kwargs)

    gercek_yaz = flat_workspace._write_record

    def izleyen_yaz(*args, **kwargs):
        gorulen["kayit"] = True
        return gercek_yaz(*args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", izleyen_mkdir)
    monkeypatch.setattr(flat_workspace, "_write_record", izleyen_yaz)
    _create(repo, state_dir, baseline)
    assert gorulen["kayit"] and gorulen["holder"], "senaryo kurulmadi"


def test_a_record_without_a_holder_is_an_orphan(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    shutil.rmtree(ws.reference_root.parent, ignore_errors=True)
    yetimler = flat_workspace.find_orphans(repo, state_dir=state_dir)
    assert ws.workspace_id in {y["workspace_id"] for y in yetimler}


def test_a_half_materialised_holder_is_an_orphan(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    shutil.rmtree(ws.implementer_root, ignore_errors=True)
    yetimler = flat_workspace.find_orphans(repo, state_dir=state_dir)
    assert ws.workspace_id in {y["workspace_id"] for y in yetimler}


def test_a_workspace_that_never_reached_ready_is_refused(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    flat_workspace._set_status(state_dir, ws.workspace_id,
                               flat_workspace.STATUS_MATERIALIZING)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.assert_binding(
            repo, state_dir=state_dir, run_id="kosu-1",
            workspace_id=ws.workspace_id, baseline_sha=baseline)


def test_a_ready_workspace_with_the_right_identities_binds(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    baglanan = flat_workspace.assert_binding(
        repo, state_dir=state_dir, run_id="kosu-1",
        workspace_id=ws.workspace_id, baseline_sha=baseline)
    assert baglanan == ws


@pytest.mark.parametrize("bozuk", ["run_id", "baseline", "workspace_id",
                                   "repo"])
def test_any_wrong_identity_refuses_the_binding(repo, state_dir, tmp_path,
                                                bozuk):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    cagri = dict(state_dir=state_dir, run_id="kosu-1",
                 workspace_id=ws.workspace_id, baseline_sha=baseline)
    hedef = repo
    if bozuk == "run_id":
        cagri["run_id"] = "baska-kosu"
    elif bozuk == "baseline":
        cagri["baseline_sha"] = "0" * 40
    elif bozuk == "workspace_id":
        cagri["workspace_id"] = "f" * 32
    else:
        hedef = _new_repo(tmp_path, "baska-depo")
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.assert_binding(hedef, **cagri)


def test_a_holder_collision_does_not_delete_the_squatter(repo, state_dir,
                                                         monkeypatch):
    """Someone else's directory is never removed to make room. Refusing
    is the only safe answer, because the collision might BE the other
    run."""
    baseline = _commit_files(repo, {"a.py": b"A"})
    sabit = "a" * 32
    monkeypatch.setattr(flat_workspace, "_mint_id", lambda: sabit)
    isgalci = flat_workspace.holder_for(sabit)
    isgalci.mkdir(parents=True, exist_ok=True)
    kanarya = isgalci / "kanarya.txt"
    kanarya.write_bytes(b"BASKASININ")
    try:
        with pytest.raises(flat_workspace.FlatWorkspaceError):
            _create(repo, state_dir, baseline)
        assert kanarya.read_bytes() == b"BASKASININ", "isgalci silindi"
    finally:
        shutil.rmtree(isgalci, ignore_errors=True)


def test_a_failed_record_write_leaves_the_filesystem_untouched(
        repo, state_dir, monkeypatch):
    baseline = _commit_files(repo, {"a.py": b"A"})

    def patlayan(*args, **kwargs):
        raise flat_workspace.FlatWorkspaceError("kurgu kayit arizasi")

    monkeypatch.setattr(flat_workspace, "_write_record", patlayan)
    once = sorted(p.name for p in flat_workspace.runner_temp_root().iterdir())
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)
    sonra = sorted(p.name for p in flat_workspace.runner_temp_root().iterdir())
    assert once == sonra, "kayit yazilamadigi halde dosya sistemi degisti"
    assert not _kayitlar(state_dir)


def test_a_failure_midway_leaves_either_nothing_or_a_findable_record(
        repo, state_dir, monkeypatch):
    baseline = _commit_files(repo, {"a.py": b"A", "b.py": b"B", "c.py": b"C"})
    gercek = git_objects._object_bytes
    durum = {"sayi": 0}

    def patlayan(*args, **kwargs):
        durum["sayi"] += 1
        if durum["sayi"] == 2:
            raise flat_workspace.FlatWorkspaceError("kurgu materyalizasyon")
        return gercek(*args, **kwargs)

    monkeypatch.setattr(git_objects, "_object_bytes", patlayan)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)
    assert durum["sayi"] >= 2, "senaryo kurulmadi"

    kayitlar = _kayitlar(state_dir)
    kalanlar = [p for p in flat_workspace.runner_temp_root().iterdir()
                if p.name.startswith(flat_workspace.TEMP_PREFIX)]
    if kalanlar:
        assert kayitlar, "holder kaldi ama kayit dusuruldu -- bulunamaz kalinti"
        yetimler = flat_workspace.find_orphans(repo, state_dir=state_dir)
        assert yetimler
    for p in kalanlar:
        shutil.rmtree(p, ignore_errors=True)


def test_recovery_finds_an_orphan_without_being_told_the_identity(
        repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    shutil.rmtree(ws.implementer_root, ignore_errors=True)
    yetimler = flat_workspace.find_orphans(repo, state_dir=state_dir)
    assert yetimler and all("workspace_id" in y for y in yetimler)


# =====================================================================
# E. DELETION IS AUTHORISED BY A RECORD, NEVER BY A PATH
# =====================================================================

def test_a_successful_removal_takes_the_holder_and_the_record(repo,
                                                              state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    holder = ws.reference_root.parent
    flat_workspace.remove(repo, state_dir=state_dir,
                          workspace_id=ws.workspace_id)
    assert not holder.exists()
    assert not _kayitlar(state_dir)


def test_a_second_removal_is_a_typed_refusal_not_a_quiet_success(repo,
                                                                 state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    flat_workspace.remove(repo, state_dir=state_dir,
                          workspace_id=ws.workspace_id)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.remove(repo, state_dir=state_dir,
                              workspace_id=ws.workspace_id)


def test_another_repository_cannot_remove_this_workspace(repo, state_dir,
                                                         tmp_path):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    yabanci = _new_repo(tmp_path, "yabanci")
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.remove(yabanci, state_dir=state_dir,
                              workspace_id=ws.workspace_id)
    assert ws.reference_root.exists(), "yabanci depo holder'i sildi"


def test_another_state_directory_grants_no_authority(repo, state_dir,
                                                     tmp_path):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    baska = tmp_path / "baska-durum"
    baska.mkdir()
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.remove(repo, state_dir=baska,
                              workspace_id=ws.workspace_id)
    assert ws.reference_root.exists()


@pytest.mark.parametrize("tur", ["symlink", "junction"])
def test_removal_unlinks_a_link_without_touching_what_it_points_at(
        repo, state_dir, tmp_path, tur):
    """The model may leave a link behind. Deleting the workspace must
    delete the link, never the target's contents."""
    if tur == "junction" and not _WINDOWS:
        pytest.skip("kavsak noktasi Windows'a ozgu")
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)

    disari = tmp_path / "disarida"
    disari.mkdir()
    kanarya = disari / "kanarya.txt"
    kanarya.write_bytes(b"DOKUNMA")
    baglanti = ws.implementer_root / "baglanti"
    if tur == "junction":
        import _winapi
        _winapi.CreateJunction(str(disari), str(baglanti))
    else:
        try:
            os.symlink(str(disari), str(baglanti), target_is_directory=True)
        except OSError:
            pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    assert baglanti.exists(), "senaryo kurulmadi"

    flat_workspace.remove(repo, state_dir=state_dir,
                          workspace_id=ws.workspace_id)
    assert kanarya.exists() and kanarya.read_bytes() == b"DOKUNMA", \
        "temizlik baglantinin hedefine dokundu"
    assert disari.exists()


def test_a_holder_that_will_not_die_keeps_its_record(repo, state_dir,
                                                     monkeypatch):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)

    def patlayan(*args, **kwargs):
        raise OSError("kurgu silme arizasi")

    monkeypatch.setattr(shutil, "rmtree", patlayan)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.remove(repo, state_dir=state_dir,
                              workspace_id=ws.workspace_id)
    monkeypatch.undo()
    assert _kayitlar(state_dir), "silinemeyen holder'in kaydi dusuruldu"


@pytest.mark.parametrize("hedef", ["ana-depo", "state", "temp-koku"])
def test_a_broad_directory_can_never_be_the_deletion_target(
        repo, state_dir, monkeypatch, hedef):
    """Structural: the id decides the path, so there is no id that can
    denote the repository, the state directory or the temp root."""
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    yol = {"ana-depo": repo, "state": state_dir,
           "temp-koku": flat_workspace.runner_temp_root()}[hedef]
    monkeypatch.setattr(flat_workspace, "holder_for", lambda _id: Path(yol))
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.remove(repo, state_dir=state_dir,
                              workspace_id=ws.workspace_id)
    assert Path(yol).exists(), f"{hedef} silindi"


def test_a_root_that_is_a_link_is_refused_by_removal(repo, state_dir,
                                                     tmp_path, monkeypatch):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    disari = tmp_path / "sahte-holder-hedefi"
    disari.mkdir()
    (disari / "kanarya.txt").write_bytes(b"DOKUNMA")
    sahte = tmp_path / "sahte-holder"
    try:
        os.symlink(str(disari), str(sahte), target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    monkeypatch.setattr(flat_workspace, "holder_for", lambda _id: sahte)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        flat_workspace.remove(repo, state_dir=state_dir,
                              workspace_id=ws.workspace_id)
    assert (disari / "kanarya.txt").exists()
    assert ws.reference_root.exists()


# =====================================================================
# F. NO GIT AFTER CREATE
# =====================================================================

@pytest.mark.parametrize("fonksiyon", ["assert_binding", "remove",
                                       "find_orphans"])
def test_no_git_runs_after_materialisation(repo, state_dir, monkeypatch,
                                           fonksiyon):
    """Behavioural, not a grep: git is made to explode, and the
    post-create surface still has to work."""
    baseline = _commit_files(repo, {"a.py": b"A", "alt/b.py": b"B"})
    ws = _create(repo, state_dir, baseline)

    cagrilar = []

    def yasak(*args, **kwargs):
        cagrilar.append(args)
        raise AssertionError("materyalizasyondan sonra git calisti")

    # the seam moved into `git_transport` with the split; patching the
    # re-export in `git_objects` would leave the real one reachable and
    # turn this into a test that cannot fail
    monkeypatch.setattr(git_transport, "run_git_bounded", yasak)
    monkeypatch.setattr(subprocess, "run", yasak)
    monkeypatch.setattr(subprocess, "Popen", yasak)

    if fonksiyon == "assert_binding":
        flat_workspace.assert_binding(
            repo, state_dir=state_dir, run_id="kosu-1",
            workspace_id=ws.workspace_id, baseline_sha=baseline)
    elif fonksiyon == "find_orphans":
        flat_workspace.find_orphans(repo, state_dir=state_dir)
    else:
        flat_workspace.remove(repo, state_dir=state_dir,
                              workspace_id=ws.workspace_id)
    assert not cagrilar


def test_no_blob_content_reaches_the_record_or_an_exception(repo, state_dir):
    icerik = SENTINEL.encode("ascii") * 4
    baseline = _commit_files(repo, {"gizli.txt": icerik})
    ws = _create(repo, state_dir, baseline)
    kayit = (Path(state_dir) / flat_workspace.REGISTRY_DIRNAME /
             f"{ws.workspace_id}.json").read_text(encoding="utf-8")
    assert SENTINEL not in kayit
    assert str(ws.reference_root) not in kayit
    assert SENTINEL not in repr(ws)


def test_an_exception_carries_no_absolute_path(repo, state_dir):
    with pytest.raises(flat_workspace.FlatWorkspaceError) as ret:
        _create(repo, state_dir, "0" * 40)
    metin = str(ret.value) + repr(ret.value)
    assert "/" not in metin and "\\" not in metin
    assert str(repo) not in metin


def test_the_two_trees_start_with_the_same_filesystem_evidence(repo,
                                                               state_dir):
    """D2's scanner is used through its public seam, with one key for
    both scans -- different keys could never compare equal and would
    make this test pass for the wrong reason."""
    from tools.agent_loop import fs_evidence

    baseline = _commit_files(repo, {"a.py": b"A", "alt/b.py": b"B"})
    ws = _create(repo, state_dir, baseline)
    anahtar = b"T" * 32
    ref = fs_evidence.scan(ws.reference_root, key=anahtar)
    imp = fs_evidence.scan(ws.implementer_root, key=anahtar)
    assert [e.path for e in ref.entries] == [e.path for e in imp.entries]
    assert [e.sha256 for e in ref.entries] == [e.sha256 for e in imp.entries]


def test_the_scan_key_is_not_written_into_the_workspace_or_the_record(
        repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    kayit = (Path(state_dir) / flat_workspace.REGISTRY_DIRNAME /
             f"{ws.workspace_id}.json").read_bytes()
    icerikler = b"".join(p.read_bytes()
                         for p in ws.implementer_root.rglob("*") if p.is_file())
    assert b"scan_key" not in kayit and b"anahtar" not in kayit
    assert len(icerikler) < 1024


def test_the_manifest_is_not_written_into_the_model_workspace(repo,
                                                              state_dir):
    """A file the model can edit cannot be the verification authority."""
    baseline = _commit_files(repo, {"a.py": b"A"})
    ws = _create(repo, state_dir, baseline)
    yollar = {p.name for p in ws.implementer_root.rglob("*")}
    for yasak in ("manifest.json", "baseline.json", "flat_manifest.json",
                  ".flat-workspace"):
        assert yasak not in yollar
    assert yollar == {"a.py"}


def test_the_repository_working_tree_is_never_written_to(repo, state_dir):
    baseline = _commit_files(repo, {"a.py": b"A"})
    once = sorted((p.relative_to(repo).as_posix(), p.stat().st_size)
                  for p in repo.rglob("*")
                  if p.is_file() and ".git" not in p.parts)
    _create(repo, state_dir, baseline)
    sonra = sorted((p.relative_to(repo).as_posix(), p.stat().st_size)
                   for p in repo.rglob("*")
                   if p.is_file() and ".git" not in p.parts)
    assert once == sonra, "materyalizasyon ana calisma agacina yazdi"

