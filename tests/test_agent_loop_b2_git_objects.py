"""PACKAGE B2B-A-D3A -- the raw Git object layer, on its own.

WHY THESE TESTS SIT APART FROM THE WORKSPACE ONES. This layer's whole
job is to distrust a repository under audit: it recomputes every object
id from the bytes that arrived and parses trees from the object itself.
The workspace layer's job is ownership and lifecycle. Splitting the
tests the same way the modules split means a failure here names the
trust boundary rather than the bookkeeping.

WHY RAW TREE OBJECTS AND NOT `ls-tree`. MEASURED: `ls-tree` is a
NORMALISING view -- a raw entry whose mode is `100640` is reported as
`100644`, and a flat entry whose name contains a slash is reported
exactly like a real subtree. Tests built on it cannot be made red by a
hostile tree at all, which is why every fixture below writes tree
objects byte by byte with `hash-object --literally`.

WHAT IS TESTED NEXT DOOR. Running git -- the container, the reader
threads, the ceilings and the cleanup verdict -- is a process lifecycle
question and lives in `test_agent_loop_b2_git_transport`. What is left
here is what the bytes MEAN.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from tools.agent_loop import flat_workspace, git_objects, git_transport

SENTINEL = "KURGU-GIZLI-NOBETCI-" + "z" * 8
_WINDOWS = os.name == "nt"


@pytest.fixture(autouse=True)
def _izole_runner_koku(tmp_path, monkeypatch):
    """Each test gets its OWN runner-owned root under `tmp_path`.

    Never the shared global root: a fixture that swept that root would
    delete a concurrent run's holder, which was measured happening."""
    kok = tmp_path / "runner-koku"
    kok.mkdir()
    monkeypatch.setattr(flat_workspace, "runner_temp_root", lambda: kok)
    return kok


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

@pytest.mark.parametrize("mode,etiket", [
    ("120000", "symlink"),
    ("160000", "gitlink"),
    ("100640", "bilinmeyen-mod"),
    ("040000", "agac-girdisi-yanlis-yerde"),
])
def test_a_mode_outside_the_contract_is_refused(repo, state_dir, mode,
                                                etiket):
    oid = _blob(repo, b"payload")
    if mode == "160000":
        oid = "0" * 40
    tree = _raw_tree(repo, [(mode, "kotu", oid)])
    baseline = _commit_of_tree(repo, tree)
    # Setup assertion against the RAW object, because `cat-file -p` is a
    # normalising view exactly like `ls-tree`: it reports the hostile
    # `100640` as `100644`, so checking there would assert nothing.
    ham = _git(repo, "cat-file", "tree", tree, binary_out=True)
    assert ham.startswith(mode.encode("ascii") + b" "), "senaryo kurulmadi"

    with pytest.raises(flat_workspace.FlatWorkspaceError) as ret:
        _create(repo, state_dir, baseline)
    assert SENTINEL not in str(ret.value)


@pytest.mark.parametrize("ad,etiket", [
    (b"..", "ust-dizin"),
    (b"/mutlak", "mutlak"),
    (b"alt/b.py", "ayirici"),
    (b"gorunmez\xe2\x80\x8b.py", "sifir-genislik"),
    (b"bos\xc2\xa0bosluk.py", "siradan-olmayan-bosluk"),
    (b"\xff\xfe-gecersiz.py", "utf8-degil"),
    (b"kontrol\x01.py", "kontrol-karakteri"),
])
def test_a_name_outside_the_contract_is_refused(repo, state_dir, ad, etiket):
    tree = _raw_tree(repo, [("100644", ad, _blob(repo, b"x"))])
    baseline = _commit_of_tree(repo, tree)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)


@pytest.mark.parametrize("bilesen", [".git", ".GIT", ".Git"])
def test_a_dot_git_component_is_refused(repo, state_dir, bilesen):
    """Refused at every spelling: Windows folds case, so `.GIT` is the
    same directory there and a materializer that only knows the lower
    case one writes into it."""
    ic = _raw_tree(repo, [("100644", "config", _blob(repo, b"x"))])
    tree = _raw_tree(repo, [("40000", bilesen, ic)])
    baseline = _commit_of_tree(repo, tree)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)


def test_a_duplicate_canonical_path_is_refused(repo, state_dir):
    oid = _blob(repo, b"x")
    tree = _raw_tree(repo, [("100644", "a.py", oid), ("100644", "a.py", oid)])
    baseline = _commit_of_tree(repo, tree)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)


def test_a_name_that_is_not_nfc_is_refused(repo, state_dir):
    ad = "e\u0301klenti.py"
    assert unicodedata.normalize("NFC", ad) != ad, "senaryo kurulmadi"
    tree = _raw_tree(repo, [("100644", ad, _blob(repo, b"x"))])
    baseline = _commit_of_tree(repo, tree)
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, baseline)


def test_a_blob_whose_bytes_do_not_hash_to_its_id_is_refused(
        repo, state_dir, monkeypatch):
    """Git's own answer is verified, not trusted: the object id is
    recomputed from the bytes that arrived."""
    baseline = _commit_files(repo, {"a.py": b"AAAA\n"})
    gercek = git_objects._object_bytes
    durum = {"cagrildi": False}

    def bozan(repo_, oid, *args, **kwargs):
        durum["cagrildi"] = True
        return b"BASKA-BAYTLAR\n"

    monkeypatch.setattr(git_objects, "_object_bytes", bozan)
    with pytest.raises(flat_workspace.FlatWorkspaceError) as ret:
        _create(repo, state_dir, baseline)
    assert durum["cagrildi"], "saldiri geri cagrimina ulasilmadi"
    assert gercek is not None
    assert "/" not in str(ret.value) and "\\" not in str(ret.value)


_GURULTULU_KAYNAK = "\n".join([
    'import os',
    'import sys',
    "sys.stderr.write(os.environ['GURULTU'])",
    'sys.stderr.flush()',
    'sys.exit(3)',
])


@pytest.mark.parametrize("nasil", ["kapsayici", "baslatilamadi",
                                   "sifir-disi-cikis"])
def test_a_failing_git_call_is_a_typed_silent_refusal(repo, state_dir,
                                                      monkeypatch, tmp_path,
                                                      nasil):
    """Every failure class leaves one fixed sentence and nothing else.

    Driven through the real bounded transport: the earlier version
    swapped a `CompletedProcess` in behind it, which stopped being a
    test of anything once the transport started owning the process
    lifecycle."""
    baseline = _commit_files(repo, {"a.py": b"AAAA\n"})
    durum = {"ulasildi": False}
    gizli = SENTINEL + " C:" + chr(92) + "gizli"

    if nasil == "sifir-disi-cikis":
        # a stand-in that fails the way git fails: loudly, on stderr,
        # in words that name the machine
        betik = tmp_path / "gurultulu_git.py"
        betik.write_text(_GURULTULU_KAYNAK, encoding="utf-8")
        gercek = git_transport.run_git_bounded

        def yonlendir(argv_, *, cwd, stdout_limit, **kwargs):
            durum["ulasildi"] = True
            return gercek([sys.executable, str(betik)], cwd=tmp_path,
                          stdout_limit=stdout_limit, **kwargs)

        monkeypatch.setattr(git_transport, "_git_env",
                            lambda: {**os.environ, "GURULTU": gizli})
        monkeypatch.setattr(git_transport, "run_git_bounded", yonlendir)
    else:
        from tools.agent_loop import process as process_mod

        def patlayan(argv_, *, cwd, env=None):
            durum["ulasildi"] = True
            if nasil == "kapsayici":
                raise process_mod.ContainmentError(gizli)
            raise OSError(gizli)

        monkeypatch.setattr(process_mod, "launch_contained", patlayan)

    with pytest.raises(flat_workspace.FlatWorkspaceError) as ret:
        _create(repo, state_dir, baseline)

    assert durum["ulasildi"], "hedef mekanizmaya ulasilmadi"
    metin = (str(ret.value) + repr(ret.value)
             + " ".join(getattr(ret.value, "__notes__", None) or []))
    assert SENTINEL not in metin, "ham surec metni istisnaya tasindi"
    assert "/" not in metin and chr(92) not in metin


@pytest.mark.parametrize("kotu", ["", "kisa", "z" * 40, "0" * 39, "0" * 41])
def test_a_baseline_that_is_not_a_full_sha_is_refused(repo, state_dir, kotu):
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, kotu)


def test_a_baseline_that_is_not_a_commit_is_refused(repo, state_dir):
    oid = _blob(repo, b"ben bir commit degilim")
    with pytest.raises(flat_workspace.FlatWorkspaceError):
        _create(repo, state_dir, oid)


def test_the_baseline_digest_binds_path_mode_object_id_and_length():
    """The digest is the authority for "same baseline", so it must move
    when any of the four fields moves."""
    temel = [("a.py", "100644", "0" * 40, 4)]
    taban = flat_workspace.baseline_digest(temel)
    for degistir in (
            [("b.py", "100644", "0" * 40, 4)],
            [("a.py", "100755", "0" * 40, 4)],
            [("a.py", "100644", "1" * 40, 4)],
            [("a.py", "100644", "0" * 40, 5)]):
        assert flat_workspace.baseline_digest(degistir) != taban
    assert flat_workspace.baseline_digest(temel) == taban


def test_the_module_computes_the_git_object_id_itself():
    """`sha1(b"blob " + len + b"\\0" + data)` -- pinned here so a
    refactor cannot quietly start trusting git's answer."""
    veri = b"AAAA\n"
    beklenen = hashlib.sha1(b"blob " + str(len(veri)).encode("ascii") +
                            b"\0" + veri).hexdigest()
    assert flat_workspace.blob_object_id(veri) == beklenen


def test_git_runs_with_replace_objects_disabled():
    kaynak = Path(git_transport.__file__).read_text(encoding="utf-8")
    assert "GIT_NO_REPLACE_OBJECTS" in kaynak


def test_the_materializer_never_shells_out_to_checkout_or_archive():
    """Structural pin. `checkout`, `checkout-index` and `archive` are
    the three commands that would hand the repository's configuration a
    vote over the copy. BOTH halves of the split are checked -- moving
    the transport into its own module must not move the ban with it."""
    for modul in (git_objects, git_transport):
        kaynak = Path(modul.__file__).read_text(encoding="utf-8")
        kod = "\n".join(s for s in kaynak.splitlines()
                        if not s.strip().startswith("#"))
        for yasak in ('"checkout"', "'checkout'", '"checkout-index"',
                      '"archive"', "'archive'", "shell=True", "os.system",
                      "os.popen"):
            assert yasak not in kod, f"materializer {yasak} kullaniyor"


def test_a_replace_ref_does_not_change_the_materialised_baseline(
        repo, state_dir):
    """`GIT_NO_REPLACE_OBJECTS=1`, proved by building a replacement and
    showing the ORIGINAL bytes come out."""
    gercek = _commit_files(repo, {"a.py": b"GERCEK\n"})
    sahte = _commit_files(repo, {"a.py": b"SAHTE-DEGISTIRILMIS\n"})
    _git(repo, "replace", gercek, sahte)
    try:
        # setup assertion: the replacement really is in effect
        degistirilmis = _git(repo, "cat-file", "-p", f"{gercek}^{{tree}}")
        assert degistirilmis, "senaryo kurulmadi"
        ws = _create(repo, state_dir, gercek)
        assert _oku(ws.reference_root, "a.py") == b"GERCEK\n", \
            "replace-ref materyalizasyonu yonlendirdi"
    finally:
        _git(repo, "replace", "-d", gercek)


def test_this_layer_owns_no_process(tmp_path):
    """Structural pin for the split. This layer says what bytes MEAN;
    the container, the reader threads and the cleanup verdict belong to
    `git_transport`, and both must keep raising the one shared error."""
    assert git_objects._git_bytes is git_transport.git_bytes
    assert git_objects.FlatWorkspaceError is git_transport.FlatWorkspaceError
    assert flat_workspace.FlatWorkspaceError is git_transport.FlatWorkspaceError
    kaynak = Path(git_objects.__file__).read_text(encoding="utf-8")
    for yasak in ("subprocess", "threading", "Popen", "launch_contained"):
        assert yasak not in kaynak, f"nesne katmani {yasak} tasiyor"

