"""The guard over the document tree, measured rather than assumed.

`tests/conftest.py` is the thing standing between a careless test and
the operator's documents, and every version of it so far has been wrong
in a way only a probe found: it masked names badly, then it fingerprinted
them reproducibly, then it started too late to see an import-time write.
Each was fixed against a scratch probe that left no trace in the repo,
which means each fix could silently rot. These are those probes, kept.

EVERYTHING RUNS IN A THROWAWAY REPOSITORY, IN A SUBPROCESS. The guard's
whole contract is about the moment before collection and the moment
after the session, and neither can be exercised from inside a session it
is already running in. The fake repositories get their own `data/`
holding invented "user" files; the real `data/` is never read, never
written and never named.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REAL_ROOT = Path(__file__).resolve().parent.parent
GUARD = REAL_ROOT / "tests" / "conftest.py"

# Deliberately distinctive, and deliberately multi-dot: the masking
# version of this guard split each component at its FIRST dot and copied
# the rest verbatim, so a name shaped like this leaked almost whole.
KULLANICI_DOSYALARI = {
    "GIZLIDIZIN.ALTI/OZELBELGEADI.SURUM.KURGU.pdf": b"GIZLIICERIK-ALFA\n",
    "IKINCIBELGE.SURUM.pdf": b"GIZLIICERIK-BETA\n",
}
# tokens that must never appear in the guard's output
SIZMAMASI_GEREKENLER = ("GIZLIDIZIN", "ALTI", "OZELBELGEADI", "SURUM",
                        "IKINCIBELGE", "GIZLIICERIK", "ALFA",
                        "BETA", "EKLENENDOSYAADI")


# NOT named test_*.py, and that is the whole point. These fake
# repositories live under pytest's own temp area, which pytest RETAINS
# for a few runs -- and if that area sits inside the repository (an
# audit run pointing --basetemp at output/ is enough), the next root
# `pytest` collects the probes as if they were real tests. A file the
# default discovery pattern does not match cannot be picked up by
# accident; the subprocess names it explicitly, which collects it
# regardless of the pattern.
PROBE_FILE = "guard_probe_case.py"


def _fake_repo(tmp_path, body, *, at_import=""):
    """A repository with the real guard, invented documents, one probe."""
    repo = tmp_path / "sahte-depo"
    (repo / "tests").mkdir(parents=True)
    shutil.copy(GUARD, repo / "tests" / "conftest.py")
    for name, content in KULLANICI_DOSYALARI.items():
        target = repo / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (repo / "tests" / PROBE_FILE).write_text(
        "import pathlib\n"
        "ROOT = pathlib.Path(__file__).resolve().parent.parent\n"
        f"{at_import}\n\n{body}\n",
        encoding="utf-8")
    return repo


def _run(repo):
    """Run the fake repository's suite, with its temp space CONTAINED.

    A session the guard fails keeps its disposable root, because the key
    that resolves the reported ids lives in it -- correct there, and a
    litter of thirty directories here, since most of these tests fail
    the guard on purpose. Pointing the subprocess's temp space at the
    fake repository puts those roots where this test's own tmp_path
    cleanup can reach them."""
    scratch = repo / "gecici"
    scratch.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({name: str(scratch) for name in ("TMPDIR", "TEMP", "TMP")})
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
         f"tests/{PROBE_FILE}"],
        cwd=str(repo), capture_output=True, text=True, env=env)
    return finished.returncode, finished.stdout + finished.stderr


PROB_TEMIZ = "def test_hicbir_sey_yazmaz():\n    assert True\n"

PROB_EKLER = (
    "def test_gercek_belge_dizinine_yazar():\n"
    "    hedef = ROOT / 'data' / 'uploads'\n"
    "    hedef.mkdir(parents=True, exist_ok=True)\n"
    "    (hedef / 'EKLENENDOSYAADI.SURUM.pdf').write_bytes(b'KURGU')\n")

PROB_DEGISTIRIR_VE_SILER = (
    "def test_mevcut_dosyalari_bozar():\n"
    "    veri = ROOT / 'data'\n"
    "    (veri / 'GIZLIDIZIN.ALTI' / 'OZELBELGEADI.SURUM.KURGU.pdf')"
    ".write_bytes(b'DEGISTIRILDI')\n"
    "    (veri / 'IKINCIBELGE.SURUM.pdf').unlink()\n")

# executed while pytest IMPORTS the module, i.e. during collection --
# the exact moment the session-fixture version of this guard could not see
YAZMA_IMPORT_ANINDA = (
    "(ROOT / 'data' / 'uploads').mkdir(parents=True, exist_ok=True)\n"
    "(ROOT / 'data' / 'uploads' / 'EKLENENDOSYAADI.SURUM.pdf')"
    ".write_bytes(b'IMPORT')\n")


def test_a_session_that_touches_nothing_passes_and_cleans_up(tmp_path):
    repo = _fake_repo(tmp_path, PROB_TEMIZ)
    code, output = _run(repo)

    assert code == 0, output
    assert "VERI AGACI DEGISTI" not in output
    # and the invented documents are exactly as they were
    for name, content in KULLANICI_DOSYALARI.items():
        assert (repo / "data" / name).read_bytes() == content


def test_a_file_added_to_the_data_tree_fails_the_session(tmp_path):
    repo = _fake_repo(tmp_path, PROB_EKLER)
    code, output = _run(repo)

    assert code != 0, output
    assert "eklenen 1" in output
    assert "degisen 0" in output and "silinen 0" in output
    assert "eklendi" in output


def test_a_changed_and_a_removed_file_are_each_caught(tmp_path):
    """Three change kinds, three separate claims. An earlier design only
    looked for NEW files, and a test that overwrote or deleted one of the
    operator's documents -- strictly worse than leaving junk behind --
    would have gone unnoticed."""
    repo = _fake_repo(tmp_path, PROB_DEGISTIRIR_VE_SILER)
    code, output = _run(repo)

    assert code != 0, output
    assert "eklenen 0" in output
    assert "degisen 1" in output
    assert "silinen 1" in output
    assert "degisti" in output and "silindi" in output


def test_a_write_during_collection_is_caught(tmp_path):
    """The regression that killed the fixture-based design: pytest imports
    test modules BEFORE session fixtures run, so a write at import time
    was neither redirected nor snapshotted -- it silently became part of
    the "before" state."""
    repo = _fake_repo(tmp_path, PROB_TEMIZ, at_import=YAZMA_IMPORT_ANINDA)
    code, output = _run(repo)

    assert code != 0, output
    assert "eklenen 1" in output


def test_the_report_carries_no_filename_and_no_content(tmp_path):
    """What a failure is allowed to say. Two earlier versions failed this:
    one printed a mask that split at the first dot and kept the rest, the
    other printed the matching bytes outright."""
    repo = _fake_repo(tmp_path, PROB_DEGISTIRIR_VE_SILER + PROB_EKLER)
    code, output = _run(repo)

    assert code != 0, output
    leaked = [token for token in SIZMAMASI_GEREKENLER if token in output]
    assert not leaked, f"rapor icerik ya da ad sizdirdi: {leaked}"


def test_two_sessions_do_not_report_the_same_id_for_the_same_file(tmp_path):
    """The ids are KEYED, not merely hashed. A plain digest of a path is a
    deterministic fingerprint: anyone who guesses a filename confirms it
    by recomputing. A per-session key makes a forwarded report inert --
    and identical ids across runs would prove the key was not doing
    anything."""
    first = _run(_fake_repo(tmp_path / "bir", PROB_EKLER))[1]
    second = _run(_fake_repo(tmp_path / "iki", PROB_EKLER))[1]

    def ids(output):
        # "  - eklendi  <path id> (icerik <content id>)"
        return {line.split()[2] for line in output.splitlines()
                if line.strip().startswith("- eklendi")}

    assert ids(first) and ids(second)
    assert ids(first).isdisjoint(ids(second))


def _redirection_problems(roots, environment, guard):
    """Every way a document root can still be pointing somewhere real.

    Written as a checker rather than as a run of assertions so that it
    can be aimed at a DELIBERATELY BROKEN mapping too. The first version
    only checked that each root differed from three known production
    paths, and a root moved to any other directory inside the
    repository -- somewhere git can see, which is the whole hazard --
    passed it."""
    disposable = Path(guard._DISPOSABLE_ROOT).resolve()
    problems = []
    for name, (value, root_name) in roots.items():
        expected = Path(guard._DOCUMENT_ROOTS[root_name]).resolve()
        actual = Path(value).resolve()
        if actual != expected:
            problems.append(f"{name}: yonlendirilen kok degil")
        elif disposable != actual and disposable not in actual.parents:
            problems.append(f"{name}: atilabilir kok altinda degil")
    for root_name, expected in guard._DOCUMENT_ROOTS.items():
        given = environment.get(root_name)
        if given is None or Path(given).resolve() != Path(expected).resolve():
            problems.append(f"{root_name}: ortam degiskeni eslesmiyor")
    return problems


def _live_roots():
    from pipeline.api import app, owui_chat
    from pipeline.index import ingest, publication

    return {
        "publication.UPLOAD_DIR": (publication.UPLOAD_DIR, "UPLOAD_DIR"),
        "app.UPLOAD_DIR": (app.UPLOAD_DIR, "UPLOAD_DIR"),
        "owui_chat.UPLOAD_DIR": (owui_chat.UPLOAD_DIR, "UPLOAD_DIR"),
        "owui_chat.EXPORT_DIR": (owui_chat.EXPORT_DIR, "EXPORT_DIR"),
        "ingest.OUTPUT_DIR": (ingest.OUTPUT_DIR, "OUTPUT_DIR"),
    }


def test_every_document_root_is_redirected_in_this_very_session():
    """Prevention, checked in-process because THIS session is the claim.

    Five readers, three environment names. Patching one module attribute
    protected one of them, and the OpenWebUI table path -- which writes
    to its root directly -- was not the one. Each root must BE the
    disposable one, not merely differ from the paths we thought of."""
    import conftest as guard

    problems = _redirection_problems(_live_roots(), os.environ, guard)
    assert not problems, f"yonlendirme eksik: {problems}"


def test_the_redirection_check_rejects_a_root_inside_the_repository():
    """The counter-example that killed the previous assertion. A root
    aimed at any other directory INSIDE the repository is exactly the
    hazard -- git can see it -- and "not equal to the three paths I
    listed" called that safe."""
    import conftest as guard

    smuggled = dict(_live_roots())
    smuggled["publication.UPLOAD_DIR"] = (REAL_ROOT / "kurgu-yukleme",
                                          "UPLOAD_DIR")
    problems = _redirection_problems(smuggled, os.environ, guard)
    assert any("publication.UPLOAD_DIR" in problem for problem in problems), (
        "depo icindeki bir kok guvenli sayildi")


def test_the_redirection_check_rejects_a_stale_environment_variable():
    """The module attributes and the environment must agree. They are
    set together and read together; a test that only looked at the
    attributes would miss a subprocess inheriting a production value."""
    import conftest as guard

    stale = dict(os.environ)
    stale["EXPORT_DIR"] = str(REAL_ROOT / "output" / "owui")
    problems = _redirection_problems(_live_roots(), stale, guard)
    assert any("EXPORT_DIR" in problem for problem in problems), (
        "uretim degerini tasiyan ortam degiskeni guvenli sayildi")


@pytest.mark.parametrize(
    ("already", "expected"),
    [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4)],
    ids=["temiz", "test-hatasi", "kesinti", "ic-hata", "kullanim-hatasi"])
def test_a_detected_change_never_hides_a_worse_exit_code(tmp_path, monkeypatch,
                                                         already, expected):
    """An interrupted run exits 2 and an internal error exits 3. Setting 1
    unconditionally traded a precise diagnosis for a vaguer one -- and the
    data finding is on the screen either way, so nothing is gained by
    overwriting the code."""
    import conftest as guard

    fake_data = tmp_path / "data"
    fake_data.mkdir()
    monkeypatch.setattr(guard, "DATA_ROOT", fake_data)
    # its own disposable root: the real one belongs to the session that
    # is running this test, and sessionfinish deletes what it is given
    monkeypatch.setattr(guard, "_DISPOSABLE_ROOT", tmp_path / "atilabilir")
    (tmp_path / "atilabilir").mkdir()
    monkeypatch.setattr(guard, "_BEFORE", {})
    (fake_data / "kurgu.txt").write_bytes(b"KURGU")   # a change to find

    class _Plugins:
        @staticmethod
        def get_plugin(_name):
            return None

    class _Session:
        exitstatus = already
        config = type("_Config", (), {"pluginmanager": _Plugins()})()

    session = _Session()
    guard.pytest_sessionfinish(session, already)
    assert session.exitstatus == expected


def test_the_session_key_is_written_only_when_something_changed(tmp_path,
                                                                monkeypatch):
    """The key turns an inert report back into something actionable, and
    it stays on the operator's disk rather than in the report. A clean
    session leaves neither the key nor the disposable root behind."""
    import conftest as guard

    fake_data = tmp_path / "data"
    fake_data.mkdir()
    disposable = tmp_path / "atilabilir"
    disposable.mkdir()
    monkeypatch.setattr(guard, "DATA_ROOT", fake_data)
    monkeypatch.setattr(guard, "_DISPOSABLE_ROOT", disposable)
    monkeypatch.setattr(guard, "_BEFORE", {})

    class _Plugins:
        @staticmethod
        def get_plugin(_name):
            return None

    class _Session:
        exitstatus = 0
        config = type("_Config", (), {"pluginmanager": _Plugins()})()

    guard.pytest_sessionfinish(_Session(), 0)          # nothing changed
    assert not disposable.exists()

    disposable.mkdir()
    (fake_data / "kurgu.txt").write_bytes(b"KURGU")
    guard.pytest_sessionfinish(_Session(), 0)          # something changed
    key = (disposable / guard.KEY_FILENAME).read_text(encoding="utf-8")
    assert len(key) == 64
    # and the key really is what resolves an id back to a path
    resolved = guard.resolve_path_id(guard._path_id("kurgu.txt"), key,
                                     root=fake_data)
    assert resolved == fake_data / "kurgu.txt"
