"""PACKAGE B2B-B1 -- the flat-workspace execution bridge.

ONE question: does the implementer adapter run the model inside a D3A
flat workspace's IMPLEMENTER root, and refuse everything else before a
process exists.

WHY A SECOND FILE. The legacy suite next door is two thousand lines
about the git-worktree branch, and that branch is still live for one
more package. Mechanically rewriting it would have made a bridge commit
unreviewable. The stub machinery is IMPORTED from it rather than copied,
so the two files cannot drift apart.

NO REAL MODEL IS CALLED. Every binary is a stub under `tmp_path`, an
autouse guard records every launch, and each refusal test proves the
count stayed at zero -- a refusal that merely happened later than the
launch is not a refusal.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

# The stub builder, the helper script and the reply payload live in the
# general execution suite; importing them keeps one definition of "a
# fake model".
import test_agent_loop_b2_execution as legacy
from tools.agent_loop import contract, execution, flat_workspace
from tools.agent_loop import process as process_module

RUN = legacy.RUN
SENTINEL = "KURGU-GIZLI-NOBETCI-" + "z" * 8
# where `legacy._fake_binary` puts its shims; the only programs that
# count as "the model ran"
STUB_HOLDER = "sahte-bin"


@pytest.fixture(autouse=True)
def _izole_runner_koku(tmp_path, monkeypatch):
    """This file creates real workspaces. None of them may land in the
    machine-wide runner root, where a concurrent run keeps its holder."""
    kok = tmp_path / "runner-koku"
    kok.mkdir()
    monkeypatch.setattr(flat_workspace, "runner_temp_root", lambda: kok)
    return kok


@pytest.fixture(autouse=True)
def launched(monkeypatch):
    """Every MODEL launch, recorded. A refusal test that does not check
    this is a test that would pass while the model ran anyway.

    Only the stub binaries count. `flat_workspace.create()` runs git
    through the same contained-launch seam, so recording every process
    would mean the list was never empty and every "nothing started"
    assertion would be measuring the fixture."""
    kayit = []
    gercek_popen = process_module.subprocess.Popen

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
        def Popen(argv, **kwargs):                     # noqa: N802 -- stdlib
            if STUB_HOLDER in str(argv[0]):
                kayit.append(list(argv))
            surecler.append(gercek_popen(argv, **kwargs))
            return surecler[-1]

    surecler = []
    monkeypatch.setattr(process_module, "subprocess", _Recorder)
    yield kayit
    kalan = [p.pid for p in surecler if p.poll() is None]
    for surec in surecler:
        if surec.poll() is None:
            surec.kill()
    assert kalan == [], f"testten sonra yasayan surec: {kalan}"


def _git(repo, *args):
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    assert done.returncode == 0, f"fikstur git komutu basarisiz: {args[0]}"
    return done.stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "kurgu-depo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "k@example.invalid")
    _git(repo, "config", "user.name", "Kurgu")
    (repo / "kurgu.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "kurgu")
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def flat(tmp_path):
    """A real repository, a real D3A workspace, and the identities that
    name it. Nothing here is a stub: the binding under test is the
    production one."""
    repo, baseline = _repo(tmp_path)
    state_dir = tmp_path / "durum"
    state_dir.mkdir()
    ws = flat_workspace.create(repo, state_dir=state_dir, run_id=RUN,
                               baseline_sha=baseline)
    return {"repo": repo, "state_dir": state_dir, "baseline": baseline,
            "ws": ws,
            "identity": {"repo": repo, "state_dir": state_dir,
                         "run_id": RUN, "workspace_id": ws.workspace_id,
                         "baseline_sha": baseline}}


def _run(binary, identity, **overrides):
    call = {"prompt": "kurgu", "budget_usd": 1.0, "timeout_seconds": 60,
            "max_output_bytes": 65536}
    call.update(overrides)
    return execution.run_implementer(binary, **identity, **call)


# =====================================================================
# THE WORKSPACE BRANCH
# =====================================================================

def test_the_model_runs_exactly_in_the_implementer_root(tmp_path, flat,
                                                        launched):
    """POSITIVE CONTROL for the whole file. The witness is the CHILD's
    own report of its working directory -- the adapter's word for where
    it ran is not evidence about where it ran."""
    iz = tmp_path / "cocugun-cwd.txt"
    binary = legacy._fake_binary(
        tmp_path, mode="raw", cwd_record=str(iz),
        hex=json.dumps(legacy._valid_reply()).encode().hex())

    outcome = _run(binary, flat["identity"])

    assert launched, "senaryo kurulmadi: hic surec baslamadi"
    gorulen = os.path.realpath(iz.read_text(encoding="utf-8").strip())
    assert gorulen == os.path.realpath(flat["ws"].implementer_root)
    assert gorulen != os.path.realpath(flat["ws"].reference_root)
    assert outcome.reply["run_id"] == RUN


def test_the_binding_is_asked_for_exactly_the_callers_identities(tmp_path,
                                                                 flat):
    """The adapter may not invent, widen or substitute an identity on
    the way to the binding."""
    binary = legacy._reply_binary(tmp_path)
    gercek = flat_workspace.assert_binding
    gorulen = {}

    def izleyen(repo, **kwargs):
        gorulen["repo"] = repo
        gorulen.update(kwargs)
        return gercek(repo, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(flat_workspace, "assert_binding", izleyen)
    try:
        _run(binary, flat["identity"])
    finally:
        monkeypatch.undo()

    assert str(gorulen["repo"]) == str(flat["repo"])
    assert str(gorulen["state_dir"]) == str(flat["state_dir"])
    assert gorulen["run_id"] == RUN
    assert gorulen["workspace_id"] == flat["ws"].workspace_id
    assert gorulen["baseline_sha"] == flat["baseline"]


def test_the_reference_tree_never_reaches_the_child_or_the_result(tmp_path,
                                                                  flat,
                                                                  launched):
    """The reference copy is what the implementer's work gets compared
    against. A model that can find it can edit it, and the comparison
    would then be against something the model chose."""
    nobetci = flat["ws"].reference_root / "nobetci.txt"
    nobetci.write_text(SENTINEL, encoding="utf-8")
    iz = tmp_path / "cocugun-cwd.txt"
    binary = legacy._fake_binary(
        tmp_path, mode="raw", cwd_record=str(iz),
        hex=json.dumps(legacy._valid_reply()).encode().hex())

    outcome = _run(binary, flat["identity"])

    referans = str(flat["ws"].reference_root)
    argv_metni = " ".join(" ".join(argv) for argv in launched)
    # the ROOT'S OWN PATH, not the bare word: pytest builds `tmp_path`
    # from the test's name, so "reference" appears in every path here
    # and an assertion on it would be measuring the fixture
    assert referans not in argv_metni, "referans kok argv'de"
    assert referans not in repr(outcome), "referans kok sonucta"
    # a token that cannot arrive by coincidence
    assert SENTINEL not in argv_metni, "referans agacin icerigi argv'de"
    assert SENTINEL not in repr(outcome), "referans agacin icerigi sonucta"
    # and the child could not reach it to change it
    assert nobetci.read_text(encoding="utf-8") == SENTINEL
    gorulen = os.path.realpath(iz.read_text(encoding="utf-8").strip())
    assert gorulen == os.path.realpath(flat["ws"].implementer_root)


@pytest.mark.parametrize("kotu", ["", "z" * 32, "a" * 31, "A" * 32, 5, b"a" * 32])
def test_a_malformed_workspace_identity_is_refused_before_launch(
        tmp_path, flat, launched, kotu):
    kimlik = dict(flat["identity"], workspace_id=kotu)
    with pytest.raises(execution.IdentityRefused):
        _run(legacy._reply_binary(tmp_path), kimlik)
    assert launched == [], "reddedilen kimlige ragmen surec basladi"


@pytest.mark.parametrize("nasil", ["eski-kimlik", "hicbiri"])
def test_a_call_that_does_not_name_this_workspace_starts_nothing(
        tmp_path, flat, launched, nasil):
    """B2B-B2C made `workspace_id` the only execution identity, so the
    two ways of failing to name one are now different KINDS of failure:
    the old keyword is refused by Python itself before the adapter has an
    opinion, and an omitted identity is a missing required argument.
    Neither starts a process, which is the part that matters."""
    kimlik = dict(flat["identity"])
    if nasil == "eski-kimlik":
        kimlik["worktree_id"] = "b" * 32
    else:
        kimlik.pop("workspace_id")

    with pytest.raises(TypeError) as ret:
        _run(legacy._reply_binary(tmp_path), kimlik)
    assert launched == [], "reddedilen cagriya ragmen surec basladi"
    metin = str(ret.value) + repr(ret.value)
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"


@pytest.mark.parametrize("alan", ["run_id", "baseline_sha", "repo"])
def test_a_workspace_that_belongs_to_something_else_is_refused(tmp_path, flat,
                                                               launched, alan):
    """The record has to name THIS repository, THIS run and THIS
    baseline. A workspace id on its own authorises nothing."""
    kimlik = dict(flat["identity"])
    if alan == "run_id":
        kimlik["run_id"] = "baska-kosu"
    elif alan == "baseline_sha":
        kimlik["baseline_sha"] = "0" * 40
    else:
        baska, _ = _repo(tmp_path / "ikinci")
        kimlik["repo"] = baska

    with pytest.raises(execution.WorkspaceNotBound):
        _run(legacy._reply_binary(tmp_path), kimlik)
    assert launched == [], "yetkisiz baglama ragmen surec basladi"


def test_a_replaced_workspace_root_is_refused_through_the_binding(tmp_path,
                                                                  flat,
                                                                  launched):
    """The replacement holds the same bytes under the same names; only
    the OBJECT differs. D3A stored the root identity for exactly this,
    and the adapter inherits the refusal rather than re-deriving it."""
    kok = flat["ws"].implementer_root
    kenara = kok.parent / (kok.name + "-eski")
    kok.rename(kenara)
    shutil.copytree(kenara, kok)
    assert (kok / "kurgu.py").is_file(), "senaryo kurulmadi"

    with pytest.raises(execution.WorkspaceNotBound) as ret:
        _run(legacy._reply_binary(tmp_path), flat["identity"])
    assert launched == [], "degistirilmis koke ragmen surec basladi"
    metin = str(ret.value) + repr(ret.value)
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"
    assert ret.value.reason == contract.StopReason.PREFLIGHT_FAILED


# =====================================================================
# THE SEAM ITSELF
# =====================================================================

def test_the_seam_takes_one_identity_and_no_path():
    """Structural pin. `workspace_id` is an identity and it is the ONLY
    execution identity; nothing that selects a directory was ever added
    beside it, and the legacy keyword is gone rather than deprecated --
    a parameter that still exists is a parameter something still
    passes."""
    import inspect

    parametreler = inspect.signature(execution.run_implementer).parameters
    assert "workspace_id" in parametreler
    assert "worktree_id" not in parametreler, "eski kimlik hala imzada"
    for kacis in ("cwd", "path", "workdir", "working_dir", "directory",
                  "holder", "implementer_root", "reference_root",
                  "schema_path", "schema"):
        assert kacis not in parametreler, f"kacis parametresi: {kacis}"
    for ad, parametre in parametreler.items():
        if ad != "binary":
            assert parametre.kind == inspect.Parameter.KEYWORD_ONLY, ad
    # required, not defaulted: an identity with a default is one a caller
    # can forget to give
    assert parametreler["workspace_id"].default is inspect.Parameter.empty
    assert not hasattr(execution, "WorktreeNotBound"), \
        "eski baglama hatasi hala disa aciliyor"
    assert issubclass(execution.WorkspaceNotBound, execution.AdapterError)
