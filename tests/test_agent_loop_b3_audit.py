"""PACKAGE B3 -- the READ-ONLY evaluator adapter.

NO REAL MODEL IS CALLED ANYWHERE IN THIS FILE. Every binary is a shim
written into `tmp_path`, and the change-set suite's own launch recorder
is imported so anything that is neither a stub nor the fixture's git
fails the test.

WHY THIS FILE IS NOT THE IMPLEMENTER BATTERY. The two adapters differ in
the one place that matters: the implementer answers on STDOUT under an
inline schema, the evaluator answers into a FILE named by
`--output-last-message`. Pipe ceilings say nothing about that file, so
the tests that matter most here are about a bound nobody had: a child
that fills a disk through a path the adapter handed it while stdout
stays empty and the exit code stays 0.

Every negative test asserts its SETUP was reached before it claims the
refusal, and every refusal test counts processes -- a gate that refuses
after launching is a different gate from one that refuses before.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

import test_agent_loop_b2_changes as legacy
from tools.agent_loop import audit, cli, contract, schemas

RUN = legacy.RUN
STUB_HOLDER = legacy.STUB_HOLDER

build_gate = legacy.build_gate
private_runner_root = legacy.private_runner_root
only_fake_models_may_run = legacy.only_fake_models_may_run


@pytest.fixture
def gate(tmp_path):
    return build_gate(tmp_path)


# A codex-shaped fake: it finds `--output-last-message` on its OWN argv
# and answers into that file, exactly as the real CLI does. Answering on
# stdout instead is what the "no stdout fallback" test uses.
_HELPER = '''\
import json, os, subprocess, sys, time
from pathlib import Path


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    argv = sys.argv[2:]
    hedef = flag(argv, "--output-last-message")
    if cfg.get("argv_record"):
        Path(cfg["argv_record"]).write_text(json.dumps(argv), encoding="utf-8")
    if cfg.get("cwd_record"):
        Path(cfg["cwd_record"]).write_text(os.getcwd(), encoding="utf-8")
    if cfg.get("schema_record"):
        # the BYTES the child was actually handed, copied from the path
        # on its own argv -- the caller's word for what it wrote is not
        # evidence about what arrived
        sema = flag(argv, "--output-schema")
        Path(cfg["schema_record"]).write_bytes(Path(sema).read_bytes())
    if cfg.get("git_gate") and "--skip-git-repo-check" not in argv \\
            and not Path(".git").exists():
        # THE REAL CLI'S OWN GATE, reproduced from a measurement of
        # codex-cli 0.147.0-alpha.6.6: exit 1 before any work, with this
        # exact sentence on stderr and nothing on stdout.
        sys.stderr.write("Reading prompt from stdin...\\n"
                         "Not inside a trusted directory and "
                         "--skip-git-repo-check was not specified.\\n")
        sys.exit(1)
    if cfg.get("stderr_hex"):
        # RAW BYTES, so a scenario may hand over output that is not valid
        # UTF-8 at all -- the classifier has to survive that, and a test
        # that could only spell decodable text could never prove it.
        sys.stderr.buffer.write(bytes.fromhex(cfg["stderr_hex"]))
        sys.stderr.buffer.flush()
    mode = cfg["mode"]
    if mode == "flood_stderr":
        blok = b"x" * 4096
        gonderilen = 0
        while gonderilen < cfg["bytes"]:
            sys.stderr.buffer.write(blok)
            sys.stderr.buffer.flush()
            gonderilen += len(blok)
    if mode == "reply":
        if cfg.get("read_stdin", True):
            sys.stdin.read()
        Path(hedef).write_text(cfg["body"], encoding="utf-8")
    elif mode == "stdout_only":
        sys.stdin.read()
        sys.stdout.write(cfg["body"])
        sys.stdout.flush()
    elif mode == "flood_file":
        blok = "x" * 65536
        with open(hedef, "w", encoding="utf-8") as handle:
            gonderilen = 0
            while gonderilen < cfg["bytes"]:
                handle.write(blok)
                handle.flush()
                gonderilen += len(blok)
        time.sleep(cfg.get("seconds", 30))
    elif mode == "flood_stdout":
        blok = b"x" * 4096
        gonderilen = 0
        while gonderilen < cfg["bytes"]:
            sys.stdout.buffer.write(blok)
            sys.stdout.buffer.flush()
            gonderilen += len(blok)
    elif mode == "sleep":
        time.sleep(cfg["seconds"])
    elif mode == "spawn":
        subprocess.Popen([sys.executable, "-c", cfg["child_code"]])
        if cfg.get("wait_for"):
            limit = time.monotonic() + 20
            while not Path(cfg["wait_for"]).exists():
                if time.monotonic() > limit:
                    sys.exit(97)
                time.sleep(0.02)
        Path(hedef).write_text(cfg["body"], encoding="utf-8")
        sys.stdin.read()
        time.sleep(cfg.get("seconds", 0))
    elif mode == "write_tree":
        Path(cfg["target"]).write_text("SIZDI\\n", encoding="utf-8")
        sys.stdin.read()
        Path(hedef).write_text(cfg["body"], encoding="utf-8")
    sys.exit(cfg.get("code", 0))


main()
'''


def _stub(tmp_path, name="sahte_codex", **config):
    holder = tmp_path / STUB_HOLDER
    holder.mkdir(exist_ok=True)
    helper = holder / "denetci_yardimci.py"
    helper.write_text(_HELPER, encoding="utf-8")
    settings = holder / f"{name}.json"
    # `default=str` so a scenario may hand a `Path` straight in: without
    # it every record-file test dies in `json.dumps` instead of reaching
    # the behaviour it was written for, and red for the wrong reason
    # still counts as red.
    settings.write_text(json.dumps(config, default=str), encoding="utf-8")
    if os.name == "nt":
        shim = holder / f"{name}.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{helper}" "{settings}" %*\r\n',
            encoding="ascii")
    else:
        shim = holder / f"{name}.sh"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{helper}" "{settings}" "$@"\n',
            encoding="ascii")
        shim.chmod(0o755)
    return shim


def _code_reply(**overrides):
    payload = {"protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
               "role": contract.Role.EVALUATOR,
               "audit_kind": contract.AuditKind.CODE,
               "status": contract.Status.APPROVED, "summary": "kurgu denetim",
               "next_action": "stop"}
    payload.update(overrides)
    return payload


def _locked_reply(run_id, **overrides):
    payload = {"protocol_version": contract.PROTOCOL_VERSION,
               "run_id": run_id, "role": contract.Role.EVALUATOR,
               "audit_kind": contract.AuditKind.LOCKED,
               "status": contract.Status.APPROVED,
               "summary_code": contract.SummaryCode.CRITERIA_MET,
               "next_action": "stop"}
    payload.update(overrides)
    return payload


def _audit(binary, gate_obj, **overrides):
    settings = {"repo": gate_obj.repo, "state_dir": gate_obj.state_dir,
                "run_id": RUN, "workspace_id": gate_obj.workspace_id,
                "baseline_sha": gate_obj.baseline,
                "audit_kind": contract.AuditKind.CODE,
                "prompt": "kurgu denetim istemi", "timeout_seconds": 60,
                "max_output_bytes": 65536}
    settings.update(overrides)
    return audit.run_evaluator(binary, **settings)


# =====================================================================
# A. POSITIVE CONTROL -- an adapter that refuses everything would pass
#    every negative test in this file
# =====================================================================

def test_a_healthy_code_audit_comes_back_validated(tmp_path, gate,
                                                   only_fake_models_may_run):
    binary = _stub(tmp_path, mode="reply",
                   body=json.dumps(_code_reply()))
    outcome = _audit(binary, gate)

    assert len(only_fake_models_may_run) == 1, "denetci hic calismadi"
    assert outcome.reply["status"] == contract.Status.APPROVED
    assert outcome.audit_kind == contract.AuditKind.CODE
    assert outcome.exit_code == 0
    assert outcome.last_message_bytes > 0
    assert outcome.schema_sha256 == schemas.SchemaBinding(
        schemas.CODE_AUDIT_RESULT_SCHEMA).sha256


def test_the_reply_is_read_from_the_file_and_never_from_stdout(
        tmp_path, gate, only_fake_models_may_run):
    """THE SEAM. `codex exec` answers into `--output-last-message`, and
    an adapter that also accepted stdout would have a second, unbounded
    road for a reply -- the one the file ceiling does not watch. A child
    that prints a PERFECTLY VALID reply and writes no file must fail."""
    binary = _stub(tmp_path, mode="stdout_only",
                   body=json.dumps(_code_reply()))
    with pytest.raises(audit.AuditError) as refusal:
        _audit(binary, gate)

    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert not isinstance(refusal.value, audit.AuditInputRefused)


def test_the_evaluator_argv_is_read_only_and_names_both_files(
        tmp_path, gate, only_fake_models_may_run):
    iz = tmp_path / "denetci-argv.json"
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()),
                   argv_record=iz)
    _audit(binary, gate)

    argv = json.loads(iz.read_text(encoding="utf-8"))
    assert "exec" in argv
    assert argv[argv.index("--sandbox") + 1] == contract.CODEX_SANDBOX_READ_ONLY
    flag, value = contract.CODEX_APPROVAL_OVERRIDE
    assert value in argv and argv[argv.index(value) - 1] == flag
    assert "--output-schema" in argv and "--output-last-message" in argv
    assert "-a" not in argv and "--ask-for-approval" not in argv


def test_the_evaluator_runs_in_the_candidate_implementer_root(
        tmp_path, gate, only_fake_models_may_run):
    """The witness is the CHILD's own report of where it ran -- the
    caller's word for that is not evidence about it -- and the reference
    copy is neither reachable nor mentioned."""
    iz = tmp_path / "denetci-cwd.txt"
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()),
                   cwd_record=iz)
    _audit(binary, gate)

    gorulen = os.path.realpath(iz.read_text(encoding="utf-8").strip())
    assert gorulen == os.path.realpath(gate.tree)
    assert gorulen != os.path.realpath(gate.reference)
    assert gorulen != os.path.realpath(gate.repo)


# =====================================================================
# B. THE BOUND NOBODY HAD -- the last-message file
# =====================================================================

def test_an_oversized_last_message_stops_the_child_while_it_is_running(
        tmp_path, gate, only_fake_models_may_run):
    """THE mechanism this module exists for. The child writes far past
    the ceiling into the answer file and then SLEEPS for thirty seconds.
    If the bound were only checked after the process ended, this test
    would take those thirty seconds and the bytes would already be on
    the disk. It is stopped instead."""
    binary = _stub(tmp_path, mode="flood_file", bytes=4_000_000, seconds=30)
    with pytest.raises(audit.OutputLimitExceeded) as refusal:
        _audit(binary, gate, max_output_bytes=65536)

    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert refusal.value.stream == "last_message"
    assert refusal.value.reason == contract.StopReason.SCHEMA_VIOLATION


def test_an_oversized_stdout_is_still_refused(tmp_path, gate,
                                              only_fake_models_may_run):
    """The pipe ceiling did not go away when the file ceiling arrived."""
    binary = _stub(tmp_path, mode="flood_stdout", bytes=4_000_000)
    with pytest.raises(audit.OutputLimitExceeded) as refusal:
        _audit(binary, gate, max_output_bytes=65536)

    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert refusal.value.stream in ("stdout", "stderr")


def test_a_missing_last_message_file_is_a_transport_failure(
        tmp_path, gate, only_fake_models_may_run):
    """Exit 0 and no answer is not an approval."""
    binary = _stub(tmp_path, mode="sleep", seconds=0)
    with pytest.raises(audit.TransportFailed):
        _audit(binary, gate)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"


def test_a_child_that_never_reads_the_prompt_is_refused(
        tmp_path, gate, only_fake_models_may_run):
    """A reply produced from an instruction the evaluator never received
    is a verdict on a different question. Exit 0 does not answer it."""
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()),
                   read_stdin=False)
    # the child may or may not win the race to exit before the write
    # completes, so the claim is narrow: IF the prompt was not delivered
    # the call must not return a verdict
    try:
        outcome = _audit(binary, gate)
    except audit.TransportFailed:
        return
    assert outcome.reply["status"] == contract.Status.APPROVED


# =====================================================================
# C. CONTAINMENT AND CLEANUP
# =====================================================================

def _audit_with_fast_clock(binary, gate_obj, monkeypatch):
    """The contract's minimum per-call timeout is 30 seconds, so the
    CLOCK moves rather than the range -- the same seam `execution` uses,
    and deliberately the only one. Proving timeout behaviour otherwise
    means either half-minute tests or a production range loosened to
    suit them."""
    gercek = audit._now
    baslangic = gercek()

    def hizli():
        return baslangic + (gercek() - baslangic) * 1000

    monkeypatch.setattr(audit, "_now", hizli)
    return _audit(binary, gate_obj, timeout_seconds=30)


def test_a_timeout_stops_the_whole_process_tree(tmp_path, gate, monkeypatch,
                                                only_fake_models_may_run):
    binary = _stub(tmp_path, mode="sleep", seconds=300)
    with pytest.raises(audit.Timeout) as refusal:
        _audit_with_fast_clock(binary, gate, monkeypatch)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert refusal.value.reason == contract.StopReason.TIMEOUT
    assert refusal.value.timeout_seconds == 30


def test_a_surviving_grandchild_is_reported_rather_than_ignored(
        tmp_path, gate, only_fake_models_may_run):
    """Something the evaluator started is still running against the
    candidate, so whatever it said about that candidate describes a tree
    that is still moving."""
    isaret = tmp_path / "torun-yasiyor.txt"
    kod = (f"open({str(isaret)!r},'w').write('1')\n"
           "import time; time.sleep(300)\n")
    binary = _stub(tmp_path, mode="spawn", child_code=kod,
                   wait_for=str(isaret), body=json.dumps(_code_reply()),
                   seconds=0)
    outcome = None
    try:
        outcome = _audit(binary, gate)
    except audit.AuditError as refusal:
        assert refusal.reason in (
            contract.StopReason.MODEL_PROCESS_FAILED,
            contract.StopReason.TIMEOUT,
            contract.StopReason.INTERRUPTED)
        return
    # the container emptied the tree, which is the other legal outcome
    assert outcome.exit_code == 0
    assert isaret.exists(), "senaryo kurulmadi: torun hic baslamadi"


def test_the_audit_holder_is_gone_after_every_outcome(tmp_path, gate,
                                                      only_fake_models_may_run):
    """The holder carries the model's own output. A run that cannot
    prove it cleaned up cannot claim the boundary this module keeps."""
    before = _holders()
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()))
    _audit(binary, gate)
    assert _holders() == before, "basarili kosu tutucu birakti"

    kotu = _stub(tmp_path, name="sahte_bozuk", mode="reply", body="JSON degil")
    with pytest.raises(audit.SchemaViolation):
        _audit(kotu, gate)
    assert _holders() == before, "basarisiz kosu tutucu birakti"


def _holders():
    root = Path(tempfile.gettempdir())
    return {p.name for p in root.glob(audit.HOLDER_PREFIX + "*")}


# =====================================================================
# D. THE SCHEMA BOUNDARY AND THE TWO AUDIT KINDS
# =====================================================================

@pytest.mark.parametrize(
    "govde",
    ['{"not": "the schema"}', "duz metin, JSON degil", "[]",
     '{"role": "evaluator"}'],
    ids=["yanlis-alanlar", "json-degil", "dizi", "eksik-alanlar"])
def test_a_reply_outside_the_schema_is_a_violation(tmp_path, gate, govde,
                                                   only_fake_models_may_run):
    binary = _stub(tmp_path, mode="reply", body=govde)
    with pytest.raises(audit.SchemaViolation):
        _audit(binary, gate)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"


def test_a_locked_reply_cannot_arrive_under_the_code_schema(
        tmp_path, gate, only_fake_models_may_run):
    """The two kinds are a TYPE boundary. A locked reply judged by the
    code schema is a locked finding carrying free text, which is the one
    thing the split exists to prevent."""
    binary = _stub(tmp_path, mode="reply",
                   body=json.dumps(_locked_reply("0" * 32)))
    with pytest.raises(audit.SchemaViolation) as refusal:
        _audit(binary, gate, audit_kind=contract.AuditKind.CODE)
    assert "denetim turunu" in str(refusal.value)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"


def test_a_locked_audit_is_bound_to_the_ids_the_runner_issued(
        tmp_path, gate, only_fake_models_may_run):
    """The static pattern proves SHAPE only: any 32 hex characters
    satisfy it, including ids the runner never minted."""
    verilen_bulgu, verilen_mekanizma = "f" * 32, "e" * 32
    kosu = "a" * 32
    bulgu = {"finding_id": verilen_bulgu, "mechanism_id": verilen_mekanizma,
             "severity": "high",
             "error_class": contract.LockedFindingClass.WRONG_ROW,
             "case_count": 3}
    saglikli = _locked_reply(
        kosu, status=contract.Status.CHANGES_REQUESTED,
        summary_code=contract.SummaryCode.REGRESSION_DETECTED,
        next_action="await_repair", findings=[bulgu])
    # `run_id` stays the LOOP's readable identifier -- it is what binds
    # the workspace. `issued_run_id` is the opaque one the runner mints
    # for the textless envelope, and pinning the two together would make
    # a locked audit impossible to bind.
    ortak = {"audit_kind": contract.AuditKind.LOCKED, "issued_run_id": kosu,
             "issued_finding_ids": [verilen_bulgu],
             "issued_mechanism_ids": [verilen_mekanizma]}

    binary = _stub(tmp_path, mode="reply", body=json.dumps(saglikli))
    outcome = _audit(binary, gate, **ortak)
    assert outcome.reply["findings"][0]["finding_id"] == verilen_bulgu

    kacak = json.loads(json.dumps(saglikli))
    kacak["findings"][0]["finding_id"] = "b" * 32
    sahte = _stub(tmp_path, name="sahte_kacak", mode="reply",
                  body=json.dumps(kacak))
    with pytest.raises(audit.SchemaViolation):
        _audit(sahte, gate, **ortak)


def test_a_locked_audit_without_issued_ids_is_refused_before_launch(
        tmp_path, gate, only_fake_models_may_run):
    """An empty enum accepts nothing, so every reply -- including an
    honest approval -- would be a schema violation. A gate that cannot
    pass is indistinguishable from a broken one."""
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()))
    with pytest.raises(audit.AuditInputRefused):
        _audit(binary, gate, audit_kind=contract.AuditKind.LOCKED,
               issued_run_id="a" * 32)
    assert not only_fake_models_may_run, "kimliksiz kilitli denetim calisti"


def test_a_code_audit_refuses_ids_that_would_bind_nothing(
        tmp_path, gate, only_fake_models_may_run):
    """The code schema has no per-call enum to put them in, so ids handed
    here would bind NOTHING while the caller believed they had bound
    something. A parameter that is silently ignored is worse than one
    that is absent."""
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()))
    for fazladan in ({"issued_run_id": "a" * 32},
                     {"issued_finding_ids": ["f" * 32]},
                     {"issued_mechanism_ids": ["e" * 32]}):
        with pytest.raises(audit.AuditInputRefused):
            _audit(binary, gate, audit_kind=contract.AuditKind.CODE,
                   **fazladan)
    assert not only_fake_models_may_run, "baglamayan kimlikle denetci calisti"


# =====================================================================
# E. THE CALL BOUNDARY -- nothing is discovered, nothing leaks
# =====================================================================

def test_the_adapter_discovers_nothing_and_takes_no_directory():
    import inspect

    parameters = inspect.signature(audit.run_evaluator).parameters
    assert set(parameters) == {
        "binary", "repo", "state_dir", "run_id", "workspace_id",
        "baseline_sha", "audit_kind", "prompt", "timeout_seconds",
        "max_output_bytes", "issued_run_id", "issued_finding_ids",
        "issued_mechanism_ids", "model"}
    for escape in ("cwd", "workdir", "schema_path", "last_message_path",
                   "holder", "root", "implementer_root", "argv", "env"):
        assert escape not in parameters, f"kacis parametresi: {escape}"
    body = Path(audit.__file__).read_text(encoding="utf-8")
    for probe in ("shutil.which", "find_executable", "capture_output"):
        assert probe not in body, f"denetci kesfi/kestirmesi: {probe}"


@pytest.mark.parametrize(
    ("alan", "deger"),
    [("timeout_seconds", 5), ("timeout_seconds", 10**9),
     ("timeout_seconds", True), ("max_output_bytes", 8),
     ("max_output_bytes", 10**9), ("prompt", ""), ("prompt", b"kurgu"),
     ("audit_kind", "uydurma_tur"), ("workspace_id", "kisa"),
     ("baseline_sha", "0" * 39)],
    ids=["sure-kucuk", "sure-buyuk", "sure-bool", "cikti-kucuk",
         "cikti-buyuk", "istem-bos", "istem-bayt", "tur-yok",
         "alan-kimligi", "taban-surum"])
def test_a_refused_input_never_starts_a_process(tmp_path, gate, alan, deger,
                                                only_fake_models_may_run):
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()))
    with pytest.raises(audit.AuditError):
        _audit(binary, gate, **{alan: deger})
    assert not only_fake_models_may_run, f"{alan} reddedildigi halde calisti"


def test_a_refusal_carries_no_path_argv_or_os_message(tmp_path, gate,
                                                      only_fake_models_may_run):
    """`OSError` text names absolute paths, and this text travels into
    reports. `raise ... from None` alone does NOT clear `__context__`, so
    the chain is inspected too."""
    binary = _stub(tmp_path, mode="reply", body="JSON degil")
    with pytest.raises(audit.SchemaViolation) as refusal:
        _audit(binary, gate)

    zincir = []
    hata = refusal.value
    while hata is not None:
        zincir.append(str(hata))
        zincir.append(repr(hata))
        hata = hata.__cause__ or hata.__context__
    metin = " ".join(zincir)
    for gizli in (str(tmp_path), str(gate.repo), str(gate.tree),
                  audit.HOLDER_PREFIX, "--output-last-message", "exec"):
        assert gizli not in metin, f"ret metni sizdirdi: {gizli}"


def test_the_workspace_binding_is_what_chooses_the_directory(
        tmp_path, gate, only_fake_models_may_run):
    """A wrong workspace identity is refused before a process exists --
    the binding is the authority, not an `is_dir()` check that the main
    checkout would also pass."""
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()))
    with pytest.raises(audit.WorkspaceNotBound):
        _audit(binary, gate, workspace_id="0" * 32)
    assert not only_fake_models_may_run, "baglama tutmadigi halde calisti"


# =====================================================================
# B4-R3 -- A FLAT WORKSPACE IS NOT A GIT REPOSITORY
# =====================================================================
#
# MEASURED on codex-cli 0.147.0-alpha.6.6 with the exact production
# argv: exit 1 after 341 ms, 0 bytes of stdout, 105 of stderr, no
# last-message file, and this text:
#
#   "Not inside a trusted directory and --skip-git-repo-check was not
#    specified."
#
# The evaluator's cwd is the workspace's implementer root, which holds
# tracked files and no `.git` -- by design, because it is a copy rather
# than a clone. The stub below reproduces that gate so the fixture fails
# for the real reason rather than because a fake was told to fail.

def test_the_evaluator_runs_where_there_is_no_git_repository(
        tmp_path, gate, only_fake_models_may_run):
    """Both directions, against a stub carrying the real CLI's gate.

    FIRST the control: the same stub, launched with the production argv
    MINUS the flag, refuses exactly as the real binary did -- so the
    gate is genuinely armed and the green half below is not vacuous.
    THEN the production path, which must pass."""
    import subprocess as real_subprocess

    assert not (Path(gate.tree) / ".git").exists(), \
        "senaryo kurulmadi: aday agacinda .git var"
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()),
                   git_gate=True)

    schema_file = tmp_path / "sema.json"
    schema_file.write_text("{}", encoding="utf-8")
    last_message = tmp_path / "son.txt"
    argv = cli.build_evaluator_argv(binary, repo=gate.tree,
                                    schema_path=schema_file,
                                    last_message_path=last_message)
    assert "--skip-git-repo-check" in argv, "uretim argv'si bayragi tasimiyor"
    stripped = [token for token in argv if token != "--skip-git-repo-check"]
    refused = real_subprocess.run(stripped, cwd=str(gate.tree), input=b"",
                                  capture_output=True)
    assert refused.returncode == 1
    assert b"--skip-git-repo-check was not specified" in refused.stderr
    assert refused.stdout == b""
    assert not last_message.exists()

    # and with the flag the production seam completes normally
    outcome = _audit(binary, gate)
    assert outcome.reply["role"] == contract.Role.EVALUATOR
    assert outcome.exit_code == 0
    assert outcome.last_message_bytes > 0


# =====================================================================
# B4-R8 -- NAMING A NON-ZERO EVALUATOR EXIT
# =====================================================================
#
# MEASURED on the first run that ever reached `auditing`: the evaluator
# ended with exit 1, 0 bytes of stdout and kilobytes of stderr, and the
# terminal event carried no failure code at all. The bytes that would
# have named the refusal were in a bounded buffer at the moment the
# failure was raised and were dropped immediately afterwards.
#
# The gate is EXACT and whole-line. These tests are written from both
# sides: the one proven marker must be named, and everything that merely
# resembles it must stay generic.

_MARKER = ("Not inside a trusted directory and --skip-git-repo-check "
           "was not specified.")
# the real refusal arrives UNDER a preamble line, which is why the whole
# buffer never equals the marker and the match has to be per line
_REAL_STDERR = f"Reading prompt from stdin...\n{_MARKER}\n"


def _failing(tmp_path, stderr_bytes, *, name="sahte_hata", code=1, **config):
    return _stub(tmp_path, name=name, mode="fail", code=code,
                 stderr_hex=stderr_bytes.hex(), **config)


def test_a_nonzero_evaluator_exit_with_unknown_stderr_stays_generic(
        tmp_path, gate, only_fake_models_may_run):
    """The regression floor. An exit nobody can name keeps the generic
    class, the generic code and every measurement it already had."""
    binary = _failing(tmp_path, b"beklenmeyen bir arizanin metni\n", code=3)
    with pytest.raises(audit.ProcessFailed) as refusal:
        _audit(binary, gate)

    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    failure = refusal.value
    assert type(failure) is audit.ProcessFailed, "genel sinif ozellestirildi"
    assert contract.FAILURE_CODES[("audit", type(failure).__name__)] == \
        contract.FailureCode.EVALUATOR_PROCESS_FAILED
    assert failure.role == contract.Role.EVALUATOR
    assert failure.exit_code == 3
    assert failure.stdout_bytes == 0 and failure.stderr_bytes > 0
    assert failure.cleanup_complete is True
    assert failure.reason == contract.StopReason.MODEL_PROCESS_FAILED


def test_the_exact_trusted_directory_refusal_is_named(
        tmp_path, gate, only_fake_models_may_run):
    """The one classification this package has evidence for -- against
    the REAL two-line shape, preamble included."""
    binary = _failing(tmp_path, _REAL_STDERR.encode("utf-8"))
    with pytest.raises(audit.RepositoryRefused) as refusal:
        _audit(binary, gate)

    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    failure = refusal.value
    assert isinstance(failure, audit.ProcessFailed), "aile disina cikti"
    assert contract.FAILURE_CODES[("audit", type(failure).__name__)] == \
        contract.FailureCode.EVALUATOR_REPOSITORY_REFUSED
    assert failure.role == contract.Role.EVALUATOR
    assert failure.event == contract.EventCode.MODEL_CALL_FINISHED
    assert failure.reason == contract.StopReason.MODEL_PROCESS_FAILED
    assert failure.exit_code == 1
    assert failure.stdout_bytes == 0
    assert failure.stderr_bytes == len(_REAL_STDERR.encode("utf-8"))
    assert failure.cleanup_complete is True
    # CRLF is the only spelling difference that may be normalised away
    crlf = _failing(tmp_path, _REAL_STDERR.replace("\n", "\r\n").encode(),
                    name="sahte_crlf")
    with pytest.raises(audit.RepositoryRefused):
        _audit(crlf, gate)


@pytest.mark.parametrize(
    "metin",
    [_MARKER.lower(),
     _MARKER.rstrip("."),
     _MARKER + " Try again.",
     f"warning: {_MARKER}",
     _MARKER.replace("  ", " ").replace(
         "trusted directory", "trusted  directory"),
     "".join(_MARKER.split())],
    ids=["kucuk-harf", "nokta-yok", "sonuna-eklenmis", "basina-eklenmis",
         "ic-bosluk", "bosluksuz"])
def test_a_sentence_that_is_not_exactly_the_marker_is_not_classified(
        tmp_path, gate, metin, only_fake_models_may_run):
    """SUBSTRING SEARCH WOULD PASS FOUR OF THESE. A classifier that
    guesses is worse than the generic code it replaces, because an
    operator acts on it."""
    binary = _failing(tmp_path, f"{metin}\n".encode("utf-8"))
    with pytest.raises(audit.ProcessFailed) as refusal:
        _audit(binary, gate)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert type(refusal.value) is audit.ProcessFailed


def test_undecodable_stderr_stays_generic_and_raises_nothing_of_its_own(
        tmp_path, gate, only_fake_models_may_run):
    """Bytes that are not UTF-8 at all, with the marker sitting in the
    middle of them. The decode failure is the classifier's silence, not
    an exception that replaces the failure that got us here."""
    payload = b"\xff\xfe gecersiz " + _MARKER.encode("utf-8") + b"\n\x80\x81"
    binary = _failing(tmp_path, payload, code=2)
    with pytest.raises(audit.ProcessFailed) as refusal:
        _audit(binary, gate)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert type(refusal.value) is audit.ProcessFailed
    assert refusal.value.exit_code == 2


def test_the_stronger_authorities_still_outrank_the_stderr_class(
        tmp_path, gate, monkeypatch, only_fake_models_may_run):
    """Three of them, each carrying the marker on stderr so the new gate
    would fire if it were allowed to.

    An OVERFLOWED buffer is a partial document; a TIMEOUT is a call that
    never produced a verdict; EXIT 0 is not a failure at all. None of
    them may be reclassified by what stderr happened to say."""
    marker = _REAL_STDERR.encode("utf-8")
    flooding = _stub(tmp_path, name="sahte_tasan", mode="flood_stderr",
                     bytes=4_000_000, code=1, stderr_hex=marker.hex())
    with pytest.raises(audit.OutputLimitExceeded) as overflow:
        _audit(flooding, gate, max_output_bytes=65536)
    assert type(overflow.value) is audit.OutputLimitExceeded

    healthy = _stub(tmp_path, name="sahte_saglikli", mode="reply",
                    body=json.dumps(_code_reply()), stderr_hex=marker.hex())
    outcome = _audit(healthy, gate)
    assert outcome.exit_code == 0, "sifir cikisli kosu yeniden siniflandirildi"

    # LAST, because the fast clock stays patched for the rest of the
    # test: a healthy call made after it would time out at a thousand
    # times real speed and go red for the fixture's reason
    slow = _stub(tmp_path, name="sahte_yavas", mode="sleep", seconds=300,
                 code=1, stderr_hex=marker.hex())
    with pytest.raises(audit.Timeout) as timeout:
        _audit_with_fast_clock(slow, gate, monkeypatch)
    assert type(timeout.value) is audit.Timeout


def test_no_stderr_byte_reaches_the_exception_that_names_it(
        tmp_path, gate, only_fake_models_may_run):
    """The classification is the ONLY thing that travels. The marker is a
    lookup key; the sentinel beside it, the absolute path and the
    vendor's own wording stay in the buffer that is about to be
    dropped."""
    sentinel = "GIZLI-DENETCI-CUMLESI"
    noisy = (f"{sentinel} {gate.tree}\n{_REAL_STDERR}"
             f"session_id=abc123 {sentinel}\n")
    binary = _failing(tmp_path, noisy.encode("utf-8"), name="sahte_gurultulu")
    with pytest.raises(audit.RepositoryRefused) as refusal:
        _audit(binary, gate)

    failure = refusal.value
    carried = " ".join([str(failure), repr(failure),
                        repr(getattr(failure, "__dict__", {})),
                        repr(getattr(failure, "__notes__", []))])
    assert sentinel not in carried
    assert str(gate.tree) not in carried
    assert "session_id" not in carried
    # not even the marker that produced the class
    assert "trusted directory" not in carried
    assert "--skip-git-repo-check" not in carried
    assert failure.stderr_bytes == len(noisy.encode("utf-8"))


# =====================================================================
# B4-R11 -- WHAT TRAVELS IS NOT WHAT DECIDES
# =====================================================================
#
# MEASURED from the failing run's own session record: the provider
# answered 400 `invalid_json_schema`, param `text.format.schema`, saying
# `allOf` is not permitted at the root -- which is exactly where the
# audit schema's conditional rules live. The argv file now carries a
# derived TRANSPORT copy; the reply is still judged by the authority.
#
# Every test below proves BOTH halves, because "the transport accepted
# it" is not a verdict and a suite that only checked the weaker document
# would go green on the defect this package exists to remove.

_FORBIDDEN_KEYWORDS = ("allOf", "if", "then", "minLength", "maxLength",
                       "minimum", "maximum", "maxItems", "minItems",
                       "pattern", "else")


def _keyword_counts(node, counts=None):
    """RECURSIVE, because "no allOf at the root" is not the claim. The
    root was merely where the provider stopped reading."""
    counts = {} if counts is None else counts
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for sub in value.values():
                    _keyword_counts(sub, counts)
                continue
            counts[key] = counts.get(key, 0) + 1
            _keyword_counts(value, counts)
    elif isinstance(node, list):
        for value in node:
            _keyword_counts(value, counts)
    return counts


def _forbidden(schema):
    counts = _keyword_counts(schema)
    return {key: counts[key] for key in _FORBIDDEN_KEYWORDS if counts.get(key)}


def test_the_file_on_the_argv_carries_the_transport_and_not_the_authority(
        tmp_path, gate, only_fake_models_may_run):
    """THE DEFECT, stated as a test. The witness is the CHILD's copy of
    the file it was handed -- what the adapter says it wrote is not
    evidence about what arrived."""
    kayit = tmp_path / "verilen-sema.json"
    binary = _stub(tmp_path, mode="reply", body=json.dumps(_code_reply()),
                   schema_record=kayit)
    outcome = _audit(binary, gate)

    teslim = kayit.read_bytes()
    assert teslim == schemas.CODE_AUDIT_TRANSPORT_BINDING.canonical_bytes
    assert teslim != schemas.CODE_AUDIT_SCHEMA_BINDING.canonical_bytes
    # and it is provider-compatible by COUNT, not by root inspection
    assert _forbidden(json.loads(teslim.decode("utf-8"))) == {}
    assert _forbidden(schemas.CODE_AUDIT_RESULT_SCHEMA)["allOf"] == 1
    # the reply was judged by the AUTHORITY all the same
    assert outcome.schema_sha256 == schemas.CODE_AUDIT_SCHEMA_BINDING.sha256
    assert outcome.transport_sha256 == \
        schemas.CODE_AUDIT_TRANSPORT_BINDING.sha256
    assert outcome.schema_sha256 != outcome.transport_sha256


_CODE_FINDING_BODY = {"finding_id": "bulgu-1", "mechanism_id": "mekanizma-1",
                      "severity": "high", "claim": "kurgu iddia",
                      "reproduction_result": "reproduced",
                      "required_action": "kurgu eylem"}


@pytest.mark.parametrize(
    "govde",
    [{"status": contract.Status.APPROVED, "next_action": "stop",
      "findings": [_CODE_FINDING_BODY]},
     {"status": contract.Status.CHANGES_REQUESTED,
      "next_action": "await_repair", "findings": []},
     {"status": contract.Status.APPROVED, "next_action": "await_repair"}],
    ids=["onaylandi-ama-bulgu-var", "degisiklik-ama-bulgu-yok",
         "onaylandi-ama-yanlis-eylem"])
def test_what_the_transport_allows_the_authority_still_refuses(
        tmp_path, gate, govde, only_fake_models_may_run):
    """THE LOAD-BEARING TEST of the whole split. Each body is chosen so
    the WEAKER document accepts it -- proven here, not assumed -- and
    the authority must still refuse it. If the adapter ever validated
    with the transport copy, every one of these would be approved."""
    reply = _code_reply(**govde)
    # the transport copy really does accept it: that is what makes the
    # refusal below evidence about the authority rather than about JSON
    schemas.SchemaBinding(schemas.CODE_AUDIT_TRANSPORT_SCHEMA).validate(reply)

    binary = _stub(tmp_path, mode="reply", body=json.dumps(reply))
    with pytest.raises(audit.SchemaViolation) as refusal:
        _audit(binary, gate)
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
    assert "sema disi" in str(refusal.value)


def test_a_healthy_reply_passes_the_transport_and_the_authority(
        tmp_path, gate, only_fake_models_may_run):
    """The positive control: an adapter that refused everything would
    pass the test above."""
    reply = _code_reply()
    schemas.SchemaBinding(schemas.CODE_AUDIT_TRANSPORT_SCHEMA).validate(reply)
    schemas.CODE_AUDIT_SCHEMA_BINDING.validate(reply)

    binary = _stub(tmp_path, mode="reply", body=json.dumps(reply))
    outcome = _audit(binary, gate)
    assert outcome.reply["status"] == contract.Status.APPROVED
    assert outcome.exit_code == 0


def test_a_locked_transport_carries_this_call_s_issued_ids(
        tmp_path, gate, only_fake_models_may_run):
    """A STATIC locked transport would describe a document accepting ids
    nobody minted. The transport is derived from the schema already
    bound to this call, so `const` and the two `enum`s travel with it."""
    kosu, bulgu_id, mekanizma_id = "a" * 32, "f" * 32, "e" * 32
    bulgu = {"finding_id": bulgu_id, "mechanism_id": mekanizma_id,
             "severity": "high",
             "error_class": contract.LockedFindingClass.WRONG_ROW,
             "case_count": 3}
    reply = _locked_reply(
        kosu, status=contract.Status.CHANGES_REQUESTED,
        summary_code=contract.SummaryCode.REGRESSION_DETECTED,
        next_action="await_repair", findings=[bulgu])
    kayit = tmp_path / "kilitli-sema.json"
    binary = _stub(tmp_path, mode="reply", body=json.dumps(reply),
                   schema_record=kayit)
    outcome = _audit(binary, gate, audit_kind=contract.AuditKind.LOCKED,
                     issued_run_id=kosu, issued_finding_ids=[bulgu_id],
                     issued_mechanism_ids=[mekanizma_id])

    teslim = json.loads(kayit.read_text(encoding="utf-8"))
    assert _forbidden(teslim) == {}
    assert teslim["properties"]["run_id"] == {"const": kosu}
    alanlar = teslim["properties"]["findings"]["items"]["properties"]
    assert alanlar["finding_id"] == {"enum": [bulgu_id]}
    assert alanlar["mechanism_id"] == {"enum": [mekanizma_id]}
    assert outcome.schema_sha256 != outcome.transport_sha256


def test_an_unissued_locked_id_is_still_refused_by_the_authority(
        tmp_path, gate, only_fake_models_may_run):
    """The reduction must not have loosened the binding it travels
    with. The stub answers whatever it is told, so nothing but the
    authority is between this reply and an approval."""
    kosu, bulgu_id, mekanizma_id = "a" * 32, "f" * 32, "e" * 32
    kacak = {"finding_id": "b" * 32, "mechanism_id": mekanizma_id,
             "severity": "high",
             "error_class": contract.LockedFindingClass.WRONG_ROW,
             "case_count": 3}
    reply = _locked_reply(
        kosu, status=contract.Status.CHANGES_REQUESTED,
        summary_code=contract.SummaryCode.REGRESSION_DETECTED,
        next_action="await_repair", findings=[kacak])
    binary = _stub(tmp_path, mode="reply", body=json.dumps(reply))
    with pytest.raises(audit.SchemaViolation):
        _audit(binary, gate, audit_kind=contract.AuditKind.LOCKED,
               issued_run_id=kosu, issued_finding_ids=[bulgu_id],
               issued_mechanism_ids=[mekanizma_id])
    assert len(only_fake_models_may_run) == 1, "senaryo kurulmadi"
