"""PACKAGE B2A -- the safe implementer subprocess adapter.

NO REAL MODEL IS CALLED ANYWHERE IN THIS FILE. Every binary is a shim
written into `tmp_path` during the test, and an autouse guard records
each launch and fails the test if anything outside `tmp_path` was ever
executed. The claim is asserted, not asserted-about.

WHY THE FAKES ARE GENERATED HERE. B2A may only create two files, so the
helper cannot live beside this module. Each test writes a launcher (a
`.cmd` on Windows, a `.sh` elsewhere -- the only shapes the preflight
runnability rule accepts) plus a small Python script it delegates to.
Behaviour that has to be byte-exact -- invalid UTF-8, an output limit
hit exactly, a flood that must not be read to the end -- is impossible
to express portably in shell, and guessing at it is how a test ends up
proving something other than what it says.

Every negative test asserts its SETUP was reached before it claims the
refusal.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import cli, contract, execution, schemas, state, worktree
from tools.agent_loop import process as process_module

RUN = "kurgu-run-1"
# the contract's own minimum per-call timeout; tests move the
# CLOCK rather than lowering it
MIN_TIMEOUT = execution.MIN_TIMEOUT_SECONDS

# Delegating shim: the adapter needs a genuinely launchable file, and the
# behaviour needs Python. `%*` / `"$@"` forward nothing of ours -- the
# adapter's argv arrives on the shim's own command line.
_HELPER = '''\
import json, os, subprocess, sys, time
from pathlib import Path


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if cfg.get("cwd_record"):
        # WHERE the adapter really ran this child, written by the child
        # itself -- the only witness whose answer is not the adapter's
        Path(cfg["cwd_record"]).write_text(os.getcwd(), encoding="utf-8")
    mode = cfg["mode"]
    if mode == "dump":
        payload = {"argv": sys.argv[2:], "cwd": os.getcwd(),
                   "env": dict(os.environ), "stdin": sys.stdin.read()}
        sys.stdout.write(json.dumps(payload))
    elif mode == "raw":
        sys.stdout.buffer.write(bytes.fromhex(cfg["hex"]))
        sys.stdout.buffer.flush()
    elif mode == "consume":
        received = sys.stdin.buffer.read()
        Path(cfg["record"]).write_text(str(len(received)), encoding="ascii")
        sys.stdout.buffer.write(bytes.fromhex(cfg["hex"]))
        sys.stdout.buffer.flush()
    elif mode == "flood":
        target = (sys.stdout if cfg["stream"] == "out" else sys.stderr).buffer
        block = b"x" * 4096
        sent = 0
        while sent < cfg["bytes"]:
            target.write(block)
            target.flush()
            sent += len(block)
    elif mode == "sleep":
        time.sleep(cfg["seconds"])
    elif mode == "spawn":
        subprocess.Popen([sys.executable, "-c", cfg["child_code"]])
        if cfg.get("wait_for"):
            # do not answer until the grandchild has PROVEN it is alive,
            # so the scenario cannot pass by never having run
            limit = time.monotonic() + 20
            while not Path(cfg["wait_for"]).exists():
                if time.monotonic() > limit:
                    sys.exit(97)
                time.sleep(0.02)
        if cfg.get("stdout_hex"):
            sys.stdout.buffer.write(bytes.fromhex(cfg["stdout_hex"]))
            sys.stdout.buffer.flush()
        sys.stdin.read()
        time.sleep(cfg["seconds"])
    sys.stderr.write(cfg.get("stderr", ""))
    sys.stderr.flush()
    sys.exit(cfg.get("code", 0))


main()
'''


def _fake_binary(tmp_path, name="sahte_claude", **config):
    """A launchable stub plus its configuration."""
    holder = tmp_path / "sahte-bin"
    holder.mkdir(exist_ok=True)
    helper = holder / "yardimci.py"
    helper.write_text(_HELPER, encoding="utf-8")
    settings = holder / f"{name}.json"
    settings.write_text(json.dumps(config), encoding="utf-8")
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


def _valid_reply(**overrides):
    payload = {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": RUN,
        "role": "implementer",
        "status": "implemented",
        "summary": "kurgu ozet",
        "next_action": "await_acceptance",
        "changed_files": ["pipeline/kurgu.py"],
    }
    payload.update(overrides)
    return payload


def _git(repo, *args):
    """The REAL subprocess module, on purpose: the recorder below only
    patches what the code under test uses."""
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture(autouse=True)
def private_worktree_root(tmp_path):
    """Every test gets its OWN runner temp root -- B1's lesson, imported
    wholesale: the real root is shared by every process on the machine,
    and the worktrees this battery now creates must be unable to collide
    with, or sweep away, anybody else's.

    Its OWN `MonkeyPatch`, not the shared fixture: one test in this file
    calls `monkeypatch.undo()` mid-test, which would silently undo this
    redirect too."""
    private = tmp_path / "runner-temp"
    private.mkdir()
    isolation = pytest.MonkeyPatch()
    isolation.setattr(tempfile, "tempdir", str(private))
    for variable in ("TMPDIR", "TEMP", "TMP"):
        isolation.setenv(variable, str(private))
    root = worktree.runner_temp_root()
    assert root.parent.resolve() == private.resolve(), \
        "gecici kok bu teste ozel degil"
    yield root
    isolation.undo()


@pytest.fixture
def bound(tmp_path, private_worktree_root):
    """A REAL repository, a REAL B1 worktree at its baseline, and the
    identity tuple that is now the ONLY way to name it. R2A removed the
    path parameter: execution derives the directory from these or it
    refuses."""
    repo = tmp_path / "kurgu-depo"
    repo.mkdir()
    for argv in (["init", "-q"],
                 ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(repo, *argv)
    (repo / "kurgu.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "kurgu")
    baseline = _git(repo, "rev-parse", "HEAD")
    state_dir = tmp_path / "durum"
    path, worktree_id = worktree.create(repo, state_dir=state_dir,
                                        run_id=RUN, baseline_sha=baseline)
    # the RUN'S binding, pinning exactly this worktree -- R2A-R1: the
    # first fixture never wrote it, so the seam could not be asked
    # whether it reads the document at all, and it did not
    state.write_binding(state_dir, {
        "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
        "repo_id": state.repo_identity(repo), "baseline_sha": baseline,
        "manifest_digest": "0" * 64, "worktree_id": worktree_id})
    yield types.SimpleNamespace(
        repo=repo, state_dir=state_dir, run_id=RUN, worktree_id=worktree_id,
        baseline_sha=baseline, path=path,
        identity={"repo": repo, "state_dir": state_dir, "run_id": RUN,
                  "worktree_id": worktree_id, "baseline_sha": baseline})
    try:
        worktree.remove(repo, state_dir=state_dir, worktree_id=worktree_id)
    except Exception:                                  # noqa: BLE001
        # negative tests sabotage the repo or the record on purpose;
        # everything lives under tmp_path and is discarded with it
        pass


@pytest.fixture
def stub_cwd(tmp_path):
    """cwd for DIRECT stub runs only -- probes that never touch the
    adapter and therefore need no binding."""
    path = tmp_path / "dogrudan-cwd"
    path.mkdir()
    return path


def _no_binding(tmp_path):
    """Identities that CANNOT bind, for refusals that must fire BEFORE
    the binding is consulted. If the validation order ever changes,
    every test using these goes red with `WorktreeNotBound` in place of
    the type it asserts -- the order is part of the claim."""
    return {"repo": tmp_path, "state_dir": tmp_path / "durum-yok",
            "run_id": RUN, "worktree_id": "0" * 32,
            "baseline_sha": "0" * 40}


# R2B removed the schema_file fixture: the schema is not a file, not a
# parameter and not a fixture any more -- it is the frozen canonical
# binding, inline on the argv, and nothing here can substitute it.


@pytest.fixture(autouse=True)
def only_fake_binaries_may_run(tmp_path, monkeypatch):
    """THE claim this whole file rests on, enforced rather than repeated.

    Every launch is recorded and its program checked against `tmp_path`.
    A test that accidentally reached a real `claude` or `codex` would
    spend money and still look green -- an earlier suite in this project
    created fake binaries and then never handed them to anything."""
    launched = []
    started = []
    real_popen = process_module.subprocess.Popen

    class _Recorder:
        DEVNULL = subprocess.DEVNULL
        PIPE = subprocess.PIPE
        TimeoutExpired = subprocess.TimeoutExpired
        SubprocessError = subprocess.SubprocessError
        CREATE_NEW_PROCESS_GROUP = getattr(subprocess,
                                           "CREATE_NEW_PROCESS_GROUP", 0)

        @staticmethod
        def run(argv, **kwargs):
            launched.append(list(argv))
            return subprocess.run(argv, **kwargs)

        @staticmethod
        def Popen(argv, **kwargs):                     # noqa: N802 -- stdlib
            launched.append(list(argv))
            process = real_popen(argv, **kwargs)
            started.append(process)
            return process

    monkeypatch.setattr(process_module, "subprocess", _Recorder)
    monkeypatch.setattr(execution, "subprocess", _Recorder)
    yield launched
    # NO TEST MAY OUTLIVE ITS PROCESSES. A negative test that sabotages
    # cleanup leaves a real child behind, and an audit found one of ours
    # still running after the suite finished. Every process this file
    # started is stopped here, with the REAL tools rather than the
    # module under test -- which may still be monkeypatched.
    survivors = [process for process in started if process.poll() is None]
    for process in survivors:
        _kill_tree_for_real(process)
    root = str(tmp_path).casefold()
    strayed = [argv[0] for argv in launched
               if not str(argv[0]).casefold().startswith(root)
               and Path(argv[0]).name.casefold() not in ("taskkill.exe",
                                                         "taskkill")]
    assert strayed == [], f"tmp_path disinda bir program calistirildi: {strayed}"
    alive = [process.pid for process in started if process.poll() is None]
    assert alive == [], f"test bittikten sonra yasayan surec: {alive}"


def _kill_tree_for_real(process):
    """The suite's own safety net, independent of the code under test."""
    try:
        if os.name == "nt":
            killer = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            subprocess.run([str(killer / "System32" / "taskkill.exe"),
                            "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(process.pid), 9)
    except Exception:                                  # noqa: BLE001
        pass
    try:
        process.kill()
        process.wait(timeout=15)
    except Exception:                                  # noqa: BLE001
        pass


@pytest.fixture
def fast_clock(monkeypatch):
    """Moves the CLOCK, not the contract.

    The frozen task schema puts the minimum per-call timeout at 30
    seconds. Proving timeout behaviour would otherwise mean half-minute
    tests or a production range widened to suit them -- and loosening a
    contract to make tests convenient is how the range stopped meaning
    anything the first time. So the adapter's one clock seam runs fast
    and every timeout here asks for the real contract minimum."""
    origin = time.monotonic()

    def racing():
        return origin + (time.monotonic() - origin) * 60.0

    monkeypatch.setattr(execution, "_now", racing)
    return racing


# =====================================================================
# THE PROCESS: exactly what was asked for, nowhere else
# =====================================================================

def test_the_adapter_launches_exactly_the_argv_the_cli_builds(
        tmp_path, bound, only_fake_binaries_may_run):
    binary = _fake_binary(tmp_path, stdout_json=None,
                          mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    execution.run_implementer(
        binary, **bound.identity, prompt="kurgu istem",
        budget_usd=1.0, timeout_seconds=60,
        max_output_bytes=65536)

    expected = cli.build_implementer_argv(binary, budget_usd=1.0)
    assert only_fake_binaries_may_run[0] == expected, \
        "adapter cli.py'nin urettigi argv'yi degistirdi"


def test_the_process_runs_inside_the_derived_worktree(
        tmp_path, bound):
    """WHERE it ran, read from the child itself: `dump` emits a report
    rather than a reply, so the refusal proves the process really
    executed, and the trace file carries its cwd out. The adapter was
    handed identities only -- the path it used had to be derived."""
    trace = tmp_path / "nerede-dump.txt"
    binary = _fake_binary(tmp_path, mode="dump", cwd_record=str(trace))
    with pytest.raises(execution.SchemaViolation):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu istem",
            budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=65536)
    assert trace.exists(), "senaryo kurulmadi: cocuk hic calismadi"
    assert Path(trace.read_text(encoding="utf-8")).resolve() == \
        bound.path.resolve()


def _dump_report(tmp_path, cwd, prompt):
    """Run the dump stub directly to read what the child observed."""
    binary = _fake_binary(tmp_path, name="dokum", mode="dump")
    argv = cli.build_implementer_argv(binary, budget_usd=1.0)
    done = subprocess.run(argv, cwd=str(cwd), input=prompt,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_prompt_travels_only_on_stdin(tmp_path, stub_cwd):
    """A prompt on a command line is a prompt in `ps` output, in a shell
    history and in any error that echoes the argv."""
    prompt = "KURGU-ISTEM-NOBETCISI-" + "q" * 10
    report = _dump_report(tmp_path, stub_cwd, prompt=prompt)

    assert report["stdin"] == prompt, "istem stdin'e ulasmadi"
    assert not any(prompt in str(token) for token in report["argv"]), \
        "istem komut satirina sizdi"
    assert not any(prompt in value for value in report["env"].values()), \
        "istem ortam degiskenine sizdi"


def test_a_missing_binary_argument_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        execution.run_implementer(                     # noqa: PLE1120
            **_no_binding(tmp_path), prompt="kurgu",
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)


@pytest.mark.parametrize("bare", ["claude", "codex", "git", "cmd"])
def test_a_bare_command_name_is_never_resolved_through_the_path(
        tmp_path, only_fake_binaries_may_run, bare):
    """These exist on PATH. The adapter still refuses them, because it
    executes a path and never searches for one."""
    with pytest.raises(execution.BinaryNotUsable):
        execution.run_implementer(
            bare, **_no_binding(tmp_path), prompt="kurgu",
            budget_usd=1.0,
            timeout_seconds=MIN_TIMEOUT, max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], "reddedilen isim yine de calisti"


def test_no_shell_is_ever_involved():
    """Structural, not behavioural: a shell that appears only on a rare
    branch would pass every functional test in this file."""
    import ast

    source = "".join(Path(module.__file__).read_text(encoding="utf-8")
                     for module in (execution, process_module))
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "shell" and \
                        getattr(keyword.value, "value", False) is True:
                    offenders.append("shell=True")
            if ast.unparse(node.func) in ("os.system", "os.popen",
                                          "subprocess.getoutput"):
                offenders.append(ast.unparse(node.func))
    assert offenders == [], f"kabuk kullanimi: {offenders}"
    for token in ("bash -c", "/bin/sh -c", "cmd /c", "cmd.exe", "powershell"):
        assert token not in source.lower(), f"kabuk cagrisi metni: {token}"


# =====================================================================
# THE REPLY: parsed, validated, never repaired
# =====================================================================

def _run_with_stdout(tmp_path, bound, payload_bytes,
                     **kwargs):
    binary = _fake_binary(tmp_path, mode="raw", hex=payload_bytes.hex(),
                          **{k: v for k, v in kwargs.items()
                             if k in ("code", "stderr")})
    settings = {"prompt": "kurgu", "budget_usd": 1.0,
                "timeout_seconds": 60, "max_output_bytes": 65536,
                "model": None}
    settings.update({k: v for k, v in kwargs.items() if k in settings})
    return execution.run_implementer(binary, **bound.identity, **settings)


def test_a_valid_reply_comes_back_as_structured_data(tmp_path, bound):
    reply = _valid_reply()
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(reply).encode("utf-8"))
    assert result.reply == reply
    assert result.exit_code == 0
    assert result.event == contract.EventCode.MODEL_CALL_FINISHED
    assert isinstance(result.duration_ms, int)
    assert result.reply["changed_files"] == ["pipeline/kurgu.py"]


@pytest.mark.parametrize(
    ("label", "payload"),
    [("bozuk-json", b"{bu JSON degil"),
     ("json-degil", b"sadece duz metin"),
     ("bos", b""),
     ("dizi", b"[1, 2, 3]")],
    ids=["bozuk-json", "duz-metin", "bos-cikti", "nesne-degil"])
def test_output_that_is_not_a_json_object_is_refused(tmp_path, bound, label,
                                                     payload):
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, bound, payload)


@pytest.mark.parametrize(
    ("label", "reply"),
    [("rol", _valid_reply(role="evaluator")),
     ("durum", _valid_reply(status="approved")),
     ("protokol", _valid_reply(protocol_version="0.9")),
     ("eksik-alan", {k: v for k, v in _valid_reply().items()
                     if k != "next_action"}),
     ("fazla-alan", dict(_valid_reply(), gizli_alan="kurgu")),
     ("serbest-metin", dict(_valid_reply(), summary="x" * 2001)),
     ("uydurma-eylem", _valid_reply(next_action="uydurma"))],
    ids=["yanlis-rol", "yanlis-durum", "yanlis-protokol", "eksik-alan",
         "fazla-alan", "asiri-uzun-ozet", "sozluk-disi-eylem"])
def test_a_reply_outside_the_frozen_schema_is_refused(tmp_path, bound, label,
                                                      reply):
    """The schema is the contract. Nothing here repairs, coerces or
    guesses -- a model that answered outside the vocabulary is a model
    whose answer nobody can act on."""
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, bound,
                         json.dumps(reply).encode("utf-8"))


def test_invalid_utf8_is_refused(tmp_path, bound):
    """Decoded strictly. A replacement character would turn undecodable
    bytes into a string that might then parse as something."""
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, bound,
                         b'{"role": "\xff\xfe implementer"}')


def test_a_nonzero_exit_is_a_typed_failure_without_the_stderr(
        tmp_path, bound):
    sentinel = "KURGU-STDERR-NOBETCISI-" + "w" * 8
    with pytest.raises(execution.ProcessFailed) as refusal:
        _run_with_stdout(tmp_path, bound,
                         json.dumps(_valid_reply()).encode("utf-8"),
                         code=7, stderr=sentinel)
    failure = refusal.value
    assert failure.exit_code == 7
    assert sentinel not in str(failure)
    assert sentinel not in repr(failure)
    assert failure.stderr_bytes >= len(sentinel)


# =====================================================================
# BOUNDS: enforced while reading, not after
# =====================================================================

def test_a_flood_on_stdout_stops_the_process(tmp_path, bound):
    limit = 8192
    binary = _fake_binary(tmp_path, mode="flood", stream="out",
                          bytes=limit * 40)
    started = time.monotonic()
    with pytest.raises(execution.OutputLimitExceeded) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=120,
            max_output_bytes=limit)
    assert refusal.value.stream == "stdout"
    assert refusal.value.stdout_bytes <= limit + process_module.READ_CHUNK_BYTES
    assert time.monotonic() - started < 100, "sinir asimi beklenmedi"


def test_a_flood_on_stderr_is_bounded_too(tmp_path, bound):
    limit = 8192
    binary = _fake_binary(tmp_path, mode="flood", stream="err",
                          bytes=limit * 40)
    with pytest.raises(execution.OutputLimitExceeded) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=120,
            max_output_bytes=limit)
    assert refusal.value.stream == "stderr"


def test_output_of_exactly_the_limit_is_accepted(tmp_path, bound):
    """The boundary in the other direction, or the rule above is just
    "refuse large replies"."""
    # padded past the contract's own floor: `max_output_bytes` below
    # 1024 is outside the frozen range and is refused before launch
    padded = _valid_reply(summary="k" * 1200)
    reply = json.dumps(padded).encode("utf-8")
    result = _run_with_stdout(tmp_path, bound, reply,
                              max_output_bytes=len(reply))
    assert result.reply["role"] == "implementer"
    assert result.stdout_bytes == len(reply)


# =====================================================================
# TIMEOUT: the process AND everything it started
# =====================================================================

_CHILD = ("import pathlib, time\n"
          "p = pathlib.Path({beat!r})\n"
          "while True:\n"
          "    p.write_text(str(time.time()))\n"
          "    time.sleep(0.1)\n")


def test_a_timeout_kills_the_process_and_its_children(tmp_path, bound, fast_clock):
    """A model process that spawned something must not leave it running.
    The child writes a heartbeat; after the kill the heartbeat has to
    stop advancing, which is observable without a process library."""
    beat = tmp_path / "kalp.txt"
    binary = _fake_binary(tmp_path, mode="spawn", seconds=120,
                          child_code=_CHILD.format(beat=str(beat)))
    with pytest.raises(execution.Timeout) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert refusal.value.timeout_seconds == MIN_TIMEOUT

    deadline = time.monotonic() + 10
    while not beat.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert beat.exists(), "senaryo kurulmadi: cocuk surec hic calismadi"

    time.sleep(1.5)
    first = beat.read_text(encoding="utf-8")
    time.sleep(1.5)
    assert beat.read_text(encoding="utf-8") == first, \
        "cocuk surec oldurulmedi; hala yaziyor"


def test_a_hanging_process_with_no_children_also_times_out(tmp_path,
                                                           bound, fast_clock):
    binary = _fake_binary(tmp_path, mode="sleep", seconds=120)
    started = time.monotonic()
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert time.monotonic() - started < 60


def test_a_cleanup_failure_does_not_replace_the_primary_failure(
        tmp_path, bound, fast_clock, monkeypatch):
    """The caller needs to know the run timed out. A kill that then
    fails is a second problem, not the headline."""
    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)

    def refusing_kill(*args, **kwargs):
        raise OSError("kurgu: surec agaci oldurulemedi")

    monkeypatch.setattr(process_module, "terminate_tree", refusing_kill)
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)


# =====================================================================
# BUDGET: refused before anything starts
# =====================================================================

@pytest.mark.parametrize(
    "budget", [0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True],
    ids=["sifir-int", "sifir", "negatif", "nan", "sonsuz", "eksi-sonsuz",
         "bool"])
def test_an_unusable_budget_refuses_before_any_process_starts(
        tmp_path, only_fake_binaries_may_run,
        budget):
    """Counted, not assumed: the refusal is only worth anything if no
    process was started, and "we would have refused later" is what a
    paid call looks like in hindsight."""
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    with pytest.raises(execution.BudgetRefused):
        execution.run_implementer(
            binary, **_no_binding(tmp_path), prompt="kurgu",
            budget_usd=budget, timeout_seconds=60,
            max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], \
        "butce reddedildikten sonra yine de bir surec baslatildi"


def test_the_exact_remaining_budget_reaches_the_argv(tmp_path, bound,
                                                     only_fake_binaries_may_run):
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    execution.run_implementer(
        binary, **bound.identity, prompt="kurgu",
        budget_usd=0.375, timeout_seconds=60,
        max_output_bytes=65536)
    argv = only_fake_binaries_may_run[0]
    assert "0.375" in argv[argv.index("--max-budget-usd") + 1]


def test_the_adapter_never_claims_to_know_what_was_spent(tmp_path,
                                                         bound):
    """`IMPLEMENTER_RESULT_SCHEMA` carries no cost field, so there is no
    contractual value to report. Reporting zero would be a number the
    caller could subtract from a budget."""
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert not hasattr(result, "spent_usd")
    assert not hasattr(result, "cost_usd")
    assert "spent" not in {field for field in vars(result)}


# =====================================================================
# WHAT MAY LEAVE THE ADAPTER
# =====================================================================

def test_no_raw_model_output_survives_in_the_result(tmp_path, bound):
    """Byte counts and codes leave; text does not. Raw stdout in a
    result object is raw stdout in whatever log the caller writes."""
    sentinel = "KURGU-CIKTI-NOBETCISI-" + "v" * 8
    reply = _valid_reply(summary=sentinel)
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(reply).encode("utf-8"))
    # the PARSED reply legitimately holds it; nothing else may
    leaked = [name for name, value in vars(result).items()
              if name != "reply" and sentinel in str(value)]
    assert leaked == [], f"ham cikti tasiyan alan: {leaked}"
    assert not any(isinstance(value, (bytes, bytearray))
                   for value in vars(result).values())


def test_the_adapter_writes_no_files_of_its_own(tmp_path, bound):
    before = {p for p in bound.path.rglob("*")}
    _run_with_stdout(tmp_path, bound,
                     json.dumps(_valid_reply()).encode("utf-8"))
    assert {p for p in bound.path.rglob("*")} == before, \
        "adapter calisma agacina dosya yazdi"


def test_only_closed_codes_and_numbers_leave_the_adapter(tmp_path,
                                                         bound):
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert result.event in contract.ALL_EVENT_CODES
    for name, value in vars(result).items():
        if name == "reply":
            continue
        assert isinstance(value, (int, str)), f"{name} sayisal/kod degil"
        if isinstance(value, str):
            if name == "schema_sha256":
                # a 64-hex code, not free text: the ONE schema fact a
                # report may carry
                assert re.fullmatch(r"[0-9a-f]{64}", value), \
                    "schema_sha256 64-hex bir kod degil"
                continue
            assert value in contract.ALL_EVENT_CODES, \
                f"{name} kapali sozluk disinda bir metin tasiyor"


# =====================================================================
# R1 -- what the first round of tests did not cover
# =====================================================================

def test_a_child_that_never_reads_stdin_still_times_out(
        tmp_path, bound, fast_clock):
    """THE P0. The prompt used to be written synchronously, before the
    readers and before the deadline loop -- so a child that read nothing
    filled the pipe buffer, the write blocked, and the timeout that was
    supposed to cover exactly this had not started yet. The adapter did
    not return at all; an outside watchdog had to kill it.

    The write is asynchronous now, and the deadline is absolute."""
    prompt = "P" * (2 * 1024 * 1024)
    binary = _fake_binary(tmp_path, mode="sleep", seconds=120)
    started = time.monotonic()
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, **bound.identity, prompt=prompt,
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    elapsed = time.monotonic() - started
    assert elapsed < 45, f"sure siniri stdin yaziminda asildi ({elapsed:.1f}s)"


@pytest.mark.parametrize(
    ("label", "prompt"),
    [("metin-degil", 123), ("bos", ""), ("bayt", b"kurgu"),
     ("vekil-karakter", "kurgu \ud800"),
     ("tavan-ustu", "P" * (execution.MAX_PROMPT_BYTES + 1))],
    ids=["metin-degil", "bos", "bayt-dizisi", "kodlanamaz", "tavan-ustu"])
def test_an_unusable_prompt_refuses_before_any_process_starts(
        tmp_path, only_fake_binaries_may_run,
        label, prompt):
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            binary, **_no_binding(tmp_path), prompt=prompt,
            budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], "reddedilen istem yine de calisti"


@pytest.mark.parametrize(
    ("label", "timeout_seconds"),
    [("nan", float("nan")), ("sonsuz", float("inf")), ("sifir", 0),
     ("negatif", -1), ("bool", True), ("metin", "60"), ("ondalik", 60.0),
     ("taban-alti", execution.MIN_TIMEOUT_SECONDS - 1),
     ("tavan-ustu", execution.MAX_TIMEOUT_SECONDS + 1)],
    ids=["nan", "sonsuz", "sifir", "negatif", "bool", "metin", "ondalik",
         "taban-alti", "tavan-ustu"])
def test_an_unusable_timeout_refuses_before_any_process_starts(
        tmp_path, only_fake_binaries_may_run,
        label, timeout_seconds):
    """`timeout_seconds=NaN` made every deadline comparison false, so a
    hung child ran until something outside stopped it. A bound that
    cannot be compared is not a bound."""
    binary = _fake_binary(tmp_path, mode="sleep", seconds=30)
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            binary, **_no_binding(tmp_path), prompt="kurgu",
            budget_usd=1.0,
            timeout_seconds=timeout_seconds, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


@pytest.mark.parametrize(
    ("label", "max_output_bytes"),
    [("nan", float("nan")), ("sonsuz", float("inf")), ("sifir", 0),
     ("negatif", -1), ("bool", True), ("ondalik", 65536.0),
     ("taban-alti", execution.MIN_OUTPUT_BYTES - 1),
     ("tavan-ustu", execution.MAX_OUTPUT_BYTES + 1)],
    ids=["nan", "sonsuz", "sifir", "negatif", "bool", "ondalik",
         "taban-alti", "tavan-ustu"])
def test_an_unusable_output_ceiling_refuses_before_any_process_starts(
        tmp_path, only_fake_binaries_may_run,
        label, max_output_bytes):
    """`max_output_bytes=NaN` read a megabyte in full and then refused it
    as bad JSON -- a parse error standing in for a ceiling that never
    applied, which is a refusal for the wrong reason after the cost was
    already paid."""
    binary = _fake_binary(tmp_path, mode="flood", stream="out",
                          bytes=1024 * 1024)
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            binary, **_no_binding(tmp_path), prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=max_output_bytes)
    assert only_fake_binaries_may_run == []


def test_the_contract_range_is_taken_from_the_frozen_schema():
    """The bounds are the task schema's, not numbers invented here."""
    task = schemas.TASK_SCHEMA["properties"]
    assert execution.MIN_OUTPUT_BYTES == task["max_output_bytes"]["minimum"]
    assert execution.MAX_OUTPUT_BYTES == task["max_output_bytes"]["maximum"]
    per_call = task["model_call_timeout_seconds"]
    assert execution.MIN_TIMEOUT_SECONDS == per_call["minimum"]
    assert execution.MAX_TIMEOUT_SECONDS == per_call["maximum"]
    assert execution.MAX_TIMEOUT_SECONDS != 60 * task[
        "max_wall_clock_minutes"]["maximum"],         "toplam duvar saati ile cagri basina sinir ayni sey degil"


def test_the_tree_killer_is_not_located_through_the_environment(
        tmp_path, bound, fast_clock, monkeypatch):
    """`%SystemRoot%` is an ordinary environment variable: anything that
    set it redirected the only cleanup path this adapter has, and the
    failure was swallowed. The system directory now comes from the
    operating system."""
    source = "".join(Path(module.__file__).read_text(encoding="utf-8")
                     for module in (execution, process_module))
    for read_from_environment in ('environ.get("SystemRoot"',
                                  'environ["SystemRoot"]',
                                  'environ.get("windir"',
                                  'getenv("SystemRoot"'):
        assert read_from_environment not in source, read_from_environment

    monkeypatch.setenv("SystemRoot", str(tmp_path / "sahte-sistem"))
    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)
    started = time.monotonic()
    with pytest.raises(execution.Timeout) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert time.monotonic() - started < 45
    assert refusal.value.cleanup_complete is True, \
        "yonlendirilmis ortamda temizlik tamamlanamadi"


def test_a_failed_cleanup_is_reported_and_does_not_hide_the_timeout(
        tmp_path, bound, fast_clock, monkeypatch):
    """Both halves: the caller still hears "timed out", and the fact
    that a process may still be running does not vanish into an
    `except`."""
    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)

    def refusing_kill(*args, **kwargs):
        raise OSError("kurgu: surec agaci oldurulemedi")

    monkeypatch.setattr(process_module, "terminate_tree", refusing_kill)
    with pytest.raises(execution.Timeout) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert refusal.value.cleanup_complete is False, \
        "basarisiz temizlik gorunmez kaldi"
    assert refusal.value.reason == contract.StopReason.TIMEOUT


def test_a_reply_produced_without_reading_the_prompt_is_refused(
        tmp_path, bound):
    """The asynchronous write is what stops a deaf child from blocking
    the deadline -- and it is also why delivery has to be CHECKED. A
    child that writes valid JSON and exits without reading answers a
    question it was never asked, and the adapter used to accept it: the
    writer's completion flag was computed and never consulted.

    The prompt is large enough that it cannot fit in a pipe buffer, so
    "the child exited early" and "the write finished anyway" cannot be
    the same run."""
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    with pytest.raises(execution.PromptNotDelivered) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="P" * (2 * 1024 * 1024),
            budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=65536)
    assert refusal.value.reason == contract.StopReason.INTERRUPTED
    assert refusal.value.exit_code == 0, \
        "surec basariyla bitti; reddedilen sey teslimat"


def test_a_large_prompt_arrives_whole(tmp_path, stub_cwd):
    """End to end: a prompt many times a pipe buffer arrives entire.

    Asserted on the BYTES THE CHILD RECEIVED rather than on the writer's
    own completion flag, because the flag is precisely what was wrong
    once.

    HONEST LIMIT, measured rather than assumed: this does NOT catch the
    partial-write defect. With a child that reads, the kernel moves the
    whole buffer even through a single raw `write`, so the truncation is
    invisible here -- verified by running this against the broken writer
    on Linux, where it still passed. The guard for that defect is
    `test_a_reply_produced_without_reading_the_prompt_is_refused`, and
    it only fails on POSIX."""
    prompt = "K" * (2 * 1024 * 1024)
    report = _dump_report(tmp_path, stub_cwd, prompt=prompt)
    assert len(report["stdin"]) == len(prompt), \
        f"istem kirpildi: {len(report['stdin'])} / {len(prompt)} bayt"
    assert report["stdin"] == prompt


def test_a_delivered_prompt_is_still_accepted(tmp_path, bound):
    """The boundary: the refusal above must not become "reject every
    reply". This child reads its stdin to the end."""
    report = _dump_report(tmp_path, bound.path,
                          prompt="kurgu istem")
    assert report["stdin"] == "kurgu istem"
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert result.reply["status"] == "implemented"


def test_a_failed_process_reports_the_contract_code_for_it(tmp_path,
                                                           bound):
    """Two wrong answers preceded this one: `preflight_failed` named a
    gate that had already succeeded, and `None` left a terminal result
    with no closed reason at all. The contract owner added
    `model_process_failed`."""
    with pytest.raises(execution.ProcessFailed) as refusal:
        _run_with_stdout(tmp_path, bound, b"", code=9)
    failure = refusal.value
    assert failure.reason == contract.StopReason.MODEL_PROCESS_FAILED
    assert failure.reason in contract.ALL_STOP_REASONS
    assert failure.reason != contract.StopReason.PREFLIGHT_FAILED
    assert failure.event == contract.EventCode.MODEL_CALL_FINISHED
    assert failure.exit_code == 9


def test_a_grandchild_never_outlives_a_SUCCESSFUL_call(tmp_path, bound):
    """THE finding. The model read its prompt, answered correctly and
    exited zero -- leaving a child running for two more minutes, which
    the adapter reported as success. Tracking the `Popen` objects we
    created cannot see that: it knows the parent and nothing the parent
    started.

    A leftover implementer still editing files while the patch is
    verified and carried to the main checkout is precisely what the
    disposable worktree exists to prevent, so the boundary is now drawn
    by the operating system and emptiness is asked of it."""
    beat = tmp_path / "torun-kalp.txt"
    child = _CHILD.format(beat=str(beat))
    reply = json.dumps(_valid_reply()).encode("utf-8")
    binary = _fake_binary(tmp_path, mode="spawn", seconds=0,
                          child_code=child, stdout_hex=reply.hex(),
                          wait_for=str(beat))

    result = execution.run_implementer(
        binary, **bound.identity, prompt="kurgu",
        budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
        max_output_bytes=65536)
    assert result.reply["status"] == "implemented", "senaryo kurulmadi"

    deadline = time.monotonic() + 10
    while not beat.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert beat.exists(), "senaryo kurulmadi: torun surec hic calismadi"
    time.sleep(1.0)
    first = beat.read_text(encoding="utf-8")
    time.sleep(1.5)
    assert beat.read_text(encoding="utf-8") == first, \
        "basarili cagridan sonra torun surec hala calisiyor"


def test_the_cleanup_budget_covers_the_tree_kill_too(tmp_path, bound, fast_clock,
                                                     monkeypatch):
    """`_stop` took a deadline and the tree-kill did not, so the killer
    kept a private thirty-second timeout: a ten-second budget measured
    twelve seconds. The remaining time is handed down."""
    seen = {}
    real_terminate = process_module.terminate_tree       # noqa: SLF001

    def recording(process, deadline=None):
        seen["deadline"] = deadline
        return real_terminate(process, deadline)

    monkeypatch.setattr(process_module, "terminate_tree", recording)
    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)
    started = time.monotonic()
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0,
            timeout_seconds=MIN_TIMEOUT, max_output_bytes=65536)
    elapsed = time.monotonic() - started
    assert seen["deadline"] is not None, "kill ortak butceyi almiyor"
    assert elapsed < process_module.REAP_SECONDS + 5, \
        f"temizlik ortak butceyi asti ({elapsed:.1f}s)"


def test_stop_drain_and_join_share_one_deadline(tmp_path, bound, fast_clock,
                                                monkeypatch):
    """The seams were tested in isolation and the adapter did not use
    them: both `container.drain()` calls were argument-free, so drain
    opened its OWN ten-second budget while `stop` and the joins shared
    another. Three cleanup steps, three budgets.

    This watches the values the adapter actually passes."""
    seen = {}
    real_stop = process_module.stop
    real_join = process_module.join_within
    real_drain = process_module.Container.drain

    def recording_stop(process, deadline):
        seen["stop"] = deadline
        return real_stop(process, deadline)

    def recording_join(threads, deadline):
        seen["join"] = deadline
        return real_join(threads, deadline)

    def recording_drain(self, deadline=None):
        seen.setdefault("drain", deadline)
        return real_drain(self, deadline)

    monkeypatch.setattr(process_module, "stop", recording_stop)
    monkeypatch.setattr(process_module, "join_within", recording_join)
    monkeypatch.setattr(process_module.Container, "drain", recording_drain)
    monkeypatch.setattr(execution, "stop", recording_stop)
    monkeypatch.setattr(execution, "join_within", recording_join)

    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu",
            budget_usd=1.0,
            timeout_seconds=MIN_TIMEOUT, max_output_bytes=65536)
    monkeypatch.undo()

    assert set(seen) == {"stop", "drain", "join"}, f"eksik dikis: {sorted(seen)}"
    assert seen["drain"] is not None, "drain kendi butcesini aciyor"
    assert seen["stop"] == seen["drain"] == seen["join"], \
        f"temizlik adimlari ayri deadline kullaniyor: {seen}"


def test_no_cleanup_seam_is_called_without_a_deadline():
    """Structural, because the behavioural test above can only watch the
    paths it happens to take."""
    import ast

    source = Path(execution.__file__).read_text(encoding="utf-8")
    bare = [node.lineno for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("container.drain")
            and not node.args]
    assert bare == [], f"deadline'siz drain cagrisi: {bare}"


def test_the_adapter_has_no_state_machine_of_its_own():
    """B2A launches one process. Advancing the run belongs to the runner
    that does not exist yet, and a second state machine here would be a
    second place the truth lives."""
    source = "".join(Path(module.__file__).read_text(encoding="utf-8")
                     for module in (execution, process_module))
    for forbidden in ("ALLOWED_TRANSITIONS", "advance(", "write_state",
                      "events.jsonl", "assert_transition"):
        assert forbidden not in source, f"durum makinesi izi: {forbidden}"


# =====================================================================
# R2A -- execution is BOUND to the recorded worktree
# =====================================================================
#
# `run_implementer` used to take any directory that existed --
# `is_dir()` was the whole check -- so the MAIN CHECKOUT was a perfectly
# acceptable working directory for the model, and B1's write-ahead
# ownership record sat unread at the one moment it was supposed to
# matter. Now the caller passes identities and the path is DERIVED; a
# caller cannot even express "run it over there".

def _record_file(state_dir, worktree_id):
    return Path(state_dir) / worktree.REGISTRY_DIRNAME / f"{worktree_id}.json"


def _rewrite_record(state_dir, worktree_id, **changes):
    """Forge or adjust an ownership record THROUGH the schema, so every
    forgery here is one `read_record` cannot tell from a real one."""
    record = worktree.read_record(state_dir, worktree_id)
    assert record is not None, "senaryo kurulmadi: kayit okunamadi"
    state.write_json_atomically(_record_file(state_dir, worktree_id),
                                dict(record, **changes),
                                worktree.RECORD_SCHEMA,
                                "calisma agaci kaydi")


def _rewrite_binding(state_dir, **changes):
    """Adjust the run's binding THROUGH the schema. Negative tests that
    hand in a non-bound id rewrite the binding to AGREE with the call,
    so the refusal they assert keeps coming from the one check they
    isolate rather than from the binding comparison in front of it."""
    state.write_binding(state_dir,
                        dict(state.read_binding(state_dir), **changes))


def _reply_binary(tmp_path):
    """A stub that would SUCCEED if it were ever launched, so a refusal
    can never be blamed on the binary."""
    return _fake_binary(tmp_path, mode="raw",
                        hex=json.dumps(_valid_reply()).encode().hex())


def _binding_refused(binary, identity, launched, sentinels):
    """The one shape every negative control must have: the typed
    refusal, zero processes, and an error text that carries no path, no
    repository name, no identity and no record content."""
    with pytest.raises(execution.WorktreeNotBound) as refusal:
        execution.run_implementer(
            binary, **identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert launched == [], "reddedilen bag yine de bir surec baslatti"
    text = str(refusal.value) + repr(refusal.value)
    assert "/" not in text and "\\" not in text, "ret metni yol tasiyor"
    for sentinel in sentinels:
        assert str(sentinel) not in text, "ret metni kimlik/yol tasiyor"
    assert refusal.value.reason == contract.StopReason.PREFLIGHT_FAILED
    assert refusal.value.event == contract.EventCode.PREFLIGHT_FAILED
    return refusal.value


def _sentinels(identity):
    return (identity["repo"], identity["worktree_id"],
            identity["baseline_sha"], "kurgu-depo")


def _plant_link(link, target):
    """A directory symlink, or an NTFS junction where symlinks need a
    privilege this test cannot assume. Proven to resolve, or the test
    fails as UNBUILT rather than passing around the scenario."""
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    assert Path(link).resolve() == Path(target).resolve(), \
        "senaryo kurulmadi: baglanti kurulamadi"


def test_the_signature_offers_no_path_parameter_at_all():
    """The fix is structural: there is nothing to validate because
    there is nothing to pass."""
    parameters = inspect.signature(execution.run_implementer).parameters
    assert set(parameters) == {
        "binary", "repo", "state_dir", "run_id", "worktree_id",
        # B2B-B1 bridge: the flat-workspace identity. An IDENTITY, not a
        # path -- the escape list below is unchanged and still empty.
        # `worktree_id` leaves when B2B-B2 removes the legacy branch.
        "workspace_id",
        "baseline_sha", "prompt", "budget_usd",
        "timeout_seconds", "max_output_bytes", "model"}
    for escape in ("worktree", "cwd", "path", "workdir", "working_dir",
                   "directory", "schema_path", "schema", "schema_json",
                   "schema_bytes", "schema_file"):
        assert escape not in parameters, f"kacis parametresi: {escape}"
    for name, parameter in parameters.items():
        if name != "binary":
            assert parameter.kind == inspect.Parameter.KEYWORD_ONLY, name


def test_the_model_runs_exactly_in_the_derived_recorded_worktree(
        tmp_path, bound):
    """POSITIVE CONTROL for the whole section: a real repository, a real
    B1 `create()`, a READY record and the correct identities -- and the
    child's own report of its cwd is the derived holder path. Without
    this, every refusal below could be a binding that refuses
    everything."""
    trace = tmp_path / "nerede.txt"
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex(),
                          cwd_record=str(trace))
    result = execution.run_implementer(
        binary, **bound.identity, prompt="kurgu",
        budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert result.reply["status"] == "implemented"
    derived = worktree.holder_for(bound.worktree_id) / \
        worktree.WORKTREE_DIRNAME
    observed = Path(trace.read_text(encoding="utf-8")).resolve()
    assert observed == derived.resolve() == bound.path.resolve()


def test_the_main_checkout_is_never_the_model_working_directory(
        tmp_path, private_worktree_root,
        only_fake_binaries_may_run):
    """THE R2 P0, rebuilt as far as the new design allows it to exist: a
    repository whose MAIN checkout sits exactly where the id derives to,
    with a READY record forged to match it. The directory is real, git
    lists it (as the main tree), HEAD is the recorded baseline -- and
    the main-checkout comparison still refuses."""
    state_dir = tmp_path / "durum"
    record = worktree.register(state_dir, repo=tmp_path, run_id=RUN,
                               baseline_sha="0" * 40)
    wid = record["worktree_id"]
    main = worktree.holder_for(wid) / worktree.WORKTREE_DIRNAME
    main.mkdir(parents=True)
    for argv in (["init", "-q"],
                 ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(main, *argv)
    (main / "kurgu.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "kurgu")
    baseline = _git(main, "rev-parse", "HEAD")
    _rewrite_record(state_dir, wid, repo_id=state.repo_identity(main),
                    baseline_sha=baseline, status=worktree.STATUS_READY)
    state.write_binding(state_dir, {
        "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
        "repo_id": state.repo_identity(main), "baseline_sha": baseline,
        "manifest_digest": "0" * 64, "worktree_id": wid})
    assert main.is_dir() and _git(main, "rev-parse", "HEAD") == baseline, \
        "senaryo kurulmadi"
    identity = {"repo": main, "state_dir": state_dir, "run_id": RUN,
                "worktree_id": wid, "baseline_sha": baseline}
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_the_repository_argument_may_never_be_the_execution_target(
        tmp_path, bound, only_fake_binaries_may_run):
    """The record's repo_id rewritten to name the WORKTREE itself, and
    `repo=` handed that same directory: git lists the tree, HEAD
    matches, the path derives correctly -- executing in the repository
    you answer to is what refuses."""
    _rewrite_record(bound.state_dir, bound.worktree_id,
                    repo_id=state.repo_identity(bound.path))
    _rewrite_binding(bound.state_dir,
                     repo_id=state.repo_identity(bound.path))
    identity = dict(bound.identity, repo=bound.path)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_random_existing_directory_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """A plain `mkdir` at the derived location, with a READY record
    forged around it: existing was the WHOLE former check, and existing
    is no longer enough -- git does not list it."""
    record = worktree.register(bound.state_dir, repo=bound.repo, run_id=RUN,
                               baseline_sha=bound.baseline_sha)
    wid = record["worktree_id"]
    stray = worktree.holder_for(wid) / worktree.WORKTREE_DIRNAME
    stray.mkdir(parents=True)
    _rewrite_record(bound.state_dir, wid, status=worktree.STATUS_READY)
    _rewrite_binding(bound.state_dir, worktree_id=wid)
    assert stray.is_dir(), "senaryo kurulmadi"
    identity = dict(bound.identity, worktree_id=wid)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_another_repositorys_worktree_record_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """Repository B's genuinely READY worktree, asked for by repository
    A. The record is real and complete -- it was just never issued to
    A. (Setup proves B's own binding is accepted, so the refusal below
    is about the repository and nothing else.)"""
    other = tmp_path / "baska-depo"
    other.mkdir()
    for argv in (["init", "-q"],
                 ["config", "user.email", "k@example.invalid"],
                 ["config", "user.name", "Kurgu"]):
        _git(other, *argv)
    (other / "kurgu.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(other, "add", "-A")
    _git(other, "commit", "-qm", "kurgu")
    other_baseline = _git(other, "rev-parse", "HEAD")
    other_state = tmp_path / "durum-b"
    path_b, id_b = worktree.create(other, state_dir=other_state,
                                   run_id=RUN, baseline_sha=other_baseline)
    state.write_binding(other_state, {
        "protocol_version": contract.PROTOCOL_VERSION, "run_id": RUN,
        "repo_id": state.repo_identity(other),
        "baseline_sha": other_baseline,
        "manifest_digest": "0" * 64, "worktree_id": id_b})
    assert worktree.assert_execution_binding(
        other, state_dir=other_state, run_id=RUN, worktree_id=id_b,
        baseline_sha=other_baseline) == path_b, "senaryo kurulmadi"
    identity = {"repo": bound.repo, "state_dir": other_state,
                "run_id": RUN, "worktree_id": id_b,
                "baseline_sha": other_baseline}
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_wrong_run_id_is_refused(tmp_path, bound,
                                   only_fake_binaries_may_run):
    identity = dict(bound.identity, run_id="baska-kosu-1")
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_record_issued_to_a_different_repository_identity_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """ONLY the repository identity is wrong: the tree is real,
    registered in the asking repository, at the right baseline -- but
    the ledger says it was issued to somebody else. The first version of
    this scenario asked with another repository and was refused by git's
    registry instead; a mutation run proved the identity check itself
    was never the thing refusing."""
    _rewrite_record(bound.state_dir, bound.worktree_id,
                    repo_id=state.repo_identity(tmp_path / "hayalet-depo"))
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_worktree_id_nobody_recorded_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    identity = dict(bound.identity, worktree_id="f" * 32)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_record_copied_under_another_id_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """A byte-identical record filed under a SECOND, genuinely created
    worktree's name: the directory that id derives to exists, git lists
    it, HEAD answers correctly -- and the record still NAMES the other
    worktree, which alone refuses. (The first version parked the copy
    under a fresh id with a bare directory, and git's registry refused
    before the name comparison ever mattered -- a mutation run caught
    it.)"""
    second_path, second_id = worktree.create(
        bound.repo, state_dir=bound.state_dir, run_id=RUN,
        baseline_sha=bound.baseline_sha)
    _record_file(bound.state_dir, second_id).write_bytes(
        _record_file(bound.state_dir, bound.worktree_id).read_bytes())
    # the run's binding AGREES with the call, so the one thing refusing
    # is the record's own name -- without this line the binding
    # comparison refused first and a mutation run showed the name check
    # was never the thing being tested
    _rewrite_binding(bound.state_dir, worktree_id=second_id)
    assert _git(second_path, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi"
    identity = dict(bound.identity, worktree_id=second_id)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_wrong_baseline_is_refused(tmp_path, bound,
                                     only_fake_binaries_may_run):
    identity = dict(bound.identity, baseline_sha="1" * 40)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_record_whose_baseline_was_rewritten_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """The tree IS at the caller's baseline; the RECORD is not. Without
    the record comparison the HEAD check happily agrees with the caller
    and a tampered ledger entry stops meaning anything -- which is
    exactly how the first mutation run found this scenario missing."""
    _rewrite_record(bound.state_dir, bound.worktree_id,
                    baseline_sha="2" * 40)
    assert _git(bound.path, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_planned_record_is_refused_even_with_the_tree_on_disk(
        tmp_path, bound, only_fake_binaries_may_run):
    """Wound back to PLANNED after a full create: the directory exists,
    git registers it, HEAD matches -- ONLY the status refuses. PLANNED
    means `create` never proved the tree it made, and an unproven tree
    is not a place to run a model."""
    _rewrite_record(bound.state_dir, bound.worktree_id,
                    status=worktree.STATUS_PLANNED)
    assert bound.path.is_dir(), "senaryo kurulmadi"
    assert _git(bound.path, "rev-parse", "HEAD") == bound.baseline_sha
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_missing_record_is_refused(tmp_path, bound,
                                     only_fake_binaries_may_run):
    _record_file(bound.state_dir, bound.worktree_id).unlink()
    assert bound.path.is_dir(), "senaryo kurulmadi: agac yerinde durmali"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_corrupt_record_is_refused(tmp_path, bound,
                                     only_fake_binaries_may_run):
    _record_file(bound.state_dir, bound.worktree_id).write_bytes(b"{bozuk")
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_copied_tree_that_git_does_not_register_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """A full copy of a real worktree -- `.git` link file included, HEAD
    answering correctly -- under its own READY record. Git's registry is
    what refuses: the bytes being right does not make it the tree git
    made."""
    record = worktree.register(bound.state_dir, repo=bound.repo, run_id=RUN,
                               baseline_sha=bound.baseline_sha)
    wid = record["worktree_id"]
    clone = worktree.holder_for(wid) / worktree.WORKTREE_DIRNAME
    shutil.copytree(bound.path, clone)
    _rewrite_record(bound.state_dir, wid, status=worktree.STATUS_READY)
    _rewrite_binding(bound.state_dir, worktree_id=wid)
    assert _git(clone, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi: kopya inandirici degil"
    identity = dict(bound.identity, worktree_id=wid)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_worktree_whose_head_moved_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """The tree is real and registered; its HEAD is simply not the
    recorded baseline any more. A model dropped into it would diff
    against history nobody froze."""
    _git(bound.path, "commit", "--allow-empty", "-qm", "kurgu-ilerledi")
    assert _git(bound.path, "rev-parse", "HEAD") != bound.baseline_sha, \
        "senaryo kurulmadi"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_failed_git_query_is_a_refusal_not_an_absence(
        tmp_path, bound, only_fake_binaries_may_run):
    """The `.git` directory renamed away: every git question now fails.
    A question that cannot be asked has no answer, and no answer binds
    nothing."""
    os.rename(bound.repo / ".git", bound.repo / "git-yok")
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_link_that_resolves_outside_the_holder_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """The worktree replaced by a link to ANOTHER registered worktree of
    the same repository: registration, HEAD and the main-checkout
    comparison all pass THROUGH the link, so path containment is the
    only thing standing -- the resolved location must be exactly where
    the runner-owned root derives it."""
    second_path, second_id = worktree.create(
        bound.repo, state_dir=bound.state_dir, run_id=RUN,
        baseline_sha=bound.baseline_sha)
    os.rename(bound.path, bound.path.parent / "wt-asil")
    _plant_link(bound.path, second_path)
    assert _git(bound.path, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi: baglanti uzerinden git cevap vermiyor"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


# ---------------------------------------------------------------------
# R2A-R1 -- the RUN'S OWN BINDING is part of the authority
# ---------------------------------------------------------------------
#
# The registry says a worktree EXISTS and who made it; `binding.json`
# says which SINGLE tree this run is pinned to. The first version of
# this seam read only the registry, so any READY sibling of the right
# repo/run/baseline was executable -- proven by an evaluator probe with
# two READY worktrees, where the one the binding did NOT name was
# accepted.

def test_a_second_ready_worktree_the_binding_does_not_name_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """THE evaluator probe, pinned: a second, genuinely READY worktree
    for the SAME repo, run and baseline. Its registry record is beyond
    reproach -- only the run's binding refuses it, because the run is
    bound to one tree, not to a family of them."""
    second_path, second_id = worktree.create(
        bound.repo, state_dir=bound.state_dir, run_id=RUN,
        baseline_sha=bound.baseline_sha)
    assert _git(second_path, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi"
    identity = dict(bound.identity, worktree_id=second_id)
    _binding_refused(_reply_binary(tmp_path), identity,
                     only_fake_binaries_may_run, _sentinels(identity))


def test_a_missing_run_binding_is_refused(tmp_path, bound,
                                          only_fake_binaries_may_run):
    """No binding, no execution -- the registry record alone was the
    whole former check, and it is not enough."""
    (bound.state_dir / state.BINDING_FILENAME).unlink()
    assert bound.path.is_dir(), "senaryo kurulmadi: agac yerinde durmali"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_corrupt_run_binding_is_refused(tmp_path, bound,
                                          only_fake_binaries_may_run):
    (bound.state_dir / state.BINDING_FILENAME).write_bytes(b"{bozuk")
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_binding_issued_to_a_different_repository_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """Only the binding's repository identity is wrong; the registry
    record agrees with the call completely."""
    _rewrite_binding(bound.state_dir,
                     repo_id=state.repo_identity(tmp_path / "hayalet-depo"))
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_binding_for_a_different_run_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    _rewrite_binding(bound.state_dir, run_id="baska-kosu-1")
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_binding_at_a_different_baseline_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """The tree IS at the caller's baseline; only the binding is not."""
    _rewrite_binding(bound.state_dir, baseline_sha="3" * 40)
    assert _git(bound.path, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def test_a_record_whose_run_was_rewritten_is_refused(
        tmp_path, bound, only_fake_binaries_may_run):
    """Only the REGISTRY record's run is wrong -- the binding agrees
    with the call, so the record comparison is the one thing refusing.
    (Isolates the record-side run check now that the binding carries a
    run comparison of its own.)"""
    _rewrite_record(bound.state_dir, bound.worktree_id,
                    run_id="baska-kosu-1")
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


def _case_sensitive_filesystem(tmp_path):
    probe = tmp_path / "kucuk-harf-sondasi.txt"
    probe.write_text("k", encoding="ascii")
    return not (tmp_path / "KUCUK-HARF-SONDASI.TXT").exists()


def test_a_case_twin_of_the_worktree_is_refused_on_case_sensitive_fs(
        tmp_path, bound, only_fake_binaries_may_run):
    """R2A-R1 P1 pinned end to end: a full copy of the real tree in a
    CASE-TWIN directory inside the same holder, with the real name
    replaced by a link to it. Under unconditional casefold the twin
    compared equal to the registered tree everywhere -- containment,
    registry, all of it -- and an unregistered directory executed. On a
    case-sensitive filesystem the twin is a DIFFERENT directory and the
    binding must say so."""
    if not _case_sensitive_filesystem(tmp_path):
        pytest.skip("dosya sistemi harfe duyarsiz; ikiz dizin kurulamiyor")
    twin = bound.path.parent / "WT"
    shutil.copytree(bound.path, twin)
    os.rename(bound.path, bound.path.parent / "wt-asil")
    _plant_link(bound.path, twin)
    assert _git(bound.path, "rev-parse", "HEAD") == bound.baseline_sha, \
        "senaryo kurulmadi: ikiz uzerinden git cevap vermiyor"
    _binding_refused(_reply_binary(tmp_path), bound.identity,
                     only_fake_binaries_may_run, _sentinels(bound.identity))


# =====================================================================
# R2B -- the argv schema, the validator and the hash are ONE thing
# =====================================================================
#
# The schema used to travel as a caller-chosen FILE PATH -- to a CLI
# flag that takes INLINE JSON -- while the validator used a separate
# live dictionary. A probe rewrote the schema file to garbage after the
# argv was built and the adapter accepted the reply without noticing:
# nothing tied what the model received to what judged its answer.

def test_the_argv_schema_is_inline_canonical_and_hashed(
        tmp_path, bound, only_fake_binaries_may_run):
    """POSITIVE CONTROL for the section, read off the argv the adapter
    ACTUALLY launched: exactly one `--json-schema`, inline canonical
    JSON right after it, equal to the frozen schema, not a path on any
    disk -- and its exact UTF-8 bytes hash to the `schema_sha256` the
    result reports."""
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"))
    argv = only_fake_binaries_may_run[0]
    assert argv.count("--json-schema") == 1
    token = argv[argv.index("--json-schema") + 1]
    assert json.loads(token) == schemas.IMPLEMENTER_RESULT_SCHEMA
    # os.path.exists, NOT Path.exists: on Linux a 3KB "filename" makes
    # pathlib raise ENAMETOOLONG instead of answering False, and the
    # question here is "is this a path", not "can stat swallow it"
    assert not os.path.exists(token), "sema hala bir dosya yolu"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert digest == result.schema_sha256 \
        == schemas.IMPLEMENTER_SCHEMA_BINDING.sha256
    assert re.fullmatch(r"[0-9a-f]{64}", result.schema_sha256)


def test_a_reply_is_judged_by_the_schema_extracted_from_the_argv(
        tmp_path, bound, only_fake_binaries_may_run):
    """Both directions of the same-schema claim: the accepted reply
    validates under the schema pulled OUT of the argv, and a reply that
    schema rejects is exactly what the adapter refuses -- no separate
    in-process schema gets a say."""
    good = _valid_reply()
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(good).encode("utf-8"))
    argv = only_fake_binaries_may_run[0]
    extracted = json.loads(argv[argv.index("--json-schema") + 1])
    Draft202012Validator(extracted).validate(result.reply)
    bad = dict(good, status="approved")            # implementer may not
    with pytest.raises(ValidationError):
        Draft202012Validator(extracted).validate(bad)
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, bound, json.dumps(bad).encode("utf-8"))


def test_a_builder_that_smuggles_a_different_schema_is_refused_before_launch(
        tmp_path, bound, only_fake_binaries_may_run, monkeypatch):
    """The deliberate divergence: the builder emits a PERMISSIVE inline
    schema instead of the frozen one. The hash comparison refuses it
    with zero processes started -- a schema nobody agreed to must not
    even cost a launch."""
    binary = _reply_binary(tmp_path)
    real_builder = cli.build_implementer_argv

    def smuggling(target, **kwargs):
        argv = real_builder(target, **kwargs)
        argv[argv.index("--json-schema") + 1] = "{}"
        return argv

    monkeypatch.setattr(cli, "build_implementer_argv", smuggling)
    with pytest.raises(execution.SchemaNotBound):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu", budget_usd=1.0,
            timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], \
        "sema uyusmazligina ragmen surec basladi"


def test_an_argv_without_exactly_one_inline_schema_is_refused(
        tmp_path, bound, only_fake_binaries_may_run, monkeypatch):
    binary = _reply_binary(tmp_path)
    real_builder = cli.build_implementer_argv

    def stripping(target, **kwargs):
        argv = real_builder(target, **kwargs)
        index = argv.index("--json-schema")
        return argv[:index] + argv[index + 2:]

    monkeypatch.setattr(cli, "build_implementer_argv", stripping)
    with pytest.raises(execution.SchemaNotBound):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu", budget_usd=1.0,
            timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


def test_a_lying_str_subclass_cannot_impersonate_the_schema(
        tmp_path, bound, only_fake_binaries_may_run, monkeypatch):
    """R2B-R1.1: `isinstance` accepts SUBCLASSES, and a subclass
    answers whatever it likes. This one carries `{}` as its real
    content while claiming equality with the canonical text and
    returning the canonical BYTES from `encode()` -- both guards
    passed, `json.loads` read the real `{}`, and a permissive validator
    accepted an arbitrary reply. An exact-type check is what closes it:
    every guard here asks the object a question, and only a real `str`
    cannot lie about the answer.

    Three assertions, because two of them alone would not notice a
    validator built from `{}`: the typed refusal, zero processes, and
    the arbitrary reply never being accepted."""
    binding = schemas.IMPLEMENTER_SCHEMA_BINDING

    class Taklitci(str):
        def __eq__(self, other):
            return other == binding.canonical_json or str.__eq__(self, other)

        def __ne__(self, other):
            return not self.__eq__(other)

        __hash__ = str.__hash__

        def encode(self, *args, **kwargs):
            return binding.canonical_bytes

    liar = Taklitci("{}")
    # the setup is only meaningful if the impersonation really works
    assert json.loads(str(liar)) == {}, "senaryo kurulmadi"
    assert liar == binding.canonical_json
    assert hashlib.sha256(liar.encode("utf-8")).hexdigest() == binding.sha256

    real_builder = cli.build_implementer_argv

    def poisoning(target, **kwargs):
        argv = real_builder(target, **kwargs)
        argv[argv.index("--json-schema") + 1] = liar
        return argv

    monkeypatch.setattr(cli, "build_implementer_argv", poisoning)
    # a reply NOTHING but an empty schema would accept
    binary = _fake_binary(
        tmp_path, mode="raw",
        hex=json.dumps({"tamamen": "keyfi"}).encode().hex())
    with pytest.raises(execution.SchemaNotBound):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu", budget_usd=1.0,
            timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], \
        "taklit semaya ragmen surec basladi"


@pytest.mark.parametrize(
    "poison", [b"{}", 123, None, "kurgu \ud800"],
    ids=["bayt", "tamsayi", "bos-deger", "vekil-karakter"])
def test_a_malformed_schema_token_is_the_same_typed_refusal(
        tmp_path, bound, only_fake_binaries_may_run, monkeypatch, poison):
    """R2B-R1: bytes, numbers, None and unencodable strings used to
    crash as AttributeError or UnicodeEncodeError -- no process
    started, but the contract says every schema divergence is
    `SchemaNotBound`, and an untyped crash is not that. Zero processes,
    same typed refusal, all four shapes."""
    binary = _reply_binary(tmp_path)
    real_builder = cli.build_implementer_argv

    def poisoning(target, **kwargs):
        argv = real_builder(target, **kwargs)
        argv[argv.index("--json-schema") + 1] = poison
        return argv

    monkeypatch.setattr(cli, "build_implementer_argv", poisoning)
    with pytest.raises(execution.SchemaNotBound):
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu", budget_usd=1.0,
            timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], \
        "bozuk sema degerine ragmen surec basladi"


def test_no_file_on_disk_can_influence_the_schema_any_more(
        tmp_path, bound, only_fake_binaries_may_run):
    """The old fixture location, recreated as garbage: the call neither
    reads it nor cares. There is no file in the schema story at all."""
    (tmp_path / "implementer.schema.json").write_text("{BOZUK",
                                                      encoding="utf-8")
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert result.reply["status"] == "implemented"
    assert result.schema_sha256 == schemas.IMPLEMENTER_SCHEMA_BINDING.sha256


# =====================================================================
# B2A -- the CALL BOUNDARY: what was checked is what is used
# =====================================================================
#
# Every input to `run_implementer` was validated in one representation
# and then used in another. A probe drove all of them: the budget was
# checked as 1.0 and reached argv as 999999, the binary was checked as
# A and launched as B, the prompt was checked as "kurgu" and a
# different payload went down stdin. Validation now RETURNS the
# canonical value and the raw arguments are deleted, so there is
# nothing left to diverge from.


class _IkiYuzluYol:
    """`__fspath__` names one program; `__str__` names another."""

    def __init__(self, denetlenen, calisan):
        self.denetlenen, self.calisan = denetlenen, calisan

    def __fspath__(self):
        return str(self.denetlenen)

    def __str__(self):
        return str(self.calisan)


def _marker_binary(tmp_path, name, marker):
    """A stub that RECORDS having run, so 'B never started' is observed
    rather than assumed."""
    return _fake_binary(tmp_path, name=name, mode="raw",
                        hex=json.dumps({"hangi": name}).encode().hex(),
                        cwd_record=str(marker))


def test_the_checked_binary_is_the_launched_binary(
        tmp_path, bound, only_fake_binaries_may_run):
    """The deception is NEUTRALISED rather than merely refused: the one
    conversion is `__fspath__`, so the program that was verified is the
    program that runs, and `__str__` is never consulted again."""
    izB = tmp_path / "B-calisti.txt"
    binary_a = _fake_binary(tmp_path, name="ikili_a", mode="raw",
                            hex=json.dumps(_valid_reply()).encode().hex())
    binary_b = _marker_binary(tmp_path, "ikili_b", izB)
    iki_yuzlu = _IkiYuzluYol(binary_a, binary_b)
    assert os.fspath(iki_yuzlu) != str(iki_yuzlu), "senaryo kurulmadi"

    result = execution.run_implementer(
        iki_yuzlu, **bound.identity, prompt="kurgu",
        budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert result.reply["status"] == "implemented"
    assert only_fake_binaries_may_run[0][0] == str(binary_a.resolve())
    time.sleep(0.5)
    assert not izB.exists(), "denetlenmeyen ikili calistirildi"


def test_a_relative_binary_is_resolved_before_the_worktree_becomes_cwd(
        tmp_path, bound, only_fake_binaries_may_run):
    """A relative path is checked against the CURRENT directory and
    then launched with the worktree as cwd -- two different meanings
    for one string. Resolution happens before the launch, so argv
    carries an absolute path."""
    binary = _fake_binary(tmp_path, name="goreli", mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    goreli = os.path.relpath(binary, os.getcwd())
    assert not os.path.isabs(goreli), "senaryo kurulmadi"

    result = execution.run_implementer(
        goreli, **bound.identity, prompt="kurgu", budget_usd=1.0,
        timeout_seconds=60, max_output_bytes=65536)
    assert result.reply["status"] == "implemented"
    launched = only_fake_binaries_may_run[0][0]
    assert os.path.isabs(launched), f"goreli yol firlatildi: {launched}"
    assert Path(launched) == binary.resolve()


def test_a_deceptive_tool_object_starts_no_process(
        tmp_path, bound, only_fake_binaries_may_run, monkeypatch):
    """The adapter's half of the allowlist finding: the refusal has to
    happen before anything is launched, not after."""
    class Taklitci:
        def __eq__(self, other):
            return other == "Read"

        def __hash__(self):
            return hash("Read")

        def __str__(self):
            return "Bash"

    real_builder = cli.build_implementer_argv

    def smuggling(target, **kwargs):
        return real_builder(target, **dict(kwargs, allowed_tools=[Taklitci()]))

    monkeypatch.setattr(cli, "build_implementer_argv", smuggling)
    with pytest.raises(cli.UnsafeInvocation):
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


@pytest.mark.parametrize(
    "budget",
    [101, 100.0001, 1000, 10 ** 400],
    ids=["tavan-ustu-int", "tavan-ustu-float", "cok-buyuk", "devasa-int"])
def test_a_budget_above_the_schema_maximum_is_refused(
        tmp_path, bound, only_fake_binaries_may_run, budget):
    """The task schema caps a run at 100 USD and the adapter enforced
    only "greater than zero" -- so a single call could be authorised
    for any amount. `10 ** 400` is here because an enormous exact
    integer must be bounded without `math.isfinite` raising."""
    with pytest.raises(execution.BudgetRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity, prompt="kurgu",
            budget_usd=budget, timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


def test_a_deceptive_budget_never_reaches_the_argv(
        tmp_path, bound, only_fake_binaries_may_run):
    """Checked as 1.0, stringified as 999999 -- an uncapped spend
    authorised by a gate that had already agreed to one dollar."""
    class YalanciButce(float):
        def __str__(self):
            return "999999"

    with pytest.raises(execution.BudgetRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity, prompt="kurgu",
            budget_usd=YalanciButce(1.0), timeout_seconds=60,
            max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


@pytest.mark.parametrize("budget", [100, 100.0, 0.375],
                         ids=["tavan-int", "tavan-float", "kesirli"])
def test_a_budget_at_or_below_the_maximum_reaches_the_argv_unchanged(
        tmp_path, bound, only_fake_binaries_may_run, budget):
    """POSITIVE CONTROL and the boundary in the other direction."""
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"),
                              budget_usd=budget)
    assert result.reply["status"] == "implemented"
    argv = only_fake_binaries_may_run[0]
    assert argv[argv.index("--max-budget-usd") + 1] == str(budget)


@pytest.mark.parametrize("field", ["timeout_seconds", "max_output_bytes"])
def test_a_deceptive_integer_bound_is_refused_before_any_process(
        tmp_path, bound, only_fake_binaries_may_run, field):
    """An `int` subclass whose comparisons always agree passed both
    range checks while carrying a value nine orders of magnitude
    outside them."""
    class YalanciSayi(int):
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    settings = {"timeout_seconds": 60, "max_output_bytes": 65536}
    settings[field] = YalanciSayi(10 ** 9)
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity, prompt="kurgu",
            budget_usd=1.0, **settings)
    assert only_fake_binaries_may_run == []


def test_the_exact_bounds_are_still_accepted_and_used(
        tmp_path, bound, only_fake_binaries_may_run):
    """Both edges of both frozen ranges, so the refusals above cannot
    become "reject everything"."""
    reply = json.dumps(_valid_reply(summary="k" * 1200)).encode("utf-8")
    result = _run_with_stdout(
        tmp_path, bound, reply,
        timeout_seconds=execution.MIN_TIMEOUT_SECONDS,
        max_output_bytes=execution.MAX_OUTPUT_BYTES)
    assert result.reply["role"] == "implementer"


def test_a_deceptive_prompt_never_reaches_stdin(
        tmp_path, bound, only_fake_binaries_may_run):
    """Checked as "kurgu", encoded into an entirely different
    instruction -- a reply to a question nobody asked."""
    class YalanciIstem(str):
        def encode(self, *args, **kwargs):
            return b"TAMAMEN BASKA BIR ISTEM"

    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity,
            prompt=YalanciIstem("kurgu"), budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


def test_the_validated_prompt_bytes_are_what_the_child_receives(
        tmp_path, bound):
    """The accepted direction, asserted on the bytes the CHILD counted
    rather than on the adapter's own bookkeeping."""
    kayit = tmp_path / "alinan.txt"
    istem = "K" * 5000
    binary = _fake_binary(
        tmp_path, name="yutan", mode="consume", record=str(kayit),
        hex=json.dumps(_valid_reply()).encode().hex())
    result = execution.run_implementer(
        binary, **bound.identity, prompt=istem, budget_usd=1.0,
        timeout_seconds=60, max_output_bytes=65536)
    assert result.reply["status"] == "implemented"
    assert kayit.read_text(encoding="ascii") == str(len(istem.encode()))


@pytest.mark.parametrize(
    "model", ["BUYUK-HARF", "-bastan-tire", 5, b"model"],
    ids=["buyuk-harf", "gecersiz-baslangic", "sayi", "bayt"])
def test_a_model_outside_the_frozen_schema_starts_no_process(
        tmp_path, bound, only_fake_binaries_may_run, model):
    """B2A-R1 STRENGTHENED this: it used to assert
    `cli.UnsafeInvocation`, which was the package pinning the very leak
    an audit then called a contract violation -- `run_implementer`
    promises a typed `AdapterError` and that is not one. The demand is
    now the promise, plus the same zero-process count."""
    with pytest.raises(execution.AdapterError) as refusal:
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536,
            model=model)
    assert not isinstance(refusal.value, cli.UnsafeInvocation)
    assert only_fake_binaries_may_run == []


def test_an_exact_model_string_reaches_the_argv(
        tmp_path, bound, only_fake_binaries_may_run):
    result = _run_with_stdout(tmp_path, bound,
                              json.dumps(_valid_reply()).encode("utf-8"),
                              model="kurgu-model-1")
    assert result.reply["status"] == "implemented"
    argv = only_fake_binaries_may_run[0]
    assert argv[argv.index("--model") + 1] == "kurgu-model-1"


def test_every_cli_refusal_reaches_the_caller_as_an_adapter_error(
        tmp_path, bound, only_fake_binaries_may_run):
    """B2A-R1: `run_implementer` promises a typed `AdapterError` for
    every refusal, and `cli.UnsafeInvocation` is not one -- an invalid
    model raised it straight through, which the runner's closed state
    machine would have had no reason code to record.

    The positive control matters here: the same call with a VALID model
    has to get PAST this gate, or the test would pass against an
    implementation that refuses everything. It dies later, at the
    schema/worktree stage, with an AdapterError of its own."""
    binary = _reply_binary(tmp_path)
    saglikli = execution.run_implementer(
        binary, **bound.identity, prompt="kurgu", budget_usd=1.0,
        timeout_seconds=60, max_output_bytes=65536, model="kurgu-model-1")
    assert saglikli.reply["status"] == "implemented", "senaryo kurulmadi"

    nobetci = "GECERSIZ MODEL!!"
    with pytest.raises(execution.AdapterError) as refusal:
        execution.run_implementer(
            binary, **bound.identity, prompt="kurgu", budget_usd=1.0,
            timeout_seconds=60, max_output_bytes=65536, model=nobetci)
    hata = refusal.value
    assert not isinstance(hata, cli.UnsafeInvocation)
    assert hata.reason in contract.ALL_STOP_REASONS
    assert hata.event in contract.ALL_EVENT_CODES
    assert nobetci not in str(hata) + repr(hata), \
        "ret metni reddedilen model adini tasiyor"


@pytest.mark.parametrize(
    "kotu", [0, -1, float("nan"), float("inf"), 101, 100.0001],
    ids=["sifir", "negatif", "nan", "sonsuz", "tavan-ustu-int",
         "tavan-ustu"])
def test_the_adapter_and_the_builder_refuse_the_same_budgets(
        tmp_path, bound, only_fake_binaries_may_run, kotu):
    """One authority, two roads: whatever the builder refuses, the
    adapter refuses -- as its own typed error, before any process."""
    with pytest.raises(cli.UnsafeInvocation):
        cli.build_implementer_argv("/kurgu/claude", budget_usd=kotu)
    with pytest.raises(execution.BudgetRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **bound.identity, prompt="kurgu",
            budget_usd=kotu, timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], \
        "reddedilen butceye ragmen surec basladi"


@pytest.mark.parametrize("field", ["run_id", "worktree_id", "baseline_sha"])
def test_a_deceptive_identity_never_reaches_the_worktree_binding(
        tmp_path, bound, only_fake_binaries_may_run, field):
    """A `str` subclass that claims equality with everything satisfied
    every comparison the binding makes. The identities are exact
    strings matching their existing patterns before the binding is
    even consulted."""
    class YalanciKimlik(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        __hash__ = str.__hash__

    identity = dict(bound.identity)
    identity[field] = YalanciKimlik(identity[field])
    with pytest.raises(execution.IdentityRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("run_id", "KURGU"), ("run_id", "ab"), ("worktree_id", "z" * 32),
     ("worktree_id", "a" * 31), ("baseline_sha", "A" * 40),
     ("baseline_sha", "a" * 39), ("run_id", 5), ("worktree_id", None)],
    ids=["buyuk-kosu", "kisa-kosu", "hex-disi", "kisa-id", "buyuk-sha",
         "kisa-sha", "sayi", "bos-deger"])
def test_an_identity_outside_its_frozen_pattern_is_refused(
        tmp_path, bound, only_fake_binaries_may_run, field, value):
    identity = dict(bound.identity)
    identity[field] = value
    with pytest.raises(execution.IdentityRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


def test_path_like_repo_and_state_dir_are_converted_once(
        tmp_path, bound, only_fake_binaries_may_run):
    """`__fspath__` is the conversion; a second, different `__str__`
    must not be able to redirect the binding afterwards."""
    identity = dict(bound.identity)
    identity["repo"] = _IkiYuzluYol(bound.repo, tmp_path / "olmayan-depo")
    identity["state_dir"] = _IkiYuzluYol(bound.state_dir,
                                         tmp_path / "olmayan-durum")
    result = execution.run_implementer(
        _reply_binary(tmp_path), **identity, prompt="kurgu", budget_usd=1.0,
        timeout_seconds=60, max_output_bytes=65536)
    assert result.reply["status"] == "implemented"


@pytest.mark.parametrize("field", ["repo", "state_dir"])
def test_a_path_argument_that_is_not_a_path_is_refused(
        tmp_path, bound, only_fake_binaries_may_run, field):
    identity = dict(bound.identity)
    identity[field] = 5
    with pytest.raises(execution.IdentityRefused):
        execution.run_implementer(
            _reply_binary(tmp_path), **identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=60, max_output_bytes=65536)
    assert only_fake_binaries_may_run == []


def test_every_launched_argv_token_is_an_exact_string(
        tmp_path, bound, only_fake_binaries_may_run):
    """Read off what was ACTUALLY launched: by the time argv exists
    there must be nothing left to convert."""
    _run_with_stdout(tmp_path, bound,
                     json.dumps(_valid_reply()).encode("utf-8"),
                     model="kurgu-model-1")
    argv = only_fake_binaries_may_run[0]
    offenders = [token for token in argv if type(token) is not str]
    assert offenders == [], f"tam metin olmayan argv ogesi: {offenders}"


def test_the_adapter_abandons_its_raw_arguments_after_canonicalisation():
    """Structural, and self-enforcing: the raw parameters are DELETED
    once the canonical call exists, so any later use of one is a
    NameError the positive-control tests hit immediately. A validator
    that leaves the original object alive is the whole defect."""
    import ast

    tree = ast.parse(Path(execution.__file__).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "run_implementer")
    silinen = {target.id for statement in function.body
               if isinstance(statement, ast.Delete)
               for target in statement.targets
               if isinstance(target, ast.Name)}
    beklenen = {"binary", "repo", "state_dir", "run_id", "worktree_id",
                "baseline_sha", "prompt", "budget_usd", "timeout_seconds",
                "max_output_bytes", "model"}
    assert beklenen <= silinen, f"ham argumanlar birakilmadi: " \
        f"{sorted(beklenen - silinen)}"


def test_the_validator_cannot_be_the_raw_module_dictionary():
    """Structural: no code path in the adapter touches the mutable
    schema dictionary -- validating against it is exactly the unbound
    state R2B removed, so its reappearance must be loud."""
    import ast

    tree = ast.parse(Path(execution.__file__).read_text(encoding="utf-8"))
    offenders = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute)
                 and node.attr == "IMPLEMENTER_RESULT_SCHEMA"]
    assert offenders == [], f"ham sema sozlugu kullanimda: {offenders}"
