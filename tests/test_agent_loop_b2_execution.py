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

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.agent_loop import cli, contract, execution, schemas
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


@pytest.fixture
def worktree_dir(tmp_path):
    """Stands in for the disposable worktree B1 would have created."""
    path = tmp_path / "kurgu-worktree"
    path.mkdir()
    return path


@pytest.fixture
def schema_file(tmp_path):
    path = tmp_path / "implementer.schema.json"
    path.write_text(json.dumps(schemas.IMPLEMENTER_RESULT_SCHEMA),
                    encoding="utf-8")
    return path


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
        tmp_path, worktree_dir, schema_file, only_fake_binaries_may_run):
    binary = _fake_binary(tmp_path, stdout_json=None,
                          mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    execution.run_implementer(
        binary, worktree=worktree_dir, prompt="kurgu istem",
        schema_path=schema_file, budget_usd=1.0, timeout_seconds=60,
        max_output_bytes=65536)

    expected = cli.build_implementer_argv(binary, schema_path=schema_file,
                                          budget_usd=1.0)
    assert only_fake_binaries_may_run[0] == expected, \
        "adapter cli.py'nin urettigi argv'yi degistirdi"


def test_the_process_runs_inside_the_supplied_worktree(
        tmp_path, worktree_dir, schema_file):
    binary = _fake_binary(tmp_path, mode="dump")
    with pytest.raises(execution.SchemaViolation):
        # `dump` emits a report, not a reply: the point here is WHERE it
        # ran, and the refusal proves the process really executed
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu istem",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=65536)

    report = _dump_report(tmp_path, worktree_dir, schema_file,
                          prompt="kurgu istem")
    assert Path(report["cwd"]).resolve() == worktree_dir.resolve()


def _dump_report(tmp_path, worktree_dir, schema_file, prompt):
    """Run the dump stub directly to read what the child observed."""
    binary = _fake_binary(tmp_path, name="dokum", mode="dump")
    argv = cli.build_implementer_argv(binary, schema_path=schema_file,
                                      budget_usd=1.0)
    done = subprocess.run(argv, cwd=str(worktree_dir), input=prompt,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_prompt_travels_only_on_stdin(tmp_path, worktree_dir,
                                          schema_file):
    """A prompt on a command line is a prompt in `ps` output, in a shell
    history and in any error that echoes the argv."""
    prompt = "KURGU-ISTEM-NOBETCISI-" + "q" * 10
    report = _dump_report(tmp_path, worktree_dir, schema_file, prompt=prompt)

    assert report["stdin"] == prompt, "istem stdin'e ulasmadi"
    assert not any(prompt in str(token) for token in report["argv"]), \
        "istem komut satirina sizdi"
    assert not any(prompt in value for value in report["env"].values()), \
        "istem ortam degiskenine sizdi"


def test_a_missing_binary_argument_is_a_type_error(worktree_dir, schema_file):
    with pytest.raises(TypeError):
        execution.run_implementer(                     # noqa: PLE1120
            worktree=worktree_dir, prompt="kurgu", schema_path=schema_file,
            budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)


@pytest.mark.parametrize("bare", ["claude", "codex", "git", "cmd"])
def test_a_bare_command_name_is_never_resolved_through_the_path(
        worktree_dir, schema_file, only_fake_binaries_may_run, bare):
    """These exist on PATH. The adapter still refuses them, because it
    executes a path and never searches for one."""
    with pytest.raises(execution.BinaryNotUsable):
        execution.run_implementer(
            bare, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0,
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

def _run_with_stdout(tmp_path, worktree_dir, schema_file, payload_bytes,
                     **kwargs):
    binary = _fake_binary(tmp_path, mode="raw", hex=payload_bytes.hex(),
                          **{k: v for k, v in kwargs.items()
                             if k in ("code", "stderr")})
    settings = {"worktree": worktree_dir, "prompt": "kurgu",
                "schema_path": schema_file, "budget_usd": 1.0,
                "timeout_seconds": 60, "max_output_bytes": 65536}
    settings.update({k: v for k, v in kwargs.items() if k in settings})
    return execution.run_implementer(binary, **settings)


def test_a_valid_reply_comes_back_as_structured_data(tmp_path, worktree_dir,
                                                     schema_file):
    reply = _valid_reply()
    result = _run_with_stdout(tmp_path, worktree_dir, schema_file,
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
def test_output_that_is_not_a_json_object_is_refused(tmp_path, worktree_dir,
                                                     schema_file, label,
                                                     payload):
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, worktree_dir, schema_file, payload)


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
def test_a_reply_outside_the_frozen_schema_is_refused(tmp_path, worktree_dir,
                                                      schema_file, label,
                                                      reply):
    """The schema is the contract. Nothing here repairs, coerces or
    guesses -- a model that answered outside the vocabulary is a model
    whose answer nobody can act on."""
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, worktree_dir, schema_file,
                         json.dumps(reply).encode("utf-8"))


def test_invalid_utf8_is_refused(tmp_path, worktree_dir, schema_file):
    """Decoded strictly. A replacement character would turn undecodable
    bytes into a string that might then parse as something."""
    with pytest.raises(execution.SchemaViolation):
        _run_with_stdout(tmp_path, worktree_dir, schema_file,
                         b'{"role": "\xff\xfe implementer"}')


def test_a_nonzero_exit_is_a_typed_failure_without_the_stderr(
        tmp_path, worktree_dir, schema_file):
    sentinel = "KURGU-STDERR-NOBETCISI-" + "w" * 8
    with pytest.raises(execution.ProcessFailed) as refusal:
        _run_with_stdout(tmp_path, worktree_dir, schema_file,
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

def test_a_flood_on_stdout_stops_the_process(tmp_path, worktree_dir,
                                             schema_file):
    limit = 8192
    binary = _fake_binary(tmp_path, mode="flood", stream="out",
                          bytes=limit * 40)
    started = time.monotonic()
    with pytest.raises(execution.OutputLimitExceeded) as refusal:
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=120,
            max_output_bytes=limit)
    assert refusal.value.stream == "stdout"
    assert refusal.value.stdout_bytes <= limit + process_module.READ_CHUNK_BYTES
    assert time.monotonic() - started < 100, "sinir asimi beklenmedi"


def test_a_flood_on_stderr_is_bounded_too(tmp_path, worktree_dir,
                                          schema_file):
    limit = 8192
    binary = _fake_binary(tmp_path, mode="flood", stream="err",
                          bytes=limit * 40)
    with pytest.raises(execution.OutputLimitExceeded) as refusal:
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=120,
            max_output_bytes=limit)
    assert refusal.value.stream == "stderr"


def test_output_of_exactly_the_limit_is_accepted(tmp_path, worktree_dir,
                                                 schema_file):
    """The boundary in the other direction, or the rule above is just
    "refuse large replies"."""
    # padded past the contract's own floor: `max_output_bytes` below
    # 1024 is outside the frozen range and is refused before launch
    padded = _valid_reply(summary="k" * 1200)
    reply = json.dumps(padded).encode("utf-8")
    result = _run_with_stdout(tmp_path, worktree_dir, schema_file, reply,
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


def test_a_timeout_kills_the_process_and_its_children(tmp_path, worktree_dir,
                                                      schema_file, fast_clock):
    """A model process that spawned something must not leave it running.
    The child writes a heartbeat; after the kill the heartbeat has to
    stop advancing, which is observable without a process library."""
    beat = tmp_path / "kalp.txt"
    binary = _fake_binary(tmp_path, mode="spawn", seconds=120,
                          child_code=_CHILD.format(beat=str(beat)))
    with pytest.raises(execution.Timeout) as refusal:
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
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
                                                           worktree_dir,
                                                           schema_file, fast_clock):
    binary = _fake_binary(tmp_path, mode="sleep", seconds=120)
    started = time.monotonic()
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert time.monotonic() - started < 60


def test_a_cleanup_failure_does_not_replace_the_primary_failure(
        tmp_path, worktree_dir, schema_file, fast_clock, monkeypatch):
    """The caller needs to know the run timed out. A kill that then
    fails is a second problem, not the headline."""
    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)

    def refusing_kill(*args, **kwargs):
        raise OSError("kurgu: surec agaci oldurulemedi")

    monkeypatch.setattr(process_module, "terminate_tree", refusing_kill)
    with pytest.raises(execution.Timeout):
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)


# =====================================================================
# BUDGET: refused before anything starts
# =====================================================================

@pytest.mark.parametrize(
    "budget", [0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True],
    ids=["sifir-int", "sifir", "negatif", "nan", "sonsuz", "eksi-sonsuz",
         "bool"])
def test_an_unusable_budget_refuses_before_any_process_starts(
        tmp_path, worktree_dir, schema_file, only_fake_binaries_may_run,
        budget):
    """Counted, not assumed: the refusal is only worth anything if no
    process was started, and "we would have refused later" is what a
    paid call looks like in hindsight."""
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    with pytest.raises(execution.BudgetRefused):
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=budget, timeout_seconds=60,
            max_output_bytes=65536)
    assert only_fake_binaries_may_run == [], \
        "butce reddedildikten sonra yine de bir surec baslatildi"


def test_the_exact_remaining_budget_reaches_the_argv(tmp_path, worktree_dir,
                                                     schema_file,
                                                     only_fake_binaries_may_run):
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    execution.run_implementer(
        binary, worktree=worktree_dir, prompt="kurgu",
        schema_path=schema_file, budget_usd=0.375, timeout_seconds=60,
        max_output_bytes=65536)
    argv = only_fake_binaries_may_run[0]
    assert "0.375" in argv[argv.index("--max-budget-usd") + 1]


def test_the_adapter_never_claims_to_know_what_was_spent(tmp_path,
                                                         worktree_dir,
                                                         schema_file):
    """`IMPLEMENTER_RESULT_SCHEMA` carries no cost field, so there is no
    contractual value to report. Reporting zero would be a number the
    caller could subtract from a budget."""
    result = _run_with_stdout(tmp_path, worktree_dir, schema_file,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert not hasattr(result, "spent_usd")
    assert not hasattr(result, "cost_usd")
    assert "spent" not in {field for field in vars(result)}


# =====================================================================
# WHAT MAY LEAVE THE ADAPTER
# =====================================================================

def test_no_raw_model_output_survives_in_the_result(tmp_path, worktree_dir,
                                                    schema_file):
    """Byte counts and codes leave; text does not. Raw stdout in a
    result object is raw stdout in whatever log the caller writes."""
    sentinel = "KURGU-CIKTI-NOBETCISI-" + "v" * 8
    reply = _valid_reply(summary=sentinel)
    result = _run_with_stdout(tmp_path, worktree_dir, schema_file,
                              json.dumps(reply).encode("utf-8"))
    # the PARSED reply legitimately holds it; nothing else may
    leaked = [name for name, value in vars(result).items()
              if name != "reply" and sentinel in str(value)]
    assert leaked == [], f"ham cikti tasiyan alan: {leaked}"
    assert not any(isinstance(value, (bytes, bytearray))
                   for value in vars(result).values())


def test_the_adapter_writes_no_files_of_its_own(tmp_path, worktree_dir,
                                                schema_file):
    before = {p for p in worktree_dir.rglob("*")}
    _run_with_stdout(tmp_path, worktree_dir, schema_file,
                     json.dumps(_valid_reply()).encode("utf-8"))
    assert {p for p in worktree_dir.rglob("*")} == before, \
        "adapter calisma agacina dosya yazdi"


def test_only_closed_codes_and_numbers_leave_the_adapter(tmp_path,
                                                         worktree_dir,
                                                         schema_file):
    result = _run_with_stdout(tmp_path, worktree_dir, schema_file,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert result.event in contract.ALL_EVENT_CODES
    for name, value in vars(result).items():
        if name == "reply":
            continue
        assert isinstance(value, (int, str)), f"{name} sayisal/kod degil"
        if isinstance(value, str):
            assert value in contract.ALL_EVENT_CODES, \
                f"{name} kapali sozluk disinda bir metin tasiyor"


# =====================================================================
# R1 -- what the first round of tests did not cover
# =====================================================================

def test_a_child_that_never_reads_stdin_still_times_out(
        tmp_path, worktree_dir, schema_file, fast_clock):
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
            binary, worktree=worktree_dir, prompt=prompt,
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
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
        tmp_path, worktree_dir, schema_file, only_fake_binaries_may_run,
        label, prompt):
    binary = _fake_binary(tmp_path, mode="raw",
                          hex=json.dumps(_valid_reply()).encode().hex())
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt=prompt,
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=60,
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
        tmp_path, worktree_dir, schema_file, only_fake_binaries_may_run,
        label, timeout_seconds):
    """`timeout_seconds=NaN` made every deadline comparison false, so a
    hung child ran until something outside stopped it. A bound that
    cannot be compared is not a bound."""
    binary = _fake_binary(tmp_path, mode="sleep", seconds=30)
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0,
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
        tmp_path, worktree_dir, schema_file, only_fake_binaries_may_run,
        label, max_output_bytes):
    """`max_output_bytes=NaN` read a megabyte in full and then refused it
    as bad JSON -- a parse error standing in for a ceiling that never
    applied, which is a refusal for the wrong reason after the cost was
    already paid."""
    binary = _fake_binary(tmp_path, mode="flood", stream="out",
                          bytes=1024 * 1024)
    with pytest.raises(execution.LimitRefused):
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=60,
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
        tmp_path, worktree_dir, schema_file, fast_clock, monkeypatch):
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
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert time.monotonic() - started < 45
    assert refusal.value.cleanup_complete is True, \
        "yonlendirilmis ortamda temizlik tamamlanamadi"


def test_a_failed_cleanup_is_reported_and_does_not_hide_the_timeout(
        tmp_path, worktree_dir, schema_file, fast_clock, monkeypatch):
    """Both halves: the caller still hears "timed out", and the fact
    that a process may still be running does not vanish into an
    `except`."""
    binary = _fake_binary(tmp_path, mode="sleep", seconds=60)

    def refusing_kill(*args, **kwargs):
        raise OSError("kurgu: surec agaci oldurulemedi")

    monkeypatch.setattr(process_module, "terminate_tree", refusing_kill)
    with pytest.raises(execution.Timeout) as refusal:
        execution.run_implementer(
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
            max_output_bytes=65536)
    assert refusal.value.cleanup_complete is False, \
        "basarisiz temizlik gorunmez kaldi"
    assert refusal.value.reason == contract.StopReason.TIMEOUT


def test_a_reply_produced_without_reading_the_prompt_is_refused(
        tmp_path, worktree_dir, schema_file):
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
            binary, worktree=worktree_dir, prompt="P" * (2 * 1024 * 1024),
            schema_path=schema_file, budget_usd=1.0, timeout_seconds=60,
            max_output_bytes=65536)
    assert refusal.value.reason == contract.StopReason.INTERRUPTED
    assert refusal.value.exit_code == 0, \
        "surec basariyla bitti; reddedilen sey teslimat"


def test_a_large_prompt_arrives_whole(tmp_path, worktree_dir, schema_file):
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
    report = _dump_report(tmp_path, worktree_dir, schema_file, prompt=prompt)
    assert len(report["stdin"]) == len(prompt), \
        f"istem kirpildi: {len(report['stdin'])} / {len(prompt)} bayt"
    assert report["stdin"] == prompt


def test_a_delivered_prompt_is_still_accepted(tmp_path, worktree_dir,
                                              schema_file):
    """The boundary: the refusal above must not become "reject every
    reply". This child reads its stdin to the end."""
    report = _dump_report(tmp_path, worktree_dir, schema_file,
                          prompt="kurgu istem")
    assert report["stdin"] == "kurgu istem"
    result = _run_with_stdout(tmp_path, worktree_dir, schema_file,
                              json.dumps(_valid_reply()).encode("utf-8"))
    assert result.reply["status"] == "implemented"


def test_a_failed_process_reports_the_contract_code_for_it(tmp_path,
                                                           worktree_dir,
                                                           schema_file):
    """Two wrong answers preceded this one: `preflight_failed` named a
    gate that had already succeeded, and `None` left a terminal result
    with no closed reason at all. The contract owner added
    `model_process_failed`."""
    with pytest.raises(execution.ProcessFailed) as refusal:
        _run_with_stdout(tmp_path, worktree_dir, schema_file, b"", code=9)
    failure = refusal.value
    assert failure.reason == contract.StopReason.MODEL_PROCESS_FAILED
    assert failure.reason in contract.ALL_STOP_REASONS
    assert failure.reason != contract.StopReason.PREFLIGHT_FAILED
    assert failure.event == contract.EventCode.MODEL_CALL_FINISHED
    assert failure.exit_code == 9


def test_a_grandchild_never_outlives_a_SUCCESSFUL_call(tmp_path, worktree_dir,
                                                       schema_file):
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
        binary, worktree=worktree_dir, prompt="kurgu",
        schema_path=schema_file, budget_usd=1.0, timeout_seconds=MIN_TIMEOUT,
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


def test_the_cleanup_budget_covers_the_tree_kill_too(tmp_path, worktree_dir,
                                                     schema_file, fast_clock,
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
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0,
            timeout_seconds=MIN_TIMEOUT, max_output_bytes=65536)
    elapsed = time.monotonic() - started
    assert seen["deadline"] is not None, "kill ortak butceyi almiyor"
    assert elapsed < process_module.REAP_SECONDS + 5, \
        f"temizlik ortak butceyi asti ({elapsed:.1f}s)"


def test_stop_drain_and_join_share_one_deadline(tmp_path, worktree_dir,
                                                schema_file, fast_clock,
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
            binary, worktree=worktree_dir, prompt="kurgu",
            schema_path=schema_file, budget_usd=1.0,
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
