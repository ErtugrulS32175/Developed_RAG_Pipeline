"""PACKAGE B2A / R1.3 -- the containment lifecycle.

The transport half of the adapter, tested where it lives. Everything
here is about one question: can a process this package started outlive
the call, on ANY path -- normal, timed out, or interrupted by an
exception nobody planned for.

NO REAL MODEL IS CALLED. Every program is a stub written into
`tmp_path`, and an autouse guard stops anything this file started and
fails the test if something survived it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from tools.agent_loop import (contract, execution, flat_workspace,
                              state)
from tools.agent_loop import process as process_module

# The grandchild: writes a heartbeat immediately, then keeps writing.
# "Still advancing" is a liveness answer that needs no process library.
_CHILD = ("import pathlib, time\n"
          "p = pathlib.Path({beat!r})\n"
          "while True:\n"
          "    p.write_text(str(time.time()))\n"
          "    time.sleep(0.05)\n")

_STUB = '''\
import json, subprocess, sys, time
from pathlib import Path


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if cfg.get("child_code"):
        subprocess.Popen([sys.executable, "-c", cfg["child_code"]])
        if cfg.get("wait_for"):
            limit = time.monotonic() + 20
            while not Path(cfg["wait_for"]).exists():
                if time.monotonic() > limit:
                    sys.exit(97)
                time.sleep(0.02)
    if cfg.get("stdout"):
        sys.stdout.write(cfg["stdout"])
        sys.stdout.flush()
    time.sleep(cfg.get("seconds", 0))
    sys.exit(0)


main()
'''


def _envelope(**overrides):
    """The RESULT ENVELOPE the real CLI was measured to return (B4-R3),
    as the JSON TEXT a stub writes to stdout.

    `claude --print --output-format json` never answers with a bare
    implementer payload, so a stub that printed one would be proving
    containment against a protocol no binary speaks."""
    # NO `next_action` (B7-R1): the transport no longer asks for it and
    # the adapter derives it from `status`.
    payload = {
        "protocol_version": "1.0", "run_id": "kurgu-run-1",
        "role": "implementer", "status": "implemented",
        "summary": "kurgu"}
    payload.update(overrides)
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "terminal_reason": "completed", "stop_reason": "tool_use",
        "num_turns": 2, "duration_ms": 7523, "total_cost_usd": 0.056757,
        "session_id": "00000000-0000-4000-8000-000000000000",
        "usage": {"input_tokens": 4, "output_tokens": 5},
        "result": json.dumps(payload), "structured_output": payload})


def _stub(tmp_path, name="sahte", **config):
    holder = tmp_path / "sahte-bin"
    holder.mkdir(exist_ok=True)
    helper = holder / "yardimci.py"
    helper.write_text(_STUB, encoding="utf-8")
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


@pytest.fixture(autouse=True)
def no_process_outlives_a_test(monkeypatch):
    """This file starts real processes on purpose. None may survive."""
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
            return subprocess.run(argv, **kwargs)

        @staticmethod
        def Popen(argv, **kwargs):                     # noqa: N802 -- stdlib
            process = real_popen(argv, **kwargs)
            started.append(process)
            return process

    monkeypatch.setattr(process_module, "subprocess", _Recorder)
    yield started
    for process in started:
        if process.poll() is None:
            _kill_for_real(process)
    alive = [process.pid for process in started if process.poll() is None]
    assert alive == [], f"testten sonra yasayan surec: {alive}"


def _kill_for_real(process):
    try:
        if os.name == "nt":
            root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            subprocess.run([str(root / "System32" / "taskkill.exe"),
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
def cwd_dir(tmp_path):
    """A PLAIN directory to launch into. It was called `cwd_dir`
    and never was one: these tests are about containment, and what they
    need is a working directory that exists."""
    path = tmp_path / "kurgu-cwd"
    path.mkdir()
    return path


@pytest.fixture
def bound_identity(tmp_path):
    """R2A MECHANICAL UPDATE, carried to B2B-B2C: `run_implementer` does
    not take a path, so the lifecycle tests that reach it hand over the
    identities of a REAL recorded D3A flat workspace -- inside a private
    temp root, so nothing here touches the machine-wide holder
    directory. No containment assertion below changed; only the thing
    that supplies a bound working directory did."""
    private = tmp_path / "runner-temp"
    private.mkdir()
    isolation = pytest.MonkeyPatch()
    isolation.setattr(tempfile, "tempdir", str(private))
    for variable in ("TMPDIR", "TEMP", "TMP"):
        isolation.setenv(variable, str(private))

    def git(*args):
        done = subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True,
                              errors="replace")
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    repo = tmp_path / "kurgu-depo"
    repo.mkdir()
    git("init", "-q")
    git("config", "user.email", "k@example.invalid")
    git("config", "user.name", "Kurgu")
    (repo / "kurgu.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "kurgu")
    baseline = git("rev-parse", "HEAD")
    state_dir = tmp_path / "durum"
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace = flat_workspace.create(repo, state_dir=state_dir,
                                      run_id="kurgu-run-1",
                                      baseline_sha=baseline)
    state.write_binding(state_dir, {
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": "kurgu-run-1", "repo_id": state.repo_identity(repo),
        "baseline_sha": baseline, "manifest_digest": "0" * 64,
        "workspace_id": workspace.workspace_id})
    yield {"repo": repo, "state_dir": state_dir, "run_id": "kurgu-run-1",
           "workspace_id": workspace.workspace_id, "baseline_sha": baseline}
    try:
        flat_workspace.remove(repo, state_dir=state_dir,
                              workspace_id=workspace.workspace_id)
    except Exception:                                  # noqa: BLE001
        pass
    isolation.undo()


def _still_writing(beat, seconds=1.5):
    first = beat.read_text(encoding="utf-8")
    time.sleep(seconds)
    return beat.read_text(encoding="utf-8") != first


# =====================================================================
# THE CONTAINER EXISTS BEFORE THE PROGRAM RUNS
# =====================================================================

def test_the_program_is_frozen_until_it_has_been_contained(tmp_path,
                                                           cwd_dir,
                                                           monkeypatch):
    """The race, closed by construction rather than narrowed.

    Containing a process AFTER creating it leaves a window in which the
    child can spawn something that lands outside the container. On
    Windows the child is created suspended, so at the moment the Job
    Object is attached it has not executed a single instruction. This
    asserts the ordering by refusing containment: if the program had
    already been running, its marker would exist."""
    if os.name != "nt":
        pytest.skip("bu yol Windows'a ozgu; POSIX'te setsid exec'ten once kosar")
    marker = tmp_path / "calisti.txt"
    stub = _stub(tmp_path, seconds=0,
                 child_code=f"import pathlib; pathlib.Path({str(marker)!r})"
                            ".write_text('x')")

    def refusing_attach(pid):
        raise process_module.ContainmentError("kurgu: is nesnesi kurulamadi")

    monkeypatch.setattr(process_module, "_attach_job", refusing_attach)
    with pytest.raises(process_module.ContainmentError):
        process_module.launch_contained([str(stub)], cwd=cwd_dir)

    time.sleep(1.0)
    assert not marker.exists(), \
        "kapsayici kurulamadi ama program yine de calisti"


def test_a_container_that_cannot_be_built_stops_the_call_before_the_model(
        tmp_path, bound_identity, monkeypatch):
    """Refused BEFORE the prompt is sent. Discovering it afterwards
    means the invoice already exists -- the previous version launched
    anyway and only noticed at the end."""
    if os.name != "nt":
        pytest.skip("kapsayici kurulum hatasi Windows yoluna ozgu")
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    marker = tmp_path / "model-calisti.txt"
    stub = _stub(tmp_path, seconds=0,
                 child_code=f"import pathlib; pathlib.Path({str(marker)!r})"
                            ".write_text('x')")

    def refusing_attach(pid):
        raise process_module.ContainmentError("kurgu: is nesnesi kurulamadi")

    monkeypatch.setattr(process_module, "_attach_job", refusing_attach)
    with pytest.raises(execution.ContainmentFailed) as refusal:
        execution.run_implementer(
            stub, **bound_identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=execution.MIN_TIMEOUT_SECONDS,
            max_output_bytes=65536)
    assert refusal.value.reason in ("model_process_failed",)
    time.sleep(1.0)
    assert not marker.exists(), "kapsayicisiz model calistirildi"


# =====================================================================
# EVERY EXIT PATH DRAINS
# =====================================================================

def test_an_exception_after_containment_still_empties_the_container(
        tmp_path, bound_identity, monkeypatch):
    """THE P0. The lifecycle after the launch was not wrapped, so an
    unexpected failure -- a thread that would not start, an interrupt --
    skipped the drain entirely and left the model process running with
    its container built and never emptied."""
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    beat = tmp_path / "kalp.txt"
    stub = _stub(tmp_path, seconds=120, child_code=_CHILD.format(beat=str(beat)),
                 wait_for=str(beat))

    real_start = process_module.threading.Thread.start

    def exploding(self):
        raise RuntimeError("kurgu: is parcacigi baslatilamadi")

    monkeypatch.setattr(process_module.threading.Thread, "start", exploding)
    with pytest.raises(RuntimeError, match="kurgu"):
        execution.run_implementer(
            stub, **bound_identity, prompt="kurgu",
            budget_usd=1.0, timeout_seconds=execution.MIN_TIMEOUT_SECONDS,
            max_output_bytes=65536)
    monkeypatch.setattr(process_module.threading.Thread, "start", real_start)

    # the grandchild never got to run here, but the PARENT certainly did
    time.sleep(0.8)
    assert not beat.exists() or not _still_writing(beat, 1.0), \
        "istisna sonrasi kapsayici bosaltilmadi"


def test_a_grandchild_does_not_survive_a_successful_call(tmp_path,
                                                         bound_identity):
    """Success is not an exemption: a bounded call that leaves an
    unbounded descendant behind was not bounded."""
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    beat = tmp_path / "kalp.txt"
    reply = _envelope()
    stub = _stub(tmp_path, seconds=0, stdout=reply, wait_for=str(beat),
                 child_code=_CHILD.format(beat=str(beat)))

    result = execution.run_implementer(
        stub, **bound_identity, prompt="kurgu",
        budget_usd=1.0, timeout_seconds=execution.MIN_TIMEOUT_SECONDS,
        max_output_bytes=65536)
    assert result.reply["status"] == "implemented"
    assert beat.exists(), "senaryo kurulmadi: torun hic calismadi"
    assert not _still_writing(beat), "basarili cagridan sonra torun yasiyor"


def test_a_grandchild_holding_the_pipes_does_not_cost_the_grace_period(
        tmp_path, bound_identity):
    """The drain used to happen AFTER the thread joins, so a descendant
    holding stdout open kept the readers alive and the adapter waited
    out the whole grace window before killing anything."""
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    beat = tmp_path / "kalp.txt"
    reply = _envelope()
    stub = _stub(tmp_path, seconds=0, stdout=reply, wait_for=str(beat),
                 child_code=_CHILD.format(beat=str(beat)))

    started = time.monotonic()
    execution.run_implementer(
        stub, **bound_identity, prompt="kurgu",
        budget_usd=1.0, timeout_seconds=execution.MIN_TIMEOUT_SECONDS,
        max_output_bytes=65536)
    elapsed = time.monotonic() - started
    assert elapsed < process_module.REAP_SECONDS, \
        f"torun pipe'lari tutarken tum grace suresi harcandi ({elapsed:.1f}s)"


# =====================================================================
# THE POSIX GROUP IS CAPTURED, NOT RE-DERIVED
# =====================================================================

def test_the_process_group_is_captured_once_at_launch():
    """`getpgid` on a pid that has already been reaped can fail, and a
    failed lookup reads as "the group is empty" -- while a descendant in
    that same group is still running. The id is taken once, while the
    process is certainly alive, and never asked for again."""
    source = Path(process_module.__file__).read_text(encoding="utf-8")
    group = source[source.index("    def _drain_group(self, deadline):"):
                   source.index("    def _drain_job(self, deadline):")]
    assert "getpgid" not in group, \
        "bosluk denetimi pgid'yi olu pid'den yeniden turetiyor"
    assert "self._pgid" in group
    assert "Container(process, pgid=process.pid)" in source


# =====================================================================
# R1.4 -- the Win32 layer answers for itself
# =====================================================================

_ERROR_NO_MORE_FILES = 18

_WIN32_USED = ("CreateJobObjectW", "SetInformationJobObject", "OpenProcess",
               "AssignProcessToJobObject", "IsProcessInJob",
               "CreateToolhelp32Snapshot", "Thread32First", "Thread32Next",
               "OpenThread", "ResumeThread", "TerminateJobObject",
               "QueryInformationJobObject", "CloseHandle")


@pytest.mark.skipif(os.name != "nt", reason="Win32 ABI'si Windows'a ozgu")
def test_every_win32_function_has_a_declared_abi():
    """Left to ctypes' defaults a function returns `c_int` and accepts
    anything, which truncates a 64-bit HANDLE to 32 bits and hands the
    rest of the module a number that names nothing. That failure is
    silent -- it looks like a handle that refers to a closed object."""
    import ctypes

    kernel32 = process_module._kernel32()             # noqa: SLF001
    signatures = process_module._SIGNATURES           # noqa: SLF001
    missing = [name for name in _WIN32_USED if name not in signatures]
    assert missing == [], f"ABI bildirimi olmayan islev: {missing}"
    for name in _WIN32_USED:
        function = getattr(kernel32, name)
        declared_args, declared_result = signatures[name]
        # compared against the DECLARATION, not against a type: on
        # Windows `wintypes.BOOL` IS `c_int`, so "not the default type"
        # cannot tell a declared BOOL from a function nobody bound
        assert list(function.argtypes) == list(declared_args), \
            f"{name}: argtypes bildirimle uyusmuyor"
        assert function.restype is declared_result, \
            f"{name}: restype bildirimle uyusmuyor"
    for name in ("CreateJobObjectW", "OpenProcess", "OpenThread",
                 "CreateToolhelp32Snapshot"):
        assert getattr(kernel32, name).restype is ctypes.c_void_p, \
            f"{name}: HANDLE isaretci genisliginde degil"
    assert process_module.INVALID_HANDLE_VALUE == (1 << (8 * ctypes.sizeof(
        ctypes.c_void_p))) - 1, "INVALID_HANDLE_VALUE isaretci genisliginde degil"


@pytest.mark.skipif(os.name != "nt", reason="Win32 ABI'si Windows'a ozgu")
def test_the_binding_point_is_the_only_route_to_kernel32():
    source = Path(process_module.__file__).read_text(encoding="utf-8")
    assert "ctypes.windll" not in source, \
        "kernel32'ye merkezi bagdan gecmeyen erisim var"
    assert "use_last_error=True" in source


class _FakeKernel:
    """Drives the return codes the real library only produces on a bad
    day. Every value here is one Microsoft documents."""

    def __init__(self, real, **overrides):
        self._real = real
        self._overrides = overrides
        self.closed = []

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real, name)


@pytest.mark.skipif(os.name != "nt", reason="resume yolu Windows'a ozgu")
@pytest.mark.parametrize(
    ("label", "previous"),
    [("zaten-askida-degildi", 0), ("hala-askida", 2),
     ("hata", 0xFFFFFFFF)],
    ids=["onceki-0", "onceki-2", "hata-kodu"])
def test_only_a_suspend_count_of_one_counts_as_started(tmp_path, cwd_dir,
                                                       monkeypatch, label,
                                                       previous):
    """`ResumeThread` returns the PREVIOUS suspend count. Only 1 means
    the count reached zero and the thread is running; 0 contradicts the
    frozen launch, >1 means still suspended, and 0xFFFFFFFF is an
    outright failure. The earlier version read none of them."""
    marker = tmp_path / "calisti.txt"
    stub = _stub(tmp_path, seconds=0,
                 child_code=f"import pathlib; pathlib.Path({str(marker)!r})"
                            ".write_text('x')")
    real = process_module._kernel32()                 # noqa: SLF001
    fake = _FakeKernel(real, ResumeThread=lambda handle: previous)
    monkeypatch.setattr(process_module, "_kernel32", lambda: fake)

    with pytest.raises(process_module.ContainmentError):
        process_module.launch_contained([str(stub)], cwd=cwd_dir)
    monkeypatch.undo()
    time.sleep(0.8)
    assert not marker.exists(), "reddedilen resume sonrasi program calisti"


def _thread_walker(pid, entries, tail_error):
    """A fake toolhelp inventory that really drives `_resume`.

    Replacing `_resume` itself proved nothing about the enumeration --
    the mechanism under test was the part being swapped out. These
    stand in for `Thread32First`/`Thread32Next` and fill the caller's
    structure, so the real loop runs."""
    import ctypes

    remaining = list(entries)
    state = {"resumed": 0, "opened": []}

    def fill(pointer):
        owner, thread_id = remaining.pop(0)
        record = pointer.contents
        record.th32OwnerProcessID = owner
        record.th32ThreadID = thread_id
        return 1

    def first(snapshot, pointer):
        ctypes.set_last_error(0)
        return fill(pointer) if remaining else 0

    def following(snapshot, pointer):
        if remaining:
            return fill(pointer)
        ctypes.set_last_error(tail_error)
        return 0

    def open_thread(access, inherit, thread_id):
        state["opened"].append(thread_id)
        return 0x1234                       # a handle the fake resume takes

    def resume(handle):
        state["resumed"] += 1
        return 1

    def close(handle):
        return 1

    return first, following, open_thread, resume, close, state


@pytest.mark.skipif(os.name != "nt", reason="resume yolu Windows'a ozgu")
def test_two_threads_are_refused_before_anything_is_resumed(monkeypatch):
    """Fail-closed and WHOLE: the inventory is walked, the decision is
    made, and only then is one thread resumed. Resuming some and
    refusing after would leave a half-started program in the
    container."""
    real = process_module._kernel32()                 # noqa: SLF001
    first, following, open_thread, resume, close, state = _thread_walker(
        4242, [(4242, 11), (4242, 12)], _ERROR_NO_MORE_FILES)
    monkeypatch.setattr(process_module, "_kernel32",
                        lambda: _FakeKernel(real, Thread32First=first,
                                            Thread32Next=following,
                                            OpenThread=open_thread,
                                            ResumeThread=resume))
    with pytest.raises(process_module.ContainmentError,
                       match="is parcacigi sayisi"):
        process_module._resume(4242)                  # noqa: SLF001
    assert state["resumed"] == 0, "reddedilmeden once resume edildi"


@pytest.mark.skipif(os.name != "nt", reason="resume yolu Windows'a ozgu")
def test_a_walk_that_ends_normally_is_accepted(monkeypatch):
    """The boundary. `ERROR_NO_MORE_FILES` is the ONLY code that means
    the enumeration finished, and a single matching thread is the
    healthy case -- without this the fix above would just be "refuse
    everything"."""
    real = process_module._kernel32()                 # noqa: SLF001
    first, following, open_thread, resume, close, state = _thread_walker(
        4242, [(4242, 11), (9999, 12)], _ERROR_NO_MORE_FILES)
    monkeypatch.setattr(process_module, "_kernel32",
                        lambda: _FakeKernel(real, Thread32First=first,
                                            Thread32Next=following,
                                            OpenThread=open_thread,
                                            ResumeThread=resume,
                                            CloseHandle=close))
    process_module._resume(4242)                      # noqa: SLF001
    assert state["resumed"] == 1
    assert state["opened"] == [11], "yanlis is parcacigi acildi"


@pytest.mark.skipif(os.name != "nt", reason="resume yolu Windows'a ozgu")
def test_a_truncated_thread_inventory_is_not_a_complete_one(monkeypatch):
    """THE finding. `Thread32Next` returning FALSE means either "no more
    threads" or "the walk failed", and treating both as the end turns a
    truncated inventory into a complete one: a process whose second
    thread was never seen looks like a single-threaded launch and gets
    resumed."""
    real = process_module._kernel32()                 # noqa: SLF001
    first, following, open_thread, resume, close, state = _thread_walker(
        4242, [(4242, 11)], 5)                        # ERROR_ACCESS_DENIED
    monkeypatch.setattr(process_module, "_kernel32",
                        lambda: _FakeKernel(real, Thread32First=first,
                                            Thread32Next=following,
                                            OpenThread=open_thread,
                                            ResumeThread=resume))
    with pytest.raises(process_module.ContainmentError, match="eksik kaldi"):
        process_module._resume(4242)                  # noqa: SLF001
    assert state["resumed"] == 0, "eksik envanterden sonra resume edildi"


@pytest.mark.skipif(os.name != "nt", reason="attach yolu Windows'a ozgu")
@pytest.mark.parametrize(
    ("label", "failure"),
    [("os-hatasi", OSError("kurgu: beklenmeyen")),
     ("kesinti", KeyboardInterrupt()),
     ("degerhatasi", ValueError("kurgu"))],
    ids=["OSError", "KeyboardInterrupt", "ValueError"])
def test_any_failure_after_the_launch_still_discards_the_process(
        tmp_path, cwd_dir, monkeypatch, label, failure):
    """Only `ContainmentError` reached the cleanup, so a Win32 call
    raising `OSError`, or an interrupt landing between the launch and
    the attach, left the suspended process on the machine."""
    discarded = []
    stub = _stub(tmp_path, seconds=30)

    def exploding_attach(pid):
        raise failure

    real_discard = process_module._discard            # noqa: SLF001

    def recording_discard(process):
        discarded.append(process)
        return real_discard(process)

    monkeypatch.setattr(process_module, "_attach_job", exploding_attach)
    monkeypatch.setattr(process_module, "_discard", recording_discard)
    with pytest.raises(type(failure)):
        process_module.launch_contained([str(stub)], cwd=cwd_dir)
    monkeypatch.undo()
    assert len(discarded) == 1, "temizlik cagrilmadi"
    assert discarded[0].poll() is not None, "askidaki surec hala yasiyor"


@pytest.mark.skipif(os.name != "nt", reason="job drain Windows'a ozgu")
def test_a_job_that_empties_after_a_moment_is_not_a_failure(tmp_path,
                                                            cwd_dir,
                                                            monkeypatch):
    """Termination is not instantaneous. Reading the active count once
    turned a job that went 1, 1, 0 into a refusal -- a false red for a
    tree that emptied normally a few milliseconds later."""
    import ctypes

    real = process_module._kernel32()                 # noqa: SLF001
    samples = [1, 1, 0]

    def staged_query(job, klass, buffer, size, returned):
        accounting = ctypes.cast(
            buffer, ctypes.POINTER(process_module._JobAccounting)).contents
        accounting.ActiveProcesses = samples.pop(0) if samples else 0
        return 1

    stub = _stub(tmp_path, seconds=0)
    process, container = process_module.launch_contained([str(stub)],
                                                         cwd=cwd_dir)
    monkeypatch.setattr(process_module, "_kernel32",
                        lambda: _FakeKernel(real,
                                            QueryInformationJobObject=staged_query))
    emptied = container.drain(time.monotonic() + 5)
    monkeypatch.undo()
    assert emptied is True, "gecici olarak dolu gorunen is nesnesi reddedildi"
    assert samples == [], "senaryo kurulmadi: sorgu bir kez bile tekrarlanmadi"


@pytest.mark.skipif(os.name != "nt", reason="job drain Windows'a ozgu")
@pytest.mark.parametrize(
    "failure",
    ["terminate", "query", "never-empties"],
    ids=["terminate-reddetti", "sorgu-hatasi", "hic-bosalmadi"])
def test_a_container_that_cannot_be_proven_empty_is_refused(tmp_path,
                                                            cwd_dir,
                                                            monkeypatch,
                                                            failure):
    """Each refusal path separately, and each within the shared deadline
    rather than forever."""
    import ctypes

    real = process_module._kernel32()                 # noqa: SLF001
    overrides = {}
    if failure == "terminate":
        overrides["TerminateJobObject"] = lambda job, code: 0
    elif failure == "query":
        overrides["QueryInformationJobObject"] = \
            lambda job, klass, buffer, size, returned: 0
    else:
        def always_busy(job, klass, buffer, size, returned):
            accounting = ctypes.cast(
                buffer,
                ctypes.POINTER(process_module._JobAccounting)).contents
            accounting.ActiveProcesses = 1
            return 1

        overrides["QueryInformationJobObject"] = always_busy

    stub = _stub(tmp_path, seconds=0)
    process, container = process_module.launch_contained([str(stub)],
                                                         cwd=cwd_dir)
    monkeypatch.setattr(process_module, "_kernel32",
                        lambda: _FakeKernel(real, **overrides))
    started = time.monotonic()
    emptied = container.drain(time.monotonic() + 2)
    elapsed = time.monotonic() - started
    monkeypatch.undo()
    assert emptied is False, "kanitlanamayan bosluk basari sayildi"
    assert elapsed < 10, f"drain ortak deadline'i asti ({elapsed:.1f}s)"
    # the handle is closed exactly once, on every path
    assert container._job is None                     # noqa: SLF001
    assert container.drain(time.monotonic() + 1) is False


@pytest.mark.skipif(os.name != "nt", reason="attach yolu Windows'a ozgu")
def test_a_suspended_process_that_cannot_be_discarded_is_reported(
        tmp_path, cwd_dir, monkeypatch):
    """`kill` raising, or `wait` timing out, is exactly the case where
    "it never ran" stops being true. The earlier chain let either of
    them replace the containment error with its own, or propagate while
    a frozen process was still on the machine."""
    stub = _stub(tmp_path, seconds=0)

    def refusing_attach(pid):
        raise process_module.ContainmentError("kurgu: is nesnesi kurulamadi")

    monkeypatch.setattr(process_module, "_attach_job", refusing_attach)
    monkeypatch.setattr(process_module, "_discard", lambda process: False)
    with pytest.raises(process_module.ContainmentEscaped) as refusal:
        process_module.launch_contained([str(stub)], cwd=cwd_dir)
    monkeypatch.undo()
    # the ORIGINAL failure is chained, never replaced
    assert isinstance(refusal.value.__cause__, process_module.ContainmentError)
    assert "is nesnesi" in str(refusal.value.__cause__)


@pytest.mark.skipif(os.name != "nt", reason="attach yolu Windows'a ozgu")
@pytest.mark.parametrize("broken", ["kill", "wait"],
                         ids=["kill-hatasi", "wait-zaman-asimi"])
def test_the_discard_result_is_measured_not_assumed(tmp_path, cwd_dir,
                                                    monkeypatch, broken):
    """A counter proving `kill` was CALLED is not proof the process is
    gone. `_discard` reports what the operating system says."""
    stub = _stub(tmp_path, seconds=30)
    process, container = process_module.launch_contained([str(stub)],
                                                         cwd=cwd_dir)
    try:
        if broken == "kill":
            monkeypatch.setattr(type(process), "kill",
                                lambda self: (_ for _ in ()).throw(
                                    OSError("kurgu: oldurulmedi")))
        else:
            monkeypatch.setattr(type(process), "wait",
                                lambda self, timeout=None: (_ for _ in ()).throw(
                                    subprocess.TimeoutExpired("kurgu", 1)))
        discarded = process_module._discard(process)   # noqa: SLF001
        monkeypatch.undo()
        # the real state decides, whichever call failed
        assert discarded is (process.poll() is not None)
    finally:
        monkeypatch.undo()
        container.drain(time.monotonic() + 5)


# =====================================================================
# THE ENVIRONMENT SEAM
#
# `env` was added for a caller that builds an ISOLATED environment. The
# whole value of that is what it leaves OUT, so the three cases below
# are about absence as much as presence.
# =====================================================================

_YANKICI = "\n".join([
    "import json, os, sys",
    "sys.stdout.write(json.dumps(dict(os.environ)))",
])

_EBEVEYNE_OZEL = "KURGU-EBEVEYN-NOBETCISI-" + "z" * 8


@pytest.fixture
def yankici(tmp_path, monkeypatch):
    """A child that prints its own environment, and a parent variable
    that must not follow it in.

    Written as a list of plain lines rather than one embedded literal:
    this project has lost a newline three times passing source through
    a generator, and a list of lines has nothing left to lose."""
    script = tmp_path / "yankici.py"
    script.write_text(_YANKICI, encoding="utf-8")
    monkeypatch.setenv("EBEVEYNE_OZEL", _EBEVEYNE_OZEL)
    return script


def _child_environment(script, tmp_path, **kwargs):
    process, container = process_module.launch_contained(
        [sys.executable, str(script)], cwd=tmp_path, **kwargs)
    try:
        out, _ = process.communicate(timeout=60)
    finally:
        container.drain(time.monotonic() + 10)
    return json.loads(out.decode("utf-8"))


def test_an_omitted_env_keeps_the_inherited_environment(yankici, tmp_path):
    """The default has to stay exactly what every existing caller
    already depends on."""
    child = _child_environment(yankici, tmp_path)
    assert child.get("EBEVEYNE_OZEL") == _EBEVEYNE_OZEL


def test_an_explicit_env_reaches_the_child(yankici, tmp_path):
    child = _child_environment(yankici, tmp_path, env=_minimal_env())
    assert child.get("KURGU_ANAHTAR") == "kurgu-deger", \
        "verilen ortam cocuga ulasmadi"


def test_a_parent_variable_left_out_of_an_explicit_env_stays_out(yankici,
                                                                 tmp_path):
    """The one that matters. Re-merging with `os.environ` would put back
    exactly what the caller removed on purpose, and the call would still
    look correct from the outside."""
    child = _child_environment(yankici, tmp_path, env=_minimal_env())
    assert "EBEVEYNE_OZEL" not in child, "ebeveyn ortami yeniden birlestirilmis"


def _minimal_env():
    """Enough for a Python child to start, and nothing else. The two
    Windows variables are not decoration: without them the interpreter
    fails to initialise, and the test would then be measuring a broken
    launch rather than an isolated one."""
    return {"KURGU_ANAHTAR": "kurgu-deger",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PATH": os.environ.get("PATH", "")}


def test_the_transport_module_runs_no_shell():
    import ast

    source = Path(process_module.__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "shell" and \
                        getattr(keyword.value, "value", False) is True:
                    offenders.append("shell=True")
            if ast.unparse(node.func) in ("os.system", "os.popen"):
                offenders.append(ast.unparse(node.func))
    assert offenders == [], f"kabuk kullanimi: {offenders}"
