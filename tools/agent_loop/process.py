"""Contained process transport. PACKAGE B2A / R1.3.

Split out of `execution.py`, which had grown to carry both the policy of
a model call and every platform detail of running one. The policy is
what an auditor needs to read; the ctypes structures are what makes it
long. Nothing here knows what a model is.

THE CONTAINER EXISTS BEFORE THE PROGRAM RUNS. That ordering is the whole
point of this module. Creating the process and then putting it in a
container leaves a window -- microseconds, but real -- in which the
child can spawn something that lands OUTSIDE the container and outlives
the call. On Windows the child is therefore created SUSPENDED: it has
not executed a single instruction when the Job Object is attached, the
membership is read back from the OS, and only then is its main thread
resumed. If any of that fails the program never runs at all, which is
both the safe answer and the cheap one -- an uncontained model call is
also a billable one.

POSIX needs none of that: `start_new_session=True` runs `setsid()` in the
child between fork and exec, so the session exists before the program
does. The process group id is captured once, at that moment, and never
re-derived -- asking `getpgid` about a pid that has already been reaped
can fail and make an occupied group look empty.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

READ_CHUNK_BYTES = 65536
REAP_SECONDS = 10.0
TASKKILL_SECONDS = 30.0


class ContainmentError(RuntimeError):
    """The process could not be placed in a container, so it was not
    allowed to run. Deliberately a plain error: the typed vocabulary
    that reaches a caller belongs to the layer above."""


class ContainmentEscaped(ContainmentError):
    """Worse: containment failed AND the process could not be cleaned
    up, so something may still be running. A separate type because the
    two need different responses -- one is a refused call, the other is
    a machine with a stray process on it. The original failure is
    chained, never replaced."""


DRAIN_POLL_SECONDS = 0.05

if os.name == "nt":                                   # pragma: no cover -- os
    import ctypes
    from ctypes import wintypes

    _CREATE_SUSPENDED = 0x00000004
    _KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_ACCOUNTING_CLASS = 1
    _JOB_EXTENDED_LIMIT_CLASS = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED = 0x1000
    _SNAP_THREAD = 0x00000004
    _THREAD_SUSPEND_RESUME = 0x0002
    # ResumeThread returns the PREVIOUS suspend count.
    _RESUMED_AND_RUNNING = 1
    _RESUME_FAILED = 0xFFFFFFFF
    # the ONLY code that means the walk finished normally
    _ERROR_NO_MORE_FILES = 18

    class _JobBasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadOperationCount", "WriteOperationCount",
                     "OtherOperationCount", "ReadTransferCount",
                     "WriteTransferCount", "OtherTransferCount")]

    class _JobExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _JobBasicLimits),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _JobAccounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64),
                    ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64),
                    ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", ctypes.c_uint32),
                    ("TotalProcesses", ctypes.c_uint32),
                    ("ActiveProcesses", ctypes.c_uint32),
                    ("TotalTerminatedProcesses", ctypes.c_uint32)]

    class _ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD),
                    ("tpBasePri", ctypes.c_long),
                    ("tpDeltaPri", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD)]

    # EVERY signature is declared. Left to ctypes' defaults, a function
    # returns `c_int` and takes anything -- which silently truncates a
    # 64-bit HANDLE to 32 bits and hands the rest of this module a
    # number that is not the object it names. Nothing about that fails
    # loudly; it fails as a handle that refers to nothing.
    _SIGNATURES = {
        "CreateJobObjectW": ([wintypes.LPVOID, wintypes.LPCWSTR],
                             wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int,
                                     wintypes.LPVOID, wintypes.DWORD],
                                    wintypes.BOOL),
        "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
                        wintypes.HANDLE),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE],
                                     wintypes.BOOL),
        "IsProcessInJob": ([wintypes.HANDLE, wintypes.HANDLE,
                            ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
        "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD],
                                     wintypes.HANDLE),
        "Thread32First": ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)],
                          wintypes.BOOL),
        "Thread32Next": ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)],
                         wintypes.BOOL),
        "OpenThread": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
                       wintypes.HANDLE),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT],
                               wintypes.BOOL),
        "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int,
                                       wintypes.LPVOID, wintypes.DWORD,
                                       ctypes.POINTER(wintypes.DWORD)],
                                      wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetSystemDirectoryW": ([wintypes.LPWSTR, wintypes.UINT],
                                wintypes.UINT),
    }

    def _bind_kernel32():
        library = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype
        return library

    _KERNEL32 = _bind_kernel32()
    # pointer-width, so the comparison is correct on 64-bit
    INVALID_HANDLE_VALUE = ctypes.cast(ctypes.c_void_p(-1),
                                       wintypes.HANDLE).value


def _kernel32():
    """The ONE binding point. A seam as well as a rule: tests drive the
    failure codes through here instead of guessing at them."""
    return _KERNEL32


def _attach_job(pid):
    """A Job Object holding this pid, with membership READ BACK.

    `AssignProcessToJobObject` returning success is not the same as the
    process being in the job -- an early version believed the return
    value and the membership query disagreed with it, because the handle
    lacked query access. So the answer comes from `IsProcessInJob`."""
    kernel32 = _kernel32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ContainmentError("is nesnesi olusturulamadi")
    try:
        limits = _JobExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
                job, _JOB_EXTENDED_LIMIT_CLASS, ctypes.byref(limits),
                ctypes.sizeof(limits)):
            raise ContainmentError("is nesnesi sinirlari kurulamadi")
        handle = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED,
            False, pid)
        if not handle:
            raise ContainmentError("surec tanitici acilamadi")
        try:
            if not kernel32.AssignProcessToJobObject(job, handle):
                raise ContainmentError("surec is nesnesine atanamadi")
            inside = wintypes.BOOL()
            if not kernel32.IsProcessInJob(handle, job, ctypes.byref(inside)) \
                    or not inside.value:
                raise ContainmentError("is nesnesi uyeligi dogrulanamadi")
        finally:
            kernel32.CloseHandle(handle)
    except BaseException:
        kernel32.CloseHandle(job)
        raise
    return job


def _resume(pid):
    """Let the frozen program start -- and prove that it did.

    `ResumeThread` returns the PREVIOUS suspend count, and only 1 means
    "the count reached zero and the thread is now running". An earlier
    version treated a successful `OpenThread` as success and never read
    the result at all, so a thread that stayed suspended, or a call that
    failed outright, both looked like a started program.

    Threads are ENUMERATED FIRST and the decision is made before any of
    them is touched. A frozen launch has exactly one thread; anything
    else is a situation this function does not understand, and resuming
    some of them and then refusing would leave a half-started program
    inside the container."""
    kernel32 = _kernel32()
    snapshot = kernel32.CreateToolhelp32Snapshot(_SNAP_THREAD, 0)
    if not snapshot or snapshot == INVALID_HANDLE_VALUE:
        raise ContainmentError("is parcacigi listesi alinamadi")
    entry = _ThreadEntry()
    entry.dwSize = ctypes.sizeof(_ThreadEntry)
    threads = []
    try:
        ctypes.set_last_error(0)
        found = kernel32.Thread32First(snapshot, ctypes.pointer(entry))
        if not found:
            raise ContainmentError("is parcacigi listesi baslatilamadi")
        while found:
            if entry.th32OwnerProcessID == pid:
                threads.append(entry.th32ThreadID)
            ctypes.set_last_error(0)
            found = kernel32.Thread32Next(snapshot, ctypes.pointer(entry))
        # A FALSE from `Thread32Next` means either "no more threads" or
        # "the walk failed". Treating both as the end turns a truncated
        # inventory into a complete one -- and a process whose second
        # thread was never seen looks like a single-threaded launch.
        code = ctypes.get_last_error()
        if code != _ERROR_NO_MORE_FILES:
            raise ContainmentError(
                f"is parcacigi listesi eksik kaldi (hata {code})")
    finally:
        kernel32.CloseHandle(snapshot)

    if len(threads) != 1:
        raise ContainmentError(
            f"askiya alinmis surecte beklenmeyen is parcacigi sayisi: "
            f"{len(threads)}")

    thread = kernel32.OpenThread(_THREAD_SUSPEND_RESUME, False, threads[0])
    if not thread:
        raise ContainmentError("is parcacigi tanitici acilamadi")
    try:
        previous = kernel32.ResumeThread(thread)
    finally:
        kernel32.CloseHandle(thread)
    if previous == _RESUME_FAILED:
        raise ContainmentError("is parcacigi devam ettirilemedi")
    if previous != _RESUMED_AND_RUNNING:
        # 0 == it was never suspended, which contradicts the frozen
        # launch; >1 == still suspended after this call
        raise ContainmentError(
            f"surec calisir duruma gecmedi (onceki aski sayisi: {previous})")


class Container:
    """The processes a call is allowed to have started.

    Emptiness is ASKED of the operating system. Keeping a list of the
    `Popen` objects we made cannot answer it: that list knows the parent
    and nothing the parent started, and a model that answered correctly
    and left a child running looked exactly like a clean run."""

    def __init__(self, process, *, job=None, pgid=None):
        self.process = process
        self._job = job
        # captured ONCE, while the process is certainly alive. Deriving
        # it later from a pid that has been reaped can fail, and a
        # failed lookup reads as "the group is empty".
        self._pgid = pgid

    def drain(self, deadline=None):
        """Stop everything inside; report whether it is now empty.

        Emptiness is POLLED until the shared deadline rather than read
        once. A job whose active count is 1, 1, 0 is a job that emptied
        normally -- termination is not instantaneous -- and answering
        from the first sample turns that into a false refusal. A count
        that stays above zero until the deadline is a real one."""
        if deadline is None:
            deadline = time.monotonic() + REAP_SECONDS
        if os.name == "nt":
            return self._drain_job(deadline)
        return self._drain_group(deadline)

    def _drain_group(self, deadline):
        if self._pgid is None:
            return False
        try:
            os.killpg(self._pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        while True:
            try:
                os.killpg(self._pgid, 0)
            except ProcessLookupError:
                return True                     # the group is gone
            except OSError:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(DRAIN_POLL_SECONDS)

    def _drain_job(self, deadline):
        kernel32 = _kernel32()
        if self._job is None:
            return False
        try:
            if not kernel32.TerminateJobObject(self._job, 1):
                return False                    # refused, and it matters
            while True:
                accounting = _JobAccounting()
                if not kernel32.QueryInformationJobObject(
                        self._job, _JOB_ACCOUNTING_CLASS,
                        ctypes.byref(accounting), ctypes.sizeof(accounting),
                        None):
                    return False
                if accounting.ActiveProcesses == 0:
                    return True                 # the OS says it is empty
                if time.monotonic() >= deadline:
                    return False
                time.sleep(DRAIN_POLL_SECONDS)
        finally:
            # exactly once, on every path. Closing it also triggers
            # KILL_ON_JOB_CLOSE, so even a refused drain still stops the
            # tree -- it just cannot claim to have proven it.
            kernel32.CloseHandle(self._job)
            self._job = None


def launch_contained(argv, *, cwd, env=None):
    """Start `argv` inside a container that already exists.

    Returns `(process, container)`. Raises `ContainmentError` WITHOUT
    running the program if the container cannot be established -- an
    uncontained model call is a call nobody can stop, and it costs money
    to discover that afterwards.

    `env=None` keeps the inherited environment, which is what every
    existing caller wants. A mapping is passed to `Popen` EXACTLY as
    given and is never merged with `os.environ` again: a caller that
    built an isolated environment did so to leave things out, and
    re-merging would put them back."""
    if os.name != "nt":
        process = subprocess.Popen(
            argv, cwd=str(cwd), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            shell=False, start_new_session=True, env=env)
        # `setsid()` ran between fork and exec, so the session existed
        # before the program did: nothing to race with
        return process, Container(process, pgid=process.pid)

    process = subprocess.Popen(
        argv, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, shell=False, env=env,
        creationflags=(_CREATE_SUSPENDED | subprocess.CREATE_NEW_PROCESS_GROUP))
    try:
        job = _attach_job(process.pid)
    except BaseException as refused:
        # EVERY failure, not just `ContainmentError`. An `OSError` from
        # a Win32 call, or a `KeyboardInterrupt` between the launch and
        # the attach, used to leave the suspended process on the machine
        # because only one exception type reached the cleanup.
        #
        # It never executed anything -- but that only stays true if the
        # process is actually gone, and `kill` and `wait` can both fail.
        if not _discard(process):
            raise ContainmentEscaped(
                "kapsayici kurulamadi ve askidaki surec temizlenemedi"
            ) from refused
        raise
    container = Container(process, job=job)
    try:
        _resume(process.pid)
    except BaseException as refused:
        if not container.drain():
            raise ContainmentEscaped(
                "surec baslatilamadi ve kapsayici bosaltilamadi"
            ) from refused
        raise
    return process, container


def _discard(process):
    """Stop a process that was never allowed to run, and CHECK.

    Returns whether it is really gone. `kill()` raising, or `wait()`
    timing out, is exactly the case where the honest answer is "no"."""
    try:
        process.kill()
    except Exception:                           # noqa: BLE001 -- checked below
        pass
    try:
        process.wait(timeout=REAP_SECONDS)
    except Exception:                           # noqa: BLE001
        pass
    return process.poll() is not None


# How a reader ENDED. Three outcomes, and the third is the one that was
# missing: a pipe that broke mid-read used to be indistinguishable from
# a short answer, so a truncated buffer arrived beside a clean exit code
# and was treated as the whole output.
READ_PENDING = "pending"
READ_COMPLETED = "completed"
READ_OVERFLOWED = "overflowed"
READ_FAILED = "read_failed"


class BoundedStream(threading.Thread):
    """Reads one pipe in chunks and stops the moment its ceiling is
    crossed. The bytes past the ceiling are never kept: what is over the
    limit is not evidence, it is the thing being refused.

    A finished reader always carries a closed `outcome`, and that
    outcome is a CONSTANT -- never the exception, never its text. What
    a broken pipe has to say about the machine is not something a
    caller should be able to print."""

    def __init__(self, label, stream, limit, tripped):
        super().__init__(daemon=True)
        self.label = label
        self.total = 0
        self.outcome = READ_PENDING
        self.buffer = bytearray()
        self._stream = stream
        self._limit = limit
        self._tripped = tripped

    @property
    def overflowed(self):
        """Kept as the name every existing caller already uses, derived
        from the outcome so the two can never disagree."""
        return self.outcome == READ_OVERFLOWED

    def run(self):
        # EVERY exit assigns the outcome, and nothing propagates out of
        # a thread: an uncaught exception here would print a traceback
        # carrying paths to stderr and still leave the reader looking
        # like a normal short read.
        sonuc = READ_FAILED
        try:
            while True:
                chunk = self._stream.read(READ_CHUNK_BYTES)
                if not chunk:
                    sonuc = READ_COMPLETED
                    break
                self.total += len(chunk)
                if self.total > self._limit:
                    sonuc = READ_OVERFLOWED
                    self._tripped.set()
                    break
                self.buffer.extend(chunk)
        except BaseException:                   # noqa: BLE001 -- recorded
            sonuc = READ_FAILED
        try:
            self._stream.close()
        except BaseException:                   # noqa: BLE001 -- recorded
            # a pipe we could not release is not a read we can call done
            sonuc = READ_FAILED
        # assigned LAST, so a thread that is no longer alive always has
        # its final answer visible to whoever joined it
        self.outcome = sonuc


class PromptWriter(threading.Thread):
    """Feeds stdin without ever blocking the deadline.

    A child that reads nothing leaves this thread parked in `write`
    until the container is drained and the pipe closes. That is fine
    here and was fatal on the main path."""

    def __init__(self, stream, payload):
        super().__init__(daemon=True)
        self.completed = False
        self._stream = stream
        self._payload = payload

    def run(self):
        try:
            # WRITTEN IN FULL, or not written. `bufsize=0` makes stdin a
            # RAW stream, and a raw write is one `write(2)`: it may move
            # only part of the buffer and return that count without
            # raising. Treating the call's return as "done" sent the
            # model a TRUNCATED prompt and reported success -- a reply
            # to a question that was cut in half. Windows hid it by
            # blocking until the whole buffer had gone; Linux did not,
            # and CI is where it surfaced.
            remaining = memoryview(self._payload)
            while remaining:
                written = self._stream.write(remaining)
                if not written:
                    return                  # the pipe stopped accepting
                remaining = remaining[written:]
            self._stream.flush()
            self.completed = True
        except (OSError, ValueError):           # the child went away
            pass
        finally:
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass


def system_directory():
    """Asked of Windows itself, not of the environment.

    The tree-killer used to be located through the system-root
    environment variable, and an environment variable is something
    anything can set. Redirecting it redirected the only cleanup path
    there was, and the failure was then swallowed. The name of that
    variable is deliberately absent from this file so its absence is
    checkable."""
    buffer = ctypes.create_unicode_buffer(512)
    written = _kernel32().GetSystemDirectoryW(buffer, 512)
    if not written:
        raise OSError("sistem dizini isletim sisteminden alinamadi")
    return Path(buffer.value)


def killer_environment(directory):
    """A minimal environment built from the OS answer, never inherited.

    Finding the executable through the OS was necessary and not
    sufficient: with the system-root variable pointing elsewhere the
    tool still started and still failed, returning 1 and killing
    nothing."""
    windows = directory.parent
    return {"SYSTEMROOT": str(windows), "windir": str(windows),
            "SystemDrive": windows.drive or "C:",
            "PATH": str(directory)}


def terminate_tree(process, deadline=None):
    """Stop the process and everything it started.

    The container is the guarantee; this is the polite request that
    comes first. `deadline` is the SHARED budget -- the kill used to
    keep a private thirty seconds outside the grace window, so a
    ten-second budget measured twelve."""
    if os.name == "nt":
        directory = system_directory()
        killer = directory / "taskkill.exe"
        if not killer.is_file():
            raise OSError("surec agaci temizleyicisi bulunamadi")
        remaining = (TASKKILL_SECONDS if deadline is None
                     else max(0.1, deadline - time.monotonic()))
        done = subprocess.run(
            [str(killer), "/F", "/T", "/PID", str(process.pid)],
            capture_output=True, timeout=remaining,
            env=killer_environment(directory))
        # 128 == already gone, which is the outcome we wanted
        if done.returncode not in (0, 128):
            raise OSError("surec agaci durdurulamadi")
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def stop(process, deadline):
    """Best effort, subordinate -- and REPORTED.

    Whatever went wrong first is what the caller needs to hear about, so
    nothing here raises. But a cleanup that quietly failed is a process
    still running against the worktree, so the outcome comes back as a
    flag instead of disappearing into an `except`."""
    complete = True
    try:
        terminate_tree(process, deadline)
    except Exception:                           # noqa: BLE001 -- subordinate
        complete = False
    try:
        process.kill()
    except Exception:                           # noqa: BLE001
        pass
    try:
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except Exception:                           # noqa: BLE001
        complete = False
    return complete


def join_within(threads, deadline):
    """One grace budget for all of them, not one each."""
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        thread.join(timeout=remaining)
    return all(not thread.is_alive() for thread in threads)
