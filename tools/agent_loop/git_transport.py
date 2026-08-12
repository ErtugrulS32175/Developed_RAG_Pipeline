"""Bounded, contained Git transport. PACKAGE B2B-A-D3A / R1B.1.

RUNNING GIT IS A PROCESS LIFECYCLE PROBLEM, not a string problem. That
is why it lives here instead of next to the object parsing it feeds:
this module owns a child process, two reader threads and a container,
and none of that has anything to say about what a tree object means.

WHY NOT `capture_output=True`. MEASURED on this machine: a producer
writing 48 MiB was read in full -- 50,331,648 bytes -- before any
application limit was consulted, and a run whose grandchild held the
pipe never returned at all. A ceiling checked after the bytes are
already in the process is not a memory bound; it is a report.

NOTHING IS ACCEPTED FROM A PROCESS THAT HAS NOT FINISHED. An answer
taken while a reader is alive, the parent is unreaped or the container
cannot be shown empty is an answer about a moment that has not ended --
so the cleanup verdict outranks the output, and a reader that FAILED is
refused even when git exited zero.

WHAT MAY LEAVE. Fixed sentences and closed reasons. Never git's stderr,
never a path, never the bytes of an object.
"""
from __future__ import annotations

import os
import threading
import time

from tools.agent_loop import contract
from tools.agent_loop import process as process_mod

_GIT_TIMEOUT_SECONDS = 120
_POLL_SECONDS = 0.01

# stderr exists here only to classify a failure; it is never returned
STDERR_CEILING = 8 << 10

_CEILING_REFUSAL = "git ciktisi sozlesme tavanini asiyor"


class FlatWorkspaceError(RuntimeError):
    """The package's single error type.

    It is defined in the LOWEST module so every layer above can raise
    exactly one class without an import cycle -- `git_objects` and
    `flat_workspace` both re-export this same object, so one
    `except FlatWorkspaceError` catches a transport refusal, a
    raw-object refusal and a ledger refusal alike.

    Fixed text and a closed reason. The message never carries a path,
    an object's bytes or git's stderr."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


def _git_env():
    env = dict(os.environ)
    env.update({
        # replacement objects would let a ref decide which bytes a
        # baseline resolves to, which is the whole thing being avoided
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "",
        "GCM_INTERACTIVE": "never", "GIT_CONFIG_NOSYSTEM": "1",
    })
    return env


def byte_ceiling(value, what):
    """A byte count is an exact positive `int`.

    Separate from the duration validator on purpose: sharing one made
    `1.5` a legal byte ceiling, and there is no such thing as one and a
    half bytes. `True` is an `int` in Python, so the type is checked
    exactly rather than with `isinstance`."""
    if type(value) is not int:
        raise FlatWorkspaceError(f"{what} tam sayi degil")
    if value <= 0:
        raise FlatWorkspaceError(f"{what} pozitif degil")
    return value


def _finite_positive(value, what):
    """A duration may be fractional, but never a bool, never NaN and
    never infinite -- `NaN` compares false to every bound, so a deadline
    built from it never arrives."""
    if type(value) is bool or not isinstance(value, (int, float)):
        raise FlatWorkspaceError(f"{what} sayisal degil")
    if value != value or value in (float("inf"), float("-inf")):
        raise FlatWorkspaceError(f"{what} sonlu degil")
    if value <= 0:
        raise FlatWorkspaceError(f"{what} pozitif degil")
    return value


def _pump(proc, tripped, okuyucular, stdout_limit, timeout):
    """Read both pipes to a bound and decide the outcome.

    RETURNS `(stdout bytes | None, refusal | None)` and NEVER RAISES.
    That signature is the point: a refusal is built here as a value and
    raised by the caller after cleanup has finished, so our own fixed
    sentence can never be chained to a raw `OSError` through
    `__context__`. Readers are recorded in `okuyucular` as they start,
    so the caller can join exactly the threads that exist."""
    try:
        deadline = time.monotonic() + timeout
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        disari = process_mod.BoundedStream("stdout", proc.stdout,
                                           stdout_limit, tripped)
        hata = process_mod.BoundedStream("stderr", proc.stderr,
                                         STDERR_CEILING, tripped)
        for akis in (disari, hata):
            akis.start()
            # recorded AFTER the start succeeds: joining a thread that
            # never started is a different bug wearing this one's face
            okuyucular.append(akis)

        while True:
            if proc.poll() is not None and not (disari.is_alive()
                                                or hata.is_alive()):
                break
            if tripped.is_set():
                return None, FlatWorkspaceError(_CEILING_REFUSAL)
            if time.monotonic() >= deadline:
                return None, FlatWorkspaceError(
                    "git komutu zaman asimina ugradi")
            time.sleep(_POLL_SECONDS)

        for akis in (disari, hata):
            if akis.outcome == process_mod.READ_OVERFLOWED:
                return None, FlatWorkspaceError(_CEILING_REFUSAL)
            if akis.outcome != process_mod.READ_COMPLETED:
                # A READER THAT FAILED IS NOT A SHORT ANSWER. Before the
                # outcome existed, a pipe that broke mid-read left a
                # truncated buffer and a clean-looking exit code, and
                # the truncation became the object's bytes.
                return None, FlatWorkspaceError("git ciktisi eksiksiz "
                                                "okunamadi")
        return bytes(disari.buffer), None
    except BaseException:                            # noqa: BLE001
        # Built, not raised: see the docstring. Whatever went wrong is
        # someone else's text and does not travel.
        return None, FlatWorkspaceError("git tasimasi tamamlanamadi")


def _durdur(proc, deadline):
    if proc.poll() is not None:
        return True
    return process_mod.stop(proc, deadline)


def _reap(proc, deadline):
    proc.wait(timeout=max(0.1, deadline - time.monotonic()))
    return True


def _cleanup(proc, container, okuyucular):
    """Attempt ALL FOUR operations, whatever any of them does.

    Each step is independent because a `finally` block that runs them
    in sequence stops at the first one that raises -- and the steps that
    would have been skipped are the ones that empty the container and
    join the readers. A step that raises is a step that did not prove
    anything, which is exactly what `False` means here."""
    deadline = time.monotonic() + process_mod.REAP_SECONDS
    adimlar = (lambda: _durdur(proc, deadline),
               lambda: container.drain(deadline),
               lambda: process_mod.join_within(okuyucular, deadline),
               lambda: _reap(proc, deadline))
    temiz = True
    for adim in adimlar:
        try:
            sonuc = adim()
        except BaseException:                        # noqa: BLE001
            sonuc = False
        temiz &= bool(sonuc)
    return temiz


def run_git_bounded(argv, *, cwd, stdout_limit,
                    timeout=_GIT_TIMEOUT_SECONDS):
    """Run git contained, read it bounded, and prove the container empty.

    THE ORDER OF THE ANSWERS IS THE CONTRACT. Cleanup is decided first,
    because a process still running against the repository outranks
    whatever it happened to print; then the transport's own refusal;
    then git's exit code; and only a run that survived all three
    returns bytes.

    Every `raise` below stands OUTSIDE an exception handler, so nothing
    raw is carried along as `__context__` or `__cause__`."""
    byte_ceiling(stdout_limit, "stdout tavani")
    _finite_positive(timeout, "zaman asimi")

    tripped = threading.Event()
    try:
        proc, container = process_mod.launch_contained(
            argv, cwd=str(cwd), env=_git_env())
    except process_mod.ContainmentError:
        # the program never ran -- that is the point of the refusal
        raise FlatWorkspaceError("git kapsayicisi kurulamadi") from None
    except OSError:
        raise FlatWorkspaceError("git komutu calistirilamadi") from None

    # FROM HERE ON THERE IS EXACTLY ONE WAY OUT, and it goes through
    # cleanup. Before this shape existed, a reader that failed to
    # construct, a `poll` that raised, or an interrupt between the two
    # `start()` calls each left a live process and a live container
    # behind with nobody holding a reference to either.
    okuyucular = []
    cikti, ret = _pump(proc, tripped, okuyucular, stdout_limit, timeout)
    temiz = _cleanup(proc, container, okuyucular)

    if not temiz:
        raise FlatWorkspaceError("git sureci temizlenemedi")
    if ret is not None:
        raise ret
    if proc.returncode != 0:
        # the command NAME, never its stderr: that text carries paths,
        # remotes and credential-helper complaints
        raise FlatWorkspaceError("git komutu basarisiz")
    return cikti


def git_bytes(repo, *args, stdout_limit) -> bytes:
    """Every git read names its own ceiling. There is no generic
    unbounded reader left to reach for."""
    return run_git_bounded(["git", "-C", str(repo), *args], cwd=repo,
                           stdout_limit=stdout_limit)
