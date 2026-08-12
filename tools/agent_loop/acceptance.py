"""Frozen acceptance commands, run in a disposable mirror. PACKAGE B2B-C1.

ONE primitive: take a candidate that has ALREADY been verified, prove it
is still exactly that by re-deriving it from the filesystem, copy it into
a throwaway tree, and run the commands the task NAMED -- in order, under
one deadline, with every stream bounded while it is read. It advances no
state machine, applies nothing to the operator's checkout, calls no
model, repairs nothing and decides nothing about what happens next.

A TASK NAMES A COMMAND; IT NEVER SPELLS ONE. `["git", "push"]` is a
perfectly well-formed argv list, which is exactly why the freedom to
write one does not belong in a file a model can influence. The argv comes
from `contract.COMMAND_REGISTRY` through `cli.resolve_registry_command`,
and a task may add repo-relative paths only where that registry says so.
There is no parameter here through which a caller can supply an argv, a
shell string, a working directory, an environment or a mirror.

THE SHELL RULE, PRECISELY, because the loose version reads as a
contradiction: `p0_gate` runs `bash scripts/p0_gate.sh`, and that is
allowed -- it is a fixed, tracked, reviewed script named in a frozen
registry. What is forbidden is an ARBITRARY or INLINE shell: `shell=True`
or any command string a task or a model could influence.

THE ORDER OF THE GATES IS THE CONTRACT. Every binding -- the exact task
bytes, their digest, the schema, the state binding, the flat workspace
binding, the verified change set's identities, and the FRESH re-derivation
of the candidate -- is settled before a mirror directory or a process
exists. A run that refuses leaves the mirror counter and the process
counter at zero.

BOUNDS ARE ENFORCED WHILE READING. `capture_output=True` was measured
taking 50,331,648 bytes into memory with no application ceiling
consulted, and a grandchild holding the pipe made the call never return
at all. Each stream is read in chunks against its own ceiling, one
absolute deadline covers every command, and a container that cannot be
shown empty is a refusal rather than a footnote.

WHAT MAY LEAVE. Closed contract codes and numbers: an exit code, a
duration, two byte counts. Never a byte a command printed, never a file's
contents, never a corpus fragment, never an absolute path, never the
operating system's own error text.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools.agent_loop import (acceptance_workspace, changes, cli, contract,
                              flat_workspace, preflight, process, schemas)

_POLL_SECONDS = 0.02

# Read from the frozen task schema rather than invented here, so this
# adapter cannot accept a call the contract would have refused.
_OUTPUT_BYTES = schemas.TASK_SCHEMA["properties"]["max_output_bytes"]
MIN_OUTPUT_BYTES = _OUTPUT_BYTES["minimum"]
MAX_OUTPUT_BYTES = _OUTPUT_BYTES["maximum"]
# The WHOLE-RUN wall clock, not the per-model-call budget: every
# acceptance command shares one deadline, so the ceiling that describes
# it is the run's, taken from the same frozen schema and converted once.
MAX_TIMEOUT_SECONDS = 60 * schemas.TASK_SCHEMA[
    "properties"]["max_wall_clock_minutes"]["maximum"]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(contract.IDENTIFIER_PATTERN)

# How a command ENDED. `read_failed` is the one that was missing from
# every earlier version of this shape: a pipe that broke mid-read is
# indistinguishable from a short answer unless it is named.
COMPLETED = "completed"
OVERFLOWED = "overflowed"
TIMED_OUT = "timed_out"
READ_FAILED = "read_failed"

_EVENTS = {COMPLETED: contract.EventCode.ACCEPTANCE_FINISHED,
           OVERFLOWED: contract.EventCode.OUTPUT_TRUNCATED,
           TIMED_OUT: contract.EventCode.INTERRUPTED,
           READ_FAILED: contract.EventCode.INTERRUPTED}

# The programs the frozen registry can name. Located from the runner's
# own environment ONCE and handed to the child as an explicit PATH, so
# nothing is discovered inside the mirror and no task can choose an
# interpreter.
_PROGRAM_SUFFIXES = ("", ".exe", ".cmd", ".bat") if os.name == "nt" else ("",)


class AcceptanceError(RuntimeError):
    """A refused or failed acceptance run.

    Carries a fixed sentence chosen here and a closed contract reason.
    Never captured output, never a path, never an OS message."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


class AcceptanceRefused(AcceptanceError):
    """An input, a bound or a binding this run may not proceed on."""


class CandidateMismatch(AcceptanceError):
    """The verified change set is not what the filesystem says now.

    Terminal rather than repairable: the thing that would judge a repair
    is the thing that disagrees."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PATH_NOT_ALLOWED)


class ContainmentFailed(AcceptanceError):
    """The container could not be established, so nothing was run."""

    def __init__(self, message):
        super().__init__(message,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED)


class ProcessTreeSurvived(AcceptanceError):
    """The command ended but its container could not be emptied.

    Not a detail to log past: something the candidate started is still
    running, so no answer from this run is returned."""

    def __init__(self, message):
        super().__init__(message,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED)


@dataclass(frozen=True, slots=True)
class AcceptanceCommandResult:
    """One command, as numbers and closed codes."""

    command_id: str
    passed: bool
    exit_code: object            # exact int, or None when none was reached
    duration_ms: int
    stdout_bytes: int
    stderr_bytes: int
    event: str


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """The whole run. `passed` is true only when every command it was
    given ran and every one of them exited zero."""

    run_id: str
    workspace_id: str
    baseline_sha: str
    passed: bool
    command_results: tuple
    total_duration_ms: int
    event: str


# ---------------------------------------------------------------------
# input canonicalization
# ---------------------------------------------------------------------

def _exact_text(value, what: str) -> str:
    try:
        return cli.exact_text(value, what=what)
    except cli.UnsafeInvocation:
        raise AcceptanceRefused(f"{what} tam bir metin degil") from None


def _exact_match(value, pattern, what: str) -> str:
    if type(value) is not str or not pattern.match(value):
        raise AcceptanceRefused(f"{what} beklenen bicimde degil")
    return value


def _canonical_limits(timeout_seconds, max_output_bytes):
    """Both bounds are checked BEFORE anything exists, and RETURNED.

    Exact `int`, which also excludes `bool` by type. An `int` SUBCLASS
    whose comparisons always agreed passed both ranges while carrying a
    value nine orders of magnitude outside them -- a bound whose
    comparisons the value itself defines is not a bound."""
    if type(timeout_seconds) is not int:
        raise AcceptanceRefused("sure siniri tam sayi degil")
    if timeout_seconds <= 0:
        raise AcceptanceRefused("sure siniri pozitif degil")
    if timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise AcceptanceRefused("sure siniri sozlesme tavanini asiyor")
    if type(max_output_bytes) is not int:
        raise AcceptanceRefused("cikti siniri tam sayi degil")
    if not MIN_OUTPUT_BYTES <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise AcceptanceRefused("cikti siniri sozlesme araligi disinda")
    return timeout_seconds, max_output_bytes


def _assert_candidate(verified, candidate, *, run_id, workspace_id,
                      baseline_sha):
    """The claim, compared to the re-derivation, field by field.

    EXACT TYPE FIRST. A duck-typed stand-in that answered every
    comparison would satisfy each check below while describing nothing,
    and the object is the whole authority for what gets copied."""
    if type(verified) is not changes.VerifiedChangeSet:
        raise AcceptanceRefused(
            "dogrulanmis degisiklik kumesi beklenen turde degil")
    for field, expected in (("run_id", run_id),
                            ("workspace_id", workspace_id),
                            ("baseline_sha", baseline_sha)):
        if getattr(verified, field) != expected:
            raise AcceptanceRefused(
                "dogrulanmis degisiklik kumesi bu cagriya ait degil")
    if verified.status != contract.Status.IMPLEMENTED:
        raise AcceptanceRefused(
            "dogrulanmis degisiklik kumesi uygulanabilir bir sonuc degil")
    if (verified.changed_files, verified.added, verified.modified,
            verified.deleted, verified.fingerprint) != (
            candidate.changed_files, candidate.added, candidate.modified,
            candidate.deleted, candidate.fingerprint):
        # BOTH the fingerprint AND the counts: the fingerprint covers
        # path, kind, mode and content, and the counts are what a report
        # would have carried if only one of the two had been compared
        raise CandidateMismatch("aday degisiklikler taze kanitla ayni degil")


def _assert_manifest(task_file: Path, expected: str) -> None:
    """The manifest bytes, re-read and compared to the digest this run
    was issued. Asked again between commands: the manifest chooses the
    commands, so one that moves mid-run chose the rest of them."""
    try:
        digest = preflight.snapshot_manifest(task_file).digest
    except (OSError, ValueError):
        raise AcceptanceRefused("gorev dosyasi okunamadi") from None
    if digest != expected:
        raise AcceptanceRefused(
            "gorev dosyasi degistirildi",
            reason=contract.StopReason.PATH_NOT_ALLOWED)


def _resolve(reference):
    """A command id becomes an argv, or the run does not start.

    The registry decides everything: which program, which flags, and
    whether this command may take path arguments at all. A task carries
    the id and, where the registry allows it, repo-relative paths in the
    order it listed them -- duplicates included, because a task that
    names the same gate twice means it twice."""
    command_id, paths = reference
    try:
        argv = cli.resolve_registry_command(command_id,
                                            contract.COMMAND_REGISTRY,
                                            paths=paths)
    except cli.UnsafeInvocation:
        raise AcceptanceRefused("kabul komutu cozumlenemedi") from None
    return command_id, tuple(argv)


# ---------------------------------------------------------------------
# the closed environment
# ---------------------------------------------------------------------

def _locate(program: str, search_path: str):
    """Resolution that depends on nothing but the directories we chose.

    Written out rather than delegated to the standard library, whose
    lookup consults the runner's own extension list -- a second
    environment reading into a decision this function exists to make
    deterministic. It is also NOT a binary discovery in the sense the
    frozen contract forbids: the only names ever passed here are the
    interpreters the registry itself spells (`python`, `bash`, `git`),
    and no model binary can reach this module at all -- the public seam
    has no parameter for one."""
    for directory in search_path.split(os.pathsep):
        for suffix in _PROGRAM_SUFFIXES:
            candidate = Path(directory) / (program + suffix)
            if candidate.is_file():
                return candidate
    return None


def _search_path() -> str:
    """The interpreter that is running this loop, plus git and bash where
    the machine actually keeps them. Located ONCE, from the runner's
    environment, and handed to the child explicitly.

    THE ALTERNATIVE IS WORSE, which is why this reads the machine at all:
    the registry's argv lists say `python`, `bash` and `git`, and a child
    launched without an explicit search path resolves those against
    whatever it inherits. Choosing the directories here and passing them
    as the child's ONLY path is the narrow version of that. No model
    binary is ever named -- this module cannot be handed one."""
    directories = [str(Path(sys.executable).resolve().parent)]
    inherited = os.environ.get("PATH", "")
    for program in ("git", "bash"):
        found = _locate(program, inherited)
        if found is not None:
            directories.append(str(found.parent))
    if os.name == "nt":
        directories.append(str(process.system_directory()))
    return os.pathsep.join(dict.fromkeys(directories))


def _acceptance_env(mirror) -> dict:
    """A CLOSED map, built here and never merged with `os.environ`.

    A task cannot supply one and neither can a caller: an inherited
    environment carries the operator's tokens, the operator's git
    identity, a `PYTHONPATH` into the operator's tree and a `HOME` full
    of configuration a candidate command would then obey. Every path in
    here belongs to the mirror.

    The git isolation travels as `GIT_CONFIG_*` VARIABLES rather than as
    command-line overrides on purpose: the frozen scanner runs git
    itself, and a setting that only applies to our own invocation would
    leave the grandchild reading whatever it found."""
    home, scratch = str(mirror.home), str(mirror.scratch)
    env = {
        "PATH": _search_path(),
        "HOME": home, "USERPROFILE": home, "XDG_CONFIG_HOME": home,
        "TMP": scratch, "TEMP": scratch, "TMPDIR": scratch,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": str(mirror.git_config),
        "GIT_CONFIG_SYSTEM": str(mirror.git_config),
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CEILING_DIRECTORIES": str(mirror.holder),
        # a hooks directory of the CALL's own, so nothing the candidate
        # ships can execute during an inventory, and no end-of-line or
        # filesystem-monitor behaviour is inherited from a machine
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "core.hookspath",
        "GIT_CONFIG_VALUE_0": str(mirror.hooks),
        "GIT_CONFIG_KEY_1": "core.autocrlf",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": "core.fsmonitor",
        "GIT_CONFIG_VALUE_2": "false",
    }
    if os.name == "nt":
        windows = process.system_directory().parent
        env.update({"SYSTEMROOT": str(windows), "windir": str(windows),
                    "SystemDrive": windows.drive or "C:",
                    # fixed rather than inherited, so the child's program
                    # resolution is the same on every machine
                    "PATHEXT": ".COM;.EXE;.BAT;.CMD"})
    return env


def _assert_programs(commands, env) -> None:
    """Every program a resolved argv names has to exist inside the
    closed PATH before anything runs. A command that cannot start is a
    gate that never ran, and a gate that never ran is not a pass."""
    for _, argv in commands:
        if _locate(argv[0], env["PATH"]) is None:
            raise AcceptanceRefused("kabul ortami gerekli programi bulamadi")


# ---------------------------------------------------------------------
# one bounded, contained process
# ---------------------------------------------------------------------

def _reclaim(child, container, started_streams, grace):
    """Stop, empty, join and reap -- attempting ALL FOUR whatever any of
    them does, and CONSUMING every answer.

    This is the cleanup for a call that never reached its own. Each step
    is independent because a sequence that stops at the first failure
    skips exactly the steps that empty the container and join the
    readers. A step that raises is a step that proved nothing, which is
    what `False` means here.

    `started_streams` holds only readers whose `start()` RETURNED:
    joining a thread that never started is a different bug wearing this
    one's face, and `join_within` on an empty list is honestly True."""
    if grace is None:
        grace = time.monotonic() + process.REAP_SECONDS
    steps = (lambda: process.stop(child, grace),
             lambda: container.drain(grace),
             lambda: process.join_within(started_streams, grace),
             # THE REAP, asked of the operating system rather than
             # assumed: a `stop` that returned is not the same as a
             # process that has been waited for.
             lambda: child.poll() is not None)
    proven = True
    for step in steps:
        try:
            answer = step()
        except BaseException:                   # noqa: BLE001 -- consumed
            answer = False
        proven &= bool(answer)
    return proven


def _run_command(argv, *, cwd, env, deadline, max_output_bytes):
    """Start `argv` inside a container that already exists, read both
    pipes to their own ceiling, and prove the container empty.

    THE ORDER OF THE ANSWERS IS THE CONTRACT. Containment first, because
    a process still running against the mirror outranks whatever it
    printed; then the reader outcomes; then the wall clock; and only a
    command that survived all three is judged by its exit code.

    FROM THE MOMENT `launch_contained` RETURNS THERE IS EXACTLY ONE WAY
    OUT. The window between that return and the poll loop -- the reader
    constructors, either `start()`, the stdin close -- used to sit
    outside every envelope: a `finally` drained the container WITHOUT
    reading the answer, never joined a reader that had already started
    and never reaped the child, so a setup failure could travel out
    looking like an ordinary refusal while the tree was still running.
    Nothing is decided in a `finally` here: the primary error and the
    cleanup verdict are held in named variables and judged afterwards.

    Returns `(outcome, exit_code, stdout_bytes, stderr_bytes)` and never
    the bytes themselves."""
    try:
        child, container = process.launch_contained(list(argv), cwd=str(cwd),
                                                    env=env)
    except process.ContainmentError:
        raise ContainmentFailed(
            "kabul sureci kapsayicisi kurulamadi") from None
    except OSError:
        raise ContainmentFailed("kabul sureci baslatilamadi") from None

    started_streams = []
    grace = None
    settled = False
    primary = None
    answer = None
    try:
        tripped = threading.Event()
        streams = [process.BoundedStream("stdout", child.stdout,
                                         max_output_bytes, tripped),
                   process.BoundedStream("stderr", child.stderr,
                                         max_output_bytes, tripped)]
        for stream in streams:
            stream.start()
            # RECORDED ONLY AFTER `start()` RETURNS, so the cleanup can
            # never be handed a thread that was never started
            started_streams.append(stream)
        # STDIN IS CLOSED, never written: an acceptance command is not
        # asked anything, and a pipe left open is a command that can wait
        # for an answer nobody is there to give.
        try:
            child.stdin.close()
        except (OSError, ValueError):
            pass

        timed_out = False
        while True:
            if tripped.is_set() or child.poll() is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(_POLL_SECONDS)

        # ONE grace budget covers the kill and every thread join, so a
        # command that is already failing cannot spend three of them.
        grace = time.monotonic() + process.REAP_SECONDS
        if tripped.is_set() or timed_out:
            process.stop(child, grace)
        # DRAIN BEFORE JOINING: a grandchild holding the pipes open keeps
        # the readers alive, so joining first waits out the whole grace
        # period and only then stops anything.
        drained = container.drain(grace)
        joined = process.join_within(started_streams, grace)
        # THE CLEANUP HAS NOW HAPPENED, and `drain` closes the job
        # handle on its way out -- so a second attempt would be trusting
        # a handle that no longer names anything and would answer False
        # for a container that is genuinely empty.
        settled = True
        counts = (streams[0].total, streams[1].total)
        if not drained:
            raise ProcessTreeSurvived(
                "kabul sureci kapsayicisi bosaltilamadi")
        if timed_out:
            answer = (TIMED_OUT, None) + counts
        elif any(stream.overflowed for stream in streams):
            answer = (OVERFLOWED, None) + counts
        # A READER THAT FAILED IS NOT A SHORT ANSWER, and neither is one
        # still alive: an answer taken while a reader has not finished is
        # an answer about a moment that has not ended.
        elif not joined or any(stream.outcome != process.READ_COMPLETED
                               for stream in streams):
            answer = (READ_FAILED, None) + counts
        else:
            answer = (COMPLETED, child.wait()) + counts
    except BaseException as raised:
        # CAPTURED, not handled here. Every raise below stands OUTSIDE
        # this handler, because `raise X from None` clears `__cause__`
        # and the suppression flag but leaves `__context__` holding the
        # original -- and a raw `OSError`'s text names absolute paths.
        primary = raised

    if primary is None:
        return answer

    reclaimed = True if settled else _reclaim(child, container,
                                              started_streams, grace)
    if not reclaimed:
        # STRONGER THAN WHATEVER FAILED FIRST. A gate going red and a
        # machine with a stray process on it are different events, and
        # only one of them needs a human -- so the cleanup verdict is
        # never allowed to hide behind the error that triggered it.
        raise ProcessTreeSurvived("kabul sureci temizlenemedi")
    if isinstance(primary, AcceptanceError):
        raise primary
    if not isinstance(primary, Exception):
        # KeyboardInterrupt and SystemExit are the operator's decision
        # rather than a finding: now that the cleanup is proven, they
        # travel out exactly as they arrived.
        raise primary
    raise AcceptanceRefused("kabul sureci akislari kurulamadi")


# ---------------------------------------------------------------------
# the public seam
# ---------------------------------------------------------------------

def run_acceptance(*, repo, state_dir, task_path, manifest_digest, run_id,
                   workspace_id, baseline_sha, verified_changes,
                   timeout_seconds, max_output_bytes) -> AcceptanceReport:
    """Run the task's acceptance commands against a disposable mirror.

    The caller passes IDENTITIES and a verified change set. It does not
    pass a workspace path, a mirror path, a working directory, an
    environment, an argv or a command list: the commands come from the
    exact manifest bytes whose digest this run was issued, the argv comes
    from the frozen registry, and the candidate comes from D3A's binding
    seam re-read through the change-set module's own authority."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    state_path = Path(_exact_text(state_dir, "durum dizini"))
    task_file = Path(_exact_text(task_path, "gorev dosyasi yolu"))
    manifest_digest = _exact_match(manifest_digest, _HEX64, "gorev ozeti")
    run_id = _exact_match(run_id, _RUN_ID, "kosu kimligi")
    workspace_id = _exact_match(workspace_id, flat_workspace.WORKSPACE_ID,
                                "calisma alani kimligi")
    baseline_sha = _exact_match(baseline_sha, _SHA1, "taban surum")
    timeout_seconds, max_output_bytes = _canonical_limits(timeout_seconds,
                                                          max_output_bytes)

    identity = {"repo": repo_path, "state_dir": state_path,
                "task_path": task_file, "manifest_digest": manifest_digest,
                "run_id": run_id, "workspace_id": workspace_id,
                "baseline_sha": baseline_sha}
    # THE FRESH DERIVATION IS THE GATE. It re-reads the manifest bytes,
    # re-asserts the state and workspace bindings, re-walks both flat
    # trees and re-authorises every path -- through the module that owns
    # all five of those questions, so there is no second opinion here.
    try:
        candidate = changes.derive_candidate_changes(**identity)
    except changes.ChangeSetError as refused:
        # ONE boundary for the change-set layer's refusals. This seam
        # promises a typed `AcceptanceError` for every refusal, and a
        # `ChangeSetError` escaping through it would break that promise:
        # a caller's closed state machine would have no code to record.
        # The SENTENCE and the REASON are the lower layer's own -- both
        # are already fixed and closed there, and re-deciding either here
        # would put two gates behind one message.
        raise AcceptanceRefused(str(refused),
                                reason=refused.reason) from None
    _assert_candidate(verified_changes, candidate, run_id=run_id,
                      workspace_id=workspace_id, baseline_sha=baseline_sha)
    # the bytes read for the derivation, compared to the bytes on disk
    # now: the first read is the one everything above was decided from
    _assert_manifest(task_file, candidate.task_digest)
    if candidate.task_digest != manifest_digest:
        raise AcceptanceRefused("gorev ozeti bu kosuya ait degil")

    commands = [_resolve(reference)
                for reference in candidate.acceptance_commands]
    workspace = flat_workspace.assert_binding(
        repo_path, state_dir=state_path, run_id=run_id,
        workspace_id=workspace_id, baseline_sha=baseline_sha)
    wants_corpus = any(command_id == "leak_scan" for command_id, _ in commands)

    started = time.monotonic()
    mirror = acceptance_workspace.create(
        repo=repo_path, workspace=workspace, run_id=run_id,
        baseline_sha=baseline_sha, candidate=candidate,
        snapshot_corpus=wants_corpus)
    try:
        env = _acceptance_env(mirror)
        _assert_programs(commands, env)
        # ONE ABSOLUTE DEADLINE, and it starts HERE rather than at the
        # top of the call. Building the mirror is this package's own
        # work -- materialising a baseline and walking two trees twice --
        # and charging it to the candidate's budget made a short timeout
        # expire before the first command was launched, which reports a
        # timed-out gate for a gate that never ran. Every process from
        # this point on, the git preparation included, shares the one
        # deadline: a command cannot buy itself more by being second.
        deadline = time.monotonic() + timeout_seconds
        _prepare_git(mirror, env, deadline, max_output_bytes)
        results = _walk(commands, mirror=mirror, env=env, deadline=deadline,
                        max_output_bytes=max_output_bytes,
                        task_file=task_file, digest=manifest_digest)
    finally:
        # EVERY exit -- success, refusal, timeout, interrupt. A mirror
        # that cannot be proven gone is a STRONGLY TYPED cleanup failure
        # rather than an ordinary red gate: one needs a rerun, the other
        # needs a human.
        acceptance_workspace.remove(mirror)

    return AcceptanceReport(
        run_id=run_id, workspace_id=workspace_id, baseline_sha=baseline_sha,
        passed=all(result.passed for result in results)
        and len(results) == len(commands),
        command_results=tuple(results),
        total_duration_ms=int((time.monotonic() - started) * 1000),
        event=contract.EventCode.ACCEPTANCE_FINISHED)


def _prepare_git(mirror, env, deadline, max_output_bytes) -> None:
    """Git metadata that sees the mirror and nothing else.

    The frozen scanner takes its file inventory from git, and the flat
    roots correctly have no `.git` at all -- so one is created HERE, for
    the mirror, from a fixed argv under the same bounded, contained
    transport every acceptance command uses. The operator's repository is
    never named, never linked and never opened."""
    outcome, exit_code, _, _ = _run_command(
        ("git", "init", "-q"), cwd=mirror.tree, env=env, deadline=deadline,
        max_output_bytes=max_output_bytes)
    if outcome != COMPLETED or exit_code != 0:
        raise AcceptanceRefused("kabul git ust verisi kurulamadi")
    acceptance_workspace.assert_git_metadata(mirror)


def _walk(commands, *, mirror, env, deadline, max_output_bytes, task_file,
          digest):
    """The commands, in the task's own order, stopping at the first one
    that does not exit zero.

    A run that kept going after a failure would report the LAST answer
    instead of the first, and the manifest is re-checked before each one
    because the manifest is what chose them."""
    results = []
    for command_id, argv in commands:
        _assert_manifest(task_file, digest)
        started = time.monotonic()
        outcome, exit_code, out_bytes, err_bytes = _run_command(
            argv, cwd=mirror.tree, env=env, deadline=deadline,
            max_output_bytes=max_output_bytes)
        results.append(AcceptanceCommandResult(
            command_id=command_id,
            passed=outcome == COMPLETED and exit_code == 0,
            exit_code=exit_code,
            duration_ms=int((time.monotonic() - started) * 1000),
            stdout_bytes=out_bytes, stderr_bytes=err_bytes,
            event=_EVENTS[outcome]))
        if not results[-1].passed:
            break
    return results
