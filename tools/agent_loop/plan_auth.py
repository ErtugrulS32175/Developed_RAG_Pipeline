"""Plan-only authentication for the two model CLIs. PACKAGE B4-R5.

ONE QUESTION: is this machine about to spend a SUBSCRIPTION, or an API
balance? Both CLIs will happily run either way, and the difference is
invisible in every artefact the loop already keeps -- a run that quietly
switched to metered credits looks exactly like one that did not.

WHAT IS PROVEN BEFORE ANY MODEL RUNS:

  * no API-key environment variable is set for either vendor,
  * `claude auth status` reports a first-party claude.ai subscription on
    a `pro` or `max` plan,
  * `codex login status` reports a ChatGPT login and nothing else.

Anything else refuses, and refuses BEFORE a model subprocess exists.
That ordering is the whole point: a check that runs after the call has
already been made is an audit, not a gate.

WHAT MAY LEAVE THIS MODULE. Three closed strings on `PlanAuth`, and
fixed sentences on the refusal. The account email, the organisation id,
the plan's billing details, the binary's path, the captured output and
the environment are all read here and none of them travels: they are
exactly the values an operator would least like to find in a journal.

THE ENVIRONMENT IS ASKED ONLY WHETHER A KEY EXISTS. Never its value --
not to log it, not to compare it, not to hash it. `os.environ.get(...)`
is followed immediately by a truth test and the string is dropped.

A DECLARED LIMIT, because an undeclared one is a lie by omission. This
proves how the CLI is authenticated AT THIS MOMENT. It cannot prove what
a vendor does when a subscription's quota runs out: if a CLI can be told
-- by a human, later, in its own interface -- to continue on metered
credits, no field in `auth status` reports that, and this module does
not invent one. What it guarantees is that no API key is in the
environment and that the session in force is a subscription session. An
unattended run that needs more than that guarantee should not start.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools.agent_loop import contract
from tools.agent_loop.process import (BoundedStream, ContainmentError,
                                      READ_COMPLETED, REAP_SECONDS,
                                      join_within, launch_contained, stop)

# Closed vocabulary. These are the only authentication shapes this loop
# will run under, and they are values rather than free text so a report
# can carry them.
CLAUDE_SUBSCRIPTION = "claude_subscription"
CHATGPT_SUBSCRIPTION = "chatgpt_subscription"
ACCEPTED_PLANS = ("pro", "max")

# Presence-only. A key here means the next call could be billed against
# an API balance, whatever the CLI's own session says.
FORBIDDEN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                 "OPENAI_API_KEY", "CODEX_API_KEY")

# Fixed argv. Neither spends tokens, and neither is a task's to choose:
# an argument a manifest could supply is an argument it could turn into
# a real invocation.
CLAUDE_STATUS_ARGV = ("auth", "status")
CODEX_STATUS_ARGV = ("login", "status")
CODEX_EXPECTED_LINE = "Logged in using ChatGPT"

# A status line is a sentence, not a document.
MAX_OUTPUT_BYTES = 4096
TIMEOUT_SECONDS = 10
_POLL_SECONDS = 0.02


class PlanAuthRefused(RuntimeError):
    """This machine may not start a model call.

    Carries a fixed sentence chosen here plus the closed contract codes
    -- never captured output, never a path, never an environment
    value."""

    def __init__(self, message, *, cleanup_complete=True):
        super().__init__(message)
        self.reason = contract.StopReason.PREFLIGHT_FAILED
        self.event = contract.EventCode.PREFLIGHT_FAILED
        self.cleanup_complete = cleanup_complete


class PlanAuthCleanupFailed(PlanAuthRefused):
    """The check itself left something running.

    Deliberately a DIFFERENT class from an ordinary refusal: "you are on
    the wrong plan" is a decision, and "a process this loop started is
    still alive" is an incident. A caller that treats them alike would
    retry the second one forever."""

    def __init__(self, message):
        super().__init__(message, cleanup_complete=False)


@dataclass(frozen=True, slots=True)
class PlanAuth:
    """What was proven, as three closed strings.

    WHAT IS NOT HERE: the email, the organisation, the token, the binary
    path, the raw status output, and any number that could be read as a
    balance. Frozen and slotted so nothing can be attached later."""

    implementer_auth: str
    implementer_plan: str
    evaluator_auth: str


def _refuse_environment():
    """Refused BEFORE a process exists, so a machine holding an API key
    never reaches a CLI at all.

    The value is never read into a variable that outlives the test: the
    question is existence, and knowing more would only create something
    to leak."""
    for name in FORBIDDEN_ENV:
        if (os.environ.get(name) or "").strip():
            # the VARIABLE NAME is this package's own constant; the
            # value is not named, not measured and not echoed
            raise PlanAuthRefused(
                f"ortamda API anahtari var: {name}")


def _probe(binary, arguments):
    """Run one free status command, contained and bounded.

    A TEMPORARY WORKING DIRECTORY, so a status query cannot read or
    write the operator's checkout even by accident. Stdin is closed
    immediately: these commands take no input, and an open pipe is a
    place for one to wait.

    Returns `(exit_code, stdout_text, stderr_text)` for LOCAL
    inspection. Nothing here decides what may be reported -- the callers
    extract closed values and drop the text."""
    holder = Path(tempfile.mkdtemp(prefix="agent-loop-auth-"))
    started = time.monotonic()
    try:
        try:
            process, container = launch_contained(
                [str(binary), *arguments], cwd=holder)
        except ContainmentError:
            raise PlanAuthRefused(
                "kimlik dogrulama cagrisi kapsanamadi") from None

        cleanup_complete = True
        drained = False
        try:
            if process.stdin is not None:
                process.stdin.close()
            tripped = threading.Event()
            streams = [
                BoundedStream("stdout", process.stdout, MAX_OUTPUT_BYTES,
                              tripped),
                BoundedStream("stderr", process.stderr, MAX_OUTPUT_BYTES,
                              tripped)]
            for stream in streams:
                stream.start()

            deadline = started + TIMEOUT_SECONDS
            timed_out = False
            while True:
                if tripped.is_set() or process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(_POLL_SECONDS)

            grace = time.monotonic() + REAP_SECONDS
            if timed_out or tripped.is_set():
                cleanup_complete = stop(process, grace)
            # DRAINED ONCE. `drain` kills the group and then polls it
            # empty, so a second call finds nothing to kill and reports
            # a container it cannot confirm -- which reads as a cleanup
            # failure on a call that cleaned up perfectly well.
            drained = container.drain(grace)
            if not drained:
                cleanup_complete = False
            if not join_within(streams, grace):
                cleanup_complete = False
            exit_code = process.wait()
        finally:
            # EVERY exit, including an exception raised mid-read: a
            # container that is never emptied is a process nobody stops.
            # Only when the drain above did not already happen.
            if not drained and not container.drain(
                    time.monotonic() + REAP_SECONDS):
                cleanup_complete = False

        if not cleanup_complete:
            raise PlanAuthCleanupFailed(
                "kimlik dogrulama cagrisi temizlenemedi")
        if timed_out:
            raise PlanAuthRefused("kimlik dogrulama cagrisi sure sinirini asti")
        for stream in streams:
            if stream.outcome != READ_COMPLETED:
                # an overflowed or unreadable status is not a status: a
                # truncated buffer beside a clean exit code is exactly
                # how a partial answer gets read as a whole one
                raise PlanAuthRefused("kimlik dogrulama ciktisi okunamadi")
        return (exit_code,
                bytes(streams[0].buffer).decode("utf-8", "replace"),
                bytes(streams[1].buffer).decode("utf-8", "replace"))
    finally:
        shutil.rmtree(holder, ignore_errors=True)


def _claude_plan(binary):
    """The implementer's plan, or a refusal.

    EXACT VALUES, not truthiness: `loggedIn` must be the boolean `True`
    and not merely something truthy, because a CLI that answered
    `"loggedIn": "false"` would otherwise pass."""
    exit_code, stdout_text, _ = _probe(binary, CLAUDE_STATUS_ARGV)
    if exit_code != 0:
        raise PlanAuthRefused("implementer kimlik durumu alinamadi")
    try:
        payload = json.loads(stdout_text)
    except ValueError:
        raise PlanAuthRefused("implementer kimlik durumu JSON degil") from None
    if type(payload) is not dict:
        raise PlanAuthRefused("implementer kimlik durumu bir nesne degil")
    if payload.get("loggedIn") is not True:
        raise PlanAuthRefused("implementer oturumu acik degil")
    if payload.get("authMethod") != "claude.ai" \
            or payload.get("apiProvider") != "firstParty":
        # an API-key or gateway session spends a balance rather than a
        # plan, whatever else it can do
        raise PlanAuthRefused("implementer abonelik oturumu kullanmiyor")
    plan = payload.get("subscriptionType")
    if plan not in ACCEPTED_PLANS:
        raise PlanAuthRefused("implementer plani kabul listesinde degil")
    # ONLY the plan crosses. The same document carries an email and an
    # organisation id, and neither is read out of this function.
    return plan


def _codex_login(binary):
    """The evaluator's login, or a refusal.

    BOTH STREAMS ARE CONSIDERED, because the installed CLI writes this
    line to stderr -- measured -- and reading stdout alone would see an
    empty buffer and have to decide what emptiness means. Exactly one
    meaningful line is required: a second line is a second claim."""
    exit_code, stdout_text, stderr_text = _probe(binary, CODEX_STATUS_ARGV)
    if exit_code != 0:
        raise PlanAuthRefused("evaluator kimlik durumu alinamadi")
    lines = [line.strip() for line
             in (stdout_text + "\n" + stderr_text).splitlines()
             if line.strip()]
    if len(lines) != 1 or lines[0] != CODEX_EXPECTED_LINE:
        raise PlanAuthRefused("evaluator abonelik oturumu kullanmiyor")
    return CHATGPT_SUBSCRIPTION


def assert_plan_only(*, implementer_binary, evaluator_binary) -> PlanAuth:
    """Prove both CLIs are on subscription sessions, or refuse.

    THE ORDER IS THE GUARANTEE. The environment is judged first, so a
    machine holding an API key never starts a status command, let alone
    a model. The binaries are the caller's exact paths: this module
    contains no discovery, so it cannot answer for a program the caller
    did not name.

    This is not a budget check and does not replace one. It fixes WHERE
    the money comes from; how much of it a run may spend is the
    manifest's ceiling and the runner's ledger."""
    _refuse_environment()
    plan = _claude_plan(implementer_binary)
    evaluator_auth = _codex_login(evaluator_binary)
    return PlanAuth(implementer_auth=CLAUDE_SUBSCRIPTION,
                    implementer_plan=plan,
                    evaluator_auth=evaluator_auth)
