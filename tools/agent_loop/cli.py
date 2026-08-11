"""Argv construction for the two CLIs. NO subprocess, NO discovery.

WHY THIS MODULE EXISTS SEPARATELY. The first draft of the test suite
created fake `claude` and `codex` executables and then never handed them
to anything -- there was no seam to hand them to. A runner written
against that suite would have found the real binaries on PATH and spent
real money, while the tests claimed "no real model is ever called".
The claim was untested because the wiring did not exist.

So the binaries are a MANDATORY ARGUMENT, here, at the bottom. There is
no discovery function in this package: nothing falls back to PATH, so a
caller that forgets to pass a binary gets a TypeError rather than a
surprise invoice. Tests pass fake paths and get fake processes by
construction, not by patching.

This module builds LISTS. It never runs them -- running belongs to the
runner (Phase B), and keeping the two apart is what lets the argv rules
be proven green before any process exists.
"""
from __future__ import annotations

from pathlib import Path

from tools.agent_loop import schemas
from tools.agent_loop.contract import (
    CODEX_APPROVAL_OVERRIDE,
    IMPLEMENTER_ALLOWED_TOOLS,
    CODEX_SANDBOX_READ_ONLY,
    FORBIDDEN_FLAGS,
    FORBIDDEN_FLAG_VALUES,
)


class UnsafeInvocation(RuntimeError):
    """An argv carrying a flag that would remove the sandbox, the
    permission checks or the approval gate."""


def assert_safe_argv(argv):
    """Refuse the flags that turn a supervised call into an unsupervised
    one. Checked on the BUILT argv rather than on the inputs: a rule
    that inspects intent misses whatever the builder actually emitted."""
    tokens = [str(token) for token in argv]
    for token in tokens:
        bare = token.split("=", 1)[0]
        if bare in FORBIDDEN_FLAGS:
            raise UnsafeInvocation(f"yasak bayrak: {bare}")
    for flag, value in FORBIDDEN_FLAG_VALUES:
        for index, token in enumerate(tokens):
            if token == flag and index + 1 < len(tokens) \
                    and tokens[index + 1] == value:
                raise UnsafeInvocation(f"yasak deger: {flag} {value}")
            if token == f"{flag}={value}":
                raise UnsafeInvocation(f"yasak deger: {token}")
    return tokens


def build_implementer_argv(binary, *, budget_usd,
                           allowed_tools=IMPLEMENTER_ALLOWED_TOOLS,
                           prompt_is_stdin=True, model=None,
                           permission_mode="acceptEdits"):
    """`claude` in non-interactive mode, bounded by schema and budget.

    THE SCHEMA IS NOT A PARAMETER. `--json-schema` takes INLINE JSON --
    measured against the installed CLI's own help -- and this builder
    used to pass a file path there, chosen by the caller and mutable on
    disk between building the argv and running it. The value is now the
    frozen canonical text of `IMPLEMENTER_RESULT_SCHEMA`, from the one
    binding the validator also uses; there is nothing for a caller to
    substitute.

    THE TOOL LIST IS NOT THE CALLER'S TO CHOOSE. `allowed_tools=["Bash"]`
    used to be accepted, and a Claude holding Bash can `git add`,
    `git commit`, `git push`, install a dependency or reach the network
    -- every one of them a human gate the runner never sees, because the
    gate lives in the runner and the action happened inside the model's
    own tool call. The implementer reads and edits files; the RUNNER
    runs the registry commands, and that split is the only reason the
    gates mean anything. The parameter survives so a test can prove the
    refusal, not so a caller can widen it.

    Every flag here was read from `claude --help` on the installed
    build (2.1.220). The prompt is NOT an argument -- it goes over
    stdin, so no model-produced text is ever part of a command line."""
    requested = tuple(allowed_tools or ())
    forbidden = [tool for tool in requested
                 if tool not in IMPLEMENTER_ALLOWED_TOOLS]
    if forbidden:
        raise UnsafeInvocation(
            f"implementer bu araclari alamaz: {sorted(forbidden)}; "
            f"izinli olanlar {list(IMPLEMENTER_ALLOWED_TOOLS)}")
    if not requested:
        raise UnsafeInvocation(
            "arac izin listesi bos olamaz: sinirsiz arac erisimi demek")
    allowed_tools = requested
    argv = [
        str(binary),
        "--print",
        "--output-format", "json",
        "--json-schema", schemas.IMPLEMENTER_SCHEMA_BINDING.canonical_json,
        "--max-budget-usd", str(budget_usd),
        "--permission-mode", permission_mode,
        "--allowedTools", *allowed_tools,
    ]
    if model:
        argv += ["--model", str(model)]
    if not prompt_is_stdin:                 # pragma: no cover -- refused
        raise UnsafeInvocation(
            "istem yalnizca stdin'den verilir; komut satirina konmaz")
    return assert_safe_argv(argv)


def build_evaluator_argv(binary, *, repo, schema_path, last_message_path,
                         model=None):
    """`codex exec`, read-only, never asking a human for approval.

    The approval policy travels as a CONFIG OVERRIDE, not as `-a never`:
    the top-level command takes `-a/--ask-for-approval`, but `codex
    exec` rejects it outright (`error: unexpected argument '-a'`).
    Mandating the flag that does not exist would break every evaluator
    call; mandating the one that does keeps an unattended loop from
    parking on a prompt nobody is there to answer."""
    argv = [
        str(binary), "exec",
        "--sandbox", CODEX_SANDBOX_READ_ONLY,
        *CODEX_APPROVAL_OVERRIDE,
        "--cd", str(Path(repo)),
        "--output-schema", str(schema_path),
        "--output-last-message", str(last_message_path),
    ]
    if model:
        argv += ["--model", str(model)]
    return assert_safe_argv(argv)


def resolve_registry_command(command_id, registry, *, paths=()):
    """A command_id becomes an argv list, or it does not run.

    The task file names a command; it never spells one. `["git",
    "push"]` is a well-formed argv list, which is exactly why the
    freedom to write one does not belong in a config file."""
    entry = registry.get(command_id)
    if entry is None:
        raise UnsafeInvocation(f"kayitli olmayan komut: {command_id!r}")
    argv = list(entry["argv"])
    if paths:
        if not entry["accepts_paths"]:
            raise UnsafeInvocation(
                f"{command_id!r} yol argumani almaz")
        for path in paths:
            candidate = str(path)
            if candidate.startswith("-") or "\\" in candidate \
                    or candidate.startswith("/") or ".." in Path(
                        candidate).parts:
                raise UnsafeInvocation(f"guvensiz yol argumani: {candidate!r}")
            argv.append(candidate)
    return argv
