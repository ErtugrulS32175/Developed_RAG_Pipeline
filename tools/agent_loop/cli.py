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

import math
import os
import re
from pathlib import Path

from tools.agent_loop import schemas
from tools.agent_loop.contract import (
    CODEX_APPROVAL_OVERRIDE,
    IMPLEMENTER_ALLOWED_TOOLS,
    CODEX_SANDBOX_READ_ONLY,
    FORBIDDEN_FLAGS,
    FORBIDDEN_FLAG_VALUES,
)

# The one permission mode this loop runs the implementer under. NOT a
# caller's choice: `--permission-mode` is a security control, and a
# value that merely compares equal to the safe one while stringifying
# into something else is exactly the class of defect this module now
# refuses. `bypassPermissions` is separately on the forbidden-value
# list, but that net only catches the values somebody remembered to
# name -- an unlisted alternate mode went straight through.
IMPLEMENTER_PERMISSION_MODE = "acceptEdits"

# Read from the frozen task schema, never re-spelled here: a second
# model grammar is a second thing to keep in sync.
_MODEL_RULE = schemas.TASK_SCHEMA["properties"]["implementer"][
    "properties"]["model"]
MODEL_PATTERN = re.compile(_MODEL_RULE["pattern"])
MODEL_MAX_LENGTH = _MODEL_RULE["maxLength"]


class UnsafeInvocation(RuntimeError):
    """An argv carrying a flag that would remove the sandbox, the
    permission checks or the approval gate -- or a value that could
    still change its meaning between the check and the command line."""


def exact_text(value, *, what):
    """EXACTLY ONE conversion, and the result must be an exact `str`.

    A path-like object is asked once, through `__fspath__`; nothing
    afterwards consults `__str__`. This is the whole boundary rule in
    one function: an object that answers `__fspath__` with binary A and
    `__str__` with binary B used to get verified as A and launched as
    B, and a subclass can customise equality, ordering, encoding and
    stringification long after a check has agreed with it."""
    if isinstance(value, os.PathLike):
        value = value.__fspath__()
    if type(value) is not str:
        raise UnsafeInvocation(f"{what} tam bir metin degil")
    return value


# The whole-run ceiling the frozen task schema sets. Derived, never
# spelled again: a literal here could drift away from the contract.
MAX_BUDGET_USD = schemas.TASK_SCHEMA["properties"]["max_budget_usd"]["maximum"]


def exact_budget(budget_usd):
    """THE budget authority, for every caller of this builder.

    It lived in `execution` alone, and the builder is a PUBLIC callable
    that tests hand straight to a subprocess: called directly it spelled
    `101`, `0`, `-5`, `nan` and `inf` onto the command line, because the
    only thing it asked was the exact type. A rule enforced on one of
    two roads is a rule for people who take that road.

    RETURNED, not merely checked: the number that was bounded has to be
    the number that gets spelled, or a `float` subclass checked as 1.0
    writes 999999 into argv.

    `math.isfinite` is asked only of floats -- an exact integer is
    bounded by comparison, and handing an enormous one to `isfinite`
    raises `OverflowError` instead of refusing."""
    if type(budget_usd) not in (int, float):     # excludes bool by type
        raise UnsafeInvocation("butce tam bir sayi degil")
    if type(budget_usd) is float and not math.isfinite(budget_usd):
        raise UnsafeInvocation("butce sonlu bir sayi degil")
    if budget_usd <= 0:
        raise UnsafeInvocation("kalan butce bir cagriyi fonlamiyor")
    if budget_usd > MAX_BUDGET_USD:
        raise UnsafeInvocation("butce sozlesme tavanini asiyor")
    return budget_usd


def exact_model(model):
    """`None`, or an exact string the frozen task schema accepts.

    ONE authority for the model grammar, used by the argv builder and
    by the adapter's canonicalisation alike -- a second grammar is a
    second thing to keep in sync. The refusal names the FIELD and never
    the value: this text travels into reports."""
    if model is None:
        return None
    if type(model) is not str or len(model) > MODEL_MAX_LENGTH \
            or not MODEL_PATTERN.fullmatch(model):
        raise UnsafeInvocation("model adi sozlesme desenine uymuyor")
    return model


def assert_safe_argv(argv):
    """Refuse the flags that turn a supervised call into an unsupervised
    one. Checked on the BUILT argv rather than on the inputs: a rule
    that inspects intent misses whatever the builder actually emitted.

    THE LAST NET, and it no longer converts anything. It used to call
    `str()` on every token, which is a deferred conversion -- the
    forbidden-value comparison then ran against text that did not exist
    yet when the caller was checked. By the time an argv reaches here
    there must be nothing left to convert."""
    tokens = []
    for token in argv:
        if type(token) is not str:
            raise UnsafeInvocation("argv tam metin olmayan bir oge tasiyor")
        tokens.append(token)
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
                           permission_mode=IMPLEMENTER_PERMISSION_MODE):
    """`claude` in non-interactive mode, bounded by schema and budget.

    THE SCHEMA IS NOT A PARAMETER. `--json-schema` takes INLINE JSON --
    measured against the installed CLI's own help -- and this builder
    used to pass a file path there, chosen by the caller and mutable on
    disk between building the argv and running it. The value is the
    frozen canonical text of a schema binding; there is nothing for a
    caller to substitute.

    WHICH schema travels changed in B4-R2. It used to be the same
    `IMPLEMENTER_RESULT_SCHEMA` the validator uses, and the real API
    refused it: the published structured-output subset does not accept
    `pattern`, `if`/`then` or the length and numeric constraints this
    contract is built out of, and two authorized diagnostics both came
    back as a 4xx client error. What goes on the argv now is
    `CLAUDE_TRANSPORT_SCHEMA` -- the same document reduced to the
    supported subset, derived deterministically and hash-pinned on its
    own. It constrains GENERATION only. The acceptance authority never
    travels, and `execution` still judges every reply with it.

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
    # `is True`, not truthiness: an object with `__bool__` is not the
    # caller promising the prompt stays off the command line.
    if prompt_is_stdin is not True:
        raise UnsafeInvocation(
            "istem yalnizca stdin'den verilir; komut satirina konmaz")
    if type(permission_mode) is not str \
            or permission_mode != IMPLEMENTER_PERMISSION_MODE:
        raise UnsafeInvocation("izin modu anlasilan guvenli deger degil")
    # EXACT strings before the membership test. The allowlist used to
    # be asked of arbitrary objects with `==`, and argv then took their
    # `__str__`: an object equal to "Read" arrived on the command line
    # as "Bash", which is a Claude that can `git push`.
    requested = tuple(allowed_tools or ())
    if not requested:
        raise UnsafeInvocation(
            "arac izin listesi bos olamaz: sinirsiz arac erisimi demek")
    if any(type(tool) is not str for tool in requested):
        raise UnsafeInvocation("arac adlari tam metin olmalidir")
    forbidden = [tool for tool in requested
                 if tool not in IMPLEMENTER_ALLOWED_TOOLS]
    if forbidden:
        # A COUNT and the frozen allowlist, never the rejected names:
        # the refused value is caller input, and this text travels into
        # reports. Echoing it back put arbitrary caller strings there.
        raise UnsafeInvocation(
            f"implementer izinli olmayan {len(forbidden)} arac istedi; "
            f"izinli olanlar {list(IMPLEMENTER_ALLOWED_TOOLS)}")
    # Repeating a flag has no agreed meaning here, so a duplicate is a
    # caller mistake rather than something to guess at.
    if len(set(requested)) != len(requested):
        raise UnsafeInvocation("arac izin listesi yinelenen ad tasiyor")
    # Numbers are stringified into argv, so the value that was BOUNDED
    # has to be the value that is spelled: a `float` subclass checked
    # as 1.0 wrote 999999 onto the command line. Bounds AND type come
    # from the one authority, so the builder cannot be the lenient road.
    budget_usd = exact_budget(budget_usd)
    model = exact_model(model)
    argv = [
        exact_text(binary, what="ikili dosya"),
        "--print",
        "--output-format", "json",
        "--json-schema", schemas.IMPLEMENTER_TRANSPORT_BINDING.canonical_json,
        "--max-budget-usd", str(budget_usd),
        "--permission-mode", permission_mode,
        "--allowedTools", *requested,
    ]
    if model is not None:
        argv += ["--model", model]
    return assert_safe_argv(argv)


def exact_launchable_text(value, *, what):
    """`exact_text`, plus the one question argv actually asks later.

    A string that cannot be encoded is not a token: it passes every type
    check here and then dies inside `Popen`, where the failure is an
    `OSError`/`UnicodeEncodeError` carrying the path -- the exact text
    this package refuses to let into a report. An unpaired surrogate is
    the shape that reaches this: `type(value) is str` is true of it, so
    the type gate alone cannot see it.

    Asked in UTF-8 rather than the platform codec on purpose. Windows
    spells argv in UTF-16 and would accept a lone surrogate, POSIX would
    not, and a builder whose refusals depend on which machine ran it is
    a builder nobody can write a test against."""
    text = exact_text(value, what=what)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # the FIELD, never the value: this text travels into reports
        raise UnsafeInvocation(f"{what} kodlanabilir bir metin degil") from None
    return text


def build_evaluator_argv(binary, *, repo, schema_path, last_message_path,
                         model=None):
    """`codex exec`, read-only, never asking a human for approval.

    The approval policy travels as a CONFIG OVERRIDE, not as `-a never`:
    the top-level command takes `-a/--ask-for-approval`, but `codex
    exec` rejects it outright (`error: unexpected argument '-a'`).
    Mandating the flag that does not exist would break every evaluator
    call; mandating the one that does keeps an unattended loop from
    parking on a prompt nobody is there to answer.

    EVERY TOKEN IS CONVERTED EXACTLY ONCE (B3). This builder was the last
    deferred conversion left in the module: it called `str()` on the
    binary, the repository, both file paths and the model, which is the
    defect the implementer road was hardened against and this road kept.
    `str()` is not a check, it is a REQUEST -- an object answers it
    whenever it is finally asked, so the value that was inspected and the
    value that reached the command line could be two different things.
    `bytes` stringified into `b'...'` and became a path with quotes in
    it; `None` became the literal `None`; a `Path` subclass could answer
    `__fspath__` with one file and `__str__` with another.

    `--output-schema` names a FILE here, unlike the implementer's inline
    `--json-schema`: `codex exec` takes one. That is why the caller owns
    the file's lifetime, and why the token is bound as text exactly once
    rather than re-derived."""
    # `exact_model` before anything is spelled: it is the frozen task
    # schema's own grammar, so the builder cannot be the lenient road
    # into a flag the contract constrains. `None` stays absent; an empty
    # string is a caller mistake rather than a silent omission, which is
    # what a bare truthiness test made it.
    model = exact_model(model)
    argv = [
        exact_launchable_text(binary, what="ikili dosya"), "exec",
        # A FIXED LITERAL, not a parameter (B4-R3). `codex exec` refuses
        # to start outside a git repository -- measured on 0.147.0-alpha:
        # exit 1 after 341ms with "Not inside a trusted directory and
        # --skip-git-repo-check was not specified", before the schema
        # file was ever read. The cwd here is the flat workspace's
        # implementer root, which holds tracked files and no `.git`
        # because it is a COPY rather than a clone.
        #
        # Nothing is loosened by skipping that check: the directory is
        # not a caller's path but one `flat_workspace.assert_binding`
        # derives from the recorded run, and `--sandbox read-only` still
        # bounds what the evaluator may do inside it. Git's notion of a
        # trusted directory is not the authority on this road, and there
        # is no keyword through which a caller, a task or a model could
        # remove this token or add a second one.
        "--skip-git-repo-check",
        "--sandbox", CODEX_SANDBOX_READ_ONLY,
        *CODEX_APPROVAL_OVERRIDE,
        "--cd", exact_launchable_text(repo, what="depo yolu"),
        "--output-schema", exact_launchable_text(schema_path,
                                                 what="sema dosyasi yolu"),
        "--output-last-message", exact_launchable_text(
            last_message_path, what="son mesaj dosyasi yolu"),
    ]
    if model is not None:
        argv += ["--model", model]
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
