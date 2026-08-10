"""Preflight. PACKAGE B1.

Every check here runs BEFORE a worktree exists and before any model
could be called. A preflight failure must cost nothing: a model invoked
before the tree was checked has already been paid for, and a worktree
created before the manifest was validated is a directory nobody
recorded.

WHAT A FAILURE MAY SAY. A stop reason from the frozen `StopReason`
vocabulary and a short, fixed explanation. Never an absolute path, never
a captured exception message, never a fragment of the manifest -- those
are the three routes by which private material reaches a report that
gets forwarded.

THE MANIFEST IS READ ONCE. The bytes that are hashed are the bytes that
are parsed, from a single read. Reading twice -- once to parse, once to
digest -- means the digest can describe a file the run never saw, which
is exactly the binding this is supposed to establish.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import contract, schemas, state as state_module


@dataclass(frozen=True)
class ManifestSnapshot:
    """One read: the parsed task and the digest of the very same bytes."""

    task: dict
    digest: str
    size: int


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    stop_reason: str | None = None
    detail: str = ""
    run_id: str | None = None
    repo_id: str | None = None
    baseline_sha: str | None = None
    manifest: ManifestSnapshot | None = field(default=None, repr=False)


def snapshot_manifest(path) -> ManifestSnapshot:
    """Read the task manifest ONCE, hash those exact bytes, parse them.

    A second read to compute the digest could pick up a different file:
    the window is small and the consequence is a run bound to a manifest
    it never validated."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    task = json.loads(raw.decode("utf-8"))
    return ManifestSnapshot(task=task, digest=digest, size=len(raw))


def manifest_changed(path, snapshot: ManifestSnapshot) -> bool:
    """Re-verified at every later phase boundary.

    Byte comparison, not semantic: the same JSON re-serialised with
    different whitespace is a file somebody edited, and the run was
    bound to what it actually read."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return True
    return hashlib.sha256(raw).hexdigest() != snapshot.digest


def _fail(stop_reason, detail=""):
    return PreflightResult(ok=False, stop_reason=stop_reason, detail=detail)


# A version query: no model call, no tokens, no cost, and it exits by
# itself. Fixed here rather than taken from the manifest, because an
# argument a task could choose is an argument a task could turn into a
# real invocation.
HANDSHAKE_ARGV = ("--version",)
HANDSHAKE_TIMEOUT_SECONDS = 20

def control_plane_pathspecs():
    """Exact prefixes AND the family patterns, both from the contract.

    The pattern lived here as a local constant while the contract still
    listed the Phase A test file one by one -- a widening the
    implementer had made up, which is the wrong place for a normative
    definition even when the definition is right. `CONTROL_PLANE_GLOBS`
    is now part of the frozen contract, so there is one source."""
    return tuple(contract.CONTROL_PLANE_PATHS) + tuple(
        contract.CONTROL_PLANE_GLOBS)


class GitUnavailable(RuntimeError):
    """A git command this gate depends on did not succeed."""


def _git(repo, *args):
    """Every git call is CHECKED.

    The safety gates read `git diff --cached` and `git status` and
    trusted the text that came back without ever looking at the exit
    code. Feeding both commands a failure and an empty stdout -- which
    is what a broken index or a contended lock produces -- made
    preflight report "nothing staged, tree clean" and pass. "We could
    not verify" is not "verified clean", and the only way to keep the
    two apart is to make the failure impossible to ignore."""
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, errors="replace")
    if done.returncode != 0:
        # the command NAME and the code, never stderr: that text can
        # carry a path, a remote or a credential helper's complaint
        raise GitUnavailable(f"git {args[0]} (rc={done.returncode})")
    return done


def _touches_control_plane(entries):
    """Does any path in this manifest list reach the loop's own code?

    ONE rule applied to every path list the manifest carries.
    `allowed_paths` had it and `dirty_tree_allowlist` did not, so the
    allowlist could name `tools/agent_loop` and launder a modified
    control plane through the dirty-tree gate."""
    blocked = contract.CONTROL_PLANE_BLOCKED_PATHS | set(
        contract.CONTROL_PLANE_PATHS)
    for entry in entries or []:
        normalised = str(entry).replace("\\", "/").rstrip("/")
        if normalised in blocked or normalised + "/" in blocked:
            return True
        if any(normalised == prefix.rstrip("/")
               or normalised.startswith(prefix)
               for prefix in contract.CONTROL_PLANE_PATHS):
            return True
        # the whole test family, not just the names listed today
        if any(fnmatch.fnmatch(normalised, pattern)
               for pattern in contract.CONTROL_PLANE_GLOBS):
            return True
    return False


def _control_plane_changes(repo_path):
    """What git says about the loop's own files, right now.

    THE authority, and deliberately not derived from the dirty-tree
    machinery: the allowlist is something the task manifest grants
    itself, and no permission a task can write should be able to make
    the code that supervises it look clean. Asking git directly means
    the answer does not depend on any manifest field at all.

    Hashing the control plane later would not help either -- the run
    would simply take the tampered files as its reference."""
    changed = []
    for entry in control_plane_pathspecs():
        out = _git(repo_path, "status", "--porcelain", "--", entry).stdout
        changed.extend(line[3:].strip() for line in out.splitlines()
                       if line.strip())
    return changed


def _covered(path, allowlist):
    """Allowlist entries match whole path SEGMENTS.

    `startswith` alone let one permitted file cover any sibling that
    happened to share its opening characters: allowing `notlar.md` also
    allowed `notlar.md.ozel`, so an uncommitted file nobody approved
    made it through the dirty-tree gate by prefix accident."""
    return any(path == entry or path.startswith(entry + "/")
               for entry in allowlist)


def _is_runnable(binary):
    """Can this be launched? Answered by LAUNCHING IT.

    Fourth attempt, and the first that is not a guess about file
    formats. The three before it each accepted something that could not
    actually start: `Path.exists()` accepted a directory; adding
    `is_file()` left a Windows branch that accepted any regular file, so
    a `.txt` passed and died with WinError 193; reading PATHEXT accepted
    a text file NAMED `.exe`, which died with WinError 216.

    Every one of those was a proxy for the real question, and each fix
    was a better proxy that a fourth input walked around. So the proxy
    is gone: the binary is started with a fixed, harmless argument and a
    timeout. `--version` makes no model call, spends no tokens and costs
    nothing, and a file that cannot be executed raises OSError here
    instead of at the first paid call.

    The EXIT CODE is not consulted -- a CLI may exit non-zero on an
    argument it does not know, and that is still a program that ran. The
    question is whether the operating system could start it.

    This launches only the path the caller supplied. It cannot discover
    one, which is what the contract's no-discovery rule protects."""
    if not binary:
        return False
    path = Path(binary)
    if not path.is_file():
        return False
    try:
        subprocess.run([str(path), *HANDSHAKE_ARGV], capture_output=True,
                       timeout=HANDSHAKE_TIMEOUT_SECONDS,
                       stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        # cannot start, or would not finish: both are "not a binary this
        # run can use", and both are cheaper to learn here
        return False
    return True


def run_preflight(task_path, *, repo, binaries, state_dir=None,
                  dirty_allowlist=(), allow_resume=False):
    """Every gate, in the order that keeps a failure free.

    Repository first (nothing else is meaningful without it), then the
    manifest, then the things the manifest claims, then the world it
    needs. The worktree is NOT created here -- that belongs to the
    caller, after this returns ok.

    Every git call inside can raise `GitUnavailable`, and that is caught
    once here rather than at each site: a gate whose evidence did not
    arrive fails, it does not proceed on the empty string."""
    try:
        return _gates(task_path, repo=repo, binaries=binaries,
                      state_dir=state_dir, dirty_allowlist=dirty_allowlist,
                      allow_resume=allow_resume)
    except GitUnavailable as unavailable:
        return _fail(contract.StopReason.PREFLIGHT_FAILED,
                     f"depo durumu dogrulanamadi: {unavailable}")


def _gates(task_path, *, repo, binaries, state_dir, dirty_allowlist,
           allow_resume):
    repo_path = Path(repo)
    if not (repo_path / ".git").exists():
        return _fail(contract.StopReason.PREFLIGHT_FAILED, "depo yok")

    try:
        head = _git(repo_path, "rev-parse", "HEAD")
    except GitUnavailable:
        return _fail(contract.StopReason.PREFLIGHT_FAILED, "HEAD cozulemedi")
    if not head.stdout.strip():
        return _fail(contract.StopReason.PREFLIGHT_FAILED, "HEAD cozulemedi")
    baseline = head.stdout.strip()

    # BEFORE the manifest is even read. This gate asks git about the
    # loop's own files and consults nothing the task provides, so no
    # field a task can write -- including its dirty-tree allowlist --
    # participates in the answer.
    tampered = _control_plane_changes(repo_path)
    if tampered:
        return _fail(contract.StopReason.CONTROL_PLANE_MODIFIED,
                     f"{len(tampered)} kontrol duzlemi dosyasi HEAD'de degil")

    manifest_file = Path(task_path)
    if not manifest_file.is_file():
        return _fail(contract.StopReason.PREFLIGHT_FAILED, "gorev dosyasi yok")
    try:
        snapshot = snapshot_manifest(manifest_file)
    except (OSError, ValueError):
        return _fail(contract.StopReason.PREFLIGHT_FAILED,
                     "gorev dosyasi okunamadi ya da JSON degil")
    try:
        Draft202012Validator(schemas.TASK_SCHEMA).validate(snapshot.task)
    except ValidationError as invalid:
        # WHICH field the schema refused decides the stop reason. The
        # frozen TASK_SCHEMA already rejects a control-plane path, so it
        # fires before the explicit check below ever runs -- and
        # reporting that as a generic "schema" failure would hide the
        # one thing the operator needs to know. The refusal is read off
        # the error's own field path, not guessed from the payload.
        where = "/".join(str(p) for p in invalid.absolute_path) or "kok"
        if where.split("/")[0] == "allowed_paths":
            return _fail(contract.StopReason.PATH_NOT_ALLOWED,
                         "izinli yollar kontrol duzlemine dokunuyor")
        return _fail(contract.StopReason.PREFLIGHT_FAILED,
                     f"gorev dosyasi sema disi (alan: {where})")

    # EVERY path list the manifest carries, through one rule. The
    # schema only constrains `allowed_paths`, and `dirty_tree_allowlist`
    # was never checked at all -- so a manifest could not ASK to edit
    # the control plane but could quietly excuse an already-modified
    # one. (The gate above is what actually guarantees this; refusing
    # the manifest as well keeps the operator from writing a permission
    # that silently means nothing.)
    for field_name in ("allowed_paths", "dirty_tree_allowlist"):
        if _touches_control_plane(snapshot.task.get(field_name)):
            return _fail(contract.StopReason.PATH_NOT_ALLOWED,
                         f"{field_name} kontrol duzlemine dokunuyor")
    if _touches_control_plane(dirty_allowlist):
        return _fail(contract.StopReason.PATH_NOT_ALLOWED,
                     "dirty_allowlist kontrol duzlemine dokunuyor")

    if snapshot.task["baseline_sha"] != baseline:
        return _fail(contract.StopReason.BASELINE_MISMATCH,
                     "HEAD gorev dosyasindaki baseline degil")

    staged = _git(repo_path, "diff", "--cached", "--name-only").stdout.split()
    if staged:
        return _fail(contract.StopReason.STAGED_CHANGES,
                     f"{len(staged)} dosya stage edilmis")

    allowlist = {str(p).replace("\\", "/").rstrip("/")
                 for p in (dirty_allowlist
                           or snapshot.task.get("dirty_tree_allowlist", []))}
    dirty = [line[3:].replace("\\", "/") for line in
             _git(repo_path, "status", "--porcelain").stdout.splitlines()
             if line.strip()]
    unexpected = [p for p in dirty if not _covered(p, allowlist)]
    if unexpected:
        return _fail(contract.StopReason.DIRTY_WORKTREE,
                     f"{len(unexpected)} beklenmeyen degisiklik")

    missing = [role for role, binary in (binaries or {}).items()
               if not _is_runnable(binary)]
    if missing or not binaries:
        return _fail(contract.StopReason.PREFLIGHT_FAILED,
                     f"eksik ikili dosya: {sorted(missing) or ['hepsi']}")

    limits = (("max_implementation_rounds", 1), ("max_repair_rounds", 1))
    for field_name, ceiling in limits:
        if snapshot.task[field_name] > ceiling:
            return _fail(contract.StopReason.PREFLIGHT_FAILED,
                         f"{field_name} sozlesme sinirini asiyor")

    repo_id = state_module.repo_identity(repo_path)
    if state_dir is not None:
        # UNCONDITIONALLY, not "if a binding happens to be there". The
        # previous form only looked when `binding.json` existed, so
        # removing that one file made any leftover state acceptable.
        try:
            state_module.assert_state_directory(
                state_dir, repo_id=repo_id, baseline_sha=baseline,
                manifest_digest=snapshot.digest, allow_resume=allow_resume)
        except state_module.StateError:
            return _fail(contract.StopReason.PREFLIGHT_FAILED,
                         "mevcut durum bu kosuya ait degil ya da eksik")

    return PreflightResult(ok=True, run_id=None, repo_id=repo_id,
                           baseline_sha=baseline, manifest=snapshot)
