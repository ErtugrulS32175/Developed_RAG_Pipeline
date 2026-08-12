"""The verified implementer change set. PACKAGE B2B-A / B2B-B2A2.

ONE mechanism: run exactly one already-safe B2A implementer call inside
a D3A FLAT WORKSPACE, then derive the change set from the FILESYSTEM
DIFFERENCE between the reference tree and the implementer tree and prove
every part of it was allowed. It runs no acceptance command, applies no
patch, touches no state machine and creates nothing in the main checkout.

THE MODEL'S `changed_files` IS A CLAIM, NOT EVIDENCE. It is compared
against what the filesystem shows, exactly and in both directions: a file
the model forgot to declare is as much a mismatch as one it invented.

WHY THE WORKSPACE'S EVIDENCE IS NOT GIT'S ANYMORE. Git held this job and
lost it twice, both times to state the model could reach: the
per-repository index, where `skip-worktree` hid a modified control-plane
file while `status` came back empty, and shared metadata, where a clean
filter declared in `.git/config` hid a change AND ran a model-supplied
command during verification. Neither was a missing check; the authority
itself was inside the blast radius. A D3A workspace has no `.git` at all,
so the question is asked of the two trees directly -- through the
handle-bound walker in `fs_evidence`, which the model cannot answer for.

WHY TWO TREES AND NOT A BASELINE DIGEST. "What did this call change" is a
question about a REFERENCE that never moved. The reference copy is
therefore read back on both sides of the call as well: if it moved, the
comparison is against something the model chose, and nothing derived from
it means anything.

WHY A SEMANTIC PROJECTION AND NOT `Manifest.digest`. Two independent
copies of one tree are SUPPOSED to differ -- different root identity,
different file identity, different timestamps, different scan counters --
so comparing the full digest would refuse every healthy workspace. What
must match is the semantic content, and that is what is compared.

WHERE GIT STILL RUNS, ON PURPOSE. The OPERATOR'S checkout is still
guarded by `git status` before and after the call. That instrument moves
to filesystem evidence in B2B-B2B; half-migrating it here would leave the
main tree guarded by neither. The rule this package enforces is therefore
not "no git" but "no git ABOUT THE FLAT ROOTS".

WHAT MAY LEAVE. Repo-relative paths that were ALLOWED, counts, closed
contract codes and hashes. Never file contents, never a patch, never
git's stderr, never a link's target, never a rejected path -- a refused
path is attacker influenced text and this travels into reports.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import (cli, contract, execution, flat_workspace,
                              fs_evidence, preflight, schemas,
                              state as state_module)

GIT_TIMEOUT_SECONDS = 120
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(contract.IDENTIFIER_PATTERN)
ADDED, MODIFIED, DELETED = "added", "modified", "deleted"

# The semantic fields two independent copies of one tree MUST agree on
# while nothing has changed. `mtime_ns`, `file_id` and the manifest's own
# root identity are per-copy BY DEFINITION and are deliberately absent:
# comparing any of them would call every healthy workspace a change.
_FILE_FIELDS = ("mode", "size", "sha256", "attributes", "reparse_tag",
                "nlink", "link_target_mac")
# A directory's size and link count are functions of what is INSIDE it
# rather than of the directory itself -- on POSIX both move when a file
# is created under them -- so including either would report one added
# file twice, once as the file and once as its parent, and refuse
# ordinary work. What is left still tells an added directory, a removed
# one, a type change and a permission change apart.
_DIR_FIELDS = ("mode", "attributes", "reparse_tag", "link_target_mac")


class ChangeSetError(RuntimeError):
    """A refusal carrying a CLOSED contract stop reason and fixed text."""

    def __init__(self, message, *, reason):
        super().__init__(message)
        self.reason = reason


class EvidenceUnavailable(ChangeSetError):
    """A question this gate depends on could not be answered.

    Never "clean": a git command that fails prints nothing, and an
    empty inventory read as "no changes" is how an unverifiable state
    becomes a verified one."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PREFLIGHT_FAILED)


class UnsafeChange(ChangeSetError):
    """The workspace, the manifest or the main checkout moved somewhere
    this run was not allowed to take it."""


class DeclarationMismatch(ChangeSetError):
    """The reply does not describe what actually happened."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.SCHEMA_VIOLATION)


@dataclass(frozen=True, slots=True)
class _Change:
    path: str        # canonical repo-relative POSIX
    kind: str        # ADDED | MODIFIED | DELETED
    mode: str        # the semantic record's own mode
    sha256: str      # of the current bytes; "" when the file is gone


@dataclass(frozen=True, slots=True)
class _Snap:
    """One projected entry: what it IS, and what it is COMPARED by."""

    kind: str
    mode: str
    sha256: str
    compare: tuple


@dataclass(frozen=True, slots=True)
class VerifiedChangeSet:
    """Bounded, structured, and free of anything a report may not
    carry: no patch, no contents, no absolute path."""

    run_id: str
    workspace_id: str
    baseline_sha: str
    status: str
    changed_files: tuple
    added: int
    modified: int
    deleted: int
    fingerprint: str          # full 64 hex; never truncated for equality
    exit_code: int
    duration_ms: int
    stdout_bytes: int
    stderr_bytes: int
    schema_sha256: str
    event: str


# ---------------------------------------------------------------------
# GIT EVIDENCE -- the MAIN CHECKOUT only, argv, checked, NUL-delimited
# ---------------------------------------------------------------------

def _git(cwd, *args) -> bytes:
    """Every git call is checked and every answer is BYTES.

    Text mode would decode with the ambient locale and split on
    newlines -- and a filename may contain a newline, which is exactly
    how a path walks out of an inventory unnoticed. The switches pin
    the answer to the repository state rather than to a cache, an
    external differ or a credential helper's opinion.

    `cwd` is the operator's repository and nothing else: no caller here
    passes a workspace root, which is what makes the boundary this
    package draws checkable from the outside."""
    argv = ["git", "-C", str(cwd), "--no-optional-locks",
            "-c", "core.fsmonitor=false", "-c", "core.quotepath=false",
            "-c", "diff.external=", "-c", "core.symlinks=true", *args]
    environment = dict(os.environ)
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "",
                        "GCM_INTERACTIVE": "never", "GIT_OPTIONAL_LOCKS": "0"})
    try:
        done = subprocess.run(argv, capture_output=True,
                              stdin=subprocess.DEVNULL, env=environment,
                              timeout=GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        # a timeout is an unanswered question, not an empty answer
        raise EvidenceUnavailable(f"git {args[0]} calistirilamadi") from None
    if done.returncode != 0:
        # the command NAME and the code, never stderr: that text carries
        # paths, remotes and credential-helper complaints
        raise EvidenceUnavailable(
            f"git {args[0]} kanit vermedi (rc={done.returncode})")
    return done.stdout


def _status(cwd):
    """The full inventory, as `(record, rename_source)` pairs.

    `--untracked-files=all` because a new file is the most ordinary
    thing a process writes and an ordinary diff never mentions it;
    `--ignore-submodules=none` because a silently skipped submodule is
    a change nobody sees; `--no-renames` so one edit is one record."""
    raw = _git(cwd, "status", "--porcelain=v2", "-z",
               "--untracked-files=all", "--ignore-submodules=none",
               "--no-renames")
    fields = raw.split(b"\0")
    records, index = [], 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        source = None
        if field[:2] == b"2 ":
            # porcelain v2 puts a rename's ORIGINAL path in the next
            # NUL-separated field; both paths matter, so both are kept
            if index < len(fields):
                source = fields[index]
                index += 1
        records.append((field, source))
    return records


def _head(cwd) -> str:
    return _git(cwd, "rev-parse", "HEAD").decode("ascii", "replace").strip()


def _index_state(cwd) -> str:
    """A digest of the index AND of git's per-entry flags.

    `status` is not the whole truth: an entry marked `skip-worktree` or
    `assume-unchanged` is one git has been TOLD to stop looking at, and
    a probe used exactly that to rewrite a protected file while
    `status` stayed empty and this gate reported nothing at all. Both
    listings were measured to be stable across ordinary edits and
    untracked additions, so comparing them costs no false refusals."""
    return hashlib.sha256(_git(cwd, "ls-files", "-s", "-z") + b"\0"
                          + _git(cwd, "ls-files", "-v", "-z")).hexdigest()


# ---------------------------------------------------------------------
# PATHS -- canonical, contained, segment-matched
# ---------------------------------------------------------------------

def _same_place(left, right) -> bool:
    """The repository's declared platform rule, not a new claim: fold
    case only where the filesystem folds it (Windows's standard
    assumption; a case-sensitive NTFS directory stays the already
    declared limit)."""
    left, right = str(left).replace("\\", "/"), str(right).replace("\\", "/")
    if os.name == "nt":
        left, right = left.casefold(), right.casefold()
    return left == right


def _fold(text) -> str:
    """ONE spelling for every scope comparison in this module.

    Three probes walked past a forbidden entry by spelling it
    differently -- `PIPELINE/GIZLI/`, `./pipeline/gizli/` and
    `pipeline//gizli/` all failed to match `pipeline/gizli/a.py`,
    because the comparisons were raw string operations and the manifest
    side was never normalised at all. Empty and `.` segments collapse,
    separators unify, and case folds exactly where the repository has
    already declared it folds.

    `..` is deliberately NOT resolved away: the frozen task schema
    refuses it in a path list, and silently collapsing it here would
    let `pipeline/../gizli` mean something this module invented."""
    parts = [part for part in str(text).replace("\\", "/").split("/")
             if part not in ("", ".")]
    joined = "/".join(parts)
    return joined.casefold() if os.name == "nt" else joined


def _covered(path: str, entries) -> bool:
    """Whole-segment matching on FOLDED spellings, the lesson
    `preflight._covered` already paid for: bare `startswith` let
    `pipeline/` cover `pipeline_private/` and one allowed file cover
    any sibling sharing its opening characters."""
    folded = _fold(path)
    for entry in entries or ():
        trimmed = _fold(entry)
        if trimmed and (folded == trimmed
                        or folded.startswith(trimmed + "/")):
            return True
    return False


def _is_control_plane(path: str) -> bool:
    """Exact prefixes AND the frozen family patterns, so a test file
    invented tomorrow is protected today. Both sides folded, so a
    differently-cased spelling cannot walk past on Windows."""
    folded = _fold(path)
    for prefix in contract.CONTROL_PLANE_PATHS:
        trimmed = _fold(prefix)
        if folded == trimmed or folded.startswith(trimmed + "/"):
            return True
    # `fnmatchcase` on already-folded text, so the case rule is this
    # module's declared one rather than fnmatch's platform guess
    return any(fnmatch.fnmatchcase(folded, _fold(pattern))
               for pattern in contract.CONTROL_PLANE_GLOBS)


def _authorize(path: str, *, allowed, forbidden, task_relative):
    """Authority order, and it is not negotiable by the manifest:
    control plane, then the task file itself, then forbidden, then
    allowed. A permission a task grants itself may never reach the code
    that supervises it."""
    if _is_control_plane(path):
        raise UnsafeChange("kontrol duzlemi degistirildi",
                           reason=contract.StopReason.CONTROL_PLANE_MODIFIED)
    if task_relative is not None and _fold(path) == _fold(task_relative):
        raise UnsafeChange("gorev dosyasi degistirildi",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    if _covered(path, forbidden) or not _covered(path, allowed):
        raise UnsafeChange("izin verilmeyen yol degistirildi",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)


# ---------------------------------------------------------------------
# FILESYSTEM EVIDENCE -- the two trees, projected and compared
# ---------------------------------------------------------------------

def _refuse_evidence(refused):
    """One boundary for the walker's refusals. Its messages are fixed
    sentences by that module's contract, so they may be repeated; its
    REASON says whether this was an unanswered question or a tree this
    evidence model does not represent."""
    if refused.reason == contract.StopReason.PATH_NOT_ALLOWED:
        return UnsafeChange(str(refused), reason=refused.reason)
    return EvidenceUnavailable(str(refused))


def _scan(root, key):
    try:
        return fs_evidence.scan(root, key=key, limits=fs_evidence.Limits())
    except fs_evidence.EvidenceError as refused:
        raise _refuse_evidence(refused) from None


def _project(manifest):
    """The semantic content of one tree, keyed by canonical path.

    ONE type gate, deliberately. A separate symlink branch in front of
    this looked like defence in depth and was not: the walker reports a
    symlink AND a junction as `kind == "link"` and gives both a keyed
    link fingerprint, so either test alone already refused everything
    the other did -- and a layer whose removal is invisible is a layer
    nobody is testing."""
    snapshot = {}
    for entry in manifest.entries:
        if (entry.kind not in ("file", "dir") or entry.reparse_tag
                or entry.link_target_mac):
            raise UnsafeChange("calisma alaninda temsil edilemeyen giris",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        if entry.path in snapshot:
            # a closed structure, stated rather than assumed: a mapping
            # would keep the last writer silently
            raise EvidenceUnavailable("ayni yol iki kez listelendi")
        alanlar = _FILE_FIELDS if entry.kind == "file" else _DIR_FIELDS
        snapshot[entry.path] = _Snap(
            kind=entry.kind, mode=entry.mode, sha256=entry.sha256,
            compare=(entry.kind,) + tuple(str(getattr(entry, alan))
                                          for alan in alanlar))
    return snapshot


def _read_pair(workspace, key):
    """Both trees, quiesced first and read with ONE call-local key.

    The key binds the link fingerprints to this call: the same key must
    be used for every scan here or two healthy manifests could never
    compare equal, and no other call can recognise them at all."""
    fs_evidence.quiesce()
    reference = _scan(workspace.reference_root, key)
    implementer = _scan(workspace.implementer_root, key)
    if reference.root_identity == implementer.root_identity:
        # the same directory twice is not two independent trees, and a
        # comparison against yourself always passes
        raise UnsafeChange("iki kok ayni nesne",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    return _project(reference), _project(implementer)


def _assert_structural(directories, changes):
    """A directory is the structural PARENT of a file, never a change of
    its own. One that appears or disappears with no file change under it
    is something this gate cannot carry into a patch, so it is refused
    rather than dropped."""
    for path in directories:
        prefix = path + "/"
        if not any(change.path.startswith(prefix) for change in changes):
            raise UnsafeChange("bos dizin degisikligi desteklenmiyor",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)


def _semantic_changes(before, after):
    """Two projections become the change set, or a refusal.

    Every unsupported shape is named: a path whose type changed, a bare
    directory-metadata edit, an empty directory appearing or vanishing.
    Guessing how to follow one of those is how an unreviewed edit
    reaches the main checkout."""
    changes, structural = [], []
    for path in sorted(set(before) | set(after)):
        left, right = before.get(path), after.get(path)
        if left is not None and right is not None \
                and left.compare == right.compare:
            continue
        if left is None:
            if right.kind == "file":
                changes.append(_Change(path=path, kind=ADDED,
                                       mode=right.mode, sha256=right.sha256))
            else:
                structural.append(path)
            continue
        if right is None:
            if left.kind == "file":
                # THE BASELINE MODE, not a blank. Recording every
                # deletion the same way erased the one field that told
                # two otherwise identical deletions apart.
                changes.append(_Change(path=path, kind=DELETED,
                                       mode=left.mode, sha256=""))
            else:
                structural.append(path)
            continue
        if left.kind != right.kind:
            raise UnsafeChange("yol turu degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        if right.kind != "file":
            raise UnsafeChange("dizin ust verisi degisikligi desteklenmiyor",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        changes.append(_Change(path=path, kind=MODIFIED, mode=right.mode,
                               sha256=right.sha256))
    _assert_structural(structural, changes)
    return tuple(sorted(changes, key=lambda item: item.path))


def _fingerprint(changes) -> str:
    """Path, kind, mode AND content, in one deterministic encoding.

    A path-only digest cannot tell two different edits to the same file
    apart, which is precisely the difference a later transfer step has
    to be able to see."""
    stream = b"".join(
        b"\0".join((change.path.encode("utf-8"), change.kind.encode("ascii"),
                    change.mode.encode("ascii"),
                    change.sha256.encode("ascii"))) + b"\n"
        for change in changes)
    return hashlib.sha256(stream).hexdigest()


# ---------------------------------------------------------------------
# SNAPSHOTS -- what must not have moved
# ---------------------------------------------------------------------

def _tree_snapshot(root: Path) -> str:
    """One digest over everything git can see about the OPERATOR'S
    checkout: HEAD, the index and its per-entry flags, the full
    inventory, and the bytes of every changed or untracked regular file
    it lists.

    Ignored content is deliberately NOT walked -- `data/` is the
    repository's own guard and the leak scanner's, and traversing a
    document tree to protect it is the wrong instrument."""
    digest = hashlib.sha256()
    digest.update(_head(root).encode("ascii"))
    digest.update(_index_state(root).encode("ascii"))
    for record, source in _status(root):
        digest.update(b"\0" + record + (source or b""))
        raw = record[2:] if record[:1] == b"?" else record.split(b" ", 8)[-1]
        if record[:1] not in (b"?", b"1"):
            continue
        candidate = root / raw.decode("utf-8", "replace")
        try:
            entry = os.lstat(candidate)
        except OSError:
            continue                    # a deletion has nothing to hash
        if stat.S_ISREG(entry.st_mode) and not getattr(
                entry, "st_reparse_tag", 0):
            try:
                payload = candidate.read_bytes()
            except OSError:
                # a file git listed but nobody can read is an
                # unanswered question, and the operating system's own
                # message is not text this module may repeat
                raise EvidenceUnavailable(
                    "anlik goruntudeki dosya okunamadi") from None
            digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


# ---------------------------------------------------------------------
# INPUT CANONICALIZATION
# ---------------------------------------------------------------------

def _exact_text(value, what: str) -> str:
    try:
        return cli.exact_text(value, what=what)
    except cli.UnsafeInvocation:
        raise EvidenceUnavailable(f"{what} tam bir metin degil") from None


def _exact_match(value, pattern, what: str) -> str:
    if type(value) is not str or not pattern.match(value):
        raise EvidenceUnavailable(f"{what} beklenen bicimde degil")
    return value


def _task_relative(task_path: Path, repo: Path) -> str:
    """The manifest's own path, normalised once, so it can be made
    immutable for this call even when the allowlist would cover it."""
    try:
        relative = task_path.resolve(strict=True).relative_to(
            repo.resolve(strict=True))
    except (OSError, ValueError):
        return None            # outside the repo: nothing in-tree to protect
    return "/".join(relative.parts)


def _bind_task(task_path, repo: Path, manifest_digest: str, baseline_sha: str):
    """The bytes that are hashed are the bytes that are parsed, and the
    hash has to be the one this run was issued.

    The manifest must live INSIDE the repository this run is bound to,
    and the baseline it names must be the baseline everything else
    names. A file outside the tree was accepted before -- so a run
    could take its permissions from a document the repository has no
    relationship with -- and its `baseline_sha` was never compared to
    anything at all."""
    path = Path(_exact_text(task_path, "gorev dosyasi yolu"))
    try:
        entry = os.lstat(path)
    except OSError:
        raise EvidenceUnavailable("gorev dosyasi yok") from None
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) \
            or getattr(entry, "st_reparse_tag", 0):
        raise EvidenceUnavailable("gorev dosyasi siradan bir dosya degil")
    try:
        snapshot = preflight.snapshot_manifest(path)
    except (OSError, ValueError):
        raise EvidenceUnavailable("gorev dosyasi okunamadi") from None
    if snapshot.digest != manifest_digest:
        raise EvidenceUnavailable("gorev dosyasi bu kosuya ait degil")
    try:
        Draft202012Validator(schemas.TASK_SCHEMA).validate(snapshot.task)
    except ValidationError:
        raise EvidenceUnavailable("gorev dosyasi sema disi") from None
    relative = _task_relative(path, repo)
    if relative is None:
        raise EvidenceUnavailable("gorev dosyasi depo agacinin disinda")
    if snapshot.task["baseline_sha"] != baseline_sha:
        raise UnsafeChange("gorev dosyasi baska bir taban surumu adliyor",
                           reason=contract.StopReason.BASELINE_MISMATCH)
    return path, snapshot, relative


def _assert_state_binding(state_dir: Path, repo: Path, run_id: str,
                          baseline_sha: str, manifest_digest: str,
                          workspace_id: str):
    """The recorded run, asked about THIS workspace.

    `workspace_id` is passed rather than left out: the binding schema
    lets exactly one execution identity exist, and `get` on the absent
    one answers `None` -- which would match a caller that asked about
    nothing at all."""
    try:
        binding = state_module.assert_binding(
            state_dir, repo_id=state_module.repo_identity(repo),
            baseline_sha=baseline_sha, manifest_digest=manifest_digest,
            workspace_id=workspace_id)
    except state_module.CorruptState:
        raise EvidenceUnavailable("kosu baglamasi yok ya da bozuk") from None
    except state_module.StateError:
        raise EvidenceUnavailable("kosu baglamasi bu cagriyla uyusmuyor") \
            from None
    if binding.get("run_id") != run_id:
        raise EvidenceUnavailable("kosu baglamasi bu cagriyla uyusmuyor")


def _bind_workspace(repo: Path, state_dir: Path, run_id: str,
                    workspace_id: str, baseline_sha: str):
    """The workspace, or a typed refusal. No path is accepted and none
    is derived here: both roots come from the object this returns."""
    try:
        return flat_workspace.assert_binding(
            repo, state_dir=state_dir, run_id=run_id,
            workspace_id=workspace_id, baseline_sha=baseline_sha)
    except flat_workspace.FlatWorkspaceError as refused:
        # the lower layer's sentences are fixed and carry no path; its
        # REASON is closed and is not re-decided here
        raise UnsafeChange(str(refused),
                           reason=getattr(refused, "reason",
                                          contract.StopReason.PREFLIGHT_FAILED)
                           ) from None


# ---------------------------------------------------------------------
# THE PUBLIC SEAM
# ---------------------------------------------------------------------

def run_verified_implementation(binary, *, repo, state_dir, task_path,
                                manifest_digest, run_id, workspace_id,
                                baseline_sha, prompt, budget_usd,
                                timeout_seconds, max_output_bytes,
                                model=None) -> VerifiedChangeSet:
    """One implementer call, then proof of what it changed.

    The caller passes IDENTITIES: no workspace path, no cwd, no
    allow/forbid list, no diff, no patch and no git result. The
    permissions come from the exact manifest bytes whose digest this
    run was issued, and the working directory comes from D3A's binding
    seam -- there is nothing here for a caller to substitute."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    state_path = Path(_exact_text(state_dir, "durum dizini"))
    manifest_digest = _exact_match(manifest_digest, _HEX64, "gorev ozeti")
    run_id = _exact_match(run_id, _RUN_ID, "kosu kimligi")
    workspace_id = _exact_match(workspace_id, flat_workspace.WORKSPACE_ID,
                                "calisma alani kimligi")
    baseline_sha = _exact_match(baseline_sha, _SHA1, "taban surum")

    task_file, snapshot, task_relative = _bind_task(
        task_path, repo_path, manifest_digest, baseline_sha)
    _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                          manifest_digest, workspace_id)
    allowed = tuple(snapshot.task["allowed_paths"])
    forbidden = tuple(snapshot.task.get("forbidden_paths", ()))

    workspace = _bind_workspace(repo_path, state_path, run_id, workspace_id,
                                baseline_sha)
    # ONE key for every scan in this call. It is never written to state,
    # never returned, never logged and never reused: it exists so the
    # four manifests can be compared at all, and a key that outlived the
    # call would be a key somebody could replay.
    key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    reference_before, implementer_before = _read_pair(workspace, key)
    if reference_before != implementer_before:
        # attribution is impossible when the two trees did not start
        # equal, so this refusal comes before a model process exists
        raise UnsafeChange("calisma alani basta ayrisik",
                           reason=contract.StopReason.DIRTY_WORKTREE)
    main_before = _tree_snapshot(repo_path)
    task_before = snapshot.digest

    def verify_after():
        """Everything that must NOT have moved, then the change set."""
        again = _bind_workspace(repo_path, state_path, run_id, workspace_id,
                                baseline_sha)
        _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                              manifest_digest, workspace_id)
        if preflight.manifest_changed(task_file, snapshot) or \
                preflight.snapshot_manifest(task_file).digest != task_before:
            raise UnsafeChange("gorev dosyasi degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        if _tree_snapshot(repo_path) != main_before:
            raise UnsafeChange("ana calisma agaci degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        reference_after, implementer_after = _read_pair(again, key)
        if reference_after != reference_before:
            # the copy the work is measured against moved, so no
            # difference derived from it describes this call
            raise UnsafeChange("referans agac degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        changes = _semantic_changes(reference_after, implementer_after)
        for change in changes:
            _authorize(change.path, allowed=allowed, forbidden=forbidden,
                       task_relative=task_relative)
        return changes

    failure = None
    try:
        outcome = execution.run_implementer(
            binary, repo=repo_path, state_dir=state_path, run_id=run_id,
            workspace_id=workspace_id, baseline_sha=baseline_sha,
            prompt=prompt, budget_usd=budget_usd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes, model=model)
    except BaseException as raised:
        failure = raised
        raise
    finally:
        # EVERY exit: a call that failed may have edited files first,
        # and a safety violation outranks the failure that hid it. The
        # original error is chained, never erased.
        try:
            actual = verify_after()
        except ChangeSetError as violation:
            raise violation from failure

    return _accept(outcome, actual, run_id=run_id, workspace_id=workspace_id,
                   baseline_sha=baseline_sha)


def _accept(outcome, actual, *, run_id, workspace_id, baseline_sha):
    """The declaration is compared to the evidence, exactly."""
    reply = outcome.reply
    if reply.get("run_id") != run_id:
        raise DeclarationMismatch("yanit baska bir kosuyu adliyor")
    declared = reply.get("changed_files", [])
    if any(type(item) is not str for item in declared):
        raise DeclarationMismatch("bildirilen dosya listesi tam metin degil")
    # CANONICAL spellings on both sides. Comparing raw text let the same
    # path arrive twice as `a/b.py` and `./a//b.py`, which is a duplicate
    # the set never saw.
    canonical = [_fold(item) for item in declared]
    if len(set(canonical)) != len(canonical):
        raise DeclarationMismatch("bildirilen dosya listesi yinelemeli")
    if set(canonical) != {_fold(change.path) for change in actual}:
        # BOTH directions: a forgotten file is as much a mismatch as an
        # invented one, and a subset comparison sees neither
        raise DeclarationMismatch("bildirilen degisiklikler gerceklesenlerle "
                                  "birebir ayni degil")
    status = reply["status"]
    if status != contract.Status.IMPLEMENTED and actual:
        raise DeclarationMismatch(
            "tamamlanmamis yanit yine de dosya degistirdi")
    kinds = [change.kind for change in actual]
    return VerifiedChangeSet(
        run_id=run_id, workspace_id=workspace_id, baseline_sha=baseline_sha,
        status=status,
        changed_files=tuple(change.path for change in actual),
        added=kinds.count(ADDED), modified=kinds.count(MODIFIED),
        deleted=kinds.count(DELETED), fingerprint=_fingerprint(actual),
        exit_code=outcome.exit_code, duration_ms=outcome.duration_ms,
        stdout_bytes=outcome.stdout_bytes, stderr_bytes=outcome.stderr_bytes,
        schema_sha256=outcome.schema_sha256, event=outcome.event)
