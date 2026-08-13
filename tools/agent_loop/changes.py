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

THE OPERATOR'S CHECKOUT IS READ THE SAME WAY (B2B-B2B). It used to be
guarded by `git status` on both sides of the call, and that guard had one
open limit its own tests could not close: an entry marked `skip-worktree`
or `assume-unchanged` BEFORE the call is one git was told to stop looking
at, so the inventory came back equally empty on both sides while the
bytes on disk moved. Nothing in this module asks git anything now.

NO ROOT IS SKIPPED. `.git`, `data`, `output`, `logs`, `uploads`,
`contracts` and any root nobody has thought of yet are content evidence.
The only reduction is the in-repository virtual environments, which are
metadata-only for the cost D2 measured -- and each keeps its `pyvenv.cfg`
as content, because that file says which interpreter runs. What the
reduced class cannot see is written down in `_main_policy`.

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
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import (cli, contract, execution, flat_workspace,
                              fs_evidence, preflight, schemas,
                              state as state_module)

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

# The one reduction the main-checkout policy makes, and the one file it
# always keeps inside a reduced root.
_ENV_SUFFIX = "_env"
_ENV_CONFIG = "pyvenv.cfg"


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


def _scan(root, key, *, metadata_only=(), content_always=()):
    """ONE walker seam for every question this module asks a filesystem.

    THE REFUSAL IS RAISED OUTSIDE THE HANDLER, deliberately. `raise ...
    from None` clears `__cause__` and sets the suppression flag, but
    `__context__` still holds the original object -- and an `OSError`'s
    text names absolute paths, which is precisely what must not ride out
    on an exception a report may print."""
    refusal = None
    try:
        return fs_evidence.scan(root, key=key, metadata_only=metadata_only,
                                content_always=content_always,
                                limits=fs_evidence.Limits())
    except fs_evidence.EvidenceError as refused:
        refusal = _refuse_evidence(refused)
    except OSError:
        refusal = EvidenceUnavailable("dosya sistemi kaniti alinamadi")
    raise refusal


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
# THE MAIN CHECKOUT -- one policy, one key, two reads
# ---------------------------------------------------------------------

def _main_policy(repo: Path):
    """Which roots of the OPERATOR'S checkout are metadata-only, decided
    ONCE and frozen for the call.

    A top-level directory whose name ends in `_env` is an in-repository
    virtual environment: 184,678 entries were measured across seven of
    them, and hashing that is 1.1 minutes warm per pass against 16
    seconds for the handle-bound metadata walk. Nothing else is reduced,
    and NO root is skipped -- `.git`, `data`, `output`, `logs`,
    `uploads`, `contracts` and any root nobody has thought of yet stay
    content evidence.

    A LINK IS NOT A DIRECTORY, and that is a MEASUREMENT rather than a
    precaution: on Windows `DirEntry.is_dir(follow_symlinks=False)`
    answers True for a junction (reparse tag 0x2000000B) and False for a
    symlink. Asking only `is_dir` would let a junction planted as
    `something_env` hand an entire outside tree the reduced class, so
    the reparse tag is checked and that check is doing real work.

    Frozen BEFORE the model runs, so a root created during the call
    cannot claim the reduction for itself -- it simply is not in the
    policy, and everything outside the policy is content."""
    refusal = None
    metadata_only, content_always = [], []
    try:
        with os.scandir(repo) as entries:
            for entry in entries:
                if not entry.name.endswith(_ENV_SUFFIX):
                    continue
                if not entry.is_dir(follow_symlinks=False) or getattr(
                        entry.stat(follow_symlinks=False),
                        "st_reparse_tag", 0):
                    continue
                metadata_only.append(entry.name)
                # EXACT, never a pattern: `pyvenv.cfg.bak` beside it and
                # a namesake one level down are ordinary environment
                # files, and `fs_evidence` compares this against the
                # canonical path it built itself
                content_always.append(f"{entry.name}/{_ENV_CONFIG}")
    except OSError:
        # the name of the directory that refused is not text this module
        # may repeat, and the refusal leaves the handler before it flies
        refusal = EvidenceUnavailable("ana calisma agaci envanteri alinamadi")
    if refusal is not None:
        raise refusal
    return tuple(sorted(metadata_only)), tuple(sorted(content_always))


@dataclass(frozen=True, slots=True)
class _MainSnapshot:
    """What the operator's checkout WAS.

    Both halves matter and neither replaces the other: the identity says
    the root is still the same OBJECT, and the digest says its contents
    are still the same. Unlike the two flat trees -- independent copies
    whose per-copy fields must differ -- this is one root read twice, so
    the walker's full digest is exactly the right comparison and no
    semantic projection is wanted here.

    The scan's counters, duration and peak-directory figures are NOT in
    it: those measure the walk, not the tree."""

    root_identity: str
    digest: str


def _main_snapshot(repo: Path, key: bytes, policy) -> _MainSnapshot:
    """One read of the operator's checkout, under the call's own key."""
    metadata_only, content_always = policy
    fs_evidence.quiesce()
    manifest = _scan(repo, key, metadata_only=metadata_only,
                     content_always=content_always)
    return _MainSnapshot(root_identity=manifest.root_identity,
                         digest=manifest.digest)


# ---------------------------------------------------------------------
# THE MAIN CHECKOUT, AS SEMANTIC CONTENT (B2B-C2)
# ---------------------------------------------------------------------
#
# `_main_snapshot` answers "did anything move", which is the only
# question an implementer call has: the answer must be NO. A step that
# deliberately changes the checkout needs the other question -- WHAT
# moved -- and that is the semantic projection, which this module already
# owns for the flat trees. These three names expose that authority
# instead of letting a second one grow next door; nothing below is a new
# rule, and every one of them routes into the same walker, the same
# projection and the same classifier.

Change = _Change


@dataclass(frozen=True, slots=True)
class MainProjection:
    """The operator's checkout as content, frozen so two of them can be
    compared and neither can be edited between the comparisons."""

    root_identity: str
    entries: tuple


def freeze_main_policy(repo):
    """The reduced-evidence policy, decided ONCE for a call.

    Frozen before anything is written, so a root created DURING the call
    cannot claim the reduction for itself -- it simply is not in the
    policy, and everything outside the policy is content."""
    return _main_policy(Path(_exact_text(repo, "depo yolu")))


def _project_main(manifest):
    """Every entry, including the ones a change set cannot carry.

    Unlike the workspace projection this does NOT refuse a link: an
    operator's checkout is allowed to contain one, and refusing here
    would make an ordinary repository unusable for no safety gain. What
    it must not do is let one move invisibly -- so a link is projected by
    its keyed fingerprint, any movement shows up as a difference, and
    `_semantic_changes` is what then refuses to describe it as an
    ordinary file change."""
    snapshot = {}
    for entry in manifest.entries:
        if entry.path in snapshot:
            raise EvidenceUnavailable("ayni yol iki kez listelendi")
        fields = _FILE_FIELDS if entry.kind == "file" else _DIR_FIELDS
        snapshot[entry.path] = _Snap(
            kind=entry.kind, mode=entry.mode, sha256=entry.sha256,
            compare=(entry.kind,) + tuple(str(getattr(entry, field))
                                          for field in fields))
    return snapshot


def main_projection(repo, *, key, policy) -> MainProjection:
    """One read of the operator's checkout, as comparable content."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    metadata_only, content_always = policy
    fs_evidence.quiesce()
    manifest = _scan(repo_path, key, metadata_only=metadata_only,
                     content_always=content_always)
    return MainProjection(
        root_identity=manifest.root_identity,
        entries=tuple(sorted(_project_main(manifest).items())))


def main_difference(before: MainProjection, after: MainProjection) -> tuple:
    """What changed between two readings of ONE root.

    The root identity is compared first and separately: two projections
    of two different objects can be diffed perfectly and describe
    nothing, and a checkout swapped mid-call is exactly that."""
    if type(before) is not MainProjection or type(after) is not MainProjection:
        raise EvidenceUnavailable("ana agac izdusumu beklenen turde degil")
    if before.root_identity != after.root_identity:
        raise UnsafeChange("ana calisma agaci kokunun kimligi degisti",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    return _semantic_changes(dict(before.entries), dict(after.entries))


def fingerprint(items) -> str:
    """The change set's own deterministic digest, over path, kind, mode
    AND content -- so two different edits to one file cannot collide."""
    return _fingerprint(items)


def canonical_path(text) -> str:
    """ONE spelling for every scope comparison anything in this loop
    makes. Exposed rather than copied: a second normaliser is a second
    opinion, and three probes have already walked past a forbidden entry
    by spelling it differently."""
    return _fold(text)


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
    # A SECOND call-local key, for a second question. The main checkout
    # and the flat roots are compared against different things, and one
    # key that answered for both would be one key worth replaying twice.
    main_policy = _main_policy(repo_path)
    main_key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    main_before = _main_snapshot(repo_path, main_key, main_policy)
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
        if _main_snapshot(repo_path, main_key, main_policy) != main_before:
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


def _exact_path_set(paths, what: str) -> frozenset:
    """A caller's declared path list, canonicalised the module's own way.

    Exact `str` on every item: a path list is compared against evidence,
    and an object that answers `__eq__` generously would agree with a
    set it had never been shown."""
    if isinstance(paths, (str, bytes)) or not hasattr(paths, "__iter__"):
        raise EvidenceUnavailable(f"{what} bir yol listesi degil")
    items = list(paths)
    if any(type(item) is not str for item in items):
        raise EvidenceUnavailable(f"{what} tam metin olmayan bir oge tasiyor")
    folded = [_fold(item) for item in items]
    if len(set(folded)) != len(folded):
        raise EvidenceUnavailable(f"{what} yinelemeli")
    return frozenset(folded)


def _accept_repair(outcome, delta, total, *, run_id, workspace_id,
                   baseline_sha):
    """Two different questions, deliberately asked of two different sets.

    THE DECLARATION IS ABOUT THIS CALL. A repair round's model edits some
    files and says so; it knows nothing about the round before it. So the
    declaration is compared against the DELTA -- what moved between the
    tree this call started with and the tree it ended with -- because
    comparing it against the cumulative set would refuse every honest
    repair that left an earlier file alone.

    THE RESULT IS ABOUT THE CANDIDATE. What a later step applies is not
    this call's delta, it is everything that separates the reference tree
    from the final tree. A repair that reverts a file to its baseline
    content is a DELETION from the candidate even though the model
    described it as an edit, and a first-round change the repair never
    touched is still in the candidate even though this call did not
    mention it. Returning the delta here is how a partially-applied
    candidate reaches the operator's checkout."""
    reply = outcome.reply
    if reply.get("run_id") != run_id:
        raise DeclarationMismatch("yanit baska bir kosuyu adliyor")
    declared = reply.get("changed_files", [])
    if any(type(item) is not str for item in declared):
        raise DeclarationMismatch("bildirilen dosya listesi tam metin degil")
    canonical = [_fold(item) for item in declared]
    if len(set(canonical)) != len(canonical):
        raise DeclarationMismatch("bildirilen dosya listesi yinelemeli")
    if set(canonical) != {_fold(change.path) for change in delta}:
        raise DeclarationMismatch("bildirilen onarim gerceklesenle "
                                  "birebir ayni degil")
    status = reply["status"]
    if status != contract.Status.IMPLEMENTED and delta:
        raise DeclarationMismatch(
            "tamamlanmamis yanit yine de dosya degistirdi")
    kinds = [change.kind for change in total]
    return VerifiedChangeSet(
        run_id=run_id, workspace_id=workspace_id, baseline_sha=baseline_sha,
        status=status,
        changed_files=tuple(change.path for change in total),
        added=kinds.count(ADDED), modified=kinds.count(MODIFIED),
        deleted=kinds.count(DELETED), fingerprint=_fingerprint(total),
        exit_code=outcome.exit_code, duration_ms=outcome.duration_ms,
        stdout_bytes=outcome.stdout_bytes, stderr_bytes=outcome.stderr_bytes,
        schema_sha256=outcome.schema_sha256, event=outcome.event)


def run_verified_repair(binary, *, repo, state_dir, task_path,
                        manifest_digest, run_id, workspace_id, baseline_sha,
                        previous_fingerprint, previous_changed_files,
                        prompt, budget_usd, timeout_seconds,
                        max_output_bytes, model=None) -> VerifiedChangeSet:
    """One repair call against an ALREADY-AUDITED candidate.

    WHY THIS IS NOT `run_verified_implementation`. That seam opens by
    requiring the reference and implementer trees to be EQUAL, which is
    what makes "attribution is possible" true for a first round: anything
    that differs afterwards was done by the call. A repair round starts
    from a tree that is deliberately NOT equal -- the first round's work
    is sitting in it -- so that gate would refuse every repair, and
    deleting it from the shared seam would remove the first round's only
    attribution guarantee. The two rounds ask different questions and get
    different functions; neither is a relaxation of the other.

    WHAT REPLACES THE EQUALITY GATE. The candidate on disk has to be the
    candidate the evaluator actually audited. It is re-derived from fresh
    filesystem evidence -- never replayed from a record -- and bound to
    the previous round by fingerprint AND by changed-file set, on top of
    the workspace, state, manifest and baseline identities every other
    seam here already checks. A repair aimed at a workspace whose
    contents moved since the audit is a repair of something nobody
    reviewed.

    WHAT COMES BACK. The FULL change set, reference tree to final tree --
    not this call's delta. See `_accept_repair` for why those are two
    different sets and why returning the wrong one applies half a
    candidate.

    The implementer is reached through `execution.run_implementer` HERE
    and nowhere else: the runner has no path to it, so every model edit
    in the loop passes one authorisation."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    state_path = Path(_exact_text(state_dir, "durum dizini"))
    manifest_digest = _exact_match(manifest_digest, _HEX64, "gorev ozeti")
    run_id = _exact_match(run_id, _RUN_ID, "kosu kimligi")
    workspace_id = _exact_match(workspace_id, flat_workspace.WORKSPACE_ID,
                                "calisma alani kimligi")
    baseline_sha = _exact_match(baseline_sha, _SHA1, "taban surum")
    previous_fingerprint = _exact_match(previous_fingerprint, _HEX64,
                                        "onceki aday parmak izi")
    expected_paths = _exact_path_set(previous_changed_files,
                                     "onceki aday dosya listesi")

    task_file, snapshot, task_relative = _bind_task(
        task_path, repo_path, manifest_digest, baseline_sha)
    _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                          manifest_digest, workspace_id)
    allowed = tuple(snapshot.task["allowed_paths"])
    forbidden = tuple(snapshot.task.get("forbidden_paths", ()))

    workspace = _bind_workspace(repo_path, state_path, run_id, workspace_id,
                                baseline_sha)
    key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    reference_before, implementer_before = _read_pair(workspace, key)
    prior = _semantic_changes(reference_before, implementer_before)
    # The candidate standing on disk must be authorised in its own right
    # before it is repaired: a workspace that already holds a forbidden
    # path is not made acceptable by editing something else in it.
    for change in prior:
        _authorize(change.path, allowed=allowed, forbidden=forbidden,
                   task_relative=task_relative)
    # TWO gates with DISTINCT sentences, not one asked twice. The
    # fingerprint covers path, kind, mode and content, so the file set is
    # implied by it -- but a test that can only see one message cannot
    # tell which gate refused, and a mechanism proven through the wrong
    # door is a mechanism nobody is testing.
    if _fingerprint(prior) != previous_fingerprint:
        raise UnsafeChange("onarim denetlenen adaya baglanamadi",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    if {_fold(change.path) for change in prior} != expected_paths:
        raise UnsafeChange("onarim baska bir dosya kumesini adliyor",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)

    main_policy = _main_policy(repo_path)
    main_key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    main_before = _main_snapshot(repo_path, main_key, main_policy)
    task_before = snapshot.digest

    def verify_after():
        """Everything that must NOT have moved, then both change sets."""
        again = _bind_workspace(repo_path, state_path, run_id, workspace_id,
                                baseline_sha)
        _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                              manifest_digest, workspace_id)
        if preflight.manifest_changed(task_file, snapshot) or \
                preflight.snapshot_manifest(task_file).digest != task_before:
            raise UnsafeChange("gorev dosyasi degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        if _main_snapshot(repo_path, main_key, main_policy) != main_before:
            raise UnsafeChange("ana calisma agaci degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        reference_after, implementer_after = _read_pair(again, key)
        if reference_after != reference_before:
            # the copy the candidate is measured against moved, so the
            # cumulative set below would describe something the model
            # chose rather than what it did
            raise UnsafeChange("referans agac degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        total = _semantic_changes(reference_after, implementer_after)
        for change in total:
            _authorize(change.path, allowed=allowed, forbidden=forbidden,
                       task_relative=task_relative)
        return _semantic_changes(implementer_before, implementer_after), total

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
        # EVERY exit: a call that failed may have edited files first, and
        # a safety violation outranks the failure that hid it.
        try:
            delta, total = verify_after()
        except ChangeSetError as violation:
            raise violation from failure

    return _accept_repair(outcome, delta, total, run_id=run_id,
                          workspace_id=workspace_id, baseline_sha=baseline_sha)


@dataclass(frozen=True, slots=True)
class CandidateChanges:
    """The candidate, RE-DERIVED from the filesystem AFTER the fact.

    Everything a `VerifiedChangeSet` says about what changed, derived a
    second time from a fresh read of the same two trees -- plus the
    per-entry records a later step needs to reproduce the candidate
    somewhere else, and the acceptance commands the exact manifest bytes
    name. No path, no bytes, no patch."""

    run_id: str
    workspace_id: str
    baseline_sha: str
    changed_files: tuple
    added: int
    modified: int
    deleted: int
    fingerprint: str
    task_digest: str
    acceptance_commands: tuple
    changes: tuple


def derive_candidate_changes(*, repo, state_dir, task_path, manifest_digest,
                             run_id, workspace_id, baseline_sha
                             ) -> CandidateChanges:
    """The change set, derived AGAIN from fresh filesystem evidence.

    WHY THIS SEAM EXISTS. A `VerifiedChangeSet` is a statement about a
    moment that has ended: the implementer call returned, and between
    that return and whatever happens next the workspace is an ordinary
    directory. Anything that acts on the candidate -- running its tests,
    copying it somewhere -- has to ask the filesystem again rather than
    replay a record, and it must ask through the SAME authority, or the
    two derivations are two different opinions.

    So this is `run_verified_implementation`'s verification half with no
    model call in front of it: the same manifest binding, the same state
    and workspace bindings, the same walker, the same classifier and the
    same authorization order. Nothing here is a second implementation of
    any of them."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    state_path = Path(_exact_text(state_dir, "durum dizini"))
    manifest_digest = _exact_match(manifest_digest, _HEX64, "gorev ozeti")
    run_id = _exact_match(run_id, _RUN_ID, "kosu kimligi")
    workspace_id = _exact_match(workspace_id, flat_workspace.WORKSPACE_ID,
                                "calisma alani kimligi")
    baseline_sha = _exact_match(baseline_sha, _SHA1, "taban surum")

    _, snapshot, task_relative = _bind_task(
        task_path, repo_path, manifest_digest, baseline_sha)
    _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                          manifest_digest, workspace_id)
    workspace = _bind_workspace(repo_path, state_path, run_id, workspace_id,
                                baseline_sha)
    # ONE call-local key, exactly as the implementer path uses: it exists
    # so the two manifests can be compared at all, and a key that
    # outlived the call would be a key somebody could replay.
    key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    reference, implementer = _read_pair(workspace, key)
    actual = _semantic_changes(reference, implementer)
    allowed = tuple(snapshot.task["allowed_paths"])
    forbidden = tuple(snapshot.task.get("forbidden_paths", ()))
    for change in actual:
        _authorize(change.path, allowed=allowed, forbidden=forbidden,
                   task_relative=task_relative)
    kinds = [change.kind for change in actual]
    return CandidateChanges(
        run_id=run_id, workspace_id=workspace_id, baseline_sha=baseline_sha,
        changed_files=tuple(change.path for change in actual),
        added=kinds.count(ADDED), modified=kinds.count(MODIFIED),
        deleted=kinds.count(DELETED), fingerprint=_fingerprint(actual),
        task_digest=snapshot.digest,
        acceptance_commands=tuple(
            (reference_item["command_id"],
             tuple(reference_item.get("paths", ())))
            for reference_item in snapshot.task["acceptance_commands"]),
        changes=actual)


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
