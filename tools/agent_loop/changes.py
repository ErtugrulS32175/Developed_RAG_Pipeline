"""The verified implementer change set. PACKAGE B2B-A.

ONE mechanism: run exactly one already-safe B2A implementer call, then
derive the change set from GIT AND THE FILESYSTEM and prove every part
of it was allowed. It runs no acceptance command, applies no patch,
touches no state machine and creates nothing in the main checkout.

THE MODEL'S `changed_files` IS A CLAIM, NOT EVIDENCE. It is compared
against what git reports, exactly and in both directions: a file the
model forgot to declare is as much a mismatch as one it invented.

WHY GIT AFTER THE CALL IS NOT ENOUGH BY ITSELF. "Ask git when the model
returns" answers a different question than "what did this call change":

  * a worktree that was ALREADY dirty makes attribution impossible, so
    the pristine check happens before a process exists;
  * `git diff` does not mention untracked files at all, and a new file
    is the most ordinary thing an implementer writes -- so the
    inventory is `status --porcelain=v2 --untracked-files=all`;
  * HEAD, the index or the `.git` link can move during the call, and
    each of those changes what "changed" even means;
  * a call that FAILED may still have edited files first, so the
    verification runs in `finally` on every exit path;
  * a control-plane edit stays terminal no matter what the task
    manifest permits -- permission a task writes for itself may not
    reach the code that supervises it.

PATH-ONLY EVIDENCE CANNOT SEE CONTENT. Two runs can touch the same file
with different bytes, so the fingerprint covers path, change kind, mode
and the SHA-256 of the current bytes.

WHAT MAY LEAVE. Repo-relative paths that were ALLOWED, counts, closed
contract codes and hashes. Never file contents, never a patch, never
git's stderr, never a rejected path -- a refused path is attacker
influenced text and this travels into reports.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import (cli, contract, execution, preflight, schemas,
                              state as state_module, worktree)

GIT_TIMEOUT_SECONDS = 120
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(contract.IDENTIFIER_PATTERN)
# The only blob modes an ordinary source edit produces. Anything else --
# a symlink (120000), a gitlink (160000) or a type change -- is refused
# rather than guessed at.
_PLAIN_MODES = ("100644", "100755")
ADDED, MODIFIED, DELETED = "added", "modified", "deleted"


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
    """The worktree, the manifest or the main checkout moved somewhere
    this run was not allowed to take it."""


class DeclarationMismatch(ChangeSetError):
    """The reply does not describe what actually happened."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.SCHEMA_VIOLATION)


@dataclass(frozen=True, slots=True)
class _Change:
    path: str        # canonical repo-relative POSIX
    kind: str        # ADDED | MODIFIED | DELETED
    mode: str        # git blob mode, or "000000" for a deletion
    sha256: str      # of the current bytes; "" when the file is gone


@dataclass(frozen=True, slots=True)
class VerifiedChangeSet:
    """Bounded, structured, and free of anything a report may not
    carry: no patch, no contents, no absolute path."""

    run_id: str
    worktree_id: str
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
# GIT EVIDENCE -- argv only, checked, NUL-delimited
# ---------------------------------------------------------------------

def _git(cwd, *args) -> bytes:
    """Every git call is checked and every answer is BYTES.

    Text mode would decode with the ambient locale and split on
    newlines -- and a filename may contain a newline, which is exactly
    how a path walks out of an inventory unnoticed. The switches pin
    the answer to the repository state rather than to a cache, an
    external differ or a credential helper's opinion."""
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
    thing an implementer writes and an ordinary diff never mentions it;
    `--ignore-submodules=none` because a silently skipped submodule is
    a change nobody sees; `--no-renames` so one edit is one record --
    a rename that git still reports is refused rather than
    reinterpreted."""
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
    a probe used exactly that to rewrite a control-plane file while
    `status` stayed empty and this gate returned an empty change set.
    Both listings were measured to be stable across ordinary edits and
    untracked additions, so comparing them costs no false refusals."""
    return hashlib.sha256(_git(cwd, "ls-files", "-s", "-z") + b"\0"
                          + _git(cwd, "ls-files", "-v", "-z")).hexdigest()


def _blinded_entries(cwd):
    """Entries git was told to ignore. `S` is skip-worktree; a
    LOWERCASE tag is assume-unchanged."""
    blinded = []
    for field in _git(cwd, "ls-files", "-v", "-z").split(b"\0"):
        if len(field) < 2 or field[1:2] != b" ":
            continue
        tag = field[:1]
        if tag == b"S" or (tag.isalpha() and tag.islower()):
            blinded.append(field)
    return blinded


def _assert_not_blinded(cwd):
    """A flagged index cannot be inventoried, so it is refused rather
    than reported as clean. The COUNT may leave; the paths may not."""
    blinded = _blinded_entries(cwd)
    if blinded:
        raise UnsafeChange(
            f"{len(blinded)} indeks girdisi git'ten gizlenmis",
            reason=contract.StopReason.PREFLIGHT_FAILED)


def _git_identity(cwd) -> str:
    """A digest of WHICH repository this directory is attached to.

    In a worktree `.git` is a file pointing at the real git directory;
    replacing it re-aims every later question at another repository."""
    marker = Path(cwd) / ".git"
    try:
        entry = os.lstat(marker)
        payload = marker.read_bytes() if stat.S_ISREG(entry.st_mode) else b"D"
    except OSError:
        raise EvidenceUnavailable("git baglantisi okunamadi") from None
    payload += _git(cwd, "rev-parse", "--absolute-git-dir")
    return hashlib.sha256(payload).hexdigest()


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


def _canonical_relative(raw: bytes, root: Path) -> str:
    """One spelling for a path git reported, or a refusal.

    Refuses before the value is ever used: an absolute path, a drive
    letter, a backslash, a `..` segment, an empty segment or a control
    character. Then the location itself is proven to sit inside this
    worktree -- resolved through the PARENT, because a deleted file has
    no node left to resolve."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise UnsafeChange("yol UTF-8 degil",
                           reason=contract.StopReason.PATH_NOT_ALLOWED) from None
    bad = ("\\" in text or text.startswith("/") or not text
           or re.match(r"^[A-Za-z]:", text)
           or any(ord(ch) < 32 for ch in text))
    parts = PurePosixPath(text).parts if not bad else ()
    if bad or not parts or any(part in ("", ".", "..") for part in parts):
        raise UnsafeChange("yol kanonik degil",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    try:
        inside = (root / text).parent.resolve(strict=False)
        base = root.resolve(strict=True)
    except OSError:
        raise EvidenceUnavailable("yol cozulemedi") from None
    if not _contained(inside, base):
        raise UnsafeChange("yol calisma agacinin disina cikiyor",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    return "/".join(parts)


def _contained(candidate: Path, base: Path) -> bool:
    """Is `candidate` the base directory or something beneath it?

    Compared as whole SEGMENTS after one resolution each, so a sibling
    that merely shares an opening prefix is outside."""
    left = str(candidate).replace("\\", "/").rstrip("/")
    right = str(base).replace("\\", "/").rstrip("/")
    if os.name == "nt":
        left, right = left.casefold(), right.casefold()
    return left == right or left.startswith(right + "/")


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
# THE CHANGE SET
# ---------------------------------------------------------------------

def _plain_file_digest(absolute: Path) -> str:
    """Refuses anything that is not an ORDINARY regular file.

    `lstat`, never `stat`: following the link is what makes a symlink
    look like the file it points at. A Windows reparse point carries a
    tag even when it reports as a directory, and both are refused
    rather than followed."""
    try:
        entry = os.lstat(absolute)
    except OSError:
        raise EvidenceUnavailable("degisen dosya okunamadi") from None
    # ONE guard, deliberately: a separate symlink branch in front of
    # this looked like defence in depth and was not -- a link is never
    # `S_ISREG`, so the second test already refused everything the
    # first did, and a self-attack that deleted the first changed
    # nothing observable. A layer whose removal is invisible is a layer
    # nobody is testing.
    if (not stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode)
            or getattr(entry, "st_reparse_tag", 0)):
        raise UnsafeChange("siradan olmayan dosya turu desteklenmiyor",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    digest = hashlib.sha256()
    try:
        with open(absolute, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        raise EvidenceUnavailable("degisen dosya okunamadi") from None
    return digest.hexdigest()


def _classify(record: bytes, source, root: Path):
    """One porcelain-v2 record becomes one `_Change`, or a refusal.

    Every unsupported shape is named and refused: a staged entry, an
    unmerged entry, a submodule, a mode-only change, a rename git still
    reported, an unknown record type. Guessing how to follow one of
    those is how an unreviewed edit reaches the main checkout."""
    kind = record[:1]
    if kind == b"u":
        raise UnsafeChange("birlesmemis giris var",
                           reason=contract.StopReason.STAGED_CHANGES)
    if kind == b"2":
        raise UnsafeChange("yeniden adlandirma kaydi desteklenmiyor",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    if kind == b"?":
        path = _canonical_relative(record[2:], root)
        absolute = root / path
        digest = _plain_file_digest(absolute)
        # git records an executable bit where the filesystem has one;
        # calling every untracked file 100644 put a mode in the
        # fingerprint that the repository would not agree with
        executable = (os.name != "nt"
                      and bool(os.lstat(absolute).st_mode & stat.S_IXUSR))
        return _Change(path=path, kind=ADDED,
                       mode="100755" if executable else "100644",
                       sha256=digest)
    if kind != b"1":
        raise UnsafeChange("bilinmeyen git kaydi",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    try:
        header, raw_path = record.split(b" ", 8)[:8], record.split(b" ", 8)[8]
        xy, submodule = header[1].decode("ascii"), header[2].decode("ascii")
        mode_head, mode_work = header[3].decode("ascii"), header[5].decode(
            "ascii")
    except (IndexError, ValueError, UnicodeDecodeError):
        raise UnsafeChange("bilinmeyen git kaydi",
                           reason=contract.StopReason.PATH_NOT_ALLOWED) from None
    # NO separate staged branch here. The index gate above already
    # refuses any staged state -- `git add` moves `ls-files -s`, and a
    # deleted index shows up there too -- so a second check on `xy[0]`
    # was a layer whose removal changed nothing observable. One gate,
    # one mutation, one reason.
    if submodule != "N...":
        raise UnsafeChange("alt modul degisikligi desteklenmiyor",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    path = _canonical_relative(raw_path, root)
    if xy[1] == "D":
        # The HEAD mode, not a blank. Recording every deletion as
        # `000000` accepted the removal of a SYMLINK as if it were an
        # ordinary file, and erased the one field that told two
        # otherwise identical deletions apart.
        if mode_head not in _PLAIN_MODES:
            raise UnsafeChange("siradan olmayan dosya modu",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        return _Change(path=path, kind=DELETED, mode=mode_head, sha256="")
    if xy[1] != "M":
        raise UnsafeChange("desteklenmeyen degisiklik turu",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    if mode_work not in _PLAIN_MODES or mode_head not in _PLAIN_MODES:
        raise UnsafeChange("siradan olmayan dosya modu",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    if mode_work != mode_head:
        # content is what a patch carries; a bare permission change is
        # not something this gate knows how to transfer
        raise UnsafeChange("yalnizca mod degisikligi desteklenmiyor",
                           reason=contract.StopReason.PATH_NOT_ALLOWED)
    return _Change(path=path, kind=MODIFIED, mode=mode_work,
                   sha256=_plain_file_digest(root / path))


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
    """One digest over everything git can see about a checkout: HEAD,
    the full inventory, and the bytes of every changed or untracked
    regular file it lists.

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


def _assert_pristine(root: Path, baseline_sha: str):
    """A disposable worktree that was already dirty makes attribution
    impossible, so this runs BEFORE a model process exists."""
    if _head(root) != baseline_sha:
        raise UnsafeChange("calisma agaci taban surumde degil",
                           reason=contract.StopReason.BASELINE_MISMATCH)
    _assert_not_blinded(root)
    records = _status(root)
    if records:
        raise UnsafeChange(f"calisma agacinda {len(records)} onceden var olan "
                           f"degisiklik",
                           reason=contract.StopReason.DIRTY_WORKTREE)


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
                          baseline_sha: str, manifest_digest: str):
    try:
        binding = state_module.read_binding(state_dir)
    except state_module.CorruptState:
        raise EvidenceUnavailable("kosu baglamasi yok ya da bozuk") from None
    expected = {"run_id": run_id, "baseline_sha": baseline_sha,
                "manifest_digest": manifest_digest,
                "repo_id": state_module.repo_identity(repo)}
    if any(binding.get(field) != value for field, value in expected.items()):
        raise EvidenceUnavailable("kosu baglamasi bu cagriyla uyusmuyor")


# ---------------------------------------------------------------------
# THE PUBLIC SEAM
# ---------------------------------------------------------------------

def run_verified_implementation(binary, *, repo, state_dir, task_path,
                                manifest_digest, run_id, worktree_id,
                                baseline_sha, prompt, budget_usd,
                                timeout_seconds, max_output_bytes,
                                model=None) -> VerifiedChangeSet:
    """One implementer call, then proof of what it changed.

    The caller passes IDENTITIES: no worktree path, no cwd, no
    allow/forbid list, no diff, no patch and no git result. The
    permissions come from the exact manifest bytes whose digest this
    run was issued, and the working directory comes from B1's binding
    seam -- there is nothing here for a caller to substitute."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    state_path = Path(_exact_text(state_dir, "durum dizini"))
    manifest_digest = _exact_match(manifest_digest, _HEX64, "gorev ozeti")
    run_id = _exact_match(run_id, _RUN_ID, "kosu kimligi")
    worktree_id = _exact_match(worktree_id, worktree.WORKTREE_ID,
                               "calisma agaci kimligi")
    baseline_sha = _exact_match(baseline_sha, _SHA1, "taban surum")

    task_file, snapshot, task_relative = _bind_task(
        task_path, repo_path, manifest_digest, baseline_sha)
    _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                          manifest_digest)
    allowed = tuple(snapshot.task["allowed_paths"])
    forbidden = tuple(snapshot.task.get("forbidden_paths", ()))

    try:
        tree = worktree.assert_execution_binding(
            repo_path, state_dir=state_path, run_id=run_id,
            worktree_id=worktree_id, baseline_sha=baseline_sha)
    except worktree.WorktreeError as refused:
        raise EvidenceUnavailable(str(refused)) from None
    identity_before = _git_identity(tree)
    _assert_pristine(tree, baseline_sha)
    index_before = _index_state(tree)
    main_before = _tree_snapshot(repo_path)
    task_before = snapshot.digest

    def verify_after():
        """Everything that must NOT have moved, then the change set."""
        try:
            again = worktree.assert_execution_binding(
                repo_path, state_dir=state_path, run_id=run_id,
                worktree_id=worktree_id, baseline_sha=baseline_sha)
        except worktree.WorktreeError as refused:
            raise EvidenceUnavailable(str(refused)) from None
        if not _same_place(again, tree) or _git_identity(tree) != \
                identity_before:
            raise UnsafeChange("calisma agaci baglantisi degisti",
                               reason=contract.StopReason.PREFLIGHT_FAILED)
        if preflight.manifest_changed(task_file, snapshot) or \
                preflight.snapshot_manifest(task_file).digest != task_before:
            raise UnsafeChange("gorev dosyasi degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        _assert_state_binding(state_path, repo_path, run_id, baseline_sha,
                              manifest_digest)
        if _tree_snapshot(repo_path) != main_before:
            raise UnsafeChange("ana calisma agaci degistirildi",
                               reason=contract.StopReason.PATH_NOT_ALLOWED)
        # BEFORE the inventory is trusted, and it is THE index gate: a
        # flagged index made `status` come back empty while a
        # control-plane file's bytes had changed on disk, and any
        # staging moves the same listing. Nothing downstream re-checks
        # the index, so this refusal is the only one and its removal is
        # always observable.
        _assert_not_blinded(tree)
        if _index_state(tree) != index_before:
            raise UnsafeChange("calisma agaci indeksi degistirildi",
                               reason=contract.StopReason.STAGED_CHANGES)
        changes = []
        for record, source in _status(tree):
            change = _classify(record, source, tree)
            _authorize(change.path, allowed=allowed, forbidden=forbidden,
                       task_relative=task_relative)
            changes.append(change)
        return tuple(sorted(changes, key=lambda item: item.path))

    failure = None
    try:
        outcome = execution.run_implementer(
            binary, repo=repo_path, state_dir=state_path, run_id=run_id,
            worktree_id=worktree_id, baseline_sha=baseline_sha, prompt=prompt,
            budget_usd=budget_usd, timeout_seconds=timeout_seconds,
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

    return _accept(outcome, actual, run_id=run_id, worktree_id=worktree_id,
                   baseline_sha=baseline_sha)


def _accept(outcome, actual, *, run_id, worktree_id, baseline_sha):
    """The declaration is compared to the evidence, exactly."""
    reply = outcome.reply
    if reply.get("run_id") != run_id:
        raise DeclarationMismatch("yanit baska bir kosuyu adliyor")
    declared = reply.get("changed_files", [])
    if any(type(item) is not str for item in declared):
        raise DeclarationMismatch("bildirilen dosya listesi tam metin degil")
    if len(set(declared)) != len(declared):
        raise DeclarationMismatch("bildirilen dosya listesi yinelemeli")
    if set(declared) != {change.path for change in actual}:
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
        run_id=run_id, worktree_id=worktree_id, baseline_sha=baseline_sha,
        status=status,
        changed_files=tuple(change.path for change in actual),
        added=kinds.count(ADDED), modified=kinds.count(MODIFIED),
        deleted=kinds.count(DELETED), fingerprint=_fingerprint(actual),
        exit_code=outcome.exit_code, duration_ms=outcome.duration_ms,
        stdout_bytes=outcome.stdout_bytes, stderr_bytes=outcome.stderr_bytes,
        schema_sha256=outcome.schema_sha256, event=outcome.event)
