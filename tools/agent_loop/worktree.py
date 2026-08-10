"""The disposable git worktree. PACKAGE B1.

The implementer will eventually edit files. It will never edit THESE
files: it works in a worktree created from the frozen baseline, and only
a verified patch comes back. Hashing the main tree after the fact
detects damage that has already happened and been left in the operator's
checkout; a separate worktree means the main tree was never the thing at
risk.

WHY `git worktree add` AND NOT A COPY. A copy would bring `data/`,
`output/`, `logs/` and `uploads/` with it -- the document tree, the
exports derived from it, and whatever a locked holdout lives in. A
worktree materialises TRACKED CONTENT ONLY, and those directories are
gitignored, so the private material is absent by construction rather
than by a filter someone has to maintain. That property is asserted in
the tests, not assumed.

OWNERSHIP IS A PERSISTED RECORD, WRITTEN BEFORE ANYTHING EXISTS. This is
the fourth design for cleanup authority, and the first three each failed
for the same underlying reason: they tried to make ownership something
you could READ OFF the target.

  1. A caller-supplied path plus a `force_unmarked` escape hatch. A flag
     that switches the ownership check off is the check not existing.
  2. A caller-supplied path plus a marker FILE. `mkdir` and "write the
     marker" are two steps, and an audit injected a failure between
     them: the holder existed, was unmarked, and cleanup then refused to
     remove a directory this module had just created.
  3. A 32-hex id, with the holder named after it and no path parameter
     at all. I claimed removing someone else's worktree had become
     unrepresentable. It had not: the id names a DIRECTORY, and the
     directory is shared by every repository on the machine. An audit
     created a worktree from repository B, called `remove` with
     repository A, watched git's removal fail because A had never
     registered it, watched this module ignore that failure, and deleted
     B's tree anyway -- leaving stale entries in B's registry.

So the id is only a name. Authority is `(repo, run, worktree)` together,
and it is written to durable storage BEFORE the first directory is
created. Removal reads that record, refuses a repository it was not
issued to, refuses to touch a tree git does not report as belonging to
this repository, and treats a failed git removal as a failure rather
than something to step over. Crash recovery enumerates the records, so
a run that died before it could write its binding is still able to find
what it left behind -- the previous design could only look up an id that
existed nowhere but the dead process's memory.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.agent_loop import contract, state as state_module

TEMP_PREFIX = "agent-loop-wt-"
ROOT_DIRNAME = "agent-loop-worktrees"
WORKTREE_DIRNAME = "wt"
REGISTRY_DIRNAME = "worktrees"
STATUS_PLANNED = "planned"
STATUS_READY = "ready"
# the same shape `state.BINDING_SCHEMA` requires: one vocabulary for the
# id, so a value that can name a directory can always be recorded
WORKTREE_ID = re.compile(r"^[0-9a-f]{32}$")

RECORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "worktree_id", "repo_id", "run_id",
                 "baseline_sha", "status"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "worktree_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        # identity, never a path: an absolute path in a state file is an
        # absolute path in a report
        "repo_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "run_id": {"type": "string", "pattern": contract.IDENTIFIER_PATTERN},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "status": {"enum": [STATUS_PLANNED, STATUS_READY]},
    },
}


class WorktreeError(RuntimeError):
    """The disposable worktree could not be created, verified or removed."""


def _no_prompt_env():
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "",
                "GCM_INTERACTIVE": "never"})
    return env


def _git(repo, *args, check=True):
    """Non-interactive git. No shell, argv only, prompts disabled."""
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, errors="replace", env=_no_prompt_env(),
    )
    if check and done.returncode != 0:
        # the git command NAME, never its stderr: that text can carry a
        # path, a remote or a credential helper's complaint
        raise WorktreeError(f"git {args[0]} basarisiz (rc={done.returncode})")
    return done


def runner_temp_root() -> Path:
    """The one directory disposable worktrees may live in.

    A NAMED subdirectory, not the system temp directory itself -- an
    earlier version returned `tempfile.gettempdir()`, which made
    containment almost meaningless because everything transient on the
    machine lives under there.

    Derived here, never handed in."""
    root = Path(tempfile.gettempdir()).resolve() / ROOT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def holder_for(worktree_id) -> Path:
    """The single directory a given id can name.

    The id is validated against a strict 32-hex pattern before it is
    joined to the runner-owned root, so no id can denote a path outside
    it -- `..` is not an id. This is containment ONLY; it says nothing
    about who owns the directory, which is what the record is for."""
    if not WORKTREE_ID.match(str(worktree_id)):
        raise WorktreeError("gecersiz calisma agaci kimligi")
    return runner_temp_root() / f"{TEMP_PREFIX}{worktree_id}"


def _record_path(state_dir, worktree_id) -> Path:
    holder_for(worktree_id)                       # validates the id
    return Path(state_dir) / REGISTRY_DIRNAME / f"{worktree_id}.json"


def read_record(state_dir, worktree_id):
    """The ownership record, or None. Never guesses."""
    try:
        return state_module.read_json_checked(
            _record_path(state_dir, worktree_id), RECORD_SCHEMA,
            "calisma agaci kaydi")
    except state_module.CorruptState:
        return None


def register(state_dir, *, repo, run_id, baseline_sha):
    """Mint and PERSIST the identity before anything exists on disk.

    Write-ahead, and that ordering is the whole point: a process that
    dies immediately after this call has still left a record that names
    the repository and the run, so its residue is findable. The previous
    design minted the id inside `create` and returned it only on
    success, which meant a crash between `mkdir` and the caller's first
    write left an orphan whose id existed nowhere at all."""
    record = {"protocol_version": contract.PROTOCOL_VERSION,
              "worktree_id": secrets.token_hex(16),
              "repo_id": state_module.repo_identity(repo),
              "run_id": run_id, "baseline_sha": baseline_sha,
              "status": STATUS_PLANNED}
    state_module.write_json_atomically(
        _record_path(state_dir, record["worktree_id"]), record,
        RECORD_SCHEMA, "calisma agaci kaydi")
    return record


def _authorised_record(state_dir, repo, worktree_id):
    """The record, checked against the repository asking to use it."""
    record = read_record(state_dir, worktree_id)
    if record is None:
        raise WorktreeError("calisma agaci kaydi yok; silinmedi")
    if record["repo_id"] != state_module.repo_identity(repo):
        # the exact hole the audit walked through: an id alone let one
        # repository delete another repository's worktree
        raise WorktreeError("kayit bu depoya ait degil; silinmedi")
    return record


def _registered_here(repo, path) -> bool:
    """Does GIT, in this repository, report this worktree as its own?

    CHECKED. With `check=False` a failed `git worktree list` returned an
    empty listing, and empty reads as "not registered here" -- so an
    unverifiable git state became a verified absence and removal carried
    on to delete the holder and the ledger entry. A question we could
    not ask has no answer, and no answer is not `False`."""
    listing = _git(repo, "worktree", "list", "--porcelain").stdout
    resolved = str(Path(path).resolve()).replace("\\", "/").casefold()
    for line in listing.splitlines():
        if not line.startswith("worktree "):
            continue
        entry = str(Path(line.split(" ", 1)[1]).resolve())
        if entry.replace("\\", "/").casefold() == resolved:
            return True
    return False


def create(repo, *, state_dir, run_id, baseline_sha):
    """A detached worktree at exactly the recorded baseline.

    Returns `(path, worktree_id)`. The record is written first, so every
    later failure is recoverable by enumeration rather than by
    remembering something.

    Verified AFTER creation: asking git for a sha and assuming it
    obliged is the same class of mistake as reading a success message
    instead of the end state."""
    record = register(state_dir, repo=repo, run_id=run_id,
                      baseline_sha=baseline_sha)
    worktree_id = record["worktree_id"]
    holder = holder_for(worktree_id)
    path = holder / WORKTREE_DIRNAME
    # exclusive: no `exist_ok`. A collision at 128 bits is not a thing to
    # paper over by reusing a directory whose contents we never verified.
    # This sits OUTSIDE the cleanup block on purpose: if the directory
    # was already there we did not make it, and the ordinary failure
    # path would have deleted somebody else's holder -- the same class
    # of damage this guard exists to prevent. Only our own record goes.
    try:
        holder.mkdir()
    except FileExistsError:
        _record_path(state_dir, worktree_id).unlink(missing_ok=True)
        raise WorktreeError("calisma agaci kabi zaten var") from None
    try:
        _git(repo, "worktree", "add", "--detach", str(path), baseline_sha)
        head = _git(path, "rev-parse", "HEAD").stdout.strip()
        if head != baseline_sha:
            raise WorktreeError("kurulan calisma agaci baseline'da degil")
        state_module.write_json_atomically(
            _record_path(state_dir, worktree_id),
            dict(record, status=STATUS_READY), RECORD_SCHEMA,
            "calisma agaci kaydi")
        return path, worktree_id
    except BaseException:
        try:
            remove(repo, state_dir=state_dir, worktree_id=worktree_id)
        except Exception:
            # a cleanup that fails must not REPLACE the failure the
            # caller is trying to understand; what it leaves behind is
            # caught by the residue assertion, and by `find_orphans`
            pass
        raise


def remove(repo, *, state_dir, worktree_id):
    """Remove a worktree THIS repository and run created.

    Three conditions, none waivable: a record exists, the record was
    issued to this repository, and -- when git knows the tree -- git's
    own removal succeeds. A failed `git worktree remove` used to be
    ignored and followed by an unconditional recursive delete, which is
    how another repository's tree was destroyed."""
    record = _authorised_record(state_dir, repo, worktree_id)
    holder = holder_for(worktree_id)
    path = holder / WORKTREE_DIRNAME
    if _registered_here(repo, path):
        # check=True: if git refuses, we stop. The directory is git's to
        # dismantle while it still has a registration pointing at it.
        _git(repo, "worktree", "remove", "--force", str(path))
    elif record["status"] == STATUS_READY and path.exists():
        # ready, present, and git does not claim it: something is wrong
        # in a way this function must not resolve by deleting
        raise WorktreeError("calisma agaci bu depoda kayitli degil; silinmedi")
    # `ignore_errors=True` followed by dropping the ledger entry was a
    # way to lose a directory: on Windows an open handle makes removal
    # fail silently, the holder stayed on disk, and the record that named
    # it was deleted anyway -- so `find_orphans` could no longer see it
    # and nothing would ever clean it up. The ledger entry is the only
    # thing that makes residue findable, so it goes LAST and only once
    # the directory is provably gone.
    shutil.rmtree(holder, ignore_errors=True)
    if holder.exists():
        raise WorktreeError("calisma agaci kabi silinemedi; kayit korundu")
    _record_path(state_dir, worktree_id).unlink(missing_ok=True)
    _git(repo, "worktree", "prune", check=False)


def find_orphans(repo, *, state_dir, run_id=None):
    """Every worktree this repository recorded, WITHOUT knowing any id.

    The crash-recovery path. Enumerating the registry is what makes a
    run that died before writing its binding recoverable at all: the
    previous design required the id as an argument, and after that kind
    of crash the id existed nowhere."""
    registry = Path(state_dir) / REGISTRY_DIRNAME
    if not registry.is_dir():
        return []
    repo_id = state_module.repo_identity(repo)
    found = []
    for entry in sorted(registry.glob("*.json")):
        record = read_record(state_dir, entry.stem)
        if record is None or record["repo_id"] != repo_id:
            continue
        if run_id is not None and record["run_id"] != run_id:
            continue
        found.append(record)
    return found


def owns(path, *, worktree_id) -> bool:
    """Containment only: is this path inside the holder that id names?

    Deliberately NOT an ownership answer any more -- that question is
    the record's, and a directory cannot answer it about itself."""
    try:
        holder = holder_for(worktree_id)
        resolved = Path(path).resolve()
    except (WorktreeError, OSError):
        return False
    return resolved == holder or resolved.parent == holder
