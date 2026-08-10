"""Single-instance lock. PACKAGE B1.

One runner owns a repository's loop at a time. Two runners against one
working tree is the concurrency the design refuses outright, so the
refusal has to be real rather than advisory.

WHY THE OWNER IS AN OPEN HANDLE AND NOT A FILE THAT EXISTS. The first
version claimed the lock with `O_CREAT | O_EXCL` and released it by
deleting the path. The claim was atomic; the RELEASE was not. Release
read the owner record, compared the token, and only then unlinked -- and
between the comparison and the unlink the lock could change hands, at
which point the departing owner deleted the incoming owner's lock and
two runners proceeded. That is not hypothetical: `break_stale_lock` was
itself a mechanism for handing the lock over while the previous owner
was still alive, and it carried the same read-then-delete split.

Narrowing that window would have left the same shape with a smaller
target. So ownership moved to something the kernel arbitrates: an
exclusive byte-range lock on an open handle. A second handle -- another
process, or another handle in this one -- is refused by the operating
system, with no window to race in.

NOTHING HERE EVER DELETES THE LOCK FILE. Deleting was the only operation
that could destroy another owner's claim, and the file's existence is no
longer what ownership means, so the deletion has no job left to do. A
`run.lock` sitting in a state directory is not a held lock; it is a byte
nobody is holding. That also removes the stale-lock problem entirely: a
crashed runner's handle is closed by the operating system, which drops
the lock, so there is no abandoned claim to break and no pid to guess
about. The previous version had to ask whether a pid was alive and
answer "assume yes" when it could not tell.

A CRASHED RUN IS STILL NOT SILENTLY RESUMED -- but that is a question
about the previous run's STATE, not about liveness, and it is answered
in `state.py` where the evidence actually lives.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

LOCK_FILENAME = "run.lock"


class LockHeld(RuntimeError):
    """Another runner owns this repository's loop."""


class LockNotOwned(RuntimeError):
    """A release attempted by something that does not hold the lock."""


if os.name == "nt":                                   # pragma: no cover -- os
    import msvcrt

    # Windows byte-range locks are MANDATORY, not advisory: a locked
    # range cannot be read by anyone else. Locking byte 0 therefore made
    # the owner record unreadable to every process except its owner --
    # `inspect` reported an unreadable lock while holding it. So the
    # lock sits at an offset past any plausible record and the two
    # ranges never overlap; the file itself stays a few dozen bytes,
    # because locking beyond end-of-file allocates nothing.
    _LOCK_OFFSET = 1 << 30

    def _try_lock(fd) -> bool:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd) -> None:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:                                                 # pragma: no cover -- os
    import fcntl

    def _try_lock(fd) -> bool:
        try:
            # flock, not lockf: BSD locks belong to the open file
            # description, so a second handle inside THIS process is
            # refused too. POSIX record locks are per-process and would
            # quietly let one runner take its own lock twice.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


@dataclass
class Lock:
    """A held lock. Ownership is the open descriptor, not the token.

    The token is written into the file for diagnostics -- so an operator
    looking at a state directory can tell which run is in there -- and
    is never what authorises anything."""

    path: Path
    fd: int
    token: str
    released: bool = False


def lock_path(state_dir) -> Path:
    """Derived, never supplied. A task manifest that could name the lock
    file could name a path outside the state directory -- and then the
    lock would be somewhere nobody else looks."""
    return Path(state_dir) / LOCK_FILENAME


def acquire(state_dir) -> Lock:
    """Claim the lock, or raise `LockHeld`. Returns the held handle."""
    path = lock_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    if not _try_lock(fd):
        os.close(fd)
        raise LockHeld("bu depo icin baska bir kosu kilidi tutuyor")
    token = secrets.token_hex(16)
    try:
        record = json.dumps({"token": token, "pid": os.getpid(),
                             "created_at": time.time()}).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, record)
        os.fsync(fd)
    except BaseException:
        # a claim we cannot describe is a claim we give back
        _unlock(fd)
        os.close(fd)
        raise
    return Lock(path=path, fd=fd, token=token)


def release(lock: Lock) -> None:
    """Drop the lock. The file is left where it is, on purpose.

    Removing it is the operation that could destroy someone else's
    claim, and it buys nothing: `acquire` reuses the file."""
    if not isinstance(lock, Lock):
        raise LockNotOwned("kilit nesnesi degil; birakilacak bir sey yok")
    if lock.released:
        raise LockNotOwned("bu kilit zaten birakildi")
    lock.released = True
    _unlock(lock.fd)
    os.close(lock.fd)


def inspect(state_dir):
    """What is in the lock file, and whether anyone is actually holding
    it. The two are independent: a readable record left by a crashed run
    describes nobody.

    "Readable" means EVERY field has the type it is supposed to have,
    not merely that the bytes parsed as JSON. A record with
    `created_at` as a string parses fine and then killed this function
    on an unguarded `float()` -- a diagnostic path that raises is worse
    than useless, because it is reached exactly when something is
    already wrong. Anything unexpected reports unreadable and carries no
    values, rather than half a record."""
    path = lock_path(state_dir)
    if not path.exists():
        return None
    held = is_held(state_dir)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        record = None
    pid = record.get("pid") if isinstance(record, dict) else None
    created = record.get("created_at") if isinstance(record, dict) else None
    # bools are ints in Python and neither is a pid or a timestamp
    if not isinstance(pid, int) or isinstance(pid, bool) \
            or not isinstance(created, (int, float)) \
            or isinstance(created, bool) or not math.isfinite(created):
        return {"readable": False, "held": held, "pid": None,
                "age_seconds": None}
    return {"readable": True, "held": held, "pid": pid,
            "age_seconds": max(0.0, time.time() - float(created))}


def is_held(state_dir) -> bool:
    """Ask the operating system, by trying to take it and giving it
    straight back. There is no other honest way to answer this: a file
    on disk and a pid in it are both things a dead run leaves behind."""
    path = lock_path(state_dir)
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        # on Windows an open handle can block even this; a lock we
        # cannot probe is reported as held rather than as free
        return True
    try:
        if not _try_lock(fd):
            return True
        _unlock(fd)
        return False
    finally:
        os.close(fd)


@contextmanager
def single_instance_lock(state_dir):
    """Hold the lock for a block, and release only what we took.

    A failure inside the block is NOT replaced by a failure to clean up:
    the release runs in its own try, and its error is subordinated to
    whatever the body raised."""
    lock = acquire(state_dir)
    body_failed = False
    try:
        yield lock
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            release(lock)
        except Exception:
            if not body_failed:
                raise
