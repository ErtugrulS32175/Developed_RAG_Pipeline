"""Handle-bound filesystem transport. PACKAGE B2B-A-D2.

A PATH IS A LOOKUP; A HANDLE IS AN OBJECT. Three rounds of this package
tried to keep a path honest and lost three times, each time to the same
move played at a different instant:

  1. no-follow on the final component     -> parent swapped one level up
  2. identity pinned to the child's lstat -> parent swapped BEFORE that
                                             lstat, so both sides agreed
                                             on the outside file
  3. identity checked on both sides of
     the listing                          -> swap placed between the
                                             post-listing check and the
                                             child's own lstat

Every one of those was a fresh question asked of a path, and a path can
be made to answer differently between any two questions. The fix is not
a fourth question. After the root is opened, THERE ARE NO PATHS: the
walk lists an open directory, and every child is opened relative to the
directory object that listed it.

MEASURED, on both platforms, before any of this was written:

  POSIX    `O_DIRECTORY|O_NOFOLLOW`, `scandir(fd)`, `stat(dir_fd=)`,
           `open(dir_fd=)` and `readlink(dir_fd=)` are all present. With
           the parent path swapped for a symlink afterwards, the fd
           showed the original directory and the outside file was
           invisible (ENOENT) while the path showed it.

  WINDOWS  `GetFileInformationByHandleEx` with `FileIdExtdDirectoryInfo`
           enumerates from the HANDLE and carries name, attributes,
           reparse tag, size, timestamps and the 128-bit FileId --
           everything an evidence record needs. `NtCreateFile` with a
           RootDirectory handle opens a child by RELATIVE name, which is
           the `openat` Win32 does not expose. With the parent swapped,
           that open returned STATUS_DELETE_PENDING: a typed refusal,
           never the outside file.

WHAT THIS MODULE WILL NOT DO. Follow a link, open by full path after the
root, read an object it has not first bound to the listing that named
it, or let the operating system's own error text -- which carries paths
-- reach a caller.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_WINDOWS = os.name == "nt"


class TransportError(RuntimeError):
    """A filesystem question this transport will not answer. Fixed text,
    never a path, never an OS message."""

    # A cleanup failure is carried STRUCTURALLY, not as prose. The flag
    # is what survives a boundary; the sentence is only for a human
    # reading a traceback.
    cleanup_failed = False


# One fixed sentence, defined once. Nothing else about a cleanup failure
# is ever written down, because free text is what leaks.
CLEANUP_NOTE = "temizlik basarisiz: nesne kapatilamadi"


def mark_cleanup_failed(exc: BaseException) -> None:
    """Record that a cleanup failed, on the error already being raised.

    MEASURED: the transport marked this with `add_note` alone, and the
    walker's `_guard` then built a NEW exception -- so the mark reached
    the boundary and stopped there (`GUARD_NOTES=[]`). A flag crosses;
    prose does not, and prose must not, because notes written further
    down may carry paths."""
    exc.cleanup_failed = True
    if CLEANUP_NOTE not in getattr(exc, "__notes__", ()):
        exc.add_note(CLEANUP_NOTE)


def cleanup_failed(exc: BaseException) -> bool:
    """Exact `True`, so an object that merely looks truthy cannot pass
    itself off as a verdict."""
    return getattr(exc, "cleanup_failed", False) is True


@dataclass(frozen=True, slots=True)
class Record:
    """One child, AS THE DIRECTORY OBJECT REPORTED IT.

    This is the authority for the child. Nothing may re-derive these
    fields from a path afterwards, because that is a second question
    and the gap between two questions is the whole vulnerability."""

    name: str
    kind: str                  # file | dir | link
    attributes: int            # Windows file attributes; 0 on POSIX
    reparse_tag: int
    mode: int                  # POSIX st_mode; 0 on Windows
    size: int
    mtime_ns: int
    file_id: str               # volume-qualified, as text
    nlink: int | None          # None when the listing cannot say


def validate_child_name(name: str) -> str:
    """A name from a listing is data, and it is used to open something.

    Measured: `NtCreateFile` happily accepted `pipeline\\a.py` as a
    RELATIVE name and opened the grandchild. A separator that survives
    this function is a path traversal with extra steps."""
    if type(name) is not str or not name:
        raise TransportError("giris adi gecersiz")
    if name in (".", ".."):
        raise TransportError("giris adi gecersiz")
    if "\\" in name or "/" in name or "\0" in name:
        raise TransportError("giris adi ayirici tasiyor")
    return name


if _WINDOWS:                                   # pragma: no cover - platform
    from tools.agent_loop import fs_transport_windows as _impl
else:                                          # pragma: no cover - platform
    from tools.agent_loop import fs_transport_posix as _impl

open_root = _impl.open_root
list_directory = _impl.list_directory
open_child_directory = _impl.open_child_directory
open_child_file = _impl.open_child_file
link_evidence = _impl.link_evidence
close_directory = _impl.close_directory
descriptor_identity = _impl.descriptor_identity
directory_identity = _impl.directory_identity
