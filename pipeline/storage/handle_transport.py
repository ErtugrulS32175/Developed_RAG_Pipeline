"""Portable handle-bound storage transport for production services.

A SECOND transport, and deliberately not an extension of the first.
This package is deliberately independent of the agent-loop control plane.
Production publication code and other storage clients can therefore share
the measured no-follow primitives without importing orchestration code.

So this module repeats the D2 lesson rather than the D2 code: A PATH IS A
LOOKUP; A HANDLE IS AN OBJECT. After the root is opened there are no
paths -- every child is created, opened, moved or removed relative to the
directory OBJECT that named it, so a parent swapped for a link between
two operations can make an operation fail and can never redirect it.

MEASURED ON BOTH PLATFORMS BEFORE A LINE OF THIS WAS WRITTEN:

  POSIX    `O_DIRECTORY|O_NOFOLLOW` opens, `openat`, `mkdirat`,
           `renameat`, `rmdirat` and `fsync` on a directory descriptor
           are all present. With the parent PATH swapped for a symlink
           afterwards, a create through the held descriptor landed in the
           ORIGINAL directory and the outside tree stayed empty.
           `O_EXCL` collided rather than truncating; `O_NOFOLLOW`
           refused a symlinked final component without following it.

  WINDOWS  `NtCreateFile` with a RootDirectory handle and FILE_CREATE
           creates a child by RELATIVE name and answers
           STATUS_OBJECT_NAME_COLLISION (0xC0000035) on an existing one.
           `NtSetInformationFile` with `FileRenameInformationEx` renames
           THE OBJECT THE HANDLE NAMES into a target directory given as
           another handle, preserving the file identity.
           `FileDispositionInformationEx` removes a directory through its
           handle and answers STATUS_DIRECTORY_NOT_EMPTY (0xC0000101)
           rather than recursing. With the parent path swapped for a
           junction, the create through the held handle landed inside the
           original directory and the outside tree stayed empty.

WHAT THIS MODULE WILL NOT DO. Follow a reparse point or a symlink, build
a path after the root, replace an object that already exists, delete a
directory that is not empty, or let the operating system's own error text
-- which names absolute paths -- reach a caller.

WHAT IT DOES NOT CLAIM. A multi-file change is not one atomic operation
and this module does not pretend otherwise: it offers PER-OBJECT atomic
moves, and the transaction on top of them is `application`'s job.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_WINDOWS = os.name == "nt"

# The read ceiling for anything this module pulls into memory. A backup
# copy is bounded WHILE it is read, never measured afterwards.
MAX_FILE_BYTES = 64 << 20


class TransportError(RuntimeError):
    """A write this transport will not perform. Fixed text, never a path,
    never an OS message."""


class AlreadyExists(TransportError):
    """An exclusive create found something already there.

    Its own type because the caller has to tell it apart: for an ADDED
    target this is the operator's file and the whole application stops;
    for a staging name it is a collision inside our own holder."""


class NotEmpty(TransportError):
    """A directory removal found children. Never recursed into: this
    module removes directories IT created, and one that has grown
    contents is no longer only ours."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One child, AS THE DIRECTORY OBJECT REPORTED IT.

    `mode` is the POSIX permission text the change set already speaks
    (`"0o644"`), and an empty string on Windows, where there are no such
    bits to carry and inventing one would be inventing a fact."""

    kind: str                  # file | dir | link | other
    identity: str              # volume-qualified, as text
    size: int
    mode: str
    reparse_tag: int


def validate_child_name(name: str) -> str:
    """A name is data, and it is about to be used to create something.

    Measured in D2: `NtCreateFile` accepted `pipeline\\a.py` as a
    RELATIVE name and reached the grandchild. A separator that survives
    this function is a path traversal with extra steps."""
    if type(name) is not str or not name:
        raise TransportError("giris adi gecersiz")
    if name in (".", ".."):
        raise TransportError("giris adi gecersiz")
    if "\\" in name or "/" in name or "\0" in name:
        raise TransportError("giris adi ayirici tasiyor")
    return name


if _WINDOWS:                                   # pragma: no cover - platform
    from pipeline.storage import handle_transport_windows as _impl
else:                                          # pragma: no cover - platform
    from pipeline.storage import handle_transport_posix as _impl

open_root = _impl.open_root
close_directory = _impl.close_directory
directory_identity = _impl.directory_identity
fsync_directory = _impl.fsync_directory
child_entry = _impl.child_entry
open_child_directory = _impl.open_child_directory
create_child_directory = _impl.create_child_directory
remove_child_directory = _impl.remove_child_directory
create_child_file = _impl.create_child_file
open_child_file = _impl.open_child_file
rename_child = _impl.rename_child
write_all = _impl.write_all
read_all = _impl.read_all
fsync_handle = _impl.fsync_handle
handle_identity = _impl.handle_identity
handle_mode = _impl.handle_mode
set_handle_mode = _impl.set_handle_mode
close_handle = _impl.close_handle


def close_handle_quietly(handle) -> bool:
    """Close, and SAY whether it worked.

    For failure paths, where a cleanup problem must not replace the error
    that got us here -- but must not vanish either."""
    try:
        close_handle(handle)
        return True
    except (TransportError, OSError):
        return False


def close_directory_quietly(directory) -> bool:
    try:
        close_directory(directory)
        return True
    except (TransportError, OSError):
        return False
