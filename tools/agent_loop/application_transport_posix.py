"""POSIX half of the handle-bound write transport. PACKAGE B2B-C2.

Everything hangs off a directory FILE DESCRIPTOR. `dir_fd` is the kernel
saying "relative to THIS object", which is the guarantee a path can never
give: measured here, a create through a held descriptor whose PATH had
been swapped for a symlink landed in the original directory and left the
outside tree empty.

NO SILENT FALLBACK. If a primitive is missing the platform is refused. A
write path that quietly degrades to `open(path)` would be the defect this
module exists to remove, wearing this module's name.

THE ONE DECLARED RESIDUE. `renameat` names its source by NAME, not by an
open descriptor -- POSIX has no `frenameat` -- so between the identity
check and the move there is a window a swap can sit inside. It is
narrowed the only way it can be: the object is opened no-follow, its
identity is taken from the DESCRIPTOR, and the caller verifies the moved
object's identity afterwards. Windows has no such window; this is written
down rather than papered over.
"""
from __future__ import annotations

import os
import stat

from tools.agent_loop import application_transport as _api

_REQUIRED = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK", "O_EXCL")
_DIR_FLAGS = None


def _assert_platform():
    for flag in _REQUIRED:
        if not hasattr(os, flag):
            raise _api.TransportError("platform no-follow acilisi desteklemiyor")
    for call in (os.open, os.stat, os.mkdir, os.rmdir, os.rename, os.unlink):
        if call not in os.supports_dir_fd:
            raise _api.TransportError("platform dizin-goreli yazma yapmiyor")


def _dir_flags() -> int:
    global _DIR_FLAGS
    if _DIR_FLAGS is None:
        _assert_platform()
        _DIR_FLAGS = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                      | os.O_CLOEXEC)
    return _DIR_FLAGS


class Directory:
    """An open directory. The ONLY thing a caller may name a child
    through."""

    __slots__ = ("fd", "identity", "closed")

    def __init__(self, fd: int, identity: str):
        self.fd = fd
        self.identity = identity
        self.closed = False


class Handle:
    """An open file. Carries no name: a handle that remembered where it
    came from would tempt somebody to reopen it."""

    __slots__ = ("fd", "closed")

    def __init__(self, fd: int):
        self.fd = fd
        self.closed = False


def _identity_text(info) -> str:
    return f"{info.st_dev}:{info.st_ino}"


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "link"
    if stat.S_ISDIR(mode):
        return "dir"
    return "file" if stat.S_ISREG(mode) else "other"


def _mode_text(mode: int) -> str:
    return f"0o{stat.S_IMODE(mode):o}"


def _close_fd_quietly(fd: int) -> bool:
    try:
        os.close(fd)
        return True
    except OSError:
        return False


def _fail(message: str, fd: int) -> "_api.TransportError":
    """Build a refusal and close the descriptor it was about, so a
    cleanup problem cannot replace the error being raised."""
    error = _api.TransportError(message)
    if not _close_fd_quietly(fd):
        error.add_note("temizlik basarisiz: nesne kapatilamadi")
    return error


# ---------------------------------------------------------------------
# directories
# ---------------------------------------------------------------------

def open_root(path) -> Directory:
    try:
        fd = os.open(path, _dir_flags())
    except OSError:
        raise _api.TransportError("kok dizin acilamadi") from None
    try:
        info = os.fstat(fd)
    except OSError:
        raise _fail("kok durumu okunamadi", fd) from None
    if not stat.S_ISDIR(info.st_mode):
        raise _fail("kok bir dizin degil", fd)
    return Directory(fd, _identity_text(info))


def close_directory(directory: Directory) -> None:
    """`closed` means CLOSED, not "close was attempted"."""
    if directory.closed:
        return
    try:
        os.close(directory.fd)
    except OSError:
        raise _api.TransportError("tanimlayici kapatilamadi") from None
    directory.closed = True


def directory_identity(directory: Directory) -> str:
    return directory.identity


def fsync_directory(directory: Directory) -> bool:
    """Make the RENAME durable, not only the bytes -- and SAY whether the
    platform actually did it."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")
    try:
        os.fsync(directory.fd)
        return True
    except OSError:
        return False


def child_entry(directory: Directory, name: str):
    """A `stat` relative to the descriptor, no-follow. `None` when there
    is nothing there -- an absent child is an ANSWER, not a failure."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")
    child = _api.validate_child_name(name)
    try:
        info = os.stat(child, dir_fd=directory.fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise _api.TransportError("giris durumu okunamadi") from None
    return _api.Entry(kind=_kind(info.st_mode), identity=_identity_text(info),
                      size=info.st_size, mode=_mode_text(info.st_mode),
                      reparse_tag=0)


def open_child_directory(directory: Directory, name: str) -> Directory:
    """Descend one level, refusing a symlink rather than following it."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden acilis")
    child = _api.validate_child_name(name)
    try:
        fd = os.open(child, _dir_flags(), dir_fd=directory.fd)
    except OSError:
        # O_NOFOLLOW turns a symlinked component into ELOOP, and
        # O_DIRECTORY turns a file into ENOTDIR: both are refusals here
        raise _api.TransportError("alt dizin acilamadi") from None
    try:
        info = os.fstat(fd)
    except OSError:
        raise _fail("acik nesne durumu okunamadi", fd) from None
    if not stat.S_ISDIR(info.st_mode):
        raise _fail("alt dizin bir dizin degil", fd)
    return Directory(fd, _identity_text(info))


def create_child_directory(directory: Directory, name: str) -> None:
    """EXCLUSIVE. `exist_ok` here would mean adopting a directory
    somebody else made, which is how a rollback ends up removing one it
    did not create."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")
    child = _api.validate_child_name(name)
    try:
        os.mkdir(child, 0o755, dir_fd=directory.fd)
    except FileExistsError:
        raise _api.AlreadyExists("dizin zaten var") from None
    except OSError:
        raise _api.TransportError("dizin olusturulamadi") from None


def remove_child_directory(directory: Directory, name: str) -> None:
    """EMPTY ONLY. The kernel enforces it, which is better than a check:
    a directory that gained contents between the test and the call is
    still refused."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")
    child = _api.validate_child_name(name)
    try:
        os.rmdir(child, dir_fd=directory.fd)
    except FileNotFoundError:
        raise _api.TransportError("dizin yok") from None
    except OSError as refused:
        if refused.errno in (39, 66, 40):          # ENOTEMPTY / EEXIST
            raise _api.NotEmpty("dizin bos degil") from None
        raise _api.TransportError("dizin kaldirilamadi") from None


# ---------------------------------------------------------------------
# files
# ---------------------------------------------------------------------

def create_child_file(directory: Directory, name: str) -> Handle:
    """EXCLUSIVE create. `O_EXCL` is what refuses the operator's own file
    at an ADDED target, and it refuses it in the KERNEL rather than in a
    check somebody can race."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")
    child = _api.validate_child_name(name)
    flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
             | os.O_CLOEXEC)
    try:
        fd = os.open(child, flags, 0o644, dir_fd=directory.fd)
    except FileExistsError:
        raise _api.AlreadyExists("dosya zaten var") from None
    except OSError:
        raise _api.TransportError("dosya olusturulamadi") from None
    return Handle(fd)


def open_child_file(directory: Directory, name: str) -> Handle:
    """Read an existing ordinary file, no-follow.

    `O_NONBLOCK` because `O_RDONLY` on a FIFO blocks until a writer
    appears, and an apply that hangs is worse than one that refuses."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden acilis")
    child = _api.validate_child_name(name)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        fd = os.open(child, flags, dir_fd=directory.fd)
    except OSError:
        raise _api.TransportError("dosya no-follow acilamadi") from None
    try:
        info = os.fstat(fd)
    except OSError:
        raise _fail("acik nesne durumu okunamadi", fd) from None
    if not stat.S_ISREG(info.st_mode):
        raise _fail("son bilesen siradan bir dosya degil", fd)
    return Handle(fd)


def rename_child(source: Directory, source_name: str, target: Directory,
                 target_name: str) -> None:
    """Move one ordinary file between two OPEN directories, NEVER
    replacing what is already at the target name.

    WHY `linkat`+`unlinkat` AND NOT `renameat`. `rename` on POSIX
    silently overwrites an existing target, and Python exposes no
    `RENAME_NOREPLACE`. The obvious repair -- stat the target, then
    rename -- is check-then-act: a file created in that window is
    destroyed, and destroyed is the one outcome rollback cannot undo.
    `link` refuses an occupied name IN THE KERNEL, atomically, which is
    the same guarantee NT gives a rename without REPLACE_IF_EXISTS. So
    the target name is claimed first and the source name released
    afterwards; the object, and therefore its identity, is the same one
    throughout.

    The residue this leaves is the opposite of the dangerous one: if the
    unlink fails, an EXTRA name exists and nothing has been lost, and the
    caller's post-verification sees it."""
    if source.closed or target.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")
    left = _api.validate_child_name(source_name)
    right = _api.validate_child_name(target_name)
    try:
        info = os.stat(left, dir_fd=source.fd, follow_symlinks=False)
    except FileNotFoundError:
        raise _api.TransportError("tasinacak nesne yok") from None
    except OSError:
        raise _api.TransportError("tasinacak nesne sorgulanamadi") from None
    if not stat.S_ISREG(info.st_mode):
        # a link, a FIFO or a directory is not something this transport
        # moves: every caller here has already proved its source is an
        # ordinary file, so this is a contradiction rather than a case
        raise _api.TransportError("tasinacak nesne siradan bir dosya degil")
    try:
        os.link(left, right, src_dir_fd=source.fd, dst_dir_fd=target.fd)
    except FileExistsError:
        raise _api.AlreadyExists("hedef ad zaten dolu") from None
    except OSError:
        raise _api.TransportError("nesne tasinamadi") from None
    try:
        os.unlink(left, dir_fd=source.fd)
    except OSError:
        raise _api.TransportError("kaynak ad birakilamadi") from None


def write_all(handle: Handle, data: bytes) -> None:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    written = 0
    while written < len(data):
        try:
            written += os.write(handle.fd, data[written:])
        except OSError:
            raise _api.TransportError("dosya yazilamadi") from None


def read_all(handle: Handle, ceiling: int) -> bytes:
    """Bounded WHILE reading. A ceiling applied after the bytes are in
    the process is a report, not a bound."""
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    chunks, total = [], 0
    while True:
        try:
            block = os.read(handle.fd, 1 << 20)
        except OSError:
            raise _api.TransportError("dosya okunamadi") from None
        if not block:
            return b"".join(chunks)
        total += len(block)
        if total > ceiling:
            raise _api.TransportError("dosya sozlesme tavanini asiyor")
        chunks.append(block)


def fsync_handle(handle: Handle) -> None:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    try:
        os.fsync(handle.fd)
    except OSError:
        raise _api.TransportError("dosya kalici hale getirilemedi") from None


def handle_identity(handle: Handle) -> str:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    try:
        return _identity_text(os.fstat(handle.fd))
    except OSError:
        raise _api.TransportError("acik nesne durumu okunamadi") from None


def handle_mode(handle: Handle) -> str:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    try:
        return _mode_text(os.fstat(handle.fd).st_mode)
    except OSError:
        raise _api.TransportError("acik nesne durumu okunamadi") from None


def set_handle_mode(handle: Handle, mode: str) -> None:
    """The recorded permission bits, applied through the DESCRIPTOR.

    `os.chmod(path)` after a create is a second lookup, and the object at
    that name is not guaranteed to be the one just created."""
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    if not mode.startswith("0o"):
        return
    try:
        bits = int(mode, 8) & 0o7777
    except ValueError:
        raise _api.TransportError("kip cozumlenemedi") from None
    if not bits:
        return
    try:
        os.fchmod(handle.fd, bits)
    except OSError:
        raise _api.TransportError("kip uygulanamadi") from None


def close_handle(handle: Handle) -> None:
    if handle.closed:
        return
    try:
        os.close(handle.fd)
    except OSError:
        raise _api.TransportError("tanimlayici kapatilamadi") from None
    handle.closed = True
