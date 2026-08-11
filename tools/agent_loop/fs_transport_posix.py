"""POSIX half of the handle-bound transport. PACKAGE B2B-A-D2.

Everything here hangs off a directory FILE DESCRIPTOR. `dir_fd` is the
kernel saying "relative to THIS object", which is the guarantee the
path-based walker could never obtain.

NO SILENT FALLBACK. If a primitive is missing, the platform is refused.
A walk that quietly degrades to paths would be the same defect wearing
this module's name.
"""
from __future__ import annotations

import os
import stat

from tools.agent_loop import fs_transport as _api

_REQUIRED = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")


def _assert_platform():
    for ad in _REQUIRED:
        if not hasattr(os, ad):
            raise _api.TransportError("platform no-follow acilisi desteklemiyor")
    if os.stat not in os.supports_dir_fd or os.open not in os.supports_dir_fd:
        raise _api.TransportError("platform dizin-goreli acilisi desteklemiyor")
    if os.scandir not in os.supports_fd:
        raise _api.TransportError("platform tanimlayiciyla listeleme yapmiyor")


_DIR_FLAGS = None


def _dir_flags() -> int:
    global _DIR_FLAGS
    if _DIR_FLAGS is None:
        _assert_platform()
        _DIR_FLAGS = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                      | os.O_CLOEXEC)
    return _DIR_FLAGS


class Directory:
    """An open directory. The ONLY thing a caller may ask about its
    children."""

    __slots__ = ("fd", "identity", "closed")

    def __init__(self, fd: int, identity: tuple):
        self.fd = fd
        self.identity = identity
        self.closed = False


def _identity_text(info) -> str:
    return f"{info.st_dev}:{info.st_ino}"


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "link"
    if stat.S_ISDIR(mode):
        return "dir"
    return "file" if stat.S_ISREG(mode) else "other"


def _record(name: str, info) -> _api.Record:
    return _api.Record(
        name=name, kind=_kind(info.st_mode), attributes=0, reparse_tag=0,
        mode=info.st_mode, size=info.st_size, mtime_ns=info.st_mtime_ns,
        file_id=_identity_text(info), nlink=info.st_nlink)


def descriptor_identity(fd: int) -> tuple:
    """The fields that must not move while an object is being read.

    Same shape on both platforms, so the walker's read-integrity check
    is one piece of code rather than two."""
    try:
        info = os.fstat(fd)
    except OSError:
        raise _api.TransportError("acik nesne durumu okunamadi") from None
    return (info.st_mode, info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_nlink, 0)


def directory_identity(directory: Directory) -> tuple:
    return directory.identity


def open_root(path) -> Directory:
    try:
        fd = os.open(path, _dir_flags())
    except OSError:
        raise _api.TransportError("kok dizin acilamadi") from None
    try:
        info = os.fstat(fd)
    except OSError:
        raise _fail("kok durumu okunamadi", fd)
    return Directory(fd, _identity_text(info))


def _close_fd(fd: int):
    """A close that cannot be proved did not happen.

    EINTR is NOT retried: on Linux the descriptor is already released
    when `close` reports it, so a blind retry closes whatever number was
    handed out next. If the outcome cannot be established, the caller is
    told rather than reassured."""
    try:
        os.close(fd)
    except OSError:
        raise _api.TransportError("tanimlayici kapatilamadi") from None



def _fold(exc: BaseException, fd: int) -> None:
    """Close on a failure path and CONSUME the result: a cleanup
    problem becomes a note on the error already being raised."""
    if not _close_fd_quietly(fd):
        _api.mark_cleanup_failed(exc)


def _fail(message: str, fd: int) -> "_api.TransportError":
    hata = _api.TransportError(message)
    _fold(hata, fd)
    return hata


def _close_fd_quietly(fd: int) -> bool:
    """For failure paths, where a cleanup problem must not replace the
    error that got us here -- but must still be visible."""
    try:
        _close_fd(fd)
        return True
    except _api.TransportError:
        return False


def close_directory(directory: Directory):
    """`closed` means CLOSED, not "close was attempted"."""
    if directory.closed:
        return
    _close_fd(directory.fd)
    directory.closed = True


def list_directory(directory: Directory) -> tuple:
    """List from the DESCRIPTOR. Child metadata comes from the same
    descriptor, never from a path built out of the name."""
    if directory.closed:
        raise _api.TransportError("kapali dizin listelendi")
    kayitlar = []
    try:
        with os.scandir(directory.fd) as girisler:
            adlar = [giris.name for giris in girisler]
    except OSError:
        raise _api.TransportError("dizin listelenemedi") from None
    for ad in adlar:
        _api.validate_child_name(ad)
        try:
            info = os.stat(ad, dir_fd=directory.fd, follow_symlinks=False)
        except OSError:
            raise _api.TransportError("giris durumu okunamadi") from None
        kayitlar.append(_record(ad, info))
    return tuple(kayitlar)


def _bind(fd: int, record: _api.Record):
    """The opened object must be the enumerated one. Checked BEFORE a
    byte is read, so a mismatch costs the caller nothing."""
    try:
        info = os.fstat(fd)
    except OSError:
        raise _fail("acik nesne durumu okunamadi", fd)
    if _identity_text(info) != record.file_id:
        raise _fail("acilan nesne listelenen nesne degil", fd)
    return info


def open_child_directory(directory: Directory, record: _api.Record) -> Directory:
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden acilis")
    if record.kind != "dir":
        raise _api.TransportError("giris bir dizin olarak listelenmedi")
    ad = _api.validate_child_name(record.name)
    try:
        fd = os.open(ad, _dir_flags(), dir_fd=directory.fd)
    except OSError:
        raise _api.TransportError("alt dizin acilamadi") from None
    info = _bind(fd, record)
    if not stat.S_ISDIR(info.st_mode):
        raise _fail("alt dizin bir dizin degil", fd)
    return Directory(fd, _identity_text(info))


def open_child_file(directory: Directory, record: _api.Record) -> int:
    """A descriptor for an ordinary child file, already bound to the
    listing that named it.

    `O_NONBLOCK` because `O_RDONLY` on a FIFO blocks until a writer
    appears, and a verification pass that hangs is worse than one that
    refuses."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden acilis")
    if record.kind != "file":
        raise _api.TransportError("giris siradan bir dosya olarak "
                                  "listelenmedi")
    ad = _api.validate_child_name(record.name)
    flags = (os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(ad, flags, dir_fd=directory.fd)
    except OSError:
        raise _api.TransportError("dosya no-follow acilamadi") from None
    info = _bind(fd, record)
    if not stat.S_ISREG(info.st_mode):
        raise _fail("son bilesen siradan bir dosya degil", fd)
    return fd


def link_evidence(directory: Directory, record: _api.Record) -> bytes:
    """The link target as OPAQUE BYTES for fingerprinting.

    Bytes rather than text, and the same shape Windows returns for a
    reparse buffer, so the caller has exactly one thing to key into an
    HMAC and no reason to ever look at a target."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden okuma")
    if record.kind != "link":
        raise _api.TransportError("giris bir baglanti olarak listelenmedi")
    ad = _api.validate_child_name(record.name)

    def kimlik():
        try:
            return _identity_text(
                os.stat(ad, dir_fd=directory.fd, follow_symlinks=False))
        except OSError:
            raise _api.TransportError("giris durumu okunamadi") from None

    # The identity is bound on BOTH sides of the read, the same way
    # Windows binds a reparse point before its control code runs. There
    # is no `readlinkat` on an already-open link in Python, so a swap
    # placed entirely inside this window and undone afterwards is the
    # declared residue -- narrower than reading a link the listing never
    # named, which is what happens without any binding at all.
    if kimlik() != record.file_id:
        raise _api.TransportError("acilan nesne listelenen nesne degil")
    try:
        hedef = os.readlink(ad, dir_fd=directory.fd)
    except OSError:
        raise _api.TransportError("baglanti hedefi okunamadi") from None
    if kimlik() != record.file_id:
        raise _api.TransportError("baglanti okunurken degisti")
    return hedef.encode("utf-8", "surrogatepass")
