"""Windows half of the handle-bound transport. PACKAGE B2B-A-D2.

Win32 has no `openat`, so the first design opened children by full path
and bound them afterwards by FileId. NT does have one: `NtCreateFile`
takes an OBJECT_ATTRIBUTES carrying a RootDirectory HANDLE and a
RELATIVE name. Measured working, so no path is built after the root at
all.

MEASURED BEHAVIOUR THIS MODULE RELIES ON:

  * `GetFileInformationByHandleEx` / `FileIdExtdDirectoryInfo` returns
    name, attributes, reparse tag, size, timestamps and a 128-bit
    FileId. Header is 88 bytes; 500 entries paginated over 18 buffers
    of 4 KiB and ended with ERROR_NO_MORE_FILES.
  * A child opened relative to a held directory handle, after that
    directory's PATH was swapped for a junction, failed with
    STATUS_DELETE_PENDING. It never reached the outside file.
  * `NtCreateFile` ACCEPTED `pipeline\\a.py` as a relative name and
    opened the grandchild -- so names from a listing are validated
    before they are used, every time.

THE ENUMERATION IS TRUSTED FOR NAMES AND NOTHING ELSE. Its per-child
metadata is a lazily flushed copy of the parent's index entry. Measured
with nothing else touching the file: a same-size in-place rewrite was
invisible in 117 of 200 attempts and an atomic replace in 137 of 200 --
the record still carried the OLD file identity. `os.stat` was then
measured to be what flushes that entry, which is why every probe that
compared enumeration against `os.stat` agreed perfectly and proved
nothing.

So every child is asked directly, through a relative open that requests
FILE_READ_ATTRIBUTES and no data access at all -- Windows' way of
spelling a stat relative to a directory handle, since it has no
`fstatat`. Measured over the real protected roots: 184,685 files in
10.96 seconds, against 12.5 seconds for the path-based inventory this
replaced, and it yields a true link count as well.
"""
from __future__ import annotations

import ctypes
import msvcrt
import os
from ctypes import wintypes

from tools.agent_loop import fs_transport as _api

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

FILE_LIST_DIRECTORY = 0x00000001
FILE_READ_DATA = 0x00000001
FILE_READ_ATTRIBUTES = 0x00000080
FILE_WRITE_DATA = 0x00000002
FILE_APPEND_DATA = 0x00000004
DELETE = 0x00010000
WRITE_DAC = 0x00040000
WRITE_OWNER = 0x00080000
GENERIC_READ = 0x80000000
SYNCHRONIZE = 0x00100000

# The three access sets this module is allowed to ask for, named so a
# test can assert on them rather than on a magic number.
#
# METADATA is the one that matters. The normative invariant is NOT that
# no handle is opened -- Windows has no `fstatat`, so a stat IS a handle
# -- it is that no right to read data is ever requested, no data
# descriptor is ever produced, and no data byte is ever read.
METADATA_ACCESS = FILE_READ_ATTRIBUTES | SYNCHRONIZE
CONTENT_ACCESS = FILE_READ_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE
DIRECTORY_ACCESS = FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE
FORBIDDEN_ACCESS = (FILE_WRITE_DATA | FILE_APPEND_DATA | DELETE | WRITE_DAC
                    | WRITE_OWNER | GENERIC_READ)

FSCTL_GET_REPARSE_POINT = 0x000900A8
MAXIMUM_REPARSE_DATA_BUFFER_SIZE = 16384
_SHARE_ALL = 0x00000007                    # read | write | delete
_OPEN_EXISTING = 3
_FLAG_BACKUP_SEMANTICS = 0x02000000
_FLAG_OPEN_REPARSE_POINT = 0x00200000
_INVALID = ctypes.c_void_p(-1).value

_FILE_OPEN = 1
_OBJ_CASE_INSENSITIVE = 0x00000040
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_OPEN_REPARSE_POINT = 0x00200000

_FileIdInfo = 18
_FileIdExtdDirectoryInfo = 19
_FileIdExtdDirectoryRestartInfo = 20

_ERROR_NO_MORE_FILES = 18

ATTR_DIRECTORY = 0x00000010
ATTR_REPARSE = 0x00000400
ATTR_DEVICE = 0x00000040

# One page. 500 names paginated fine at 4 KiB; 64 KiB keeps the syscall
# count down without making a malformed buffer harder to reason about.
_PAGE = 1 << 16
# A page cannot hold more records than it has room for headers. The cap
# exists so a buffer whose offsets form a cycle ends as a refusal
# instead of a hang.
_MAX_PER_PAGE = _PAGE // 88 + 2


class _FILE_ID_128(ctypes.Structure):
    _fields_ = [("Identifier", ctypes.c_ubyte * 16)]


class _FILE_ID_EXTD_DIR_INFO(ctypes.Structure):
    _fields_ = [
        ("NextEntryOffset", wintypes.ULONG), ("FileIndex", wintypes.ULONG),
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong), ("EndOfFile", ctypes.c_longlong),
        ("AllocationSize", ctypes.c_longlong),
        ("FileAttributes", wintypes.ULONG), ("FileNameLength", wintypes.ULONG),
        ("EaSize", wintypes.ULONG), ("ReparsePointTag", wintypes.ULONG),
        ("FileId", _FILE_ID_128), ("FileName", wintypes.WCHAR * 1),
    ]


class _FILE_ID_INFO(ctypes.Structure):
    _fields_ = [("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", _FILE_ID_128)]


class _FILE_BASIC_INFO(ctypes.Structure):
    _fields_ = [("CreationTime", ctypes.c_longlong),
                ("LastAccessTime", ctypes.c_longlong),
                ("LastWriteTime", ctypes.c_longlong),
                ("ChangeTime", ctypes.c_longlong),
                ("FileAttributes", wintypes.ULONG)]


class _FILE_STANDARD_INFO(ctypes.Structure):
    _fields_ = [("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.ULONG),
                ("DeletePending", ctypes.c_ubyte),
                ("Directory", ctypes.c_ubyte)]


class _FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.ULONG),
                ("ReparseTag", wintypes.ULONG)]


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p)]


class _OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", ctypes.c_void_p),
                ("ObjectName", ctypes.POINTER(_UNICODE_STRING)),
                ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", ctypes.c_void_p),
                ("SecurityQualityOfService", ctypes.c_void_p)]


class _IO_STATUS_BLOCK(ctypes.Structure):
    _fields_ = [("Status", ctypes.c_void_p),
                ("Information", ctypes.c_void_p)]


# Pointer-width return and argument types, every one of them: a HANDLE
# declared as int is silently truncated on 64-bit, which this package
# has already paid for once in the process container.
_k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p]
_k32.CreateFileW.restype = ctypes.c_void_p
_k32.CloseHandle.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.restype = wintypes.BOOL
_k32.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
_k32.GetFileInformationByHandleEx.restype = wintypes.BOOL
_k32.DeviceIoControl.argtypes = [
    ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
    ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ctypes.c_void_p]
_k32.DeviceIoControl.restype = wintypes.BOOL
_ntdll.NtCreateFile.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
    ctypes.POINTER(_OBJECT_ATTRIBUTES), ctypes.POINTER(_IO_STATUS_BLOCK),
    ctypes.c_void_p, wintypes.ULONG, wintypes.ULONG, wintypes.ULONG,
    wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
_ntdll.NtCreateFile.restype = ctypes.c_long

_HEADER = _FILE_ID_EXTD_DIR_INFO.FileName.offset


class Directory:
    """An open directory handle and the volume it lives on."""

    __slots__ = ("handle", "volume", "identity", "closed")

    def __init__(self, handle, volume: int, identity: str):
        self.handle = handle
        self.volume = volume
        self.identity = identity
        self.closed = False


def _close(handle):
    """Closing is an OPERATION, not a gesture.

    The first version discarded `CloseHandle`'s result, so a handle that
    refused to close was reported as released. A cleanup that cannot be
    proved did not happen."""
    if not _k32.CloseHandle(ctypes.c_void_p(handle)):
        raise _api.TransportError("nesne kapatilamadi")



def _fold(exc: BaseException, handle) -> None:
    """Close on a failure path and CONSUME the result.

    A cleanup problem may not replace the error already being raised,
    and may not vanish either -- so it becomes a note on that error."""
    if not _close_quietly(handle):
        _api.mark_cleanup_failed(exc)


def _fail(message: str, handle) -> "_api.TransportError":
    """Build a refusal, closing the handle and folding any cleanup
    failure into the refusal being built."""
    hata = _api.TransportError(message)
    _fold(hata, handle)
    return hata


def _close_quietly(handle) -> bool:
    """For the failure paths, where a close failure must not replace the
    error that got us here -- but must still be VISIBLE, so it is
    returned rather than swallowed."""
    try:
        _close(handle)
        return True
    except _api.TransportError:
        return False


def _identity_of(handle) -> tuple:
    info = _FILE_ID_INFO()
    if not _k32.GetFileInformationByHandleEx(
            ctypes.c_void_p(handle), _FileIdInfo, ctypes.byref(info),
            ctypes.sizeof(info)):
        raise _api.TransportError("nesne kimligi okunamadi")
    return info.VolumeSerialNumber, bytes(info.FileId.Identifier)


def _identity_text(volume: int, file_id: bytes) -> str:
    return f"{volume:016x}:{file_id.hex()}"


def _kind(attributes: int, reparse: int) -> str:
    if attributes & ATTR_REPARSE or reparse:
        return "link"
    if attributes & ATTR_DEVICE:
        return "other"
    return "dir" if attributes & ATTR_DIRECTORY else "file"


def _assert_plain_directory(handle):
    """An open handle names a plain directory -- not a file, and not a
    reparse point wearing the directory attribute."""
    etiket = _query(handle, 9, _FILE_ATTRIBUTE_TAG_INFO)
    if etiket.FileAttributes & ATTR_REPARSE or etiket.ReparseTag:
        raise _api.TransportError("dizin bir yeniden ayrisma noktasi")
    if not etiket.FileAttributes & ATTR_DIRECTORY:
        raise _api.TransportError("nesne bir dizin degil")


def open_root(path) -> Directory:
    metin = os.fspath(path)
    if not metin.startswith("\\\\?\\"):
        metin = "\\\\?\\" + os.path.abspath(metin)
    handle = _k32.CreateFileW(
        metin, FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        _SHARE_ALL, None, _OPEN_EXISTING,
        _FLAG_BACKUP_SEMANTICS | _FLAG_OPEN_REPARSE_POINT, None)
    if handle in (None, _INVALID):
        raise _api.TransportError("kok dizin acilamadi")
    try:
        # BACKUP_SEMANTICS opens a plain file just as happily, so the
        # root's own type is checked rather than assumed
        _assert_plain_directory(handle)
        volume, file_id = _identity_of(handle)
    except _api.TransportError as exc:
        _fold(exc, handle)
        raise
    return Directory(handle, volume, _identity_text(volume, file_id))


def close_directory(directory: Directory):
    """`closed` means CLOSED, not "close was attempted".

    The first version set the flag first, so a handle that refused to
    close was marked released and leaked in silence."""
    if directory.closed:
        return
    _close(directory.handle)
    directory.closed = True


def directory_identity(directory: Directory) -> tuple:
    return directory.identity


def descriptor_identity(fd: int) -> tuple:
    try:
        info = os.fstat(fd)
    except OSError:
        raise _api.TransportError("acik nesne durumu okunamadi") from None
    return (info.st_mode, info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_nlink,
            getattr(info, "st_file_attributes", 0))


def _query(handle, sinif: int, yapi):
    bilgi = yapi()
    if not _k32.GetFileInformationByHandleEx(
            ctypes.c_void_p(handle), sinif, ctypes.byref(bilgi),
            ctypes.sizeof(bilgi)):
        raise _api.TransportError("nesne bilgisi okunamadi")
    return bilgi


def _attributes_only_open(directory: "Directory", name: str):
    """A relative open with NO data access at all.

    This is what a `stat` relative to a directory handle looks like on
    Windows, which has no `fstatat`. Neither FILE_DIRECTORY_FILE nor
    FILE_NON_DIRECTORY_FILE is requested, so the same call serves a
    file, a directory and a reparse point."""
    ad = _api.validate_child_name(name)
    tampon = ctypes.create_unicode_buffer(ad)
    isim = _UNICODE_STRING(len(ad) * 2, len(ad) * 2,
                           ctypes.cast(tampon, ctypes.c_void_p))
    nitelik = _OBJECT_ATTRIBUTES(
        ctypes.sizeof(_OBJECT_ATTRIBUTES), ctypes.c_void_p(directory.handle),
        ctypes.pointer(isim), _OBJ_CASE_INSENSITIVE, None, None)
    durum = _IO_STATUS_BLOCK()
    handle = ctypes.c_void_p()
    nt = _ntdll.NtCreateFile(
        ctypes.byref(handle), FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        ctypes.byref(nitelik), ctypes.byref(durum), None, 0, _SHARE_ALL,
        _FILE_OPEN, _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT,
        None, 0)
    if nt != 0:
        raise _api.TransportError("giris goreli olarak sorgulanamadi")
    return handle.value


def _authoritative_record(directory: "Directory", name: str) -> _api.Record:
    """The child's metadata FROM THE CHILD, not from the parent's index.

    MEASURED, and this is why the enumeration is used for names only: a
    parent's index entry is a lazily flushed copy. With nothing else
    touching the file, a same-size in-place rewrite was invisible in
    117 of 200 attempts and an atomic replace -- which changes the file
    identity itself -- in 137 of 200. `os.stat` was then measured to be
    what flushes the entry, which is why every earlier probe that
    compared enumeration against `os.stat` saw perfect agreement and
    proved nothing.

    A metadata class that misses a change is worse than no metadata
    class, so the object is asked directly. Attributes only, no data
    access: 184,685 files in 10.96 seconds, against the 12.5-second
    path-based inventory this design replaced."""
    handle = _attributes_only_open(directory, name)
    try:
        basic = _query(handle, 0, _FILE_BASIC_INFO)
        standard = _query(handle, 1, _FILE_STANDARD_INFO)
        etiket = _query(handle, 9, _FILE_ATTRIBUTE_TAG_INFO)
        kimlik = _query(handle, _FileIdInfo, _FILE_ID_INFO)
    except BaseException as exc:
        # a cleanup problem must not replace the error that got us here
        _fold(exc, handle)
        raise
    _close(handle)
    attributes = basic.FileAttributes
    return _api.Record(
        name=name, kind=_kind(attributes, etiket.ReparseTag),
        attributes=attributes, reparse_tag=etiket.ReparseTag, mode=0,
        size=0 if standard.Directory else standard.EndOfFile,
        mtime_ns=(basic.LastWriteTime - 116444736000000000) * 100,
        file_id=_identity_text(kimlik.VolumeSerialNumber,
                               bytes(kimlik.FileId.Identifier)),
        nlink=standard.NumberOfLinks)


def _page_names(tampon):
    """Parse ONE enumeration buffer.

    Every field that indexes into the buffer is checked before it is
    used. A malformed page is a refusal: this data crosses a trust
    boundary the same way a network packet does."""
    adlar, ofset, adet = [], 0, 0
    boy = len(tampon)
    while True:
        adet += 1
        if adet > _MAX_PER_PAGE:
            raise _api.TransportError("listeleme tamponu tutarsiz")
        if ofset < 0 or ofset + _HEADER > boy:
            raise _api.TransportError("listeleme tamponu tutarsiz")
        kayit = _FILE_ID_EXTD_DIR_INFO.from_buffer(tampon, ofset)
        ad_bayt = kayit.FileNameLength
        if ad_bayt <= 0 or ad_bayt % 2 or ofset + _HEADER + ad_bayt > boy:
            raise _api.TransportError("listeleme tamponu tutarsiz")
        ad = ctypes.wstring_at(ctypes.addressof(tampon) + ofset + _HEADER,
                               ad_bayt // 2)
        sonraki = kayit.NextEntryOffset
        if ad not in (".", ".."):
            adlar.append(_api.validate_child_name(ad))
        if sonraki == 0:
            return adlar
        # must advance PAST this record and stay inside the page, or the
        # walk can be made to loop or to read someone else's memory
        if sonraki < _HEADER + ad_bayt or ofset + sonraki + _HEADER > boy:
            raise _api.TransportError("listeleme tamponu tutarsiz")
        ofset += sonraki


def list_directory(directory: Directory) -> tuple:
    """Names from the enumeration; every other field from the child.

    The enumeration is trusted for exactly one thing -- which names
    exist -- because its per-child metadata is a lazily flushed copy
    that was measured missing 117 of 200 same-size rewrites and 137 of
    200 atomic replaces."""
    if directory.closed:
        raise _api.TransportError("kapali dizin listelendi")
    tampon = ctypes.create_string_buffer(_PAGE)
    sinif = _FileIdExtdDirectoryRestartInfo
    adlar = []
    while True:
        ctypes.set_last_error(0)
        ok = _k32.GetFileInformationByHandleEx(
            ctypes.c_void_p(directory.handle), sinif, tampon, _PAGE)
        if not ok:
            if ctypes.get_last_error() == _ERROR_NO_MORE_FILES:
                break
            # ONLY no-more-files ends a listing normally. Everything
            # else is an incomplete answer, and an incomplete inventory
            # that looks complete is the failure mode this package has
            # been bitten by more than once.
            raise _api.TransportError("dizin listelenemedi")
        sinif = _FileIdExtdDirectoryInfo
        adlar.extend(_page_names(tampon))

    gorulen = set()
    for ad in adlar:
        if ad in gorulen:
            raise _api.TransportError("listelemede yinelenen ad")
        gorulen.add(ad)
    return tuple(_authoritative_record(directory, ad) for ad in adlar)


def _relative_open(directory: Directory, name: str, *, is_directory: bool):
    """`openat`, spelled the way NT spells it."""
    ad = _api.validate_child_name(name)
    tampon = ctypes.create_unicode_buffer(ad)
    isim = _UNICODE_STRING(len(ad) * 2, len(ad) * 2,
                           ctypes.cast(tampon, ctypes.c_void_p))
    nitelik = _OBJECT_ATTRIBUTES(
        ctypes.sizeof(_OBJECT_ATTRIBUTES), ctypes.c_void_p(directory.handle),
        ctypes.pointer(isim), _OBJ_CASE_INSENSITIVE, None, None)
    durum = _IO_STATUS_BLOCK()
    handle = ctypes.c_void_p()
    erisim = (FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE
              if is_directory
              else FILE_READ_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE)
    secenek = (_FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
               | (_FILE_DIRECTORY_FILE if is_directory
                  else _FILE_NON_DIRECTORY_FILE))
    nt = _ntdll.NtCreateFile(ctypes.byref(handle), erisim,
                             ctypes.byref(nitelik), ctypes.byref(durum), None,
                             0, _SHARE_ALL, _FILE_OPEN, secenek, None, 0)
    if nt != 0:
        # NTSTATUS carries no path, but it is still the OS talking --
        # one fixed sentence leaves this module
        raise _api.TransportError("giris goreli olarak acilamadi")
    return handle.value


def _close_descriptor_quietly(descriptor: int) -> bool:
    """Closes, and SAYS whether it worked."""
    try:
        os.close(descriptor)
        return True
    except OSError:
        return False


def _fold_descriptor(exc: BaseException, descriptor: int) -> None:
    """The descriptor twin of `_fold`.

    MEASURED without it: on all three of the descriptor-setup refusals
    -- unreadable `fstat`, reparse point, wrong type -- a failing close
    REPLACED the primary error, and the caller was told the descriptor
    would not close rather than why the file was refused."""
    if not _close_descriptor_quietly(descriptor):
        _api.mark_cleanup_failed(exc)


def _fail_descriptor(message: str, descriptor: int) -> "_api.TransportError":
    hata = _api.TransportError(message)
    _fold_descriptor(hata, descriptor)
    return hata


def _bind(handle, directory: Directory, record: _api.Record):
    """The opened object must be the enumerated one, checked before a
    single byte is read."""
    try:
        volume, file_id = _identity_of(handle)
    except _api.TransportError as exc:
        _fold(exc, handle)
        raise
    if _identity_text(volume, file_id) != record.file_id:
        raise _fail("acilan nesne listelenen nesne degil", handle)


def open_child_directory(directory: Directory,
                         record: _api.Record) -> Directory:
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden acilis")
    if record.kind != "dir":
        raise _api.TransportError("giris bir dizin olarak listelenmedi")
    handle = _relative_open(directory, record.name, is_directory=True)
    _bind(handle, directory, record)
    try:
        # FILE_OPEN_REPARSE_POINT opens the junction ITSELF, which does
        # carry the directory attribute -- so descending would list
        # whatever it points at. The type is checked from the handle.
        _assert_plain_directory(handle)
        volume, file_id = _identity_of(handle)
    except _api.TransportError as exc:
        _fold(exc, handle)
        raise
    return Directory(handle, volume, _identity_text(volume, file_id))


def open_child_file(directory: Directory, record: _api.Record) -> int:
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden acilis")
    if record.kind != "file":
        raise _api.TransportError("giris siradan bir dosya olarak "
                                  "listelenmedi")
    handle = _relative_open(directory, record.name, is_directory=False)
    _bind(handle, directory, record)
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except OSError:
        # ownership never transferred, so the handle is still ours
        raise _fail("dosya tanimlayicisi olusturulamadi", handle)
    try:
        info = os.fstat(descriptor)
    except OSError:
        raise _fail_descriptor("acik nesne durumu okunamadi",
                               descriptor) from None
    if getattr(info, "st_file_attributes", 0) & ATTR_REPARSE:
        raise _fail_descriptor("son bilesen yeniden ayrisma noktasi",
                               descriptor)
    if not (info.st_mode & 0o100000):          # S_IFREG
        raise _fail_descriptor("son bilesen siradan bir dosya degil",
                               descriptor)
    return descriptor


def link_evidence(directory: Directory, record: _api.Record) -> bytes:
    """The reparse point's own data, as OPAQUE BYTES for fingerprinting.

    Never parsed into a target path and never returned as text: the
    caller keys these bytes into an HMAC and stores only that. A reparse
    target can name a private location, and this package has already
    been caught once shipping an unkeyed digest of a short path.

    Handle-bound throughout. The entry was named by the parent handle's
    listing, it is reopened RELATIVE to that handle with
    FILE_OPEN_REPARSE_POINT, its volume-qualified FileId is bound to the
    listing BEFORE the control code runs, and the access requested is
    attributes only -- measured sufficient: symlink 268 bytes, directory
    symlink and junction 276, and a plain file answers
    ERROR_NOT_A_REPARSE_POINT.

    An unknown tag is fingerprinted like any other and never followed.
    Following one is not something this module does for ANY tag."""
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinden okuma")
    if record.kind != "link":
        raise _api.TransportError("giris bir baglanti olarak listelenmedi")
    handle = _attributes_only_open(directory, record.name)
    # identity FIRST, and OUTSIDE the block below: the control code must
    # never run against an object the listing did not name, and `_bind`
    # owns the handle's cleanup on its own failure
    _bind(handle, directory, record)
    try:
        tampon = ctypes.create_string_buffer(MAXIMUM_REPARSE_DATA_BUFFER_SIZE)
        donen = wintypes.DWORD(0)
        if not _k32.DeviceIoControl(
                ctypes.c_void_p(handle), FSCTL_GET_REPARSE_POINT, None, 0,
                tampon, MAXIMUM_REPARSE_DATA_BUFFER_SIZE,
                ctypes.byref(donen), None):
            raise _api.TransportError("ayrisma noktasi verisi okunamadi")
        uzunluk = donen.value
        if uzunluk <= 0 or uzunluk > MAXIMUM_REPARSE_DATA_BUFFER_SIZE:
            raise _api.TransportError("ayrisma noktasi verisi tutarsiz")
        ham = tampon.raw[:uzunluk]
    except BaseException as exc:
        _fold(exc, handle)
        raise
    _close(handle)
    return ham
