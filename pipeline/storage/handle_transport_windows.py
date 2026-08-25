"""Windows half of the production handle-bound storage transport.

Win32 has no `openat`, no `mkdirat` and no `renameat`. NT has all three:
`NtCreateFile` takes an OBJECT_ATTRIBUTES carrying a RootDirectory HANDLE
and a RELATIVE name, and `NtSetInformationFile` renames or removes THE
OBJECT A HANDLE NAMES -- the rename even takes its destination directory
as another handle. So no path is built after the root.

MEASURED BEHAVIOUR THIS MODULE RELIES ON:

  * `NtCreateFile` + FILE_CREATE creates a child by relative name and
    answers STATUS_OBJECT_NAME_COLLISION (0xC0000035) when one is already
    there. That refusal is the ADDED-collision gate, enforced by the
    kernel rather than by a check somebody can race.
  * `NtSetInformationFile` / `FileRenameInformationEx` with
    RootDirectory set to the destination handle moved a file and
    PRESERVED its 128-bit FileId; with REPLACE_IF_EXISTS withheld, an
    occupied destination collided instead of being overwritten.
  * `FileDispositionInformationEx` removed a directory through its own
    handle and answered STATUS_DIRECTORY_NOT_EMPTY (0xC0000101) rather
    than recursing.
  * With the parent directory's PATH swapped for a junction after the
    handle was taken, a create through that handle landed in the
    ORIGINAL directory; the junction's target stayed empty.
  * Without FILE_OPEN_REPARSE_POINT the same open FOLLOWS the junction,
    which is why every open here passes it.

NTSTATUS CARRIES NO PATH, but it is still the operating system talking,
so exactly two of them are read for meaning -- collision and
not-empty -- and everything else becomes one fixed sentence.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from pipeline.storage import handle_transport as _api

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

FILE_LIST_DIRECTORY = 0x00000001
FILE_READ_DATA = 0x00000001
FILE_WRITE_DATA = 0x00000002
FILE_TRAVERSE = 0x00000020
FILE_ADD_FILE = 0x00000002
FILE_ADD_SUBDIRECTORY = 0x00000004
FILE_READ_ATTRIBUTES = 0x00000080
FILE_WRITE_ATTRIBUTES = 0x00000100
DELETE = 0x00010000
SYNCHRONIZE = 0x00100000

_SHARE_ALL = 0x00000007
_OPEN_EXISTING = 3
_FLAG_BACKUP_SEMANTICS = 0x02000000
_FLAG_OPEN_REPARSE_POINT = 0x00200000
_INVALID = ctypes.c_void_p(-1).value

_FILE_OPEN = 1
_FILE_CREATE = 2
_OBJ_CASE_INSENSITIVE = 0x00000040
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_OPEN_REPARSE_POINT = 0x00200000

_FileBasicInfo = 0
_FileStandardInfo = 1
_FileAttributeTagInfo = 9
_FileIdInfo = 18
_FileDispositionInformation = 13
_FileDispositionInformationEx = 64
_FileRenameInformation = 10
_FileRenameInformationEx = 65

_FILE_RENAME_POSIX_SEMANTICS = 0x00000002
_FILE_DISPOSITION_DELETE = 0x00000001
_FILE_DISPOSITION_POSIX_SEMANTICS = 0x00000002

# The only two statuses read for MEANING. Everything else is one fixed
# sentence, because an NTSTATUS table in a refusal is the OS talking.
_STATUS_OBJECT_NAME_COLLISION = 0xC0000035
_STATUS_DIRECTORY_NOT_EMPTY = 0xC0000101
_STATUS_OBJECT_NAME_NOT_FOUND = 0xC0000034
_STATUS_OBJECT_PATH_NOT_FOUND = 0xC000003A
_STATUS_NOT_A_DIRECTORY = 0xC0000103
# the two an older kernel answers for an information class it lacks
_STATUS_INVALID_INFO_CLASS = 0xC0000003
_STATUS_NOT_SUPPORTED = 0xC00000BB

ATTR_DIRECTORY = 0x00000010
ATTR_REPARSE = 0x00000400
ATTR_DEVICE = 0x00000040
ATTR_NORMAL = 0x00000080


class _FILE_ID_128(ctypes.Structure):
    _fields_ = [("Identifier", ctypes.c_ubyte * 16)]


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
    _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_void_p)]


class _FILE_RENAME_INFORMATION(ctypes.Structure):
    """The union's first member is `ULONG Flags` on the Ex class and
    `BOOLEAN ReplaceIfExists` on the classic one -- which is why the
    classic path below writes a ZERO there rather than reusing the Ex
    flags: `FILE_RENAME_POSIX_SEMANTICS` is 2, and a non-zero byte in
    that position means REPLACE."""

    _fields_ = [("Flags", wintypes.ULONG), ("RootDirectory", ctypes.c_void_p),
                ("FileNameLength", wintypes.ULONG),
                ("FileName", wintypes.WCHAR * 1)]


class _FILE_DISPOSITION_INFORMATION_EX(ctypes.Structure):
    _fields_ = [("Flags", wintypes.ULONG)]


class _FILE_DISPOSITION_INFORMATION(ctypes.Structure):
    _fields_ = [("DeleteFile", ctypes.c_ubyte)]


# Pointer-width types on every argument and return: a HANDLE declared as
# int is silently truncated on 64-bit, which this package has already
# paid for once in the process container.
_k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p]
_k32.CreateFileW.restype = ctypes.c_void_p
_k32.CloseHandle.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.restype = wintypes.BOOL
_k32.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
_k32.GetFileInformationByHandleEx.restype = wintypes.BOOL
_k32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
_k32.FlushFileBuffers.restype = wintypes.BOOL
_k32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_k32.WriteFile.restype = wintypes.BOOL
_k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_k32.ReadFile.restype = wintypes.BOOL
_ntdll.NtCreateFile.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
    ctypes.POINTER(_OBJECT_ATTRIBUTES), ctypes.POINTER(_IO_STATUS_BLOCK),
    ctypes.c_void_p, wintypes.ULONG, wintypes.ULONG, wintypes.ULONG,
    wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
_ntdll.NtCreateFile.restype = ctypes.c_long
_ntdll.NtSetInformationFile.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(_IO_STATUS_BLOCK), ctypes.c_void_p,
    wintypes.ULONG, ctypes.c_int]
_ntdll.NtSetInformationFile.restype = ctypes.c_long

_DIRECTORY_ACCESS = (FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_ADD_FILE
                     | FILE_ADD_SUBDIRECTORY | FILE_READ_ATTRIBUTES
                     | SYNCHRONIZE)
_FILE_WRITE_ACCESS = (FILE_WRITE_DATA | FILE_READ_DATA | FILE_READ_ATTRIBUTES
                      | FILE_WRITE_ATTRIBUTES | SYNCHRONIZE | DELETE)
_FILE_READ_ACCESS = FILE_READ_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE
_RENAME_ACCESS = DELETE | FILE_READ_ATTRIBUTES | SYNCHRONIZE
_REMOVE_ACCESS = DELETE | FILE_READ_ATTRIBUTES | SYNCHRONIZE

_RENAME_HEADER = _FILE_RENAME_INFORMATION.FileName.offset


class Directory:
    """An open directory handle."""

    __slots__ = ("handle", "identity", "closed")

    def __init__(self, handle, identity: str):
        self.handle = handle
        self.identity = identity
        self.closed = False


class Handle:
    """An open file handle. Carries no name."""

    __slots__ = ("handle", "closed")

    def __init__(self, handle):
        self.handle = handle
        self.closed = False


def _close(handle) -> None:
    """Closing is an OPERATION, not a gesture. The result is read,
    because a handle that refused to close is a handle that leaked."""
    if not _k32.CloseHandle(ctypes.c_void_p(handle)):
        raise _api.TransportError("nesne kapatilamadi")


def _close_quietly(handle) -> bool:
    try:
        _close(handle)
        return True
    except _api.TransportError:
        return False


def _fail(message: str, handle, kind=_api.TransportError):
    """Build a refusal OF THE CALLER'S TYPE and close the handle it was
    about, folding a cleanup problem in as a note rather than letting it
    replace the error being raised.

    The type is a parameter because `AlreadyExists` and `NotEmpty` are
    ANSWERS the caller acts on -- collapsing them into the base class
    here would put two different questions behind one refusal."""
    error = kind(message)
    if not _close_quietly(handle):
        error.add_note("temizlik basarisiz: nesne kapatilamadi")
    return error


def _query(handle, klass: int, structure):
    info = structure()
    if not _k32.GetFileInformationByHandleEx(
            ctypes.c_void_p(handle), klass, ctypes.byref(info),
            ctypes.sizeof(info)):
        raise _api.TransportError("nesne bilgisi okunamadi")
    return info


def _identity_text(handle) -> str:
    info = _query(handle, _FileIdInfo, _FILE_ID_INFO)
    return (f"{info.VolumeSerialNumber:016x}:"
            f"{bytes(info.FileId.Identifier).hex()}")


def _kind(attributes: int, reparse: int) -> str:
    if attributes & ATTR_REPARSE or reparse:
        return "link"
    if attributes & ATTR_DEVICE:
        return "other"
    return "dir" if attributes & ATTR_DIRECTORY else "file"


def _relative(directory: "Directory", name: str, access: int, disposition: int,
              options: int, attributes: int = ATTR_NORMAL):
    """`openat`, `creatat` and `mkdirat`, spelled the way NT spells them.

    Returns `(status, handle)` rather than raising, because two statuses
    -- collision and not-found -- are ANSWERS to the caller's question
    and only the rest are failures."""
    child = _api.validate_child_name(name)
    buffer = ctypes.create_unicode_buffer(child)
    unicode_name = _UNICODE_STRING(len(child) * 2, len(child) * 2,
                                   ctypes.cast(buffer, ctypes.c_void_p))
    attribute = _OBJECT_ATTRIBUTES(
        ctypes.sizeof(_OBJECT_ATTRIBUTES), ctypes.c_void_p(directory.handle),
        ctypes.pointer(unicode_name), _OBJ_CASE_INSENSITIVE, None, None)
    status_block = _IO_STATUS_BLOCK()
    handle = ctypes.c_void_p()
    status = _ntdll.NtCreateFile(
        ctypes.byref(handle), access, ctypes.byref(attribute),
        ctypes.byref(status_block), None, attributes, _SHARE_ALL, disposition,
        options, None, 0)
    return status & 0xFFFFFFFF, handle.value


def _assert_open(directory: "Directory") -> None:
    if directory.closed:
        raise _api.TransportError("kapali dizin uzerinde islem")


# ---------------------------------------------------------------------
# directories
# ---------------------------------------------------------------------

def open_root(path) -> Directory:
    text = os.fspath(path)
    if not text.startswith("\\\\?\\"):
        text = "\\\\?\\" + os.path.abspath(text)
    handle = _k32.CreateFileW(text, _DIRECTORY_ACCESS, _SHARE_ALL, None,
                              _OPEN_EXISTING,
                              _FLAG_BACKUP_SEMANTICS | _FLAG_OPEN_REPARSE_POINT,
                              None)
    if handle in (None, _INVALID):
        raise _api.TransportError("kok dizin acilamadi")
    try:
        # BACKUP_SEMANTICS opens a plain file just as happily, and
        # OPEN_REPARSE_POINT opens a junction as ITSELF -- so the root's
        # own type is checked from the handle rather than assumed
        tag = _query(handle, _FileAttributeTagInfo, _FILE_ATTRIBUTE_TAG_INFO)
        if tag.FileAttributes & ATTR_REPARSE or tag.ReparseTag:
            raise _api.TransportError("kok bir yeniden ayrisma noktasi")
        if not tag.FileAttributes & ATTR_DIRECTORY:
            raise _api.TransportError("kok bir dizin degil")
        identity = _identity_text(handle)
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    return Directory(handle, identity)


def close_directory(directory: Directory) -> None:
    """`closed` means CLOSED, not "close was attempted"."""
    if directory.closed:
        return
    _close(directory.handle)
    directory.closed = True


def directory_identity(directory: Directory) -> str:
    return directory.identity


def fsync_directory(directory: Directory) -> bool:
    """ALWAYS FALSE, and honestly so.

    Windows exposes no buffer flush for a directory handle, so a rename's
    durability across a power cut cannot be demonstrated here. Returning
    True after a no-op would be the claim without the evidence -- the
    same distinction `state.durability_of` already draws."""
    _assert_open(directory)
    return False


def child_entry(directory: Directory, name: str):
    """A `stat` relative to the handle, which is what Windows has instead
    of `fstatat`: attributes only, no data access, reparse points opened
    as themselves. `None` when there is nothing there."""
    _assert_open(directory)
    status, handle = _relative(
        directory, name, FILE_READ_ATTRIBUTES | SYNCHRONIZE, _FILE_OPEN,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT)
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        return None
    if status != 0:
        raise _api.TransportError("giris goreli olarak sorgulanamadi")
    try:
        basic = _query(handle, _FileBasicInfo, _FILE_BASIC_INFO)
        standard = _query(handle, _FileStandardInfo, _FILE_STANDARD_INFO)
        tag = _query(handle, _FileAttributeTagInfo, _FILE_ATTRIBUTE_TAG_INFO)
        identity = _identity_text(handle)
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    _close(handle)
    return _api.Entry(
        kind=_kind(basic.FileAttributes, tag.ReparseTag), identity=identity,
        size=0 if standard.Directory else standard.EndOfFile,
        # Windows has no permission bits for the change set to carry, and
        # inventing one would be inventing a fact
        mode="", reparse_tag=tag.ReparseTag)


def open_child_directory(directory: Directory, name: str) -> Directory:
    _assert_open(directory)
    status, handle = _relative(
        directory, name, _DIRECTORY_ACCESS, _FILE_OPEN,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
        | _FILE_DIRECTORY_FILE)
    if status != 0:
        raise _api.TransportError("alt dizin acilamadi")
    try:
        # FILE_OPEN_REPARSE_POINT opened the junction ITSELF, and a
        # junction does carry the directory attribute -- so descending
        # without this check would list whatever it points at
        tag = _query(handle, _FileAttributeTagInfo, _FILE_ATTRIBUTE_TAG_INFO)
        if tag.FileAttributes & ATTR_REPARSE or tag.ReparseTag:
            raise _api.TransportError("alt dizin bir yeniden ayrisma noktasi")
        if not tag.FileAttributes & ATTR_DIRECTORY:
            raise _api.TransportError("alt dizin bir dizin degil")
        identity = _identity_text(handle)
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    return Directory(handle, identity)


def create_child_directory(directory: Directory, name: str) -> None:
    """EXCLUSIVE, by FILE_CREATE. Adopting an existing directory is how a
    rollback ends up removing one it did not create."""
    _assert_open(directory)
    status, handle = _relative(
        directory, name, _DIRECTORY_ACCESS, _FILE_CREATE,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_DIRECTORY_FILE,
        attributes=ATTR_DIRECTORY)
    if status == _STATUS_OBJECT_NAME_COLLISION:
        raise _api.AlreadyExists("dizin zaten var")
    if status != 0:
        raise _api.TransportError("dizin olusturulamadi")
    _close(handle)


def remove_child_directory(directory: Directory, name: str) -> None:
    """EMPTY ONLY -- and the kernel is what enforces it, so a directory
    that gained contents between the test and the call is still refused."""
    _assert_open(directory)
    status, handle = _relative(
        directory, name, _REMOVE_ACCESS, _FILE_OPEN,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
        | _FILE_DIRECTORY_FILE)
    if status != 0:
        raise _api.TransportError("dizin acilamadi")
    try:
        tag = _query(handle, _FileAttributeTagInfo, _FILE_ATTRIBUTE_TAG_INFO)
        if tag.FileAttributes & ATTR_REPARSE or tag.ReparseTag:
            raise _api.TransportError("dizin bir yeniden ayrisma noktasi")
        removal = _dispose(handle)
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    if removal == _STATUS_DIRECTORY_NOT_EMPTY:
        raise _fail("dizin bos degil", handle, _api.NotEmpty)
    if removal != 0:
        raise _fail("dizin kaldirilamadi", handle)
    _close(handle)


def remove_child_file(directory: Directory, name: str,
                      expected_identity: str) -> None:
    """Remove the exact ordinary file through its own no-follow handle."""
    _assert_open(directory)
    if type(expected_identity) is not str or not expected_identity:
        raise _api.TransportError("beklenen nesne kimligi gecersiz")
    status, handle = _relative(
        directory, name, _REMOVE_ACCESS, _FILE_OPEN,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
        | _FILE_NON_DIRECTORY_FILE)
    if status != 0:
        raise _api.TransportError("dosya acilamadi")
    try:
        tag = _query(handle, _FileAttributeTagInfo, _FILE_ATTRIBUTE_TAG_INFO)
        if (tag.FileAttributes & (ATTR_REPARSE | ATTR_DIRECTORY)
                or tag.ReparseTag):
            raise _api.TransportError("silinecek nesne siradan dosya degil")
        if _identity_text(handle) != expected_identity:
            raise _api.TransportError("silinecek nesne kimligi degisti")
        removal = _dispose(handle)
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    if removal != 0:
        raise _fail("dosya kaldirilamadi", handle)
    _close(handle)


def _dispose(handle) -> int:
    """Mark an object for removal THROUGH ITS HANDLE, POSIX semantics
    first so the name is gone the moment the call returns."""
    status_block = _IO_STATUS_BLOCK()
    info = _FILE_DISPOSITION_INFORMATION_EX(
        _FILE_DISPOSITION_DELETE | _FILE_DISPOSITION_POSIX_SEMANTICS)
    status = _ntdll.NtSetInformationFile(
        ctypes.c_void_p(handle), ctypes.byref(status_block),
        ctypes.byref(info), ctypes.sizeof(info),
        _FileDispositionInformationEx) & 0xFFFFFFFF
    if status not in (_STATUS_INVALID_INFO_CLASS, _STATUS_NOT_SUPPORTED):
        return status
    classic = _FILE_DISPOSITION_INFORMATION(1)
    return _ntdll.NtSetInformationFile(
        ctypes.c_void_p(handle), ctypes.byref(status_block),
        ctypes.byref(classic), ctypes.sizeof(classic),
        _FileDispositionInformation) & 0xFFFFFFFF


# ---------------------------------------------------------------------
# files
# ---------------------------------------------------------------------

def create_child_file(directory: Directory, name: str) -> Handle:
    """EXCLUSIVE create. FILE_CREATE is what refuses the operator's own
    file at an ADDED target, in the kernel rather than in a check
    somebody can race."""
    _assert_open(directory)
    status, handle = _relative(
        directory, name, _FILE_WRITE_ACCESS, _FILE_CREATE,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_NON_DIRECTORY_FILE)
    if status == _STATUS_OBJECT_NAME_COLLISION:
        raise _api.AlreadyExists("dosya zaten var")
    if status != 0:
        raise _api.TransportError("dosya olusturulamadi")
    return Handle(handle)


def open_child_file(directory: Directory, name: str) -> Handle:
    """An existing ordinary file, opened as ITSELF: a reparse point is
    refused rather than followed, and a directory cannot arrive here
    because FILE_NON_DIRECTORY_FILE refuses one."""
    _assert_open(directory)
    status, handle = _relative(
        directory, name, _FILE_READ_ACCESS, _FILE_OPEN,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
        | _FILE_NON_DIRECTORY_FILE)
    if status != 0:
        raise _api.TransportError("dosya no-follow acilamadi")
    try:
        tag = _query(handle, _FileAttributeTagInfo, _FILE_ATTRIBUTE_TAG_INFO)
        if tag.FileAttributes & ATTR_REPARSE or tag.ReparseTag:
            raise _api.TransportError("son bilesen yeniden ayrisma noktasi")
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    return Handle(handle)


def rename_child(source: Directory, source_name: str, target: Directory,
                 target_name: str) -> None:
    """Move one object between two OPEN directories, NEVER replacing.

    The destination directory travels as a HANDLE in the rename structure
    itself, so neither end of this operation is a path. REPLACE_IF_EXISTS
    is withheld deliberately: an occupied destination means somebody else
    took the name, and overwriting it is the one outcome a rollback
    cannot undo."""
    _assert_open(source)
    _assert_open(target)
    _api.validate_child_name(source_name)
    child = _api.validate_child_name(target_name)
    status, handle = _relative(
        source, source_name, _RENAME_ACCESS, _FILE_OPEN,
        _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT)
    if status in (_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND):
        raise _api.TransportError("tasinacak nesne yok")
    if status != 0:
        raise _api.TransportError("tasinacak nesne acilamadi")
    try:
        moved = _rename(handle, target, child)
    except _api.TransportError as refused:
        if not _close_quietly(handle):
            refused.add_note("temizlik basarisiz: nesne kapatilamadi")
        raise
    if moved == _STATUS_OBJECT_NAME_COLLISION:
        raise _fail("hedef ad zaten dolu", handle, _api.AlreadyExists)
    if moved != 0:
        raise _fail("nesne tasinamadi", handle)
    _close(handle)


def _rename(handle, target: Directory, name: str) -> int:
    """The rename structure, built twice if the kernel is old.

    The classic class reads the same four bytes as a BOOLEAN
    `ReplaceIfExists`, so the fallback writes a ZERO there rather than
    reusing the Ex flags -- `FILE_RENAME_POSIX_SEMANTICS` is 2, and a
    non-zero byte in that position means REPLACE."""
    def build(flags: int):
        name_bytes = len(name) * 2
        size = _RENAME_HEADER + name_bytes + 2
        raw = ctypes.create_string_buffer(size)
        info = ctypes.cast(raw,
                           ctypes.POINTER(_FILE_RENAME_INFORMATION)).contents
        info.Flags = flags
        info.RootDirectory = ctypes.c_void_p(target.handle)
        info.FileNameLength = name_bytes
        ctypes.memmove(ctypes.addressof(raw) + _RENAME_HEADER,
                       ctypes.create_unicode_buffer(name), name_bytes)
        return raw, size

    status_block = _IO_STATUS_BLOCK()
    raw, size = build(_FILE_RENAME_POSIX_SEMANTICS)
    status = _ntdll.NtSetInformationFile(
        ctypes.c_void_p(handle), ctypes.byref(status_block), raw, size,
        _FileRenameInformationEx) & 0xFFFFFFFF
    if status not in (_STATUS_INVALID_INFO_CLASS, _STATUS_NOT_SUPPORTED):
        return status
    raw, size = build(0)
    return _ntdll.NtSetInformationFile(
        ctypes.c_void_p(handle), ctypes.byref(status_block), raw, size,
        _FileRenameInformation) & 0xFFFFFFFF


def write_all(handle: Handle, data: bytes) -> None:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    view = (ctypes.c_char * len(data)).from_buffer_copy(data)
    written = wintypes.DWORD(0)
    total = 0
    while total < len(data):
        if not _k32.WriteFile(ctypes.c_void_p(handle.handle),
                              ctypes.byref(view, total), len(data) - total,
                              ctypes.byref(written), None):
            raise _api.TransportError("dosya yazilamadi")
        if written.value == 0:
            raise _api.TransportError("dosya yazilamadi")
        total += written.value


def read_all(handle: Handle, ceiling: int) -> bytes:
    """Bounded WHILE reading. A ceiling applied after the bytes are in
    the process is a report, not a bound."""
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    block = ctypes.create_string_buffer(1 << 20)
    got = wintypes.DWORD(0)
    chunks, total = [], 0
    while True:
        if not _k32.ReadFile(ctypes.c_void_p(handle.handle), block,
                             len(block), ctypes.byref(got), None):
            raise _api.TransportError("dosya okunamadi")
        if got.value == 0:
            return b"".join(chunks)
        total += got.value
        if total > ceiling:
            raise _api.TransportError("dosya sozlesme tavanini asiyor")
        chunks.append(block.raw[:got.value])


def fsync_handle(handle: Handle) -> None:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    if not _k32.FlushFileBuffers(ctypes.c_void_p(handle.handle)):
        raise _api.TransportError("dosya kalici hale getirilemedi")


def handle_identity(handle: Handle) -> str:
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    return _identity_text(handle.handle)


def handle_mode(handle: Handle) -> str:
    """Empty, always. There are no POSIX permission bits here to report,
    and the change set records `""` on this platform for the same
    reason."""
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")
    return ""


def set_handle_mode(handle: Handle, mode: str) -> None:
    """A no-op with a reason. Windows has no bits to restore, so applying
    anything here would be inventing a fact -- the mirror layer already
    draws exactly this line."""
    if handle.closed:
        raise _api.TransportError("kapali nesne uzerinde islem")


def close_handle(handle: Handle) -> None:
    if handle.closed:
        return
    _close(handle.handle)
    handle.closed = True
