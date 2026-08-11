"""Direct filesystem evidence. PACKAGE B2B-A-D2.

THE AUTHORITY IS THE FILESYSTEM, READ THROUGH HANDLES. Git held this
job twice and lost it twice, both times to state the model could reach:
the per-repository index, where `skip-worktree` hid a modified
control-plane file, and shared metadata, where a clean filter declared
in `.git/config` hid a change AND ran a model-supplied command during
verification. Neither was a missing check; the authority itself was
inside the blast radius.

THEN THE PATH WAS THE AUTHORITY, and lost three times to one move
played at three different instants:

  1. no-follow on the final component  -> the PARENT was swapped, and
     the last element was still an ordinary file: the outside one.
  2. identity pinned to the child's own `lstat` -> the parent was
     swapped BEFORE that `lstat`, so both sides agreed on the outside
     file. Measured: race_fired=True, accepted=True, outside_hashed=True.
  3. identity checked on both sides of the listing -> the swap was
     placed after the second check and before the child's `lstat`.

Each fix was another question asked of a path, and a path can answer
differently between any two questions. So after the root is opened
there are no paths here at all: the walk lists an OPEN DIRECTORY, and
every child is opened relative to the directory object that listed it,
then bound to the record that named it before a byte is read. See
`fs_transport`.

Nothing in this module runs git, opens a repository or launches a
subprocess. It holds directory handles and reads descriptors.

TWO EVIDENCE CLASSES, and the split is a MEASURED cost decision:

  CONTENT   every metadata field plus a streaming SHA-256.
  METADATA  every metadata field, and NO right to read data.

THE METADATA INVARIANT, stated exactly, because an earlier version of
this file stated it wrongly. It is NOT "the file is never opened".
Windows has no `fstatat`: a stat relative to a directory handle IS a
handle, opened with FILE_READ_ATTRIBUTES. What is guaranteed is
narrower and checkable:

  * no right to read data is ever requested -- the access mask is
    exactly FILE_READ_ATTRIBUTES | SYNCHRONIZE;
  * that handle is never converted into a data descriptor;
  * not one data byte is ever read from it.

The class exists for the in-repository virtual environments: seven of
them, 184,685 files, 19,951,871,042 logical bytes, largest single file
1,659,834,880. Hashing them was measured at 1.1 minutes per pass warm
and 51 minutes cold, against 12.5 seconds for the old path-based
inventory. The handle-bound walk was measured at 16.19 seconds per pass
over the protected roots. Trusting the directory enumeration's own
metadata would have cost 2.55 seconds and been WRONG: it missed a
same-size rewrite in 117 of 200 attempts and an atomic replace in 137
of 200. The cost is dominated by file COUNT, so no size threshold
rescues it.

WHAT THE METADATA CLASS CLAIMS, exactly: it detects addition, deletion,
rename, size change, mode change, attribute change, file-identity
change and timestamp change. It does NOT detect a rewrite that
preserves the entire tuple. That residue is bounded by two measured
facts and one contract: the timestamp quantum here is about a
millisecond, a one-millisecond wait produced 200 distinct timestamps
out of 200 attempts, and the implementer's tools are exactly Read,
Glob, Grep, Edit, Write -- no shell, so no `utime`.

WHAT MAY LEAVE. Repo-relative paths, closed codes, counts and digests.
Never file contents, never an absolute path, never a link's target,
never the operating system's own error text.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

from tools.agent_loop import contract, fs_transport

# The measured quantum on the reference machine was about one
# millisecond: a 1 ms wait gave 200 distinct timestamps out of 200
# attempts, while no wait at all collided 158 times. Ten times the
# measurement is the margin.
QUIESCENCE_NS = 10_000_000

# The link fingerprint key: exactly `bytes`, exactly this long. A
# `bytes` subclass can report any length while carrying nothing, and a
# `bytearray` can be edited between the before and after manifests.
KEY_BYTES = 32

CONTENT = "content"
METADATA = "metadata"

# Name classes that must never appear in an inventory: control and
# format characters, and the line/paragraph separators.
_FORBIDDEN_CATEGORIES = ("Cc", "Cf", "Zl", "Zp")
# Every space separator EXCEPT the ordinary one: U+00A0 renders exactly
# like a space and named a file the first version happily accepted.
_ORDINARY_SPACE = " "


@dataclass(frozen=True, slots=True)
class Limits:
    """Bounds derived from measurement, and SPLIT by evidence class.

    One pair of ceilings could not serve both. The content class covers
    162 tracked files totalling 1.81 MB; the metadata class covers the
    virtual environments, whose largest single file is 1,659,834,880
    bytes. A 64 MiB content ceiling applied to the metadata class
    refused a real library that was never going to be opened at all.

    Every one of these is a REFUSAL, never a truncation."""

    max_entries: int = 400000                   # olculen 184.685 + dizinler
    max_content_file_bytes: int = 64 << 20
    max_metadata_file_bytes: int = 8 << 30      # olculen 1,66 GB'nin ustu
    max_content_total_bytes: int = 1024 << 20
    max_logical_total_bytes: int = 64 << 30     # olculen ~20 GB'nin ustu
    max_depth: int = 128


@dataclass(frozen=True, slots=True)
class Entry:
    path: str                  # canonical repo-relative POSIX
    kind: str                  # file | dir | link
    mode: str
    size: int
    mtime_ns: int
    file_id: str               # volume-qualified, as text
    nlink: int                 # -1 when the platform listing cannot say
    attributes: int            # Windows file attributes; 0 on POSIX
    reparse_tag: int
    link_target_mac: str       # KEYED per call; "" when not a link
    sha256: str                # "" in the metadata class


@dataclass(frozen=True, slots=True)
class Manifest:
    root_identity: str         # the root OBJECT's own identity
    entries: tuple
    digest: str
    file_count: int
    total_bytes: int
    content_hashed: int
    metadata_only: int
    reparse_count: int
    peak_directories: int      # the resource claim, published
    duration_ms: int


class EvidenceError(RuntimeError):
    """A tree this walker cannot describe. Fixed text, closed reason."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


class UnsupportedEntry(EvidenceError):
    """A filesystem object this evidence model does not represent.

    Refused rather than guessed at: a symlink, a reparse point, a FIFO
    and a hardlinked file each mean something different, and inventing
    a representation for one is how an unreviewed change slips in."""

    def __init__(self, message):
        super().__init__(message,
                         reason=contract.StopReason.PATH_NOT_ALLOWED)


def quiesce():
    """Wait out the measured timestamp quantum.

    Called after a baseline manifest and BEFORE anything may write, so
    a later write cannot land in the same timestamp as the reading."""
    time.sleep(QUIESCENCE_NS / 1e9)


def _canonical(parts) -> str:
    """One spelling, or a refusal.

    NFC is a DECISION, not a normalisation: a name that is not already
    composed is refused rather than quietly rewritten, because silently
    changing a name makes the manifest describe something the
    filesystem does not have."""
    for part in parts:
        if part in ("", ".", "..") or "\\" in part or "/" in part:
            raise UnsupportedEntry("yol kanonik degil")
        for character in part:
            category = unicodedata.category(character)
            if category in _FORBIDDEN_CATEGORIES:
                raise UnsupportedEntry(
                    "yol gorunmez ya da kontrol karakteri tasiyor")
            if category == "Zs" and character != _ORDINARY_SPACE:
                raise UnsupportedEntry(
                    "yol siradan olmayan bosluk karakteri tasiyor")
        if unicodedata.normalize("NFC", part) != part:
            raise UnsupportedEntry("yol NFC biciminde degil")
    text = "/".join(parts)
    if not text or PurePosixPath(text).is_absolute():
        raise UnsupportedEntry("yol kanonik degil")
    return text


def _hash_descriptor(descriptor: int, ceiling: int):
    """Digest an already-bound descriptor.

    The caller has proved this descriptor names the enumerated object.
    All that is left is to notice if it MOVES while being read: a
    digest taken across a change describes neither version, so the
    mixture is refused rather than recorded."""
    before = _guard(fs_transport.descriptor_identity, descriptor,
                    message="acik nesne durumu okunamadi",
                    kind=EvidenceError)
    digest = hashlib.sha256()
    read = 0
    while True:
        try:
            block = os.read(descriptor, 1 << 20)
        except OSError:
            raise EvidenceError("dosya okunamadi") from None
        if not block:
            break
        read += len(block)
        if read > ceiling:
            raise EvidenceError("dosya sozlesme tavanini asiyor")
        digest.update(block)
    if _guard(fs_transport.descriptor_identity, descriptor,
              message="acik nesne durumu okunamadi",
              kind=EvidenceError) != before:
        raise EvidenceError("dosya okunurken degisti")
    return digest.hexdigest(), before


class _Frame:
    """One open directory and where the walk is inside it."""

    __slots__ = ("directory", "parts", "records", "index")

    def __init__(self, directory, parts, records):
        self.directory = directory
        self.parts = parts
        self.records = records
        self.index = 0


def _list(directory):
    return _guard(fs_transport.list_directory, directory,
                  message="dizin listelenemedi", kind=UnsupportedEntry)


def _close_quietly(directory) -> bool:
    try:
        fs_transport.close_directory(directory)
        return True
    except (fs_transport.TransportError, OSError):
        return False


def _close_descriptor_quietly(descriptor: int) -> bool:
    """Closes, and SAYS whether it worked. The first version returned
    nothing at all, which turned every call site into a silent drop."""
    try:
        os.close(descriptor)
        return True
    except OSError:
        return False


def _fold(exc: BaseException, ok: bool) -> None:
    """Consume a cleanup result on a failure path: it may not replace
    the error being raised, and it may not disappear."""
    if not ok:
        fs_transport.mark_cleanup_failed(exc)


def _guard(call, *args, message, kind):
    """EVERY transport call goes through here.

    A `TransportError` already carries a fixed sentence. Anything else
    -- above all a raw `OSError`, whose text names absolute paths and
    travels straight into a report -- is replaced by one.

    MEASURED before this existed: all seven seams the walker calls
    leaked the operating system's own message verbatim, and two tests
    accepted `OSError` alongside the typed error, so the leak checks
    they carried never ran."""
    try:
        return call(*args)
    except fs_transport.TransportError as exc:
        yeni = kind(str(exc))
        # The FLAG crosses, and one fixed sentence is re-added from
        # this side. The lower layer's `__notes__` are deliberately NOT
        # copied: notes are free text, and free text written closer to
        # the filesystem is exactly where a path would ride out.
        if fs_transport.cleanup_failed(exc):
            fs_transport.mark_cleanup_failed(yeni)
        raise yeni from None
    except OSError:
        raise kind(message) from None


def _close_frames(frames) -> int:
    """Close what is left, innermost first, and COUNT what refused.

    Every frame is attempted even after one fails: a cleanup that stops
    at the first problem leaks everything behind it."""
    kalan = 0
    while frames:
        frame = frames.pop()
        if not _close_quietly(frame.directory):
            kalan += 1
    return kalan


def _link_mac(evidence: bytes, key: bytes) -> str:
    """The link's own data as a KEYED code, never as text and never as
    a plain digest: a link may point at a private location, and an
    unkeyed hash of a short path is a dictionary away from readable.

    The bytes are used AS BYTES. An earlier version still wrote
    `str(evidence)` after the transport started returning bytes, which
    keyed the HMAC over `b'...'` -- a stable, wrong answer, which is the
    kind that survives every test that only compares two scans."""
    if type(evidence) is not bytes:
        raise EvidenceError("baglanti kaniti bayt degil")
    return hmac.new(key, evidence, hashlib.sha256).hexdigest()


def _covered_by(path: str, prefixes) -> bool:
    for prefix in prefixes or ():
        trimmed = str(prefix).replace("\\", "/").strip("/")
        if trimmed and (path == trimmed or path.startswith(trimmed + "/")):
            return True
    return False


def _evidence_class(path: str, metadata_only, exact_content) -> str:
    """An EXACT path in `content_always` outranks a metadata prefix.

    `pyvenv.cfg` sits inside a virtual environment whose contents are
    metadata-only for measured cost reasons, and it is also the file
    that says which interpreter runs. The prefix must not swallow it."""
    if path in exact_content:
        return CONTENT
    return METADATA if _covered_by(path, metadata_only) else CONTENT


def _entry_from(path: str, record, *, sha256="", link_mac="", nlink=None):
    return Entry(
        path=path, kind=record.kind,
        mode=oct(stat.S_IMODE(record.mode)) if record.mode else "0o0",
        size=record.size, mtime_ns=record.mtime_ns, file_id=record.file_id,
        nlink=(record.nlink if nlink is None else nlink)
        if (record.nlink is not None or nlink is not None) else -1,
        attributes=record.attributes, reparse_tag=record.reparse_tag,
        link_target_mac=link_mac, sha256=sha256)


def scan(root, *, key, metadata_only=(), content_always=(),
         limits=None) -> Manifest:
    """Open `root` ONCE, then describe the tree of that open object.

    `key` binds the link fingerprints to ONE call: the same key must be
    used for the before and after manifests or their links can never
    compare equal, and another call cannot recognise them at all."""
    limits = limits or Limits()
    # exact type, exact length -- checked in that order, so a subclass
    # never gets to answer `len()` at all
    if type(key) is not bytes or len(key) != KEY_BYTES:
        raise EvidenceError("baglanti anahtari sozlesmeye uymuyor")
    started = time.monotonic()
    exact_content = {str(item).replace("\\", "/").strip("/")
                     for item in content_always or ()}

    kok = _guard(fs_transport.open_root, root,
                 message="kok dizin acilamadi", kind=EvidenceError)

    entries = []
    logical = content_total = 0
    content_hashed = metadata_count = reparse_count = 0
    now_ns = time.time_ns()
    # DEPTH-FIRST, and that is a RESOURCE decision as much as a shape.
    # The first version queued every directory it met and held all of
    # them open until the scan ended: on the protected roots that is
    # 24,209 simultaneous handles for no reason at all. A frame is
    # closed the moment its last child is dealt with, so the number of
    # open directories is bounded by DEPTH, not by the size of the tree.
    frames = [_Frame(kok, (), _list(kok))]
    peak = 1
    try:
        while frames:
            frame = frames[-1]
            if frame.index >= len(frame.records):
                frames.pop()
                # a directory that will not close is a scan that
                # cannot be called complete
                _guard(fs_transport.close_directory, frame.directory,
                       message="dizin kapatilamadi", kind=EvidenceError)
                continue
            record = frame.records[frame.index]
            frame.index += 1
            directory, parts = frame.directory, frame.parts

            child_parts = parts + (record.name,)
            path = _canonical(child_parts)
            if record.mtime_ns > now_ns + QUIESCENCE_NS:
                raise EvidenceError("dosya zaman damgasi gelecekte")
            if len(entries) >= limits.max_entries:
                raise EvidenceError("giris sayisi sozlesme tavanini asiyor")

            if record.kind == "link":
                # Fingerprinted, NEVER followed, and never for any tag:
                # an unknown reparse tag is described by the same keyed
                # code as a symlink and descended into exactly as often,
                # which is never.
                kanit = _guard(fs_transport.link_evidence, directory,
                               record, message="baglanti kaniti alinamadi",
                               kind=EvidenceError)
                reparse_count += 1
                entries.append(_entry_from(path, record,
                                           link_mac=_link_mac(kanit, key)))
                continue

            if record.kind == "dir":
                if len(child_parts) > limits.max_depth:
                    raise EvidenceError("derinlik sozlesme tavanini asiyor")
                entries.append(_entry_from(path, record))
                alt = _guard(fs_transport.open_child_directory,
                             directory, record,
                             message="alt dizin acilamadi",
                             kind=UnsupportedEntry)
                try:
                    kayitlar = _list(alt)
                except BaseException as exc:
                    # the child is not on the stack yet, so nothing else
                    # would ever close it
                    _fold(exc, _close_quietly(alt))
                    raise
                frames.append(_Frame(alt, child_parts, kayitlar))
                peak = max(peak, len(frames))
                continue

            if record.kind != "file":
                raise UnsupportedEntry("siradan olmayan dosya turu")
            if record.nlink is not None and record.nlink != 1:
                raise UnsupportedEntry("sert baglantili dosya "
                                       "temsil edilemiyor")
            logical += record.size
            if logical > limits.max_logical_total_bytes:
                raise EvidenceError("toplam boyut sozlesme tavanini asiyor")

            if _evidence_class(path, metadata_only,
                               exact_content) == CONTENT:
                if record.size > limits.max_content_file_bytes:
                    raise EvidenceError("dosya sozlesme tavanini asiyor")
                content_total += record.size
                if content_total > limits.max_content_total_bytes:
                    raise EvidenceError(
                        "icerik toplami sozlesme tavanini asiyor")
                descriptor = _guard(fs_transport.open_child_file,
                                    directory, record,
                                    message="dosya acilamadi",
                                    kind=UnsupportedEntry)
                try:
                    digest, acik_kimlik = _hash_descriptor(
                        descriptor, limits.max_content_file_bytes)
                except BaseException as exc:
                    _fold(exc,
                          _close_descriptor_quietly(descriptor))
                    raise
                if not _close_descriptor_quietly(descriptor):
                    raise EvidenceError("tanimlayici kapatilamadi")
                # the open object could report a link count the
                # listing could not
                if acik_kimlik[5] != 1:
                    raise UnsupportedEntry("sert baglantili dosya "
                                           "temsil edilemiyor")
                content_hashed += 1
                entries.append(_entry_from(path, record, sha256=digest,
                                           nlink=acik_kimlik[5]))
            else:
                # NO RIGHT TO READ DATA is requested for this file, no
                # data descriptor is made from it, and no byte is read.
                # That -- not "never opened" -- is the cost decision.
                if record.size > limits.max_metadata_file_bytes:
                    raise EvidenceError("dosya sozlesme tavanini asiyor")
                metadata_count += 1
                entries.append(_entry_from(path, record))
    except BaseException as birincil:
        kalan = _close_frames(frames)
        if kalan:
            # A cleanup failure must not REPLACE the error that got us
            # here, and must not vanish either.
            birincil.add_note(f"temizlik basarisiz: {kalan} dizin kapanmadi")
        raise
    kalan = _close_frames(frames)
    if kalan:
        raise EvidenceError("tarama sonunda dizin kapatilamadi")

    ordered = tuple(sorted(entries, key=lambda entry: entry.path))
    stream = kok.identity.encode("ascii") + b"\n"
    stream += b"".join(
        b"\0".join((entry.path.encode("utf-8"), entry.kind.encode("ascii"),
                    entry.mode.encode("ascii"), str(entry.size).encode(),
                    str(entry.mtime_ns).encode(), entry.file_id.encode(),
                    str(entry.nlink).encode(), str(entry.attributes).encode(),
                    str(entry.reparse_tag).encode(),
                    entry.link_target_mac.encode("ascii"),
                    entry.sha256.encode("ascii"))) + b"\n"
        for entry in ordered)
    return Manifest(
        root_identity=kok.identity, entries=ordered,
        digest=hashlib.sha256(stream).hexdigest(),
        file_count=sum(1 for entry in ordered if entry.kind == "file"),
        total_bytes=logical, content_hashed=content_hashed,
        metadata_only=metadata_count, reparse_count=reparse_count,
        peak_directories=peak,
        duration_ms=int((time.monotonic() - started) * 1000))


def diff(before: Manifest, after: Manifest) -> tuple:
    """Every path whose record is not byte-identical, in either
    direction -- and the ROOT itself, reported as `.` when the
    directory the manifest describes is no longer the same object."""
    left = {entry.path: entry for entry in before.entries}
    right = {entry.path: entry for entry in after.entries}
    changed = [path for path in set(left) | set(right)
               if left.get(path) != right.get(path)]
    if before.root_identity != after.root_identity:
        changed.append(".")
    return tuple(sorted(changed))
