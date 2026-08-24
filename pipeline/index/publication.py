"""The shared publication service: the ONE way a candidate reaches disk.

The API and the CLI both come through here. They used to publish
separately -- the endpoint wrote the upload directory itself and the CLI
had no publication step at all -- and the two paths drifted until they
carried different guarantees. One service means one lock order, one
crash-window recovery, and one place where the disk target is decided.

THE SEQUENCE, under a single held lock:

    stage_candidate      the row learns about bytes it does not have yet
                         (candidate_state = STAGED)
    <bytes to disk>      temp file, then an atomic rename
    finalize_...         row and disk now agree (PUBLISHED)

Between stage and finalize the candidate is not processable at all --
which is what closes the gap a process request used to fall into,
reading a candidate whose bytes were not published yet, refusing
correctly, and marking the document error while the upload returned 200.
Both crash windows recover by simply running this again: the same bytes
keep the same candidate id, so a re-run finishes the publication instead
of starting a new one.

THE DESTINATION IS NOT THE CALLER'S. It is derived from the CANONICAL
filename ``stage_candidate`` returns -- the spelling the database
resolved to, which may differ from the one that was offered. A caller
able to name the destination could put the bytes somewhere the row does
not point at, which is precisely the split this whole package exists to
prevent. And the canonical name, though it comes from our own database,
is still INPUT: it is re-checked for basename-ness and containment
before a single byte is written, on POSIX and Windows spellings alike.
"""
import hashlib
import os
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from pipeline.index import db
from pipeline.index.attempt_contract import CandidateSuperseded
from pipeline.storage import handle_transport as _object_transport

load_dotenv()

# Read through the module attribute at call time, never bound into a
# default argument: tests and deployments both need to point it
# elsewhere, and a value captured at import cannot be moved.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))

# Version sources are bounded while they cross the storage seam.  The API's
# upload limit is intentionally not imported here: this lower-level service
# also serves CLI/worker callers and must carry its own closed ceiling.
MAX_VERSION_SOURCE_BYTES = 64 << 20
_SOURCE_OBJECT_NAME = "source.bin"
_SOURCE_TREE = ("tenants", "documents", "versions")

# Windows resolves these to DEVICES, not files: opening "NUL" writes to
# the void and reports success, and "PRN.pdf" is still PRN. The API
# learned this and refused them at its own door; the door moved here, and
# the rules had to move with it -- a protection that lives at only one of
# two entrances is a protection with a way around it.
_WINDOWS_DEVICE_NAMES = frozenset({
    "con", "prn", "aux", "nul", "clock$", "conin$", "conout$",
    *(
        f"{prefix}{suffix}"
        for prefix in ("com", "lpt")
        for suffix in (*map(str, range(1, 10)), "¹", "²", "³")
    ),
})


# Zs/Zl/Zp are the space separators; Cf is the format class that holds
# the zero-width space, the byte-order mark and the direction overrides.
# A name may contain a plain ASCII space and nothing else from here.
_INVISIBLE_CATEGORIES = frozenset({"Zs", "Zl", "Zp", "Cf"})


class UnsafeCanonicalName(ValueError):
    """The database's canonical filename is not a plain basename inside
    the upload directory. Trusted source, still untrusted input."""


class VersionSourceRefused(RuntimeError):
    """An immutable source operation that could not be proven safe.

    Messages are fixed policy vocabulary.  Neither an absolute path nor an
    operating-system error is allowed to cross this boundary.
    """


class VersionSourceDifferent(VersionSourceRefused):
    """The immutable target already carries different bytes."""


class VersionSourceMissing(VersionSourceRefused):
    """The requested immutable source object does not exist."""


class VersionSourceCorrupt(VersionSourceRefused):
    """Stored bytes do not match their database-owned digest."""


@dataclass(frozen=True, slots=True)
class VersionSourceObject:
    """Closed result of one immutable source publication."""

    tenant_id: str
    document_id: str
    version_id: str
    sha256: str
    size: int
    created: bool
    directory_fsync: bool


@dataclass(frozen=True, slots=True)
class VersionSourceProof:
    """Fresh existence/hash proof for a database activation gate."""

    tenant_id: str
    document_id: str
    version_id: str
    sha256: str
    size: int


def _uuid_text(value, label: str) -> str:
    """Return one canonical UUID segment; never accept path-like text."""
    try:
        parsed = uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError):
        raise VersionSourceRefused(f"{label} gecerli bir uuid degil") from None
    canonical = str(parsed)
    if str(value) != canonical:
        raise VersionSourceRefused(f"{label} kanonik uuid degil")
    return canonical


def _digest_text(value) -> str:
    if (type(value) is not str or len(value) != 64
            or value != value.lower()
            or any(char not in "0123456789abcdef" for char in value)):
        raise VersionSourceRefused("kaynak ozeti gecersiz")
    return value


def _source_limit(value) -> int:
    if type(value) is not int or not 1 <= value <= MAX_VERSION_SOURCE_BYTES:
        raise VersionSourceRefused("kaynak boyut tavani gecersiz")
    return value


def version_source_path(object_root, tenant_id, document_id,
                        version_id) -> Path:
    """Describe the UUID-only immutable object path without opening it.

    This function is for records and diagnostics.  Publication and reads do
    not use the returned path: after the root is opened, every lookup is
    relative to a held directory handle.
    """
    tenant = _uuid_text(tenant_id, "tenant kimligi")
    document = _uuid_text(document_id, "belge kimligi")
    version = _uuid_text(version_id, "surum kimligi")
    return (Path(object_root) / _SOURCE_TREE[0] / tenant / _SOURCE_TREE[1]
            / document / _SOURCE_TREE[2] / version / _SOURCE_OBJECT_NAME)


def _close_directories(directories) -> None:
    failed = False
    for directory in reversed(directories):
        if not _object_transport.close_directory_quietly(directory):
            failed = True
    if failed:
        raise VersionSourceRefused("kaynak dizin tanimlayicisi kapatilamadi")


def _open_or_create(parent, name: str):
    entry = _object_transport.child_entry(parent, name)
    if entry is not None:
        if entry.kind != "dir" or entry.reparse_tag:
            raise VersionSourceRefused("kaynak agaci siradan dizinlerden olusmuyor")
        return _object_transport.open_child_directory(parent, name)
    try:
        _object_transport.create_child_directory(parent, name)
    except _object_transport.AlreadyExists:
        # A concurrent creator may win between the evidence query and the
        # exclusive mkdir.  Adopt it only through a fresh no-follow handle;
        # a link or ordinary file is still refused by that open.
        pass
    return _object_transport.open_child_directory(parent, name)


def _open_source_parent(object_root, tenant: str, document: str,
                        version: str, *, create: bool):
    try:
        root = _object_transport.open_root(object_root)
        opened = [root]
        names = (_SOURCE_TREE[0], tenant, _SOURCE_TREE[1], document,
                 _SOURCE_TREE[2], version)
        for name in names:
            if create:
                child = _open_or_create(opened[-1], name)
            else:
                entry = _object_transport.child_entry(opened[-1], name)
                if entry is None:
                    raise VersionSourceMissing("kaynak nesnesi yok")
                if entry.kind != "dir" or entry.reparse_tag:
                    raise VersionSourceCorrupt(
                        "kaynak agaci siradan dizinlerden olusmuyor")
                child = _object_transport.open_child_directory(
                    opened[-1], name)
            opened.append(child)
        return opened
    except VersionSourceRefused:
        if "opened" in locals():
            for directory in reversed(opened):
                _object_transport.close_directory_quietly(directory)
        raise
    except (OSError, _object_transport.TransportError):
        if "opened" in locals():
            for directory in reversed(opened):
                _object_transport.close_directory_quietly(directory)
        raise VersionSourceRefused("kaynak nesne agaci acilamadi") from None


def _read_named(parent, name: str, maximum: int):
    entry = _object_transport.child_entry(parent, name)
    if entry is None:
        raise VersionSourceMissing("kaynak nesnesi yok")
    if entry.kind != "file" or entry.reparse_tag or entry.size > maximum:
        raise VersionSourceCorrupt("kaynak nesnesi dogrulanamadi")
    handle = None
    try:
        handle = _object_transport.open_child_file(parent, name)
        identity = _object_transport.handle_identity(handle)
        if identity != entry.identity:
            raise VersionSourceCorrupt("kaynak nesnesi okuma sirasinda degisti")
        body = _object_transport.read_all(handle, maximum)
    except VersionSourceRefused:
        raise
    except (OSError, _object_transport.TransportError):
        raise VersionSourceCorrupt("kaynak nesnesi okunamadi") from None
    finally:
        if handle is not None and not _object_transport.close_handle_quietly(
                handle):
            raise VersionSourceRefused("kaynak tanimlayicisi kapatilamadi")
    return body, identity


def read_version_source(object_root, tenant_id, document_id, version_id, *,
                        expected_sha256, max_bytes=MAX_VERSION_SOURCE_BYTES):
    """Read an ingest source through held no-follow handles and verify it."""
    tenant = _uuid_text(tenant_id, "tenant kimligi")
    document = _uuid_text(document_id, "belge kimligi")
    version = _uuid_text(version_id, "surum kimligi")
    expected = _digest_text(expected_sha256)
    maximum = _source_limit(max_bytes)
    directories = _open_source_parent(object_root, tenant, document, version,
                                      create=False)
    primary = None
    try:
        body, _identity = _read_named(directories[-1], _SOURCE_OBJECT_NAME,
                                     maximum)
        if hashlib.sha256(body).hexdigest() != expected:
            raise VersionSourceCorrupt("kaynak ozeti kayitla eslesmiyor")
        return body
    except BaseException as refused:
        primary = refused
        raise
    finally:
        try:
            _close_directories(directories)
        except VersionSourceRefused as cleanup:
            if primary is None:
                raise
            primary.add_note(str(cleanup))


def verify_version_source(object_root, tenant_id, document_id, version_id, *,
                          expected_sha256,
                          max_bytes=MAX_VERSION_SOURCE_BYTES):
    """Prove an immutable source exists and matches before DB activation."""
    tenant = _uuid_text(tenant_id, "tenant kimligi")
    document = _uuid_text(document_id, "belge kimligi")
    version = _uuid_text(version_id, "surum kimligi")
    expected = _digest_text(expected_sha256)
    body = read_version_source(
        object_root, tenant, document, version,
        expected_sha256=expected, max_bytes=max_bytes)
    return VersionSourceProof(
        tenant, document, version, expected, len(body))


def publish_version_source(object_root, tenant_id, document_id, version_id,
                           body, *, expected_sha256,
                           max_bytes=MAX_VERSION_SOURCE_BYTES):
    """Publish one immutable UUID-addressed source object.

    The temporary name is exclusive and server-generated.  Its bytes are
    fsync'd, reopened no-follow, bounded and digest-checked before a
    no-replace atomic rename.  A retry with identical bytes is idempotent;
    an occupied target with different bytes is never replaced.
    """
    tenant = _uuid_text(tenant_id, "tenant kimligi")
    document = _uuid_text(document_id, "belge kimligi")
    version = _uuid_text(version_id, "surum kimligi")
    expected = _digest_text(expected_sha256)
    maximum = _source_limit(max_bytes)
    if type(body) is not bytes or len(body) > maximum:
        raise VersionSourceRefused("kaynak govdesi gecersiz veya cok buyuk")
    if hashlib.sha256(body).hexdigest() != expected:
        raise VersionSourceRefused("kaynak govdesi ozetiyle eslesmiyor")

    directories = _open_source_parent(object_root, tenant, document, version,
                                      create=True)
    parent = directories[-1]
    primary = None
    try:
        existing = _object_transport.child_entry(parent, _SOURCE_OBJECT_NAME)
        if existing is not None:
            stored, _identity = _read_named(parent, _SOURCE_OBJECT_NAME,
                                             maximum)
            if stored != body:
                raise VersionSourceDifferent(
                    "surum kaynagi farkli baytlarla zaten var")
            return VersionSourceObject(tenant, document, version, expected,
                                       len(body), False, False)

        temporary = f"source-{uuid.uuid4()}.tmp"
        handle = None
        try:
            handle = _object_transport.create_child_file(parent, temporary)
            created_identity = _object_transport.handle_identity(handle)
            _object_transport.write_all(handle, body)
            _object_transport.fsync_handle(handle)
        except (OSError, _object_transport.TransportError):
            raise VersionSourceRefused("kaynak gecici nesnesi yazilamadi") from None
        finally:
            if handle is not None and not _object_transport.close_handle_quietly(
                    handle):
                raise VersionSourceRefused("kaynak tanimlayicisi kapatilamadi")

        staged, staged_identity = _read_named(parent, temporary, maximum)
        if staged_identity != created_identity or staged != body:
            raise VersionSourceCorrupt("gecici kaynak nesnesi dogrulanamadi")
        try:
            _object_transport.rename_child(parent, temporary, parent,
                                           _SOURCE_OBJECT_NAME)
        except _object_transport.AlreadyExists:
            stored, _identity = _read_named(parent, _SOURCE_OBJECT_NAME,
                                             maximum)
            if stored != body:
                raise VersionSourceDifferent(
                    "surum kaynagi farkli baytlarla zaten var")
            return VersionSourceObject(tenant, document, version, expected,
                                       len(body), False, False)
        except (OSError, _object_transport.TransportError):
            raise VersionSourceRefused("kaynak nesnesi yayinlanamadi") from None

        stored, final_identity = _read_named(parent, _SOURCE_OBJECT_NAME,
                                             maximum)
        if final_identity != created_identity or stored != body:
            raise VersionSourceCorrupt("yayinlanan kaynak dogrulanamadi")
        directory_fsync = _object_transport.fsync_directory(parent)
        return VersionSourceObject(tenant, document, version, expected,
                                   len(body), True, directory_fsync)
    except BaseException as refused:
        primary = refused
        raise
    finally:
        try:
            _close_directories(directories)
        except VersionSourceRefused as cleanup:
            if primary is None:
                raise
            primary.add_note(str(cleanup))


def tenant_upload_root(upload_dir, tenant_id=db.DEFAULT_TENANT_ID):
    """Stable storage namespace; the legacy tenant keeps its old layout."""
    tenant = uuid.UUID(str(tenant_id))
    root = Path(upload_dir)
    return root if tenant == db.DEFAULT_TENANT_ID else root / "tenants" / str(tenant)


def _validated_canonical_name(canonical) -> str:
    """Pure basename validation shared by legacy compatibility seams.

    Rejected on sight:

      * a separator in EITHER platform's spelling, a drive letter, a UNC
        prefix, the dot names;
      * a leading dot -- that is how the temp files below are named, and
        a canonical name shaped like one could make a publication delete
        another's in-flight temp file;
      * a Windows DEVICE name, with or without an extension: writing to
        "NUL" succeeds and stores nothing, which is silent data loss
        wearing a success message;
      * a trailing dot or trailing space -- Windows strips them when
        opening, so "kurgu.pdf." and "kurgu.pdf " both reach
        "kurgu.pdf", and two rows would quietly share one file;
      * any NON-ASCII whitespace or invisible formatting character
        anywhere, and leading/trailing whitespace of any kind: a name
        that DISPLAYS as another name is how one document quietly
        replaces another in a listing nobody double-checks. The test is
        the Unicode CATEGORY, not ``str.isspace()`` -- a zero-width
        space is category Cf and ``isspace()`` calls it False, so the
        obvious check misses exactly the character chosen for being
        invisible.

    It returns a NAME, never a path.  Handle-bound migration can therefore
    validate metadata without first resolving or following the object it will
    later open.
    """
    raw = str(canonical)
    stem = raw.split(".", 1)[0].rstrip(" .").casefold()
    if (not raw
            or "/" in raw
            or "\\" in raw
            or ":" in raw
            or raw.startswith(".")
            or raw != Path(raw).name
            or raw in (".", "..")
            or raw.rstrip(". ") != raw
            or raw.strip() != raw
            or any(char != " "
                   and (char.isspace()
                        or unicodedata.category(char) in _INVISIBLE_CATEGORIES)
                   for char in raw)
            or stem in _WINDOWS_DEVICE_NAMES
            or any(ord(char) < 32 for char in raw)):
        raise UnsafeCanonicalName(
            "kanonik ad guvenli bir dosya adi degil; disk hedefi kurulmadi")
    return raw


def _checked_destination(canonical: str,
                         tenant_id=db.DEFAULT_TENANT_ID) -> Path:
    """The legacy flat destination check, retained for compatibility."""
    raw = _validated_canonical_name(canonical)
    root = tenant_upload_root(UPLOAD_DIR, tenant_id)
    destination = (root / raw).resolve()
    if destination.parent != root.resolve():
        raise UnsafeCanonicalName(
            "kanonik ad yukleme dizininin disina cikiyor")
    return destination


def source_path(upload_dir, canonical, tenant_id=db.DEFAULT_TENANT_ID):
    """Resolve an existing source without changing the module upload root."""
    raw = _validated_canonical_name(canonical)
    root = tenant_upload_root(upload_dir, tenant_id).resolve()
    candidate = (root / raw).resolve()
    if candidate.parent != root:
        raise UnsafeCanonicalName("kanonik ad tenant dizininden cikiyor")
    return candidate


def _open_legacy_parent(object_root, tenant: str):
    opened = []
    try:
        root = _object_transport.open_root(object_root)
        opened.append(root)
        if tenant == str(db.DEFAULT_TENANT_ID):
            return opened
        for name in ("tenants", tenant):
            entry = _object_transport.child_entry(opened[-1], name)
            if entry is None:
                raise VersionSourceMissing("legacy kaynak nesnesi yok")
            if entry.kind != "dir" or entry.reparse_tag:
                raise VersionSourceCorrupt(
                    "legacy kaynak agaci siradan dizinlerden olusmuyor")
            opened.append(_object_transport.open_child_directory(
                opened[-1], name))
        return opened
    except VersionSourceRefused:
        for directory in reversed(opened):
            _object_transport.close_directory_quietly(directory)
        raise
    except (OSError, _object_transport.TransportError):
        for directory in reversed(opened):
            _object_transport.close_directory_quietly(directory)
        raise VersionSourceRefused("legacy kaynak agaci acilamadi") from None


def migrate_legacy_version_source(
        object_root, tenant_id, document_id, version_id, canonical_filename,
        *, expected_sha256, max_bytes=MAX_VERSION_SOURCE_BYTES):
    """Upgrade one verified v2 flat source into immutable v3 storage.

    The legacy file is never removed or rewritten here.  Once immutable bytes
    exist, a retry proves and returns them even if a later operator has removed
    the old compatibility copy.
    """
    tenant = _uuid_text(tenant_id, "tenant kimligi")
    document = _uuid_text(document_id, "belge kimligi")
    version = _uuid_text(version_id, "surum kimligi")
    expected = _digest_text(expected_sha256)
    maximum = _source_limit(max_bytes)
    canonical = _validated_canonical_name(canonical_filename)

    try:
        proof = verify_version_source(
            object_root, tenant, document, version,
            expected_sha256=expected, max_bytes=maximum)
    except VersionSourceMissing:
        proof = None
    if proof is not None:
        return VersionSourceObject(
            tenant, document, version, expected, proof.size, False, False)

    directories = _open_legacy_parent(object_root, tenant)
    primary = None
    try:
        body, _identity = _read_named(directories[-1], canonical, maximum)
        if hashlib.sha256(body).hexdigest() != expected:
            raise VersionSourceCorrupt(
                "legacy kaynak ozeti kayitla eslesmiyor")
    except BaseException as refused:
        primary = refused
        raise
    finally:
        try:
            _close_directories(directories)
        except VersionSourceRefused as cleanup:
            if primary is None:
                raise
            primary.add_note(str(cleanup))

    return publish_version_source(
        object_root, tenant, document, version, body,
        expected_sha256=expected, max_bytes=maximum)


def ensure_bound_version_source(
        conn, object_root, tenant_id, document_id, version_id,
        canonical_filename, *, expected_sha256,
        max_bytes=MAX_VERSION_SOURCE_BYTES):
    """Prove one candidate binding, then make its immutable source readable.

    New publications already have the immutable object, so this is only a
    verification read.  A pre-v3 candidate may still have the same proven
    bytes in the flat compatibility location; that case is migrated while the
    document's publication lock is held.  Re-reading the row under that lock
    prevents a stale attempt or queue job from assigning today's flat bytes to
    yesterday's version identity.
    """
    canonical = _validated_canonical_name(canonical_filename)
    tenant = _uuid_text(tenant_id, "tenant kimligi")
    document = _uuid_text(document_id, "belge kimligi")
    version = _uuid_text(version_id, "surum kimligi")
    expected = _digest_text(expected_sha256)
    maximum = _source_limit(max_bytes)

    with db.document_publish_lock(conn, canonical):
        current = db.get_document(conn, document)
        if (current is None
                or current.get("filename") != canonical
                or current.get("candidate_id") != version
                or current.get("content_sha256") != expected):
            raise VersionSourceCorrupt(
                "surum kaynagi belge aday bagina uymuyor")
        return migrate_legacy_version_source(
            object_root, tenant, document, version, canonical,
            expected_sha256=expected, max_bytes=maximum)


def publish_candidate(conn, filename: str, file_type: str, body: bytes,
                      allow_replace: bool = False,
                      tenant_id=db.DEFAULT_TENANT_ID):
    """Stage, publish one immutable version source, finalize -- one lock.

    Returns ``(document_id, candidate_id, canonical_filename)``.

    The row is staged FIRST.  Its ``candidate_id`` is the immutable version
    id, and together with the document/tenant ids is the ENTIRE storage
    address.  The canonical filename remains metadata used by parsers and
    responses; it is revalidated but never becomes a new-write path.
    """
    # The lock belongs to the SERVICE, not to its callers: held across
    # the database decision AND the disk write, it is what makes the two
    # one publication. Held by an endpoint instead, it would have to be
    # re-implemented by every other caller -- and the CLI, which had
    # none, is exactly how the two paths drifted apart.
    with db.document_publish_lock(conn, filename):
        content_sha256 = hashlib.sha256(body).hexdigest()
        document_id, candidate_id, canonical = db.stage_candidate(
            conn, filename, file_type,
            content_sha256=content_sha256,
            allow_replace=allow_replace)

        # Preserve the old central metadata refusal: an unsafe canonical
        # name cannot later be handed to a suffix-based parser.  The result
        # is deliberately discarded -- no filename-derived target exists on
        # the immutable write road.
        _checked_destination(canonical, tenant_id)
        publish_version_source(
            UPLOAD_DIR, tenant_id, document_id, candidate_id, body,
            expected_sha256=content_sha256)

        # A REFUSED finalisation is not a quiet outcome. It means a newer
        # candidate was staged while these bytes were being written: the
        # disk moved, the row did not follow, and returning the success
        # triple would tell the caller a publication happened that the
        # database declined. Fail closed and say which half is which.
        if not db.finalize_candidate_publication(conn, document_id,
                                                 candidate_id):
            raise CandidateSuperseded(
                "yayin sonlandirilamadi: yazma sirasinda daha yeni bir "
                "aday evrelendi; surum kaynagi korundu ama bu aday "
                "yayimlanmadi")
    return document_id, candidate_id, canonical
