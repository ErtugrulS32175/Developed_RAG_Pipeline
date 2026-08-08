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
from pathlib import Path

from dotenv import load_dotenv

from pipeline.index import db
from pipeline.index.attempt_contract import CandidateSuperseded

load_dotenv()

# Read through the module attribute at call time, never bound into a
# default argument: tests and deployments both need to point it
# elsewhere, and a value captured at import cannot be moved.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))

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


def _checked_destination(canonical: str) -> Path:
    """The one place a filename becomes a path.

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

    Then the resolved path must sit DIRECTLY in the upload directory:
    the string checks catch the spellings we thought of, containment
    catches the rest.
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
    root = Path(UPLOAD_DIR)
    destination = (root / raw).resolve()
    if destination.parent != root.resolve():
        raise UnsafeCanonicalName(
            "kanonik ad yukleme dizininin disina cikiyor")
    return destination


def publish_candidate(conn, filename: str, file_type: str, body: bytes,
                      allow_replace: bool = False):
    """Stage, put the bytes in place, finalize -- one lock, three steps.

    Returns ``(document_id, candidate_id, canonical_filename)``.

    The order is deliberate: the row is staged FIRST, so a crash between
    the steps leaves a candidate that is visibly unpublished rather than
    bytes nobody recorded. The destination is validated BEFORE anything
    is written, including the temp file: a publication that refuses must
    leave the upload directory exactly as it found it, and "wrote the
    body to a temp file, then raised" is not refusing.
    """
    # The lock belongs to the SERVICE, not to its callers: held across
    # the database decision AND the disk write, it is what makes the two
    # one publication. Held by an endpoint instead, it would have to be
    # re-implemented by every other caller -- and the CLI, which had
    # none, is exactly how the two paths drifted apart.
    with db.document_publish_lock(conn, filename):
        document_id, candidate_id, canonical = db.stage_candidate(
            conn, filename, file_type,
            content_sha256=hashlib.sha256(body).hexdigest(),
            allow_replace=allow_replace)

        destination = _checked_destination(canonical)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.yayin-{uuid.uuid4().hex}")
        try:
            temporary.write_bytes(body)
            os.replace(temporary, destination)
        finally:
            # os.replace consumed it on the happy path; this is the
            # failed one
            temporary.unlink(missing_ok=True)

        # A REFUSED finalisation is not a quiet outcome. It means a newer
        # candidate was staged while these bytes were being written: the
        # disk moved, the row did not follow, and returning the success
        # triple would tell the caller a publication happened that the
        # database declined. Fail closed and say which half is which.
        if not db.finalize_candidate_publication(conn, document_id,
                                                 candidate_id):
            raise CandidateSuperseded(
                "yayin sonlandirilamadi: yazma sirasinda daha yeni bir "
                "aday evrelendi; disk guncellendi ama bu aday "
                "yayimlanmadi")
    return document_id, candidate_id, canonical
