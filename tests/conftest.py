"""The operator's document tree is not a test workspace.

Two guards, and BOTH have to be in place before a single test module is
imported -- which is why almost nothing here is a fixture.

  1. PREVENTION. The environment variables that name every
     document-bearing directory are pointed at a disposable root, at
     THIS MODULE'S import, before anything under `pipeline/` has been
     imported. Those modules read their roots from the environment once,
     at their own import time, so a redirect that lands later lands too
     late.

  2. DETECTION. An inventory of `data/` is taken before collection and
     compared after the session. Anything ADDED, CHANGED or REMOVED
     while the tests ran fails the run, whatever it contains.

WHY NOT FIXTURES. The first version used session fixtures, and session
fixtures run AFTER pytest has imported the test modules. A write that
happens at import time therefore escaped both halves: it was not
redirected, and because the "before" snapshot had not been taken yet, it
counted as part of the starting state and was never seen at all. Module
import and `pytest_configure` both run before collection; fixtures do
not.

WHY THE ENVIRONMENT AND NOT THE MODULE ATTRIBUTES. Patching
`publication.UPLOAD_DIR` protected exactly one of five readers. The API
and the OpenWebUI table path have their own module-level copies of the
same root, and the table path writes to it directly. One environment
variable, read before any of them import, covers every reader there is
and every reader added later.

WHAT A FAILURE MAY SAY. Change type and two KEYED ids, and nothing else.
Not the path, not a masked path, not an unkeyed hash of either. The
first version masked the path and split each name at its FIRST dot, so a
name with several dots arrived largely intact. The second used a plain
SHA-256 of the path -- which is a deterministic fingerprint, not an
anonymous id: anyone who can guess a filename can confirm it by
recomputing the digest. The key below is random per session, so the ids
in a report cannot be attacked with a dictionary at all, and the report
stays actionable because the key is written to the local disposable root
for `resolve_path_id` to use.

This file is checked by tests/test_data_tree_guard.py, which runs it in
throwaway repositories via subprocess. A measuring instrument that is
not itself measured is just an assumption.
"""
import hashlib
import hmac
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"

# ---------------------------------------------------------------------
# PREVENTION -- at import, before any pipeline module can read these.
# ---------------------------------------------------------------------

_DISPOSABLE_ROOT = Path(tempfile.mkdtemp(prefix="ragtest-testler-"))

# Every environment name that steers document-bearing content to disk.
# `UPLOAD_DIR` is read by the publication service, the API and the
# OpenWebUI table path; `OUTPUT_DIR` and `EXPORT_DIR` receive exports
# derived from document CONTENT, which makes them the same kind of
# directory even though they are not the same directory.
_DOCUMENT_ROOTS = {
    "UPLOAD_DIR": _DISPOSABLE_ROOT / "uploads",
    "OUTPUT_DIR": _DISPOSABLE_ROOT / "output",
    "EXPORT_DIR": _DISPOSABLE_ROOT / "export",
}

for _name, _path in _DOCUMENT_ROOTS.items():
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_path)

# Defence in depth, for the case this module was NOT first: if something
# (a plugin, another conftest) already imported a pipeline module, its
# root is already bound to the production value and no environment
# change will move it. Nothing is imported here -- only modules that are
# already loaded are touched.
_ALREADY_BOUND = (
    ("pipeline.index.publication", "UPLOAD_DIR", "UPLOAD_DIR"),
    ("pipeline.api.app", "UPLOAD_DIR", "UPLOAD_DIR"),
    ("pipeline.api.owui_chat", "UPLOAD_DIR", "UPLOAD_DIR"),
    ("pipeline.api.owui_chat", "EXPORT_DIR", "EXPORT_DIR"),
    ("pipeline.index.ingest", "OUTPUT_DIR", "OUTPUT_DIR"),
)
for _module_name, _attribute, _root_name in _ALREADY_BOUND:
    _module = sys.modules.get(_module_name)
    if _module is not None and hasattr(_module, _attribute):
        setattr(_module, _attribute, _DOCUMENT_ROOTS[_root_name])


# ---------------------------------------------------------------------
# DETECTION
# ---------------------------------------------------------------------

# Random per session. Without it, an id is a deterministic fingerprint:
# a reader who guesses a filename can confirm the guess by hashing
# it. With it, a report is inert to everyone who does not hold the key,
# including whoever the report is forwarded to.
_KEY = secrets.token_bytes(32)

KEY_FILENAME = "anahtar.txt"

_BEFORE = {}


def _keyed(material):
    """A full-length keyed id. Truncation happens only in the report."""
    if isinstance(material, str):
        material = material.encode("utf-8")
    return hmac.new(_KEY, material, hashlib.sha256).hexdigest()


def _path_id(relative, key=None):
    """An opaque id for a path -- never the path itself."""
    posix = Path(relative).as_posix()
    if key is None:
        return _keyed(posix)
    return hmac.new(bytes.fromhex(key), posix.encode("utf-8"),
                    hashlib.sha256).hexdigest()


def _inventory(root):
    """``{keyed path id: keyed content id}`` for every file under root.

    Both halves are keyed and stored at FULL length. The inventory is
    keyed rather than merely reported that way so the structure this
    guard carries around holds no filename either -- and full length
    because a 48-bit id could in principle collide, and a collision here
    would hide a change instead of reporting one.

    Whole-content hashes rather than size and mtime: a file rewritten to
    the same length on a filesystem with coarse timestamps would slip
    past the cheaper check, and this guard exists for the case where
    something went wrong."""
    found = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = hashlib.sha256(path.read_bytes()).digest()
        except OSError:
            # unreadable is a STATE, not an absence: recorded so a file
            # that becomes unreadable mid-session still counts as changed
            content = b"okunamadi"
        found[_path_id(path.relative_to(root))] = _keyed(content)
    return found


def resolve_path_id(path_id, key, root=DATA_ROOT):
    """LOCAL OPERATOR TOOL. Returns the real path behind a reported id.

    ``key`` is the hex string the failing session wrote to
    ``<disposable root>/anahtar.txt``; without it the ids cannot be
    resolved by anyone, which is the point of keying them.

    This is the one thing here that yields a document's name, so it is
    deliberately not called by anything: run it yourself, read the
    answer on your own screen, and do not paste it into a report, an
    issue, a commit message or a chat."""
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if _path_id(path.relative_to(root), key=key) == path_id:
            return path
    return None


def pytest_configure(config):
    """Snapshot `data/` BEFORE collection imports any test module."""
    _BEFORE.clear()
    _BEFORE.update(_inventory(DATA_ROOT))


def pytest_sessionfinish(session, exitstatus):
    after = _inventory(DATA_ROOT)
    added = set(after) - set(_BEFORE)
    removed = set(_BEFORE) - set(after)
    changed = {key for key in set(_BEFORE) & set(after)
               if _BEFORE[key] != after[key]}

    if not (added or removed or changed):
        shutil.rmtree(_DISPOSABLE_ROOT, ignore_errors=True)
        return

    # The disposable root SURVIVES a failure, because the key lives in
    # it. It is what turns an inert report back into something an
    # operator can act on, and it stays on their disk rather than in the
    # report.
    key_file = _DISPOSABLE_ROOT / KEY_FILENAME
    try:
        key_file.write_text(_KEY.hex(), encoding="utf-8")
    except OSError:                         # pragma: no cover
        pass

    lines = ([f"eklendi  {key[:16]} (icerik {after[key][:16]})"
              for key in sorted(added)]
             + [f"degisti  {key[:16]} (icerik {after[key][:16]})"
                for key in sorted(changed)]
             + [f"silindi  {key[:16]} (icerik {_BEFORE[key][:16]})"
                for key in sorted(removed)])
    message = (
        f"VERI AGACI DEGISTI -- eklenen {len(added)}, degisen "
        f"{len(changed)}, silinen {len(removed)}. Bir test gercek belge "
        f"dizinine yaziyor; sizinti taramasi bu dosyalarda fail-closed "
        f"durur.\nKimlikler oturuma ozel bir anahtarla uretildi; yerel "
        f"olarak cozmek icin tests/conftest.resolve_path_id ve "
        f"{key_file}\n  - " + "\n  - ".join(lines))
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line(message, red=True, bold=True)
    else:                                   # pragma: no cover -- no terminal
        print(message, file=sys.stderr)

    # ONLY when the session was otherwise clean. An interrupted run
    # exits 2 and an internal error exits 3; overwriting either with 1
    # would trade a precise diagnosis for a vaguer one, and this finding
    # is already on the screen regardless of the code.
    if exitstatus == 0:
        session.exitstatus = 1
