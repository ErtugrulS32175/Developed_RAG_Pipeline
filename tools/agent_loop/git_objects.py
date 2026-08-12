"""Raw Git objects. PACKAGE B2B-A-D3A.

THE TRUST BOUNDARY. Everything in this module reads bytes that a
repository produced, and a repository under audit is not a trusted
source. So git's answers are recomputed here rather than believed:
every object's id is derived again from the bytes that actually
arrived, and the tree is parsed from the object itself.

WHY NOT `ls-tree`. MEASURED: it is a NORMALISING view. A raw entry
whose mode is `100640` is reported as `100644`, and a flat entry whose
name contains a slash is reported exactly like a real subtree -- so a
hostile tree is invisible through it. The same rule that forbids
trusting blob bytes without recomputing their id forbids trusting that
view. `checkout`, `checkout-index` and `archive` are barred for a
different reason: each runs filters, EOL conversion or hooks, which
hands the repository a vote over the copy.

WHERE THIS RUNS. Only during `flat_workspace.create()`. Nothing here is
reachable from the verification surface that runs after a workspace
exists.

WHAT IS NO LONGER HERE. Running git is a process lifecycle problem --
a child, two reader threads, a container and a cleanup verdict -- and
none of it has an opinion about what a tree object means. It lives in
`git_transport`, which also owns the error type both layers raise.

WHAT MAY LEAVE. Fixed sentences and closed reasons. Never git's stderr,
never a path, never an object's bytes.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import PurePosixPath

from tools.agent_loop import contract
from tools.agent_loop import git_transport

# Re-exported so a caller of this layer never has to know the transport
# exists, and so `except FlatWorkspaceError` keeps catching both.
FlatWorkspaceError = git_transport.FlatWorkspaceError
STDERR_CEILING = git_transport.STDERR_CEILING
_byte_ceiling = git_transport.byte_ceiling
_git_bytes = git_transport.git_bytes
_git_env = git_transport._git_env

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

MODE_FILE = "100644"
MODE_EXEC = "100755"
MODE_TREE = "40000"
_ALLOWED_BLOB_MODES = (MODE_FILE, MODE_EXEC)
_MAX_TREE_DEPTH = 64

_FORBIDDEN_CATEGORIES = ("Cc", "Cf", "Zl", "Zp")
_ORDINARY_SPACE = " "

# Ceilings, per object class. One generic unbounded read was the
# defect: MEASURED, `capture_output=True` took 50,331,648 bytes into
# memory before any application limit was consulted, so the 64 MiB blob
# ceiling was not a memory bound at all. Every read below names the
# ceiling that fits the object class it asked for.
TYPE_CEILING = 256                # `cat-file -t` answers one word
COMMIT_CEILING = 64 << 10         # header plus message
TREE_CEILING = 8 << 20            # structural, not content
# blob: `Limits.max_file_bytes`, a D3-specific measured decision --
# largest blob at HEAD is 98,162 bytes, roughly 680x headroom


def object_id(kind: str, data: bytes) -> str:
    """Git's own object id, recomputed here.

    SHA-1 because that is what the object id IS; this is an integrity
    check against the repository's answer, not a security hash."""
    header = kind.encode("ascii") + b" " + \
        str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()

def blob_object_id(data: bytes) -> str:
    return object_id("blob", data)

def _object_bytes(repo, object_id: str, *, limit) -> bytes:
    """Raw blob bytes, under the caller's measured blob ceiling.

    Isolated behind one name so a test can corrupt exactly this."""
    return _git_bytes(repo, "cat-file", "blob", object_id,
                      stdout_limit=_byte_ceiling(limit, "blob tavani"))

_CEILINGS = {"commit": COMMIT_CEILING, "tree": TREE_CEILING,
             "blob": None}


def _verified_object(repo, kind: str, oid: str, *,
                     blob_limit=None) -> bytes:
    """Raw object bytes whose id is recomputed before they are used."""
    data = _git_bytes(repo, "cat-file", kind, oid,
                      stdout_limit=(_CEILINGS[kind]
                                    or _byte_ceiling(
                                        blob_limit,
                                        "blob tavani")))
    if type(data) is not bytes:
        raise FlatWorkspaceError("nesne baytlari okunamadi")
    if object_id(kind, data) != oid:
        raise FlatWorkspaceError("nesne kimligi baytlarla uyusmuyor")
    return data

def _baseline_tree(repo, baseline_sha: str) -> str:
    """The commit's tree, taken from the RAW commit object.

    `rev-parse <sha>^{tree}` would be a second normalising view; the
    commit object's first line says `tree <hex>` and that is the
    authority."""
    if type(baseline_sha) is not str or not _FULL_SHA.match(baseline_sha):
        raise FlatWorkspaceError("taban surumu tam SHA degil",
                                 reason=contract.StopReason.BASELINE_MISMATCH)
    tur = _git_bytes(repo, "cat-file", "-t", baseline_sha,
                      stdout_limit=TYPE_CEILING).strip()
    if tur != b"commit":
        raise FlatWorkspaceError("taban surumu bir commit degil",
                                 reason=contract.StopReason.BASELINE_MISMATCH)
    govde = _verified_object(repo, "commit", baseline_sha)
    ilk = govde.split(b"\n", 1)[0]
    if not ilk.startswith(b"tree ") or len(ilk) != 45:
        raise FlatWorkspaceError("commit nesnesi agac satiri tasimiyor")
    agac = ilk[5:].decode("ascii", "replace")
    if not _FULL_SHA.match(agac):
        raise FlatWorkspaceError("commit agac kimligi tam SHA degil")
    return agac

def _parse_tree(data: bytes):
    """A tree object, byte by byte.

    MEASURED, and the reason this exists instead of `ls-tree`: that
    command NORMALISES. A raw entry with mode `100640` is reported as
    `100644`, and a flat entry whose name contains a slash is reported
    exactly like a real subtree -- so a hostile tree is invisible
    through it. The same rule as blob bytes: git's answer is verified,
    not trusted."""
    girisler = []
    i = 0
    while i < len(data):
        bosluk = data.find(b" ", i)
        if bosluk < 0:
            raise FlatWorkspaceError("agac nesnesi cozumlenemedi")
        sifir = data.find(b"\0", bosluk)
        if sifir < 0 or sifir + 21 > len(data):
            raise FlatWorkspaceError("agac nesnesi cozumlenemedi")
        mode = data[i:bosluk].decode("ascii", "replace")
        ad = data[bosluk + 1:sifir]
        oid = data[sifir + 1:sifir + 21].hex()
        if not ad:
            raise FlatWorkspaceError("agac girdisinde bos ad")
        girisler.append((mode, ad, oid))
        i = sifir + 21
    return girisler

def _canonical_component(raw: bytes) -> str:
    """ONE path component from a tree entry, or a refusal.

    The bytes are the repository's own name and can be anything at all.
    A separator here is not a nested path: tree objects are flat, so a
    name containing `/` is a malformed entry that `ls-tree` would have
    presented as an ordinary nested path."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise FlatWorkspaceError("yol gecerli UTF-8 degil",
                                 reason=contract.StopReason.PATH_NOT_ALLOWED) \
            from None
    if not text or "/" in text or "\\" in text:
        raise FlatWorkspaceError("yol kanonik degil",
                                 reason=contract.StopReason.PATH_NOT_ALLOWED)
    if PurePosixPath(text).is_absolute():
        raise FlatWorkspaceError("yol kanonik degil",
                                 reason=contract.StopReason.PATH_NOT_ALLOWED)
    for part in (text,):
        if part in ("", ".", ".."):
            raise FlatWorkspaceError(
                "yol kanonik degil",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        # `.git` at ANY spelling: Windows folds case, so `.GIT` is the
        # same directory there and writing into it is writing into the
        # repository control plane
        if part.casefold() == ".git":
            raise FlatWorkspaceError(
                "yol git denetim duzlemini adliyor",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        for character in part:
            category = unicodedata.category(character)
            if category in _FORBIDDEN_CATEGORIES:
                raise FlatWorkspaceError(
                    "yol gorunmez ya da kontrol karakteri tasiyor",
                    reason=contract.StopReason.PATH_NOT_ALLOWED)
            if category == "Zs" and character != _ORDINARY_SPACE:
                raise FlatWorkspaceError(
                    "yol siradan olmayan bosluk karakteri tasiyor",
                    reason=contract.StopReason.PATH_NOT_ALLOWED)
        if unicodedata.normalize("NFC", part) != part:
            raise FlatWorkspaceError(
                "yol NFC biciminde degil",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
    return text

def baseline_digest(entries) -> str:
    """A deterministic digest over path, mode, object id and length.

    Serialised field by field in canonical path order -- never through
    `repr`, a dict's iteration order or anything locale-dependent, all
    of which have produced "identical" baselines that were not."""
    stream = b""
    for path, mode, object_id, length in sorted(entries,
                                                key=lambda item: item[0]):
        stream += b"\0".join((path.encode("utf-8"), mode.encode("ascii"),
                              object_id.encode("ascii"),
                              str(length).encode("ascii"))) + b"\n"
    return hashlib.sha256(stream).hexdigest()

def _walk_tree(repo, tree_oid, parents, girisler, gorulen, limits, depth):
    """Depth-first over RAW tree objects, validating as it goes."""
    if depth > _MAX_TREE_DEPTH:
        raise FlatWorkspaceError("agac derinligi sozlesme tavanini asiyor")
    for mode, ad_ham, oid in _parse_tree(
            _verified_object(repo, "tree", tree_oid)):
        if not _FULL_SHA.match(oid):
            raise FlatWorkspaceError("nesne kimligi tam SHA degil")
        # ONE component, so a name carrying a separator is a malformed
        # entry rather than a nested path
        bilesen = _canonical_component(ad_ham)
        yol = "/".join(parents + (bilesen,))
        if mode == MODE_TREE:
            _walk_tree(repo, oid, parents + (bilesen,), girisler, gorulen,
                       limits, depth + 1)
            continue
        if mode not in _ALLOWED_BLOB_MODES:
            raise FlatWorkspaceError(
                "agac girdisi sozlesme disi bir mod tasiyor",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        anahtar = yol.casefold() if os.name == "nt" else yol
        if anahtar in gorulen:
            raise FlatWorkspaceError(
                "agacta yinelenen kanonik yol",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        gorulen.add(anahtar)
        girisler.append((yol, mode, oid))
        if len(girisler) > limits.max_entries:
            raise FlatWorkspaceError("giris sayisi sozlesme tavanini asiyor")

def _read_tree(repo, baseline_sha: str, limits):
    girisler, gorulen = [], set()
    _walk_tree(repo, _baseline_tree(repo, baseline_sha), (), girisler,
               gorulen, limits, 0)
    if not girisler:
        raise FlatWorkspaceError("taban surumu bos")
    return girisler
