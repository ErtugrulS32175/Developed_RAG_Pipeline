"""What a READY workspace has to prove. PACKAGE B2B-A-D3A / R1B.3.

TWO LOOPS RETURNING IS NOT EVIDENCE. `create()` used to call a workspace
READY because the reference and implementer materialisation loops had
both finished without raising. That says the writes were ISSUED. It says
nothing about what is on disk afterwards -- and the whole point of the
two trees is that a later comparison between them means something, which
it cannot if they did not start equal.

So READY now requires the filesystem to agree, read back through the D2
handle-bound walker.

WHY A PROJECTION AND NOT `Manifest.digest`. Two independent copies of
one tree are SUPPOSED to differ: different root identity, different file
identity, different timestamps, different scan counters. Comparing the
full digest would refuse every healthy workspace. What must match is the
semantic content -- path, kind, mode, size, the content hash and the
stable attributes -- so that is what is compared and that is what the
stored digest covers.

WHAT THE MARKER IS FOR, and what it is NOT. The ledger remains the
authority for ownership. The marker is a second, independent statement
of the same four identities, living inside the holder and outside both
model-visible roots, so a holder swapped in from somewhere else fails to
answer for itself. It is defence in depth. It never authorises anything
on its own.

NO PATHS AFTER THE HOLDER IS OPENED. Verification here follows the same
rule the D2 audit arrived at the hard way: the holder is opened once,
and the marker and both roots are opened relative to THAT object. An
`lstat` followed by an `open` of the same path is two questions, and the
gap between them is the entire vulnerability.

WHAT MAY LEAVE. Fixed sentences, closed reasons, digests and opaque
identities. Never a path, never a file's bytes, never the operating
system's own error text.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import contract, fs_evidence, fs_transport
from tools.agent_loop import state as state_module
# ONE error class for the whole package, defined in its lowest module so
# every layer can raise it without an import cycle.
from tools.agent_loop.git_transport import FlatWorkspaceError

REFERENCE_DIRNAME = "reference"
IMPLEMENTER_DIRNAME = "implementer"
# beside the two roots, never inside one: anything the model can reach
# is something the model can write
MARKER_NAME = "workspace-owner.json"
MARKER_VERSION = 1
MARKER_CEILING = 4096

_WINDOWS = os.name == "nt"

# The fields two independent copies of one tree MUST agree on. Deliberate
# omissions: `mtime_ns` and `file_id` are per-copy by definition, and so
# is the manifest's own root identity, so including any of them would
# refuse every healthy workspace.
_PROJECTED = ("path", "kind", "mode", "size", "sha256", "attributes",
              "reparse_tag", "nlink", "link_target_mac")

MARKER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "marker_version", "workspace_id",
                 "repo_id", "run_id", "baseline_sha"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "marker_version": {"const": MARKER_VERSION},
        "workspace_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        # identities, never paths: an absolute path in a marker is an
        # absolute path in whatever prints the marker
        "repo_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "run_id": {"type": "string", "pattern": contract.IDENTIFIER_PATTERN},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
    },
}
_MARKER_VALIDATOR = Draft202012Validator(MARKER_SCHEMA)


# ---------------------------------------------------------------------
# talking to the transport
# ---------------------------------------------------------------------

def _transport(call, *args, message):
    """EVERY transport call goes through here.

    A `TransportError` already carries a fixed sentence; anything else --
    above all a raw `OSError`, whose text names absolute paths -- is
    replaced by one. The cleanup FLAG crosses the boundary; the lower
    layer's notes deliberately do not, because free text written closer
    to the filesystem is exactly where a path rides out."""
    try:
        return call(*args)
    except fs_transport.TransportError as exc:
        yeni = FlatWorkspaceError(str(exc),
                                  reason=contract.StopReason.PATH_NOT_ALLOWED)
        if fs_transport.cleanup_failed(exc):
            fs_transport.mark_cleanup_failed(yeni)
        raise yeni from None
    except OSError:
        raise FlatWorkspaceError(
            message, reason=contract.StopReason.PATH_NOT_ALLOWED) from None


def _close_directory_quietly(directory) -> bool:
    try:
        fs_transport.close_directory(directory)
        return True
    except (fs_transport.TransportError, OSError):
        return False


def _close_descriptor_quietly(descriptor: int) -> bool:
    try:
        os.close(descriptor)
        return True
    except OSError:
        return False


def _fold(exc: BaseException, ok: bool) -> None:
    """Consume a cleanup result on a failure path: it may not replace the
    error being raised, and it may not disappear."""
    if not ok:
        fs_transport.mark_cleanup_failed(exc)


# ---------------------------------------------------------------------
# the evidence gate
# ---------------------------------------------------------------------

def _scan(root, key):
    try:
        return fs_evidence.scan(root, key=key, limits=fs_evidence.Limits())
    except fs_evidence.EvidenceError as exc:
        # already a fixed sentence by that module's contract
        raise FlatWorkspaceError(str(exc), reason=exc.reason) from None


def _projection(manifest):
    """The semantic content of one root, and a refusal for anything this
    workspace model does not represent."""
    satirlar = []
    for entry in manifest.entries:
        if entry.kind not in ("file", "dir"):
            raise FlatWorkspaceError(
                "calisma alaninda beklenmeyen giris turu",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        # A second line, not the first: the walker reports every symlink
        # AND every junction as `kind == "link"`, so the check above is
        # what actually catches those. This one is here for a file or a
        # directory that somehow arrives carrying a reparse tag.
        if entry.reparse_tag or entry.link_target_mac:
            raise FlatWorkspaceError(
                "calisma alaninda beklenmeyen ayrisma noktasi",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        satirlar.append(tuple(str(getattr(entry, alan))
                              for alan in _PROJECTED))
    return tuple(sorted(satirlar))


def _digest_of(projection) -> str:
    digest = hashlib.sha256()
    for satir in projection:
        digest.update(b"\0".join(alan.encode("utf-8") for alan in satir)
                      + b"\n")
    return digest.hexdigest()


def compare_roots(reference, implementer):
    """Read both trees back and require them to agree.

    Returns `(evidence_digest, reference_identity, implementer_identity)`.
    The link key is minted here, used for exactly these two scans and
    never stored: it exists so the two manifests can be compared at all,
    and a key that outlived the call would be a key somebody could
    replay."""
    fs_evidence.quiesce()
    key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    sol = _scan(reference, key)
    sag = _scan(implementer, key)

    solun, sagin = _projection(sol), _projection(sag)
    if solun != sagin:
        raise FlatWorkspaceError("iki agac ayni icerikle baslamiyor")
    if sol.root_identity == sag.root_identity:
        # the same directory twice is not two independent trees, and a
        # comparison against yourself always passes
        raise FlatWorkspaceError("iki kok ayni nesne")
    return _digest_of(solun), sol.root_identity, sag.root_identity


# ---------------------------------------------------------------------
# the runner-owned marker
# ---------------------------------------------------------------------

def marker_payload(record) -> dict:
    """Exactly the four identities plus the two versions. Never a path,
    never a digest, never the evidence key."""
    return {"protocol_version": contract.PROTOCOL_VERSION,
            "marker_version": MARKER_VERSION,
            "workspace_id": record["workspace_id"],
            "repo_id": record["repo_id"], "run_id": record["run_id"],
            "baseline_sha": record["baseline_sha"]}


def write_marker(holder, record) -> None:
    """Atomically, under the holder and outside both roots."""
    try:
        state_module.write_json_atomically(
            holder / MARKER_NAME, marker_payload(record), MARKER_SCHEMA,
            "calisma alani sahiplik isareti")
    except state_module.StateError:
        raise FlatWorkspaceError("sahiplik isareti yazilamadi") from None
    except OSError:
        raise FlatWorkspaceError("sahiplik isareti yazilamadi") from None


def _read_marker(kok, record):
    """The marker's bytes, through the holder handle that listed it."""
    if record is None:
        raise FlatWorkspaceError("calisma alani sahiplik isareti yok")
    if record.kind != "file" or record.reparse_tag:
        raise FlatWorkspaceError(
            "sahiplik isareti siradan bir dosya degil",
            reason=contract.StopReason.PATH_NOT_ALLOWED)
    descriptor = _transport(fs_transport.open_child_file, kok, record,
                            message="sahiplik isareti acilamadi")
    try:
        veri = b""
        while len(veri) <= MARKER_CEILING:
            try:
                parca = os.read(descriptor, MARKER_CEILING + 1)
            except OSError:
                raise FlatWorkspaceError(
                    "sahiplik isareti okunamadi") from None
            if not parca:
                break
            veri += parca
        if len(veri) > MARKER_CEILING:
            raise FlatWorkspaceError(
                "sahiplik isareti sozlesme tavanini asiyor")
    except BaseException as birincil:
        _fold(birincil, _close_descriptor_quietly(descriptor))
        raise
    if not _close_descriptor_quietly(descriptor):
        raise FlatWorkspaceError("sahiplik isareti kapatilamadi")
    try:
        marker = json.loads(veri.decode("utf-8"))
        _MARKER_VALIDATOR.validate(marker)
    except (UnicodeDecodeError, ValueError, ValidationError):
        raise FlatWorkspaceError(
            "sahiplik isareti sozlesmeye uymuyor") from None
    return marker


# ---------------------------------------------------------------------
# ownership, answered by one open object
# ---------------------------------------------------------------------

def _same_name(name: str, expected: str) -> bool:
    if _WINDOWS:
        return name.casefold() == expected.casefold()
    return name == expected


def _open_root_child(kok, records, expected: str):
    """One materialised root, opened THROUGH the holder handle."""
    for record in records:
        if not _same_name(record.name, expected):
            continue
        if record.kind != "dir" or record.reparse_tag:
            raise FlatWorkspaceError(
                "calisma alani koku siradan bir dizin degil",
                reason=contract.StopReason.PATH_NOT_ALLOWED)
        return _transport(fs_transport.open_child_directory, kok, record,
                          message="calisma alani koku acilamadi")
    raise FlatWorkspaceError("calisma alani koku yok")


def _refuse_git(directory) -> None:
    for record in _transport(fs_transport.list_directory, directory,
                             message="calisma alani koku listelenemedi"):
        if _same_name(record.name, ".git"):
            raise FlatWorkspaceError(
                "calisma alaninda git denetim duzlemi var",
                reason=contract.StopReason.PATH_NOT_ALLOWED)


def inspect_holder(holder):
    """Open the holder ONCE and answer everything from that object.

    Returns `(marker, reference_identity, implementer_identity)`. The
    holder itself is opened with the transport's no-follow root open, so
    a holder replaced by a symlink or a junction is refused before a
    single child is looked at."""
    kok = _transport(fs_transport.open_root, holder,
                     message="calisma alani dizini acilamadi")
    try:
        records = _transport(fs_transport.list_directory, kok,
                             message="calisma alani listelenemedi")
        marker = _read_marker(
            kok, next((r for r in records
                       if _same_name(r.name, MARKER_NAME)), None))
        kimlikler = []
        for beklenen in (REFERENCE_DIRNAME, IMPLEMENTER_DIRNAME):
            alt = _open_root_child(kok, records, beklenen)
            try:
                _refuse_git(alt)
                kimlikler.append(fs_transport.directory_identity(alt))
            except BaseException as birincil:
                _fold(birincil, _close_directory_quietly(alt))
                raise
            if not _close_directory_quietly(alt):
                raise FlatWorkspaceError("calisma alani koku kapatilamadi")
    except BaseException as birincil:
        _fold(birincil, _close_directory_quietly(kok))
        raise
    if not _close_directory_quietly(kok):
        raise FlatWorkspaceError("calisma alani dizini kapatilamadi")
    return marker, kimlikler[0], kimlikler[1]
