"""Immutable source-object storage for document versions.

The identity is tenant/document/version UUIDs owned by the server.  A user
filename is metadata and has no place in the storage API or object path.
"""
import hashlib
import inspect
import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from pipeline.index import publication


TENANT = uuid.UUID("10000000-0000-4000-8000-000000000001")
DOCUMENT = uuid.UUID("20000000-0000-4000-8000-000000000002")
VERSION = uuid.UUID("30000000-0000-4000-8000-000000000003")
TENANT_B = uuid.UUID("40000000-0000-4000-8000-000000000004")
BODY = b"immutable-version-source"
DIGEST = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "objects"
    root.mkdir()
    return root


def _publish(root, body=BODY, digest=DIGEST, maximum=1024):
    return publication.publish_version_source(
        root, TENANT, DOCUMENT, VERSION, body,
        expected_sha256=digest, max_bytes=maximum)


def _read(root, digest=DIGEST, maximum=1024):
    return publication.read_version_source(
        root, TENANT, DOCUMENT, VERSION,
        expected_sha256=digest, max_bytes=maximum)


def test_the_object_path_contains_only_server_owned_uuid_segments(store):
    path = publication.version_source_path(
        store, TENANT, DOCUMENT, VERSION)

    assert path.relative_to(store).parts == (
        "tenants", str(TENANT), "documents", str(DOCUMENT), "versions",
        str(VERSION), "source.bin")
    for function in (publication.version_source_path,
                     publication.publish_version_source,
                     publication.read_version_source,
                     publication.verify_version_source):
        assert "filename" not in inspect.signature(function).parameters


def test_production_storage_has_no_agent_loop_runtime_dependency():
    root = Path(publication.__file__).parents[1]
    sources = [Path(publication.__file__), *sorted((root / "storage").glob(
        "*.py"))]

    assert all("tools.agent_loop" not in path.read_text(encoding="utf-8")
               for path in sources)


@pytest.mark.parametrize(
    "value",
    ["not-a-uuid", "ABCDEFAB-CDEF-4ABC-8DEF-ABCDEFABCDEF", "../x"],
)
def test_noncanonical_identifiers_are_refused_before_storage(store, value):
    with pytest.raises(publication.VersionSourceRefused):
        publication.publish_version_source(
            store, value, DOCUMENT, VERSION, BODY,
            expected_sha256=DIGEST, max_bytes=1024)
    assert list(store.iterdir()) == []


def test_publish_is_atomic_verified_and_readable_for_ingest(store,
                                                            monkeypatch):
    calls = []
    real_rename = publication._object_transport.rename_child
    real_fsync = publication._object_transport.fsync_handle

    def fsync(handle):
        calls.append("file_fsync")
        return real_fsync(handle)

    def rename(source, source_name, target, target_name):
        calls.append(("rename", source_name, target_name))
        return real_rename(source, source_name, target, target_name)

    monkeypatch.setattr(publication._object_transport, "fsync_handle", fsync)
    monkeypatch.setattr(publication._object_transport, "rename_child", rename)

    result = _publish(store)

    assert result == publication.VersionSourceObject(
        str(TENANT), str(DOCUMENT), str(VERSION), DIGEST, len(BODY), True,
        result.directory_fsync)
    assert calls[0] == "file_fsync"
    assert calls[1][0] == "rename"
    assert calls[1][1].startswith("source-")
    assert calls[1][1].endswith(".tmp")
    assert calls[1][2] == "source.bin"
    assert _read(store) == BODY
    parent = publication.version_source_path(
        store, TENANT, DOCUMENT, VERSION).parent
    assert [item.name for item in parent.iterdir()] == ["source.bin"]


def test_same_byte_retry_is_idempotent_and_never_renames_again(store,
                                                               monkeypatch):
    first = _publish(store)
    monkeypatch.setattr(
        publication._object_transport, "rename_child",
        lambda *args: pytest.fail("idempotent retry tried to rename"))

    second = _publish(store)

    assert first.created is True
    assert second.created is False
    assert second.sha256 == DIGEST
    assert second.sha256 == first.sha256
    assert _read(store) == BODY


def test_activation_proof_is_fresh_closed_and_contains_no_source_bytes(store):
    _publish(store)

    proof = publication.verify_version_source(
        store, TENANT, DOCUMENT, VERSION,
        expected_sha256=DIGEST, max_bytes=1024)

    assert proof == publication.VersionSourceProof(
        str(TENANT), str(DOCUMENT), str(VERSION), DIGEST, len(BODY))
    assert not hasattr(proof, "body")
    assert not hasattr(proof, "path")


def test_activation_proof_refuses_a_wrong_database_digest(store):
    _publish(store)

    with pytest.raises(publication.VersionSourceCorrupt):
        publication.verify_version_source(
            store, TENANT, DOCUMENT, VERSION,
            expected_sha256="0" * 64, max_bytes=1024)


def test_existing_different_bytes_are_never_replaced(store):
    _publish(store)
    different = b"different-source"

    with pytest.raises(publication.VersionSourceDifferent):
        publication.publish_version_source(
            store, TENANT, DOCUMENT, VERSION, different,
            expected_sha256=hashlib.sha256(different).hexdigest(),
            max_bytes=1024)

    assert _read(store) == BODY


@pytest.mark.parametrize(
    ("body", "digest", "maximum"),
    [(bytearray(BODY), DIGEST, 1024),
     (BODY, "0" * 64, 1024),
     (BODY, DIGEST, len(BODY) - 1),
     (BODY, DIGEST.upper(), 1024)],
)
def test_invalid_or_unproven_bytes_are_refused_before_a_temp_exists(
        store, body, digest, maximum):
    with pytest.raises(publication.VersionSourceRefused):
        publication.publish_version_source(
            store, TENANT, DOCUMENT, VERSION, body,
            expected_sha256=digest, max_bytes=maximum)
    assert list(store.iterdir()) == []


def test_ingest_read_refuses_digest_drift_and_a_bounded_oversize(store):
    _publish(store)

    with pytest.raises(publication.VersionSourceCorrupt):
        _read(store, digest="0" * 64)
    with pytest.raises(publication.VersionSourceCorrupt):
        _read(store, maximum=len(BODY) - 1)


def test_missing_source_is_a_closed_typed_result(store):
    with pytest.raises(publication.VersionSourceMissing) as caught:
        _read(store)
    assert str(store) not in str(caught.value)


def _directory_link(link: Path, target: Path):
    if os.name != "nt":
        os.symlink(target, link, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, check=False)
    if completed.returncode:
        pytest.skip("junction could not be created on this Windows host")


def test_a_linked_root_is_refused_without_writing_outside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "objects"
    _directory_link(linked, outside)

    with pytest.raises(publication.VersionSourceRefused):
        _publish(linked)

    assert list(outside.iterdir()) == []


def test_a_link_in_the_uuid_tree_is_refused_without_following_it(store,
                                                                 tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _directory_link(store / "tenants", outside)

    with pytest.raises(publication.VersionSourceRefused):
        _publish(store)

    assert list(outside.iterdir()) == []


def test_a_link_at_the_final_source_is_never_read_or_replaced(store,
                                                              tmp_path):
    target = tmp_path / "canary"
    target.write_bytes(b"outside-canary")
    final = publication.version_source_path(store, TENANT, DOCUMENT, VERSION)
    final.parent.mkdir(parents=True)
    try:
        os.symlink(target, final)
    except OSError:
        pytest.skip("file symlink could not be created on this host")

    with pytest.raises(publication.VersionSourceCorrupt):
        _publish(store)

    assert target.read_bytes() == b"outside-canary"


@pytest.mark.parametrize("tenant", [publication.db.DEFAULT_TENANT_ID, TENANT_B],
                         ids=["default-root", "tenant-root"])
def test_legacy_flat_source_migrates_to_the_uuid_object_tree(store, tenant):
    legacy_parent = (store if tenant == publication.db.DEFAULT_TENANT_ID
                     else store / "tenants" / str(tenant))
    legacy_parent.mkdir(parents=True, exist_ok=True)
    legacy = legacy_parent / "Original Report.PDF"
    legacy.write_bytes(BODY)

    result = publication.migrate_legacy_version_source(
        store, tenant, DOCUMENT, VERSION, legacy.name,
        expected_sha256=DIGEST, max_bytes=1024)

    assert result.created is True
    assert publication.read_version_source(
        store, tenant, DOCUMENT, VERSION,
        expected_sha256=DIGEST, max_bytes=1024) == BODY
    assert legacy.read_bytes() == BODY


def test_legacy_migration_retry_needs_no_remaining_flat_copy(store):
    legacy = store / "rapor.pdf"
    legacy.write_bytes(BODY)
    first = publication.migrate_legacy_version_source(
        store, publication.db.DEFAULT_TENANT_ID, DOCUMENT, VERSION,
        legacy.name, expected_sha256=DIGEST, max_bytes=1024)
    legacy.unlink()

    second = publication.migrate_legacy_version_source(
        store, publication.db.DEFAULT_TENANT_ID, DOCUMENT, VERSION,
        "rapor.pdf", expected_sha256=DIGEST, max_bytes=1024)

    assert first.created is True
    assert second.created is False


def test_bound_source_migration_rechecks_the_candidate_under_publish_lock(
        store, monkeypatch):
    calls = []

    @contextmanager
    def publish_lock(conn, filename):
        calls.append(("lock", conn, filename))
        yield

    monkeypatch.setattr(publication.db, "document_publish_lock", publish_lock)
    monkeypatch.setattr(publication.db, "get_document", lambda conn, wanted: {
        "id": wanted,
        "filename": "report.pdf",
        "candidate_id": str(VERSION),
        "content_sha256": DIGEST,
    })
    monkeypatch.setattr(
        publication, "migrate_legacy_version_source",
        lambda *args, **kwargs: calls.append(("migrate", args, kwargs)) or
        "proof")
    conn = object()

    assert publication.ensure_bound_version_source(
        conn, store, TENANT, DOCUMENT, VERSION, "report.pdf",
        expected_sha256=DIGEST, max_bytes=1024) == "proof"
    assert calls[0] == ("lock", conn, "report.pdf")
    assert calls[1][0] == "migrate"


@pytest.mark.parametrize(
    "changed",
    [None,
     {"filename": "other.pdf", "candidate_id": str(VERSION),
      "content_sha256": DIGEST},
     {"filename": "report.pdf", "candidate_id": str(TENANT_B),
      "content_sha256": DIGEST},
     {"filename": "report.pdf", "candidate_id": str(VERSION),
      "content_sha256": "0" * 64}],
)
def test_bound_source_migration_refuses_stale_database_identity(
        store, monkeypatch, changed):
    @contextmanager
    def publish_lock(_conn, _filename):
        yield

    monkeypatch.setattr(publication.db, "document_publish_lock", publish_lock)
    monkeypatch.setattr(publication.db, "get_document",
                        lambda _conn, _wanted: changed)
    monkeypatch.setattr(
        publication, "migrate_legacy_version_source",
        lambda *_args, **_kwargs: pytest.fail("stale binding was migrated"))

    with pytest.raises(publication.VersionSourceCorrupt):
        publication.ensure_bound_version_source(
            object(), store, TENANT, DOCUMENT, VERSION, "report.pdf",
            expected_sha256=DIGEST, max_bytes=1024)


def test_missing_legacy_source_is_a_closed_refusal(store):
    with pytest.raises(publication.VersionSourceMissing):
        publication.migrate_legacy_version_source(
            store, publication.db.DEFAULT_TENANT_ID, DOCUMENT, VERSION,
            "missing.pdf", expected_sha256=DIGEST, max_bytes=1024)


@pytest.mark.parametrize("maximum", [len(BODY) - 1, 1024])
def test_legacy_source_must_match_its_bound_and_database_digest(
        store, maximum):
    (store / "rapor.pdf").write_bytes(BODY)
    expected = DIGEST if maximum < len(BODY) else "0" * 64

    with pytest.raises(publication.VersionSourceCorrupt):
        publication.migrate_legacy_version_source(
            store, publication.db.DEFAULT_TENANT_ID, DOCUMENT, VERSION,
            "rapor.pdf", expected_sha256=expected, max_bytes=maximum)


def test_a_legacy_source_symlink_is_never_followed(store, tmp_path):
    canary = tmp_path / "canary.pdf"
    canary.write_bytes(BODY)
    linked = store / "rapor.pdf"
    try:
        os.symlink(canary, linked)
    except OSError:
        pytest.skip("file symlink could not be created on this host")

    with pytest.raises(publication.VersionSourceCorrupt):
        publication.migrate_legacy_version_source(
            store, publication.db.DEFAULT_TENANT_ID, DOCUMENT, VERSION,
            "rapor.pdf", expected_sha256=DIGEST, max_bytes=1024)

    assert canary.read_bytes() == BODY


def test_a_legacy_file_swap_between_evidence_and_handle_is_refused(
        store, monkeypatch):
    (store / "rapor.pdf").write_bytes(BODY)
    monkeypatch.setattr(publication._object_transport, "handle_identity",
                        lambda _handle: "different-object")

    with pytest.raises(publication.VersionSourceCorrupt,
                       match="okuma sirasinda degisti"):
        publication.migrate_legacy_version_source(
            store, publication.db.DEFAULT_TENANT_ID, DOCUMENT, VERSION,
            "rapor.pdf", expected_sha256=DIGEST, max_bytes=1024)

    assert not publication.version_source_path(
        store, publication.db.DEFAULT_TENANT_ID,
        DOCUMENT, VERSION).exists()
