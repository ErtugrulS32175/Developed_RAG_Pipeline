"""The immutable source object reaches ingest without a mutable flat path."""
import hashlib
import uuid
from types import SimpleNamespace

import pytest

from pipeline.index import ingest, publication


TENANT = uuid.UUID("10000000-0000-4000-8000-000000000001")
DOCUMENT = uuid.UUID("20000000-0000-4000-8000-000000000002")
VERSION = uuid.UUID("30000000-0000-4000-8000-000000000003")
OTHER_VERSION = uuid.UUID("40000000-0000-4000-8000-000000000004")
BODY = b"version-bound-ingest"
DIGEST = hashlib.sha256(BODY).hexdigest()


def _attempt(document=DOCUMENT, version=VERSION, digest=DIGEST):
    return SimpleNamespace(
        document_id=str(document), candidate_id=str(version),
        candidate_sha=digest)


def test_verified_bytes_keep_the_canonical_name_and_extension(tmp_path,
                                                               monkeypatch):
    observed = {}

    def snapshot(body, name):
        observed["body"] = body
        observed["name"] = name
        directory = tmp_path / "private"
        directory.mkdir()
        path = directory / name
        path.write_bytes(body)
        return path, directory

    def core(path, attempt):
        observed["path_name"] = path.name
        observed["attempt"] = attempt
        assert path.read_bytes() == BODY
        return "done", None

    monkeypatch.setattr(ingest, "_snapshot_bytes", snapshot)
    monkeypatch.setattr(ingest, "ingest_attempt", core)
    attempt = _attempt()

    result = ingest.ingest_verified_source(
        BODY, "Yillik Rapor.PDF", attempt, expected_sha256=DIGEST)

    assert result == ("done", None)
    assert observed == {
        "body": BODY,
        "name": "Yillik Rapor.PDF",
        "path_name": "Yillik Rapor.PDF",
        "attempt": attempt,
    }
    assert not (tmp_path / "private").exists()


def test_version_ingest_reads_only_the_uuid_bound_object(monkeypatch):
    calls = []
    attempt = _attempt()

    def read(root, tenant, document, version, *, expected_sha256, max_bytes):
        calls.append((root, tenant, document, version, expected_sha256,
                      max_bytes))
        return BODY

    monkeypatch.setattr(publication, "read_version_source", read)
    monkeypatch.setattr(
        ingest, "ingest_verified_source",
        lambda body, name, bound, *, expected_sha256:
        (body, name, bound, expected_sha256))

    result = ingest.ingest_version_source(
        "object-root", TENANT, DOCUMENT, VERSION, "rapor.pdf", attempt,
        expected_sha256=DIGEST, max_bytes=1234)

    assert calls == [
        ("object-root", TENANT, DOCUMENT, VERSION, DIGEST, 1234)]
    assert result == (BODY, "rapor.pdf", attempt, DIGEST)


def test_a_version_cannot_be_ingested_under_another_document_attempt(
        monkeypatch):
    abandoned = []
    monkeypatch.setattr(ingest, "abandon_attempt",
                        lambda attempt, note: abandoned.append(note))
    monkeypatch.setattr(
        publication, "read_version_source",
        lambda *args, **kwargs: pytest.fail("unbound object was read"))
    other = uuid.UUID("40000000-0000-4000-8000-000000000004")

    with pytest.raises(RuntimeError, match="baska bir belge"):
        ingest.ingest_version_source(
            "object-root", TENANT, DOCUMENT, VERSION, "rapor.pdf",
            _attempt(other), expected_sha256=DIGEST)
    assert abandoned == ["RuntimeError"]


def test_same_digest_version_cannot_use_another_versions_attempt(monkeypatch):
    abandoned = []
    monkeypatch.setattr(ingest, "abandon_attempt",
                        lambda attempt, note: abandoned.append(note))
    monkeypatch.setattr(
        publication, "read_version_source",
        lambda *args, **kwargs: pytest.fail("wrong version object was read"))

    with pytest.raises(RuntimeError, match="baska bir aday"):
        ingest.ingest_version_source(
            "object-root", TENANT, DOCUMENT, VERSION, "rapor.pdf",
            _attempt(version=OTHER_VERSION), expected_sha256=DIGEST)
    assert abandoned == ["RuntimeError"]


@pytest.mark.parametrize(
    ("body", "name", "digest"),
    [(BODY + b"x", "rapor.pdf", DIGEST),
     (BODY, "../rapor.pdf", DIGEST),
     (BODY, "rapor", DIGEST),
     (bytearray(BODY), "rapor.pdf", DIGEST)],
)
def test_unproven_bytes_or_parser_names_never_reach_the_ingest_core(
        monkeypatch, body, name, digest):
    abandoned = []
    monkeypatch.setattr(ingest, "abandon_attempt",
                        lambda attempt, note: abandoned.append(note))
    monkeypatch.setattr(
        ingest, "ingest_attempt",
        lambda *args: pytest.fail("unproven source reached ingest"))

    with pytest.raises(RuntimeError):
        ingest.ingest_verified_source(
            body, name, _attempt(), expected_sha256=digest)
    assert abandoned == ["RuntimeError"]


def test_attempt_digest_must_bind_the_verified_version(monkeypatch):
    abandoned = []
    monkeypatch.setattr(ingest, "abandon_attempt",
                        lambda attempt, note: abandoned.append(note))
    monkeypatch.setattr(
        ingest, "ingest_attempt",
        lambda *args: pytest.fail("wrong candidate reached ingest"))

    with pytest.raises(RuntimeError, match="adayina bagli degil"):
        ingest.ingest_verified_source(
            BODY, "rapor.pdf", _attempt(digest="0" * 64),
            expected_sha256=DIGEST)
    assert abandoned == ["RuntimeError"]


def test_a_failed_immutable_read_closes_the_held_attempt(monkeypatch):
    abandoned = []
    monkeypatch.setattr(
        publication, "read_version_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            publication.VersionSourceCorrupt("closed refusal")))
    monkeypatch.setattr(ingest, "abandon_attempt",
                        lambda attempt, note: abandoned.append(note))

    with pytest.raises(publication.VersionSourceCorrupt):
        ingest.ingest_version_source(
            "object-root", TENANT, DOCUMENT, VERSION, "rapor.pdf",
            _attempt(), expected_sha256=DIGEST)

    assert abandoned == ["VersionSourceCorrupt"]


def test_the_legacy_path_api_keeps_its_signature():
    import inspect

    assert tuple(inspect.signature(ingest.main).parameters) == (
        "path", "expected_candidate", "attempt")
