"""Content-free PostgreSQL backup, verification, and guarded restore CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import psycopg


SNAPSHOT_VERSION = 1
MAX_MANIFEST_BYTES = 16_384
_MANIFEST_FIELDS = {
    "snapshot_version", "archive_name", "size", "sha256",
}


class SnapshotError(RuntimeError):
    """Closed operational failure; raw tool output is never propagated."""


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _digest(path: Path) -> tuple[int, str]:
    if not _regular(path):
        raise SnapshotError("archive_not_regular")
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size == 0:
        raise SnapshotError("archive_empty")
    return size, digest.hexdigest()


def _connection_env(name: str) -> tuple[dict[str, str], str, str]:
    raw = os.environ.get(name, "")
    try:
        parsed = urlparse(raw)
        port = parsed.port or 5432
    except ValueError as error:
        raise SnapshotError("database_configuration_invalid") from error
    if (parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname
            or not parsed.username or not parsed.path.strip("/")):
        raise SnapshotError("database_configuration_invalid")
    database = unquote(parsed.path.strip("/"))
    env = os.environ.copy()
    # The child needs decomposed libpq fields, not either secret-bearing source
    # variable. This also prevents an unrelated restore DSN reaching pg_dump.
    env.pop("PG_DSN", None)
    env.pop("PG_RESTORE_DSN", None)
    env.update({
        "PGHOST": parsed.hostname,
        "PGPORT": str(port),
        "PGUSER": unquote(parsed.username),
        "PGDATABASE": database,
    })
    if parsed.password is not None:
        env["PGPASSWORD"] = unquote(parsed.password)
    return env, database, raw


def _manifest_path(archive: Path) -> Path:
    return archive.with_name(archive.name + ".manifest.json")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def verify(archive, manifest=None) -> dict[str, object]:
    archive = Path(archive)
    manifest = Path(manifest) if manifest else _manifest_path(archive)
    if not _regular(manifest) or manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise SnapshotError("manifest_not_regular")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotError("manifest_invalid") from error
    if type(record) is not dict or set(record) != _MANIFEST_FIELDS:
        raise SnapshotError("manifest_invalid")
    if (type(record["snapshot_version"]) is not int
            or record["snapshot_version"] != SNAPSHOT_VERSION
            or type(record["archive_name"]) is not str
            or record["archive_name"] != archive.name
            or type(record["size"]) is not int
            or type(record["sha256"]) is not str
            or len(record["sha256"]) != 64):
        raise SnapshotError("manifest_invalid")
    size, digest = _digest(archive)
    if size != record["size"] or digest != record["sha256"]:
        raise SnapshotError("archive_digest_mismatch")
    return record


def backup(output) -> dict[str, object]:
    output = Path(output)
    manifest = _manifest_path(output)
    if output.exists() or manifest.exists() or not output.parent.is_dir():
        raise SnapshotError("output_not_available")
    env, _database, _raw_dsn = _connection_env("PG_DSN")
    token = secrets.token_hex(12)
    archive_tmp = output.parent / f".{output.name}.{token}.tmp"
    manifest_tmp = output.parent / f".{manifest.name}.{token}.tmp"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(archive_tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            completed = subprocess.run(
                ["pg_dump", "--format=custom", "--no-owner", "--no-acl"],
                stdout=handle, stderr=subprocess.PIPE, env=env,
                timeout=3600, check=False)
            handle.flush()
            os.fsync(handle.fileno())
        if completed.returncode != 0:
            raise SnapshotError("pg_dump_failed")
        size, digest = _digest(archive_tmp)
        record = {
            "snapshot_version": SNAPSHOT_VERSION,
            "archive_name": output.name,
            "size": size,
            "sha256": digest,
        }
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
        _write_exclusive(manifest_tmp, payload)
        os.replace(archive_tmp, output)
        try:
            os.replace(manifest_tmp, manifest)
        except OSError:
            output.unlink(missing_ok=True)
            raise
        return verify(output, manifest)
    except (OSError, subprocess.SubprocessError) as error:
        raise SnapshotError("backup_failed") from error
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        archive_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _database_is_empty(dsn: str) -> bool:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "WHERE c.relkind IN ('r','p') "
            "AND n.nspname NOT IN ('pg_catalog','information_schema') "
            "AND n.nspname NOT LIKE 'pg_toast%'")
        return cur.fetchone()[0] == 0


def restore(archive, manifest=None, confirmation="") -> dict[str, object]:
    if confirmation != "EMPTY_DATABASE":
        raise SnapshotError("restore_confirmation_required")
    archive = Path(archive)
    record = verify(archive, manifest)
    env, database, raw_dsn = _connection_env("PG_RESTORE_DSN")
    try:
        if not _database_is_empty(raw_dsn):
            raise SnapshotError("restore_database_not_empty")
        with archive.open("rb") as handle:
            completed = subprocess.run(
                ["pg_restore", "--exit-on-error", "--no-owner", "--no-acl",
                 "--dbname", database],
                stdin=handle, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                env=env, timeout=3600, check=False)
    except (OSError, psycopg.Error, subprocess.SubprocessError) as error:
        raise SnapshotError("restore_failed") from error
    if completed.returncode != 0:
        raise SnapshotError("pg_restore_failed")
    return {"snapshot_version": SNAPSHOT_VERSION, "status": "restored",
            "sha256": record["sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup_cmd = commands.add_parser("backup")
    backup_cmd.add_argument("--output", required=True)
    verify_cmd = commands.add_parser("verify")
    verify_cmd.add_argument("--archive", required=True)
    verify_cmd.add_argument("--manifest")
    restore_cmd = commands.add_parser("restore")
    restore_cmd.add_argument("--archive", required=True)
    restore_cmd.add_argument("--manifest")
    restore_cmd.add_argument("--confirm", required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "backup":
            result = backup(args.output)
            result = {"snapshot_version": SNAPSHOT_VERSION,
                      "status": "created", "sha256": result["sha256"]}
        elif args.command == "verify":
            result = verify(args.archive, args.manifest)
            result = {"snapshot_version": SNAPSHOT_VERSION,
                      "status": "verified", "sha256": result["sha256"]}
        else:
            result = restore(args.archive, args.manifest, args.confirm)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except SnapshotError as error:
        print(json.dumps({"snapshot_version": SNAPSHOT_VERSION,
                          "status": "failed", "code": str(error)},
                         sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
