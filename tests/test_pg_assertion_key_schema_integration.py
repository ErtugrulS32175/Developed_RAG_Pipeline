"""Real-PostgreSQL migration refusals for assertion key lifecycle state."""
import os
import uuid

import pytest

from pipeline.control import db as control_db
from pipeline.index import db as index_db


DSN = os.getenv("RAGTEST_CONTROL_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(not DSN, reason="control PostgreSQL is absent")


def _database_dsn(base, database):
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    values = conninfo_to_dict(base)
    values["dbname"] = database
    return make_conninfo(**values)


@pytest.fixture
def isolated_database():
    import psycopg
    from psycopg import sql

    name = "ragtest_key_schema_" + uuid.uuid4().hex[:12]
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(name)))
        connection = psycopg.connect(_database_dsn(DSN, name))
        yield connection
    finally:
        if connection is not None:
            connection.close()
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(name)))
        admin.close()


def test_control_upgrade_refuses_unknown_legacy_verify_window_atomically(
        isolated_database):
    import psycopg

    connection = isolated_database
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA public")
        cursor.execute("CREATE SCHEMA rag_control")
        cursor.execute(
            "CREATE TABLE rag_control.control_service_account_assertion_keys ("
            "key_version integer PRIMARY KEY, secret bytea NOT NULL, "
            "state text NOT NULL, not_before timestamptz NOT NULL, "
            "verify_until timestamptz, created_at timestamptz NOT NULL "
            "DEFAULT statement_timestamp())")
        cursor.execute(
            "INSERT INTO rag_control.control_service_account_assertion_keys "
            "VALUES (1,%s,'verify_only',statement_timestamp()-interval '1 day',"
            "statement_timestamp()+interval '1 day')", (b"k" * 32,))
    connection.commit()
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState,
                       match="overlap_unknown"):
        control_db.init_schema(connection)
    connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='rag_control' "
            "AND table_name='control_service_account_assertion_keys' "
            "AND column_name='verify_started_at'")
        assert cursor.fetchone()[0] == 0
        cursor.execute(
            "SELECT to_regclass('rag_control.control_schema_state')")
        assert cursor.fetchone()[0] is None
    connection.rollback()


def test_data_upgrade_refuses_unknown_legacy_verify_window_atomically(
        isolated_database, monkeypatch):
    import psycopg

    connection = isolated_database
    monkeypatch.setenv("RAG_DB_CONTEXT_SECRET", "c" * 32)
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION vector WITH SCHEMA public")
        cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA public")
        cursor.execute(
            "CREATE TABLE rag_service_account_assertion_keys ("
            "key_version integer PRIMARY KEY, secret bytea NOT NULL, "
            "state text NOT NULL, not_before timestamptz NOT NULL, "
            "verify_until timestamptz, created_at timestamptz NOT NULL "
            "DEFAULT statement_timestamp())")
        cursor.execute(
            "INSERT INTO rag_service_account_assertion_keys VALUES "
            "(1,%s,'verify_only',statement_timestamp()-interval '1 day',"
            "statement_timestamp()+interval '1 day')", (b"k" * 32,))
    connection.commit()
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState,
                       match="overlap_unknown"):
        index_db.init_schema(connection)
    connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema=current_schema() "
            "AND table_name='rag_service_account_assertion_keys' "
            "AND column_name='verify_started_at'")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT to_regclass('rag_schema_state')")
        assert cursor.fetchone()[0] is None
    connection.rollback()


@pytest.fixture
def default_privilege_roles(isolated_database):
    """The bootstrap script leaves ALTER DEFAULT PRIVILEGES in force for
    every later migration, so a table the NEXT migration creates arrives
    with runtime DML unless the schema revokes it itself. This mirrors
    that state: a migrator role with default privileges granting
    rag_runtime DML on new tables, before the schema is applied."""
    from psycopg import sql

    migrator = "ragtest_key_migrator_" + uuid.uuid4().hex[:12]
    runtime = "rag_runtime"
    password = "eval-migration-only"
    connection = isolated_database
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION vector WITH SCHEMA public")
        cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA public")
        cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(migrator), sql.Literal(password)))
        cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(runtime), sql.Literal(password)))
        cursor.execute(sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(
            sql.Identifier(migrator)))
        cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(
            sql.Identifier(runtime)))
        cursor.execute(sql.SQL(
            "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public "
            "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}").format(
                sql.Identifier(migrator), sql.Identifier(runtime)))
    connection.commit()
    try:
        yield migrator, runtime, password
    finally:
        connection.rollback()
        with connection.cursor() as cursor:
            for role in (runtime, migrator):
                cursor.execute(sql.SQL("DROP OWNED BY {}").format(
                    sql.Identifier(role)))
                cursor.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        connection.commit()


def test_migration_revokes_the_rotation_ledger_from_a_default_privileged_runtime(
        isolated_database, default_privilege_roles, monkeypatch):
    """Forward upgrade after bootstrap: the ledger table is created while
    default privileges hand rag_runtime DML on every new table. The
    schema must take that back itself, exactly as it does for the key
    and secret tables, and the readiness check must then pass."""
    import psycopg

    migrator, runtime, password = default_privilege_roles
    monkeypatch.setenv("RAG_DB_CONTEXT_SECRET", "c" * 32)
    database = isolated_database.info.dbname
    as_migrator = psycopg.connect(
        _database_dsn(DSN, database), user=migrator, password=password)
    try:
        index_db.init_schema(as_migrator)
        as_migrator.commit()
    finally:
        as_migrator.close()
    as_runtime = psycopg.connect(
        _database_dsn(DSN, database), user=runtime, password=password)
    try:
        with as_runtime.cursor() as cursor:
            # the default privileges really were in force: an ordinary
            # table did receive runtime DML from the migration
            cursor.execute(
                "SELECT has_table_privilege(current_user, 'documents', "
                "'SELECT,INSERT,UPDATE,DELETE')")
            assert cursor.fetchone() == (True,)
            for table in ("rag_service_account_assertion_keys",
                          "rag_service_account_assertion_rotations",
                          "rag_context_secrets",
                          "org_identity_tenant_bindings"):
                cursor.execute(
                    "SELECT has_table_privilege(current_user, %s, "
                    "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,"
                    "TRIGGER')", (table,))
                assert cursor.fetchone() == (False,), table
        as_runtime.rollback()
        assert index_db.runtime_role_is_safe(as_runtime) is True
    finally:
        as_runtime.close()


ROTATION_COLUMNS = (
    "(rotation_id,previous_key_version,target_key_version,"
    "target_key_fingerprint,verify_started_at,verify_until,phase")
KEY_COLUMNS = "(key_version,secret,state,not_before,rotation_id)"


@pytest.mark.parametrize(("side", "key_table", "rotation_table"), [
    ("data", "rag_service_account_assertion_keys",
     "rag_service_account_assertion_rotations"),
    ("control", "rag_control.control_service_account_assertion_keys",
     "rag_control.control_service_account_assertion_rotations"),
])
def test_a_second_rotation_can_follow_a_retired_tombstone(
        isolated_database, monkeypatch, side, key_table, rotation_table):
    """A ledger that can hold ONE rotation is not a ledger. Measured before
    this test existed: after v1->v2 completed, no v2->v3 could be recorded
    in any form, because a retired tombstone insisted on keeping its
    target bound while the next rotation needed the same key as its
    previous member. Now the target moves on once the tombstone is
    retired; the tombstone keeps its versions, its previous member, its
    immutability and its refusal of foreign members."""
    import psycopg

    connection = isolated_database
    monkeypatch.setenv("RAG_DB_CONTEXT_SECRET", "c" * 32)
    (index_db if side == "data" else control_db).init_schema(connection)
    connection.commit()
    first, second = uuid.uuid4(), uuid.uuid4()
    secret, fingerprint = b"k" * 32, b"f" * 32
    window = ("statement_timestamp(),"
              "statement_timestamp()+interval '300 seconds'")

    def commit(statements):
        with connection.cursor() as cursor:
            for text, params in statements:
                cursor.execute(text, params)
        connection.commit()

    def refused(statements, error=psycopg.errors.CheckViolation,
                match="rotation_keys_unbound"):
        with pytest.raises(error, match=match):
            commit(statements)
        connection.rollback()

    def ledger_rotation(rotation, previous, target):
        return (f"INSERT INTO {rotation_table} {ROTATION_COLUMNS}) "
                f"VALUES (%s,{previous},{target},%s,{window},'staged')",
                (rotation, fingerprint))

    def key(version, state, rotation):
        return (f"INSERT INTO {key_table} {KEY_COLUMNS} "
                f"VALUES ({version},%s,'{state}',statement_timestamp(),%s)",
                (secret, rotation))

    commit([ledger_rotation(first, 1, 2), key(1, "active", first),
            key(2, "staged", first)])
    commit([(f"UPDATE {key_table} SET state='verify_only', "
             f"verify_started_at=statement_timestamp(), "
             f"verify_until=statement_timestamp()+interval '300 seconds' "
             f"WHERE key_version=1", ()),
            (f"UPDATE {key_table} SET state='active' WHERE key_version=2", ()),
            (f"UPDATE {rotation_table} SET phase='activated' "
             f"WHERE rotation_id=%s", (first,))])
    commit([(f"UPDATE {key_table} SET state='retired', verify_started_at=NULL, "
             f"verify_until=NULL WHERE key_version=1", ()),
            (f"UPDATE {rotation_table} SET phase='completed', "
             f"completed_at=statement_timestamp() WHERE rotation_id=%s",
             (first,))])
    successor = [ledger_rotation(second, 2, 3), key(3, "staged", second),
                 (f"UPDATE {key_table} SET rotation_id=%s WHERE key_version=2",
                  (second,))]
    # a completed rotation still owns both members: the successor waits
    refused(successor)
    commit([(f"UPDATE {rotation_table} SET phase='retired', "
             f"retired_at=statement_timestamp() WHERE rotation_id=%s",
             (first,))])
    commit(successor)
    # the tombstone keeps refusing foreign members and any edit
    refused([key(99, "retired", first)])
    refused([(f"UPDATE {rotation_table} SET previous_key_version=5 "
              f"WHERE rotation_id=%s", (first,))],
            error=psycopg.errors.ObjectNotInPrerequisiteState,
            match="transition_refused")
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT rotation_id, array_agg(key_version ORDER BY key_version) "
            f"FROM {key_table} GROUP BY rotation_id")
        members = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.execute(
            f"SELECT phase FROM {rotation_table} ORDER BY created_at")
        phases = [row[0] for row in cursor.fetchall()]
    connection.rollback()
    assert members == {first: [1], second: [2, 3]}
    assert phases == ["retired", "staged"]


def test_init_refuses_pgcrypto_preinstalled_outside_public_without_receipt(
        isolated_database):
    import psycopg

    connection = isolated_database
    with connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA extension_attacker")
        cursor.execute("CREATE EXTENSION pgcrypto WITH SCHEMA extension_attacker")
    connection.commit()
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState,
                       match="pgcrypto_namespace_refused"):
        control_db.init_schema(connection)
    connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT to_regclass('rag_control.control_schema_state')")
        assert cursor.fetchone()[0] is None
    connection.rollback()
