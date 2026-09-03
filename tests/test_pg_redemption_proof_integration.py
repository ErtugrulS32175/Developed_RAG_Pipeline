"""Real two-database proof for the service-account assertion foundation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import hmac
import os
import struct
import uuid

import pytest

from pipeline.control import db as control_db
from pipeline.index import db as index_db
from pipeline.service_account_assertions import ServiceAccountAssertion


DSN = os.getenv("RAGTEST_CONTROL_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(not DSN, reason="control PostgreSQL is absent")
KEY = b"assertion-proof-key-material-32b"
TENANT = uuid.UUID("10000000-0000-0000-0000-000000000001")
ACTOR_DIGEST = b"a" * 32
ACTOR = uuid.UUID("20000000-0000-0000-0000-000000000002")
APPROVAL = uuid.UUID("50000000-0000-0000-0000-000000000005")
ACCOUNT = uuid.UUID("30000000-0000-0000-0000-000000000003")
BUSINESS_APPROVAL = uuid.UUID("50000000-0000-0000-0000-000000000006")
BUSINESS_ACCOUNT = uuid.UUID("30000000-0000-0000-0000-000000000004")
APPROVAL_REVISION = 3
CREDENTIAL_DIGEST = b"c" * 32
NONCE = b"n" * 16


def _database_dsn(base, database):
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    values = conninfo_to_dict(base)
    values["dbname"] = database
    return make_conninfo(**values)


@pytest.fixture(scope="module")
def proof_databases():
    import psycopg
    from psycopg import sql

    suffix = uuid.uuid4().hex[:12]
    data_name = "ragtest_proof_data_" + suffix
    control_name = "ragtest_proof_control_" + suffix
    admin = psycopg.connect(DSN, autocommit=True)
    data = control = None
    try:
        with admin.cursor() as cursor:
            for name in (data_name, control_name):
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(name)))
        data = psycopg.connect(_database_dsn(DSN, data_name))
        control = psycopg.connect(_database_dsn(DSN, control_name))
        index_db.init_schema(data)
        control_db.init_schema(control)
        assert data.info.dbname != control.info.dbname
        with data.cursor() as cursor:
            cursor.execute(
                "INSERT INTO org_tenants (id,name,policy_epoch) "
                "VALUES (%s,'proof tenant',9)", (TENANT,))
            cursor.execute(
                "INSERT INTO org_identities (id,issuer,subject,state) "
                "VALUES (%s,'proof-issuer','proof-subject','active')",
                (ACTOR,))
            cursor.execute(
                "INSERT INTO org_positions "
                "(id,tenant_id,title,kind,can_monitor_descendants,"
                "protected_from_monitoring) "
                "VALUES (%s,%s,'root','root',true,true)", (ACTOR, TENANT))
            cursor.execute(
                "INSERT INTO org_memberships "
                "(tenant_id,identity_id,position_id,display_label,"
                "app_role,state) VALUES (%s,%s,%s,'proof admin','admin',"
                "'active')", (TENANT, ACTOR, ACTOR))
            cursor.execute(
                "INSERT INTO org_architects (tenant_id,identity_id,active) "
                "VALUES (%s,%s,true)", (TENANT, ACTOR))
            cursor.execute(
                "INSERT INTO rag_service_account_assertion_keys "
                "(key_version, secret, state, not_before) "
                "VALUES (7, %s, 'active', statement_timestamp() - "
                "interval '1 minute')", (KEY,))
        data.commit()
        with control.cursor() as cursor:
            cursor.execute(
                "INSERT INTO rag_control."
                "control_service_account_assertion_keys "
                "(key_version, secret, state, not_before) "
                "VALUES (7, %s, 'active', statement_timestamp() - "
                "interval '1 minute')", (KEY,))
            cursor.execute(
                "INSERT INTO rag_control.control_regions "
                "VALUES ('proof-region','active',1)")
            cursor.execute(
                "INSERT INTO rag_control.control_tenants "
                "(tenant_id,lifecycle,deployment_profile,policy_revision) "
                "VALUES (%s,'active','enterprise',9)", (TENANT,))
            cursor.execute(
                "INSERT INTO rag_control.control_tenant_routes "
                "(tenant_id,route_kind,region_code,connection_ref,state) "
                "VALUES (%s,'shared_rls','proof-region','proof:route',"
                "'active')", (TENANT,))
            cursor.execute(
                "INSERT INTO rag_control.control_tenant_quotas VALUES "
                "(%s,10,2,2,1000,10,1,1,1000,'declared',1)", (TENANT,))
            cursor.execute(
                "INSERT INTO rag_control.control_platform_operators "
                "(role,state) VALUES ('platform_security','active') "
                "RETURNING operator_id")
            operator = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO rag_control.control_service_account_approvals "
                "(approval_id,tenant_id,service_account_id,action,state,"
                "platform_operator_id,reason_code,scopes,account_expires_at,"
                "credential_expires_at,control_policy_revision,expires_at,"
                "request_digest,resulting_fact_digest) VALUES "
                "(%s,%s,%s,'issue','approved',%s,'security_provisioning',"
                "ARRAY['rag.query'],statement_timestamp()+interval '30 days',"
                "statement_timestamp()+interval '7 days',9,"
                "statement_timestamp()+interval '10 minutes',%s,%s)",
                (BUSINESS_APPROVAL, TENANT, BUSINESS_ACCOUNT, operator,
                 b"r" * 32, b"f" * 32))
        control.commit()
        yield data, control
    finally:
        if data is not None:
            data.close()
        if control is not None:
            control.close()
        with admin.cursor() as cursor:
            for name in (data_name, control_name):
                cursor.execute(sql.SQL(
                    "DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)))
        admin.close()


def _expected_payload(issued, nonce=NONCE):
    purpose = b"approval_redeem_issue"
    return (
        b"ragtest.service-account.assertion.v1"
        + struct.pack("!I", len(purpose)) + purpose
        + struct.pack("!I", 7) + TENANT.bytes + ACTOR_DIGEST
        + struct.pack("!q", 9) + bytes((1,)) + APPROVAL.bytes
        + bytes((1,)) + struct.pack("!q", APPROVAL_REVISION)
        + bytes((1,)) + ACCOUNT.bytes
        + bytes((1,)) + CREDENTIAL_DIGEST + bytes((0,))
        + struct.pack("!q", issued) + struct.pack("!q", issued + 30)
        + nonce)


def _payload(connection, qualified, issued, nonce=NONCE):
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT {qualified}("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("approval_redeem_issue", 7, TENANT, ACTOR_DIGEST, 9,
             APPROVAL, APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None,
             issued, issued + 30, nonce))
        return bytes(cursor.fetchone()[0])


def _shape(purpose, *, approval=None, revision=None, account=None,
           credential=None, limit=None):
    return (
        purpose, 7, TENANT, ACTOR_DIGEST, 9, approval, revision, account,
        credential, limit, 2_000_000_000, 2_000_000_030, NONCE)


def _shape_payload(connection, qualified, values):
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT {qualified}("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", values)
        return bytes(cursor.fetchone()[0])


def test_two_physical_databases_share_the_pinned_payload_and_mac(
        proof_databases):
    data, control = proof_databases
    issued = 2_000_000_000
    expected = _expected_payload(issued)
    data_payload = _payload(
        data, "rag_service_account_assertion_payload", issued)
    control_payload = _payload(
        control,
        "rag_control.control_service_account_assertion_payload", issued)
    assert data_payload == control_payload == expected
    expected_mac = hmac.new(KEY, expected, hashlib.sha256).digest()
    for connection in (data, control):
        with connection.cursor() as cursor:
            cursor.execute("SELECT public.hmac(%s,%s,'sha256')",
                           (expected, KEY))
            assert bytes(cursor.fetchone()[0]) == expected_mac
        connection.rollback()


@pytest.mark.parametrize("values", [
    _shape("approval_list", limit=10),
    _shape("approval_get", approval=APPROVAL, revision=APPROVAL_REVISION,
           account=ACCOUNT),
    _shape("approval_redeem_issue", approval=APPROVAL,
           revision=APPROVAL_REVISION, account=ACCOUNT,
           credential=CREDENTIAL_DIGEST),
    _shape("approval_redeem_rotate", approval=APPROVAL,
           revision=APPROVAL_REVISION, account=ACCOUNT,
           credential=CREDENTIAL_DIGEST),
])
def test_all_valid_operation_shapes_have_cross_database_parity(
        proof_databases, values):
    data, control = proof_databases
    assert _shape_payload(
        data, "rag_service_account_assertion_payload", values
    ) == _shape_payload(
        control,
        "rag_control.control_service_account_assertion_payload", values)
    data.rollback()
    control.rollback()


@pytest.mark.parametrize("values", [
    _shape("unknown", limit=10),
    _shape("approval_list", approval=APPROVAL, limit=10),
    _shape("approval_list", limit=0),
    _shape("approval_get", approval=APPROVAL, account=ACCOUNT),
    _shape("approval_get", approval=APPROVAL, revision=APPROVAL_REVISION,
           account=ACCOUNT, credential=CREDENTIAL_DIGEST),
    _shape("approval_redeem_issue", approval=APPROVAL,
           revision=APPROVAL_REVISION, account=ACCOUNT),
    _shape("approval_redeem_rotate", approval=APPROVAL,
           revision=APPROVAL_REVISION, account=ACCOUNT,
           credential=CREDENTIAL_DIGEST, limit=1),
])
def test_invalid_operation_shapes_fail_identically_in_both_databases(
        proof_databases, values):
    import psycopg

    data, control = proof_databases
    sqlstates = []
    for connection, qualified in (
            (data, "rag_service_account_assertion_payload"),
            (control,
             "rag_control.control_service_account_assertion_payload")):
        with pytest.raises(psycopg.errors.InvalidParameterValue) as refused:
            _shape_payload(connection, qualified, values)
        sqlstates.append(refused.value.sqlstate)
        connection.rollback()
    assert sqlstates == ["22023", "22023"]


def test_data_database_mints_only_the_closed_tenant_authority(
        proof_databases):
    data, _control = proof_databases
    index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
    data.commit()
    with data.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM rag_mint_service_account_assertion("
            "%s,%s,%s,%s,%s,%s,%s,%s)",
            (ACTOR, 9, "approval_redeem_issue", APPROVAL,
             APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None))
        row = cursor.fetchone()
    assert row is not None
    (version, purpose, key_version, tenant, actor_digest, epoch,
     approval, approval_revision, account, credential, limit,
     issued, expires, nonce, mac) = row
    assert (version, purpose, key_version, tenant, epoch, approval,
            approval_revision, account, bytes(credential), limit,
            expires - issued) == (
                1, "approval_redeem_issue", 7, TENANT, 9, APPROVAL,
                APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None, 30)
    expected_actor = hmac.new(
        KEY, b"ragtest.service-account.actor.v1" + TENANT.bytes + ACTOR.bytes,
        hashlib.sha256).digest()
    assert bytes(actor_digest) == expected_actor
    expected_payload = (
        b"ragtest.service-account.assertion.v1"
        + struct.pack("!I", len(purpose.encode("ascii")))
        + purpose.encode("ascii") + struct.pack("!I", key_version)
        + TENANT.bytes + expected_actor + struct.pack("!q", epoch)
        + bytes((1,)) + APPROVAL.bytes
        + bytes((1,)) + struct.pack("!q", APPROVAL_REVISION)
        + bytes((1,)) + ACCOUNT.bytes
        + bytes((1,)) + CREDENTIAL_DIGEST + bytes((0,))
        + struct.pack("!q", issued)
        + struct.pack("!q", expires) + bytes(nonce))
    assert bytes(mac) == hmac.new(
        KEY, expected_payload, hashlib.sha256).digest()
    data.rollback()


def test_purpose_specific_mints_drive_asserted_list_get_and_issue(
        proof_databases, monkeypatch):
    data, control = proof_databases
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "a" * 32)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 32)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 32)
    index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
    data.commit()

    listed_proof = index_db.mint_service_account_approval_list_assertion(
        data, actor_id=ACTOR, expected_policy_epoch=9, limit=10)
    data.commit()
    approvals = control_db.list_redeemable_service_account_approvals(
        control, listed_proof)
    control.commit()
    with pytest.raises(control_db.ControlPlaneConflict):
        control_db.list_redeemable_service_account_approvals(
            control, listed_proof)
    control.rollback()
    assert len(approvals) == 1
    approval = approvals[0]
    assert (approval.approval_id, approval.service_account_id) == (
        BUSINESS_APPROVAL, BUSINESS_ACCOUNT)

    index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
    data.commit()
    get_proof = index_db.mint_service_account_approval_get_assertion(
        data, actor_id=ACTOR, expected_policy_epoch=9,
        approval_id=BUSINESS_APPROVAL, approval_revision=1,
        service_account_id=BUSINESS_ACCOUNT)
    data.commit()
    measured = control_db.get_redeemable_service_account_approval(
        control, get_proof)
    control.commit()
    assert measured == approval

    credential = b"z" * 32
    index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
    data.commit()
    redeem_proof = (
        index_db.mint_service_account_approval_redeem_issue_assertion(
            data, actor_id=ACTOR, expected_policy_epoch=9,
            approval_id=BUSINESS_APPROVAL, approval_revision=1,
            service_account_id=BUSINESS_ACCOUNT,
            credential_digest=credential))
    data.commit()
    result = control_db.redeem_service_account_approval(
        control, approval, assertion=redeem_proof)
    control.commit()
    assert (result.service_account_id, result.account_revision,
            result.credential_version) == (BUSINESS_ACCOUNT, 1, 1)
    with pytest.raises(control_db.ControlPlaneConflict):
        control_db.redeem_service_account_approval(
            control, approval, assertion=redeem_proof)
    control.rollback()
    with control.cursor() as cursor:
        cursor.execute(
            "DELETE FROM rag_control."
            "control_service_account_assertion_nonces "
            "WHERE tenant_id = %s", (TENANT,))
    control.commit()


def test_old_assertionless_control_signatures_are_absent(proof_databases):
    _data, control = proof_databases
    signatures = (
        "rag_control.control_list_redeemable_service_account_approvals("
        "uuid,integer)",
        "rag_control.control_redeem_service_account_issue("
        "uuid,uuid,uuid,bigint,bytea,bigint,bytea,bytea,bytea)",
        "rag_control.control_redeem_service_account_rotation("
        "uuid,uuid,uuid,bigint,bytea,bigint,bytea,bytea,bytea)",
    )
    with control.cursor() as cursor:
        for signature in signatures:
            cursor.execute("SELECT to_regprocedure(%s)", (signature,))
            assert cursor.fetchone()[0] is None
    control.rollback()


def test_service_context_cannot_mint_a_human_assertion(proof_databases):
    import psycopg

    data, _control = proof_databases
    index_db.set_tenant_context(data, TENANT, service=True, actor_id=ACTOR)
    data.commit()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with data.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM rag_mint_service_account_assertion("
                "%s,%s,%s,%s,%s,%s,%s,%s)",
                (ACTOR, 9, "approval_redeem_issue", APPROVAL,
                 APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None))
    data.rollback()


def test_suspended_identity_cannot_mint_a_human_assertion(proof_databases):
    import psycopg

    data, _control = proof_databases
    with data.cursor() as cursor:
        cursor.execute(
            "UPDATE org_identities SET state = 'suspended' WHERE id = %s",
            (ACTOR,))
    data.commit()
    try:
        index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
        data.commit()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with data.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM rag_mint_service_account_assertion("
                    "%s,%s,%s,%s,%s,%s,%s,%s)",
                    (ACTOR, 9, "approval_redeem_issue", APPROVAL,
                     APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None))
        data.rollback()
    finally:
        with data.cursor() as cursor:
            cursor.execute(
                "UPDATE org_identities SET state = 'active' WHERE id = %s",
                (ACTOR,))
        data.commit()


def test_temporary_key_table_cannot_shadow_mint_authority(proof_databases):
    data, _control = proof_databases
    attack_key = b"temporary-shadow-assertion-key32b"
    index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
    data.commit()
    with data.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE rag_service_account_assertion_keys ("
            "key_version integer, secret bytea, state text, "
            "not_before timestamptz, verify_until timestamptz, "
            "created_at timestamptz)")
        cursor.execute(
            "INSERT INTO pg_temp.rag_service_account_assertion_keys "
            "VALUES (99,%s,'active',statement_timestamp() - interval "
            "'1 minute',NULL,statement_timestamp())", (attack_key,))
        cursor.execute(
            "SELECT * FROM rag_mint_service_account_assertion("
            "%s,%s,%s,%s,%s,%s,%s,%s)",
            (ACTOR, 9, "approval_redeem_issue", APPROVAL,
             APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None))
        row = cursor.fetchone()
        cursor.execute(
            "SELECT proconfig FROM pg_proc WHERE oid = "
            "'rag_mint_service_account_assertion(uuid,bigint,text,uuid,"
            "bigint,uuid,bytea,integer)'::regprocedure")
        config = cursor.fetchone()[0]
    assert row[2] == 7
    assert config == ["search_path=pg_catalog, public, pg_temp"]
    data.rollback()


@pytest.mark.parametrize(("position", "replacement"), [
    (1, "approval_redeem_rotate"),
    (7, APPROVAL_REVISION + 1),
    (8, uuid.UUID("30000000-0000-0000-0000-000000000004")),
])
def test_business_field_substitution_cannot_reuse_a_mac(
        proof_databases, position, replacement):
    import psycopg

    _data, control = proof_databases
    nonce = uuid.uuid4().bytes
    issued = int(datetime.now(timezone.utc).timestamp())
    payload = _payload(
        control, "rag_control.control_service_account_assertion_payload",
        issued, nonce)
    params = [
        1, "approval_redeem_issue", 7, TENANT, ACTOR_DIGEST, 9, APPROVAL,
        APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None,
        issued, issued + 30, nonce,
        hmac.new(KEY, payload, hashlib.sha256).digest(),
    ]
    params[position] = replacement
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT rag_control."
                "control_consume_service_account_assertion("
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                params)
    control.rollback()


def test_list_limit_substitution_cannot_reuse_a_mac(proof_databases):
    import psycopg

    _data, control = proof_databases
    issued = int(datetime.now(timezone.utc).timestamp())
    nonce = uuid.uuid4().bytes
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT rag_control.control_service_account_assertion_payload("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            ("approval_list", 7, TENANT, ACTOR_DIGEST, 9,
             None, None, None, None, 10, issued, issued + 30, nonce))
        payload = bytes(cursor.fetchone()[0])
    params = (
        1, "approval_list", 7, TENANT, ACTOR_DIGEST, 9,
        None, None, None, None, 100, issued, issued + 30, nonce,
        hmac.new(KEY, payload, hashlib.sha256).digest())
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT rag_control."
                "control_consume_service_account_assertion("
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                params)
    control.rollback()


def test_bad_proof_does_not_burn_the_nonce_and_replay_is_refused(
        proof_databases):
    import psycopg

    _data, control = proof_databases
    issued = int(datetime.now(timezone.utc).timestamp())
    payload = _payload(
        control, "rag_control.control_service_account_assertion_payload",
        issued)
    params = (
        1, "approval_redeem_issue", 7, TENANT, ACTOR_DIGEST, 9, APPROVAL,
        APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None,
        issued, issued + 30, NONCE)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT rag_control."
                "control_consume_service_account_assertion("
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                params + (b"x" * 32,))
    control.rollback()
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM rag_control."
            "control_service_account_assertion_nonces")
        assert cursor.fetchone()[0] == 0
        cursor.execute(
            "SELECT rag_control.control_consume_service_account_assertion("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            params + (hmac.new(KEY, payload, hashlib.sha256).digest(),))
        assert bytes(cursor.fetchone()[0]) == ACTOR_DIGEST
    control.commit()
    with pytest.raises(psycopg.errors.SerializationFailure):
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT rag_control."
                "control_consume_service_account_assertion("
                "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                params + (hmac.new(KEY, payload, hashlib.sha256).digest(),))
    control.rollback()
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM rag_control."
            "control_service_account_assertion_nonces")
        assert cursor.fetchone()[0] == 1
    control.rollback()


def test_concurrent_replay_has_exactly_one_nonce_winner(proof_databases):
    import psycopg

    _data, control = proof_databases
    nonce = b"z" * 16
    issued = int(datetime.now(timezone.utc).timestamp())
    payload = _payload(
        control, "rag_control.control_service_account_assertion_payload",
        issued, nonce)
    control.rollback()
    params = (
        1, "approval_redeem_issue", 7, TENANT, ACTOR_DIGEST, 9, APPROVAL,
        APPROVAL_REVISION, ACCOUNT, CREDENTIAL_DIGEST, None,
        issued, issued + 30, nonce,
        hmac.new(KEY, payload, hashlib.sha256).digest())

    def consume():
        connection = psycopg.connect(_database_dsn(DSN, control.info.dbname))
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT rag_control."
                    "control_consume_service_account_assertion("
                    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    params)
            connection.commit()
            return "accepted"
        except psycopg.Error as exc:
            connection.rollback()
            return exc.sqlstate
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: consume(), range(2)))
    assert sorted(results) == ["40001", "accepted"]
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM rag_control."
            "control_service_account_assertion_nonces WHERE nonce = %s",
            (nonce,))
        assert cursor.fetchone()[0] == 1
    control.rollback()


def test_temporary_secret_table_cannot_forge_the_rls_context(
        proof_databases, monkeypatch):
    data, _control = proof_databases
    attack_key = b"temporary-shadow-attack-key-32b"
    issued = int(datetime.now(timezone.utc).timestamp())
    nonce = uuid.uuid4().hex
    material = f"{TENANT}||0|{issued}|{nonce}".encode("ascii")
    signature = hmac.new(attack_key, material, hashlib.sha256).hexdigest()
    with data.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE rag_context_secrets ("
            "singleton boolean, secret bytea)")
        cursor.execute(
            "INSERT INTO pg_temp.rag_context_secrets VALUES (true, %s)",
            (attack_key,))
        for name, value in (
                ("rag.tenant_id", str(TENANT)), ("rag.actor_id", ""),
                ("rag.service", "0"), ("rag.context_issued_at", str(issued)),
                ("rag.context_nonce", nonce),
                ("rag.context_signature", signature)):
            cursor.execute("SELECT set_config(%s,%s,false)", (name, value))
        cursor.execute("SELECT rag_context_valid()")
        assert cursor.fetchone()[0] is False
        cursor.execute(
            "SELECT proconfig FROM pg_proc WHERE oid = "
            "'rag_context_valid()'::regprocedure")
        config = cursor.fetchone()[0]
        assert config and config[0].startswith("search_path=pg_catalog, ")
        assert config[0].endswith(", pg_temp")
    data.rollback()


def test_key_and_nonce_tables_are_not_public_authorities(proof_databases):
    data, control = proof_databases
    with data.cursor() as cursor:
        cursor.execute(
            "SELECT has_table_privilege('public', "
            "'rag_service_account_assertion_keys', "
            "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'), "
            "has_table_privilege('public', "
            "'rag_service_account_assertion_rotations', "
            "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'), "
            "has_function_privilege('public', "
            "'rag_mint_service_account_assertion(uuid,bigint,text,uuid,"
            "bigint,uuid,bytea,integer)',"
            "'EXECUTE')")
        assert cursor.fetchone() == (False, False, False)
    data.rollback()
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT has_table_privilege('public', "
            "'rag_control.control_service_account_assertion_keys', "
            "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'), "
            "has_table_privilege('public', "
            "'rag_control.control_service_account_assertion_rotations', "
            "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'), "
            "has_table_privilege('public', "
            "'rag_control.control_service_account_assertion_nonces', "
            "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'), "
            "has_function_privilege('public', 'rag_control."
            "control_consume_service_account_assertion(smallint,text,integer,"
            "uuid,bytea,bigint,uuid,bigint,uuid,bytea,integer,bigint,bigint,"
            "bytea,bytea)', "
            "'EXECUTE')")
        assert cursor.fetchone() == (False, False, False, False)
    control.rollback()


@pytest.mark.parametrize(("side", "table"), [
    ("data", "rag_service_account_assertion_keys"),
    ("control", "rag_control.control_service_account_assertion_keys"),
])
def test_key_state_machine_has_one_active_and_bounded_verify_only_key(
        proof_databases, side, table):
    import psycopg

    data, control = proof_databases
    connection = data if side == "data" else control
    rotation_table = table.replace("assertion_keys", "assertion_rotations")

    def bind_rotation(rotation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {rotation_table} "
                "(rotation_id,previous_key_version,target_key_version,"
                "target_key_fingerprint,verify_started_at,verify_until,phase) "
                "VALUES (%s,7,8,%s,statement_timestamp(),"
                "statement_timestamp()+interval '300 seconds','staged')",
                (rotation, b"f" * 32))
            cursor.execute(
                f"UPDATE {table} SET rotation_id=%s WHERE key_version=7",
                (rotation,))

    with pytest.raises(psycopg.errors.UniqueViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} "
                "(key_version,secret,state,not_before) "
                "VALUES (8,%s,'active',statement_timestamp())", (KEY,))
    connection.rollback()
    rotation = uuid.uuid4()
    bind_rotation(rotation)
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {table} "
            "(key_version,secret,state,not_before,verify_started_at,"
            "verify_until,rotation_id) VALUES (8,%s,'verify_only',"
            "statement_timestamp()-interval '1 minute',"
            "statement_timestamp(),"
            "statement_timestamp()+interval '300 seconds',%s)",
            (KEY, rotation))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} "
                "(key_version,secret,state,not_before,verify_started_at,"
                "verify_until,rotation_id) VALUES (9,%s,'verify_only',"
                "statement_timestamp()-interval '1 minute',"
                "statement_timestamp(),"
                "statement_timestamp()+interval '1 second',%s)",
                (KEY, rotation))
    connection.rollback()
    rotation = uuid.uuid4()
    bind_rotation(rotation)
    with pytest.raises(psycopg.errors.CheckViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} "
                "(key_version,secret,state,not_before,verify_started_at,"
                "verify_until,rotation_id) VALUES (8,%s,'verify_only',"
                "statement_timestamp()-interval '1 minute',"
                "statement_timestamp(),"
                "statement_timestamp()+interval '301 seconds',%s)",
                (KEY, rotation))
    connection.rollback()
    rotation = uuid.uuid4()
    bind_rotation(rotation)
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {table} "
            "(key_version,secret,state,not_before,rotation_id) "
            "VALUES (8,%s,'staged',statement_timestamp(),%s)",
            (KEY, rotation))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} "
                "(key_version,secret,state,not_before,rotation_id) "
                "VALUES (9,%s,'staged',statement_timestamp(),%s)",
                (KEY, rotation))
    connection.rollback()
    rotation = uuid.uuid4()
    bind_rotation(rotation)
    with pytest.raises(psycopg.errors.CheckViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table} "
                "(key_version,secret,state,not_before,verify_until,rotation_id) "
                "VALUES (8,%s,'verify_only',statement_timestamp(),NULL,%s)",
                (KEY, rotation))
    connection.rollback()


@pytest.mark.parametrize(("side", "key_table", "rotation_table"), [
    ("data", "rag_service_account_assertion_keys",
     "rag_service_account_assertion_rotations"),
    ("control", "rag_control.control_service_account_assertion_keys",
     "rag_control.control_service_account_assertion_rotations"),
])
def test_rotation_ledger_requires_both_bound_keys_and_only_one_live_row(
        proof_databases, side, key_table, rotation_table):
    import psycopg

    data, control = proof_databases
    connection = data if side == "data" else control
    orphan = uuid.uuid4()
    with pytest.raises(psycopg.errors.CheckViolation,
                       match="rotation_keys_unbound"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {rotation_table} "
                "(rotation_id,previous_key_version,target_key_version,"
                "target_key_fingerprint,verify_started_at,verify_until,phase) "
                "VALUES (%s,7,8,%s,statement_timestamp(),"
                "statement_timestamp()+interval '300 seconds','staged')",
                (orphan, b"f" * 32))
        connection.commit()
    connection.rollback()


@pytest.mark.parametrize(("side", "key_table", "rotation_table"), [
    ("data", "rag_service_account_assertion_keys",
     "rag_service_account_assertion_rotations"),
    ("control", "rag_control.control_service_account_assertion_keys",
     "rag_control.control_service_account_assertion_rotations"),
])
def test_bound_rotation_resists_later_key_detach_delete_and_window_extension(
        proof_databases, side, key_table, rotation_table):
    import psycopg

    data, control = proof_databases
    connection = data if side == "data" else control
    rotation = uuid.uuid4()
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {rotation_table} "
            "(rotation_id,previous_key_version,target_key_version,"
            "target_key_fingerprint,verify_started_at,verify_until,phase) "
            "VALUES (%s,7,18,%s,statement_timestamp(),"
            "statement_timestamp()+interval '300 seconds','staged')",
            (rotation, b"f" * 32))
        cursor.execute(
            f"UPDATE {key_table} SET rotation_id=%s WHERE key_version=7",
            (rotation,))
        cursor.execute(
            f"INSERT INTO {key_table} "
            "(key_version,secret,state,not_before,rotation_id) "
            "VALUES (18,%s,'staged',statement_timestamp(),%s)",
            (KEY, rotation))
    connection.commit()
    with pytest.raises(psycopg.errors.CheckViolation,
                       match="rotation_keys_unbound"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {key_table} SET rotation_id=NULL "
                "WHERE key_version=7")
        connection.commit()
    connection.rollback()
    with pytest.raises(psycopg.errors.CheckViolation,
                       match="rotation_keys_unbound"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {key_table} WHERE key_version=18")
        connection.commit()
    connection.rollback()
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState,
                       match="transition_refused"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {rotation_table} SET verify_until="
                "verify_until+interval '1 second' WHERE rotation_id=%s",
                (rotation,))
    connection.rollback()
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState,
                       match="tombstone_immutable"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {rotation_table} WHERE rotation_id=%s",
                (rotation,))
    connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {rotation_table} SET phase='aborted' "
            "WHERE rotation_id=%s", (rotation,))
    connection.commit()
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {key_table} SET rotation_id=NULL WHERE key_version=7")
        cursor.execute(f"DELETE FROM {key_table} WHERE key_version=18")
    connection.commit()

    first = uuid.uuid4()
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {rotation_table} "
            "(rotation_id,previous_key_version,target_key_version,"
            "target_key_fingerprint,verify_started_at,verify_until,phase) "
            "VALUES (%s,7,8,%s,statement_timestamp(),"
            "statement_timestamp()+interval '300 seconds','staged')",
            (first, b"f" * 32))
        cursor.execute(
            f"UPDATE {key_table} SET rotation_id=%s WHERE key_version=7",
            (first,))
        cursor.execute(
            f"INSERT INTO {key_table} "
            "(key_version,secret,state,not_before,rotation_id) "
            "VALUES (8,%s,'staged',statement_timestamp(),%s)", (KEY, first))
        cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
    with pytest.raises(psycopg.errors.UniqueViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {rotation_table} "
                "(rotation_id,previous_key_version,target_key_version,"
                "target_key_fingerprint,verify_started_at,verify_until,phase) "
                "VALUES (%s,7,9,%s,statement_timestamp(),"
                "statement_timestamp()+interval '300 seconds','staged')",
                (uuid.uuid4(), b"g" * 32))
    connection.rollback()


@pytest.mark.parametrize(("side", "key_table", "rotation_table"), [
    ("data", "rag_service_account_assertion_keys",
     "rag_service_account_assertion_rotations"),
    ("control", "rag_control.control_service_account_assertion_keys",
     "rag_control.control_service_account_assertion_rotations"),
])
def test_rotation_membership_rechecks_both_sides_and_rejects_foreign_keys(
        proof_databases, side, key_table, rotation_table):
    import psycopg

    data, control = proof_databases
    connection = data if side == "data" else control
    live_rotation = uuid.uuid4()
    aborted_rotation = uuid.uuid4()
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {rotation_table} "
            "(rotation_id,previous_key_version,target_key_version,"
            "target_key_fingerprint,verify_started_at,verify_until,phase) "
            "VALUES (%s,7,410,%s,statement_timestamp(),"
            "statement_timestamp()+interval '300 seconds','staged')",
            (live_rotation, b"l" * 32))
        cursor.execute(
            f"UPDATE {key_table} SET rotation_id=%s WHERE key_version=7",
            (live_rotation,))
        cursor.execute(
            f"INSERT INTO {key_table} "
            "(key_version,secret,state,not_before,rotation_id) "
            "VALUES (410,%s,'staged',statement_timestamp(),%s)",
            (KEY, live_rotation))
        cursor.execute(
            f"INSERT INTO {rotation_table} "
            "(rotation_id,previous_key_version,target_key_version,"
            "target_key_fingerprint,verify_started_at,verify_until,phase) "
            "VALUES (%s,7,411,%s,statement_timestamp(),"
            "statement_timestamp()+interval '300 seconds','aborted')",
            (aborted_rotation, b"a" * 32))
    connection.commit()

    # The destination accepts version 7 as one of its declared members.  The
    # update must still fail because its OLD rotation would become unbound.
    with pytest.raises(psycopg.errors.CheckViolation,
                       match="rotation_keys_unbound"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {key_table} SET rotation_id=%s WHERE key_version=7",
                (aborted_rotation,))
        connection.commit()
    connection.rollback()

    # An aborted tombstone may have zero, one, or two declared members, never
    # an unrelated key that would falsify its immutable membership record.
    with pytest.raises(psycopg.errors.CheckViolation,
                       match="rotation_keys_unbound"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {key_table} "
                "(key_version,secret,state,not_before,rotation_id) "
                "VALUES (412,%s,'retired',statement_timestamp(),%s)",
                (KEY, aborted_rotation))
        connection.commit()
    connection.rollback()

    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {rotation_table} SET phase='aborted' "
            "WHERE rotation_id=%s", (live_rotation,))
        cursor.execute(
            f"UPDATE {key_table} SET rotation_id=NULL WHERE key_version=7")
        cursor.execute(f"DELETE FROM {key_table} WHERE key_version=410")
    connection.commit()

    retired_rotation = uuid.uuid4()
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {rotation_table} "
            "(rotation_id,previous_key_version,target_key_version,"
            "target_key_fingerprint,verify_started_at,verify_until,phase,"
            "completed_at,retired_at) VALUES (%s,7,413,%s,"
            "statement_timestamp(),statement_timestamp()+interval '300 seconds',"
            "'retired',statement_timestamp(),statement_timestamp())",
            (retired_rotation, b"r" * 32))
        cursor.execute(
            f"INSERT INTO {key_table} "
            "(key_version,secret,state,not_before,rotation_id) "
            "VALUES (413,%s,'retired',statement_timestamp(),%s)",
            (KEY, retired_rotation))
    connection.commit()
    with pytest.raises(psycopg.errors.CheckViolation,
                       match="rotation_keys_unbound"):
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {key_table} "
                "(key_version,secret,state,not_before,rotation_id) "
                "VALUES (414,%s,'retired',statement_timestamp(),%s)",
                (KEY, retired_rotation))
        connection.commit()
    connection.rollback()


def test_control_refuses_a_verify_only_key_before_its_window_begins(
        proof_databases):
    _data, control = proof_databases
    rotation = uuid.uuid4()
    nonce = uuid.uuid4().bytes
    issued = int(datetime.now(timezone.utc).timestamp())
    values = (
        "approval_list", 8, TENANT, ACTOR_DIGEST, 9,
        None, None, None, None, 10, issued, issued + 30, nonce)
    payload = _shape_payload(
        control,
        "rag_control.control_service_account_assertion_payload", values)
    proof = ServiceAccountAssertion(
        1, "approval_list", 8, TENANT, ACTOR_DIGEST, 9,
        None, None, None, None, 10, issued, issued + 30, nonce,
        hmac.new(KEY, payload, hashlib.sha256).digest())
    with control.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control."
            "control_service_account_assertion_rotations "
            "(rotation_id,previous_key_version,target_key_version,"
            "target_key_fingerprint,verify_started_at,verify_until,phase) "
            "VALUES (%s,7,8,%s,statement_timestamp()+interval '60 seconds',"
            "statement_timestamp()+interval '120 seconds','admitted')",
            (rotation, b"f" * 32))
        cursor.execute(
            "UPDATE rag_control.control_service_account_assertion_keys "
            "SET rotation_id=%s WHERE key_version=7", (rotation,))
        cursor.execute(
            "INSERT INTO rag_control.control_service_account_assertion_keys "
            "(key_version,secret,state,not_before,verify_started_at,"
            "verify_until,rotation_id) VALUES (8,%s,'verify_only',"
            "statement_timestamp(),statement_timestamp()+interval '60 seconds',"
            "statement_timestamp()+interval '120 seconds',%s)",
            (KEY, rotation))
    control.commit()
    try:
        with pytest.raises(control_db.ControlPlaneRefused):
            control_db.list_redeemable_service_account_approvals(control, proof)
        control.rollback()
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM rag_control."
                "control_service_account_assertion_nonces WHERE nonce=%s",
                (nonce,))
            assert cursor.fetchone()[0] == 0
        control.rollback()
    finally:
        with control.cursor() as cursor:
            cursor.execute(
                "UPDATE rag_control."
                "control_service_account_assertion_rotations "
                "SET phase='aborted' WHERE rotation_id=%s", (rotation,))
            cursor.execute(
                "DELETE FROM rag_control.control_service_account_assertion_keys "
                "WHERE key_version=8")
            cursor.execute(
                "UPDATE rag_control.control_service_account_assertion_keys "
                "SET rotation_id=NULL WHERE key_version=7")
        control.commit()

def test_control_prunes_only_a_bounded_expired_nonce_batch_after_valid_mac(
        proof_databases):
    data, control = proof_databases
    with control.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO rag_control.control_service_account_assertion_nonces "
            "(key_version,purpose,nonce,tenant_id,expires_at) "
            "VALUES (7,'approval_list',%s,%s,"
            "statement_timestamp()-interval '1 second')",
            [(number.to_bytes(16, "big"), TENANT) for number in range(130)],
        )
    control.commit()
    index_db.set_tenant_context(data, TENANT, actor_id=ACTOR)
    data.commit()
    proof = index_db.mint_service_account_approval_list_assertion(
        data, actor_id=ACTOR, expected_policy_epoch=9, limit=10)
    data.commit()
    invalid = replace(proof, mac=b"x" * 32)
    with pytest.raises(control_db.ControlPlaneRefused):
        control_db.list_redeemable_service_account_approvals(control, invalid)
    control.rollback()
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM rag_control."
            "control_service_account_assertion_nonces "
            "WHERE expires_at <= statement_timestamp()")
        assert cursor.fetchone()[0] == 130
    control.rollback()
    control_db.list_redeemable_service_account_approvals(control, proof)
    control.commit()
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM rag_control."
            "control_service_account_assertion_nonces "
            "WHERE expires_at <= statement_timestamp()")
        assert cursor.fetchone()[0] == 2
    control.rollback()
    second = index_db.mint_service_account_approval_list_assertion(
        data, actor_id=ACTOR, expected_policy_epoch=9, limit=10)
    data.commit()
    control_db.list_redeemable_service_account_approvals(control, second)
    control.commit()
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM rag_control."
            "control_service_account_assertion_nonces "
            "WHERE expires_at <= statement_timestamp()")
        assert cursor.fetchone()[0] == 0
    control.rollback()


def test_schema_readiness_refuses_pgcrypto_namespace_drift(proof_databases):
    data, control = proof_databases
    for connection, ready in (
            (data, index_db.schema_is_current),
            (control, lambda conn: control_db._assert_control_schema_receipt(
                conn) is None)):
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA assertion_extension_drift")
            cursor.execute(
                "ALTER EXTENSION pgcrypto SET SCHEMA assertion_extension_drift")
        connection.commit()
        try:
            if connection is data:
                assert ready(connection) is False
            else:
                with pytest.raises(control_db.ControlPlaneRefused,
                                   match="pgcrypto"):
                    ready(connection)
        finally:
            connection.rollback()
            with connection.cursor() as cursor:
                cursor.execute("ALTER EXTENSION pgcrypto SET SCHEMA public")
                cursor.execute("DROP SCHEMA assertion_extension_drift")
            connection.commit()
