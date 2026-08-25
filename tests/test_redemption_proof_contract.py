"""Closed contract for the cross-database service-account proof."""
from pathlib import Path
import struct
import uuid

from pipeline.control import db as control_db
from pipeline.index import db as index_db


ROOT = Path(__file__).resolve().parent.parent
DATA_SCHEMA = (ROOT / "pipeline" / "index" / "schema.sql").read_text(
    encoding="utf-8")
CONTROL_SCHEMA = (ROOT / "pipeline" / "control" / "schema.sql").read_text(
    encoding="utf-8")


def _function(schema, name, terminator):
    return schema.split(name, 1)[1].split(terminator, 1)[0]


def test_schema_versions_advance_for_the_proof_authorities():
    assert index_db.SCHEMA_VERSION == 12
    assert control_db.CONTROL_SCHEMA_VERSION == 5


def test_the_two_databases_pin_one_binary_payload_layout():
    data = _function(
        DATA_SCHEMA, "FUNCTION rag_service_account_assertion_payload(",
        "$service_account_assertion_payload$;")
    control = _function(
        CONTROL_SCHEMA,
        "FUNCTION rag_control.control_service_account_assertion_payload(",
        "$service_account_assertion_payload$;")
    data_return = data.split("    RETURN ", 1)[1]
    control_return = control.split("    RETURN ", 1)[1]
    assert data_return == control_return
    for forbidden in ("json", "::text", "concat", "format("):
        assert forbidden not in data.lower()
    for authority in (data, control):
        assert "int4send" in authority
        assert "int8send" in authority
        assert "uuid_send" in authority
        assert "octet_length(requested_nonce) <> 16" in authority
        assert "requested_expires_at - requested_issued_at <> 30" in authority


def test_the_payload_golden_vector_is_independent_of_sql_text_formatting():
    purpose = b"approval_redeem_issue"
    payload = (
        b"ragtest.service-account.assertion.v1"
        + struct.pack("!I", len(purpose)) + purpose
        + struct.pack("!I", 7)
        + uuid.UUID("10000000-0000-0000-0000-000000000001").bytes
        + b"a" * 32 + struct.pack("!q", 9)
        + bytes((1,))
        + uuid.UUID("50000000-0000-0000-0000-000000000005").bytes
        + bytes((1,)) + struct.pack("!q", 3)
        + bytes((1,))
        + uuid.UUID("30000000-0000-0000-0000-000000000003").bytes
        + bytes((1,)) + b"c" * 32 + bytes((0,))
        + struct.pack("!q", 2_000_000_000)
        + struct.pack("!q", 2_000_000_030)
        + b"n" * 16
    )
    assert payload.hex() == (
        "726167746573742e736572766963652d6163636f756e742e617373657274696f"
        "6e2e763100000015617070726f76616c5f72656465656d5f6973737565000000"
        "0710000000000000000000000000000001616161616161616161616161616161"
        "6161616161616161616161616161616161000000000000000901500000000000"
        "0000000000000000000501000000000000000301300000000000000000000000"
        "0000000301636363636363636363636363636363636363636363636363636363"
        "6363636363000000000077359400000000007735941e6e6e6e6e6e6e6e6e6e6e"
        "6e6e6e6e6e6e")


def test_key_material_is_private_and_separate_from_existing_domains():
    assert "rag_service_account_assertion_keys" in DATA_SCHEMA
    assert "control_service_account_assertion_keys" in CONTROL_SCHEMA
    assert "octet_length(secret) = 32" in DATA_SCHEMA
    assert "octet_length(secret) = 32" in CONTROL_SCHEMA
    assert "(state = 'verify_only') = (verify_until IS NOT NULL)" in (
        DATA_SCHEMA)
    assert "(state = 'verify_only') = (verify_until IS NOT NULL)" in (
        CONTROL_SCHEMA)
    assert "control_one_active_assertion_key" in CONTROL_SCHEMA
    assert "REVOKE ALL ON rag_service_account_assertion_keys FROM PUBLIC" in (
        DATA_SCHEMA)
    assert ("REVOKE ALL ON "
            "rag_control.control_service_account_assertion_keys FROM PUBLIC"
            in CONTROL_SCHEMA)
    mint = _function(
        DATA_SCHEMA, "FUNCTION rag_mint_service_account_assertion(",
        "$mint_service_account_assertion$;")
    assert "rag_context_secrets" not in mint
    assert "CONTROL_AUDIT_HMAC_SECRET" not in mint
    assert "CONTROL_SERVICE_ACCOUNT_HMAC_SECRET" not in mint
    runtime_script = (ROOT / "scripts" / "init_runtime_role.sh").read_text(
        encoding="utf-8")
    assert "REVOKE ALL ON rag_service_account_assertion_keys" in runtime_script
    assert "GRANT EXECUTE ON FUNCTION rag_mint" not in runtime_script


def test_mint_requires_and_locks_the_complete_human_authority():
    mint = _function(
        DATA_SCHEMA, "FUNCTION rag_mint_service_account_assertion(",
        "$mint_service_account_assertion$;")
    for fragment in (
            "architect.active = true", "membership.state = 'active'",
            "identity.state = 'active'",
            "membership.app_role = 'admin'",
            "rag_effective_actor() = requested_actor_id",
            "NOT rag_service_access()",
            "FOR UPDATE OF tenant, architect, identity, membership",
            "key.state = 'active'", "public.gen_random_bytes(16)"):
        assert fragment in mint
    assert "issued + 30" in mint
    assert "requested_actor_id" not in mint.split("RETURN QUERY", 1)[1]


def test_context_and_proof_comparisons_do_not_short_circuit_on_secret_bytes():
    for schema, helper in (
            (DATA_SCHEMA, "FUNCTION rag_secure_bytea_equal_32("),
            (CONTROL_SCHEMA,
             "FUNCTION rag_control.control_secure_bytea_equal(")):
        body = _function(schema, helper, "$secure_bytea_equal$;")
        assert "FOR byte_index IN 0..31 LOOP" in body
        assert "difference := difference" in body
        assert "RETURN difference = 0" in body
    context = _function(
        DATA_SCHEMA, "FUNCTION rag_context_valid()", "$context_valid$;")
    assert "rag_secure_bytea_equal_32(public.hmac(" in context
    assert "public.hmac(" in context and ") = decode(sig" not in context
    assert "DO $harden_context_search_path$" in DATA_SCHEMA
    assert "SET search_path TO pg_catalog, %I, pg_temp" in DATA_SCHEMA
    assert "DO $harden_assertion_search_path$" in DATA_SCHEMA
    assert "ALTER FUNCTION %I.rag_mint_service_account_assertion(" in (
        DATA_SCHEMA)


def test_control_verifies_before_consuming_one_global_nonce():
    table = _function(
        CONTROL_SCHEMA,
        "TABLE IF NOT EXISTS\nrag_control.control_service_account_assertion_nonces",
        "REVOKE ALL ON")
    assert "PRIMARY KEY (key_version, nonce)" in table
    consume = _function(
        CONTROL_SCHEMA,
        "FUNCTION rag_control.control_consume_service_account_assertion(",
        "$consume_service_account_assertion$;")
    verify = consume.index("control_secure_bytea_equal")
    insert = consume.index(
        "INSERT INTO rag_control.control_service_account_assertion_nonces")
    assert verify < insert
    assert "ON CONFLICT (key_version, nonce) DO NOTHING" in consume
    assert "requested_expires_at <= now_epoch" in consume
    assert "requested_issued_at > now_epoch + 5" in consume
    assert "key.state IN ('active', 'verify_only')" in consume


def test_each_operation_binds_every_mutable_business_dimension():
    for authority in (DATA_SCHEMA, CONTROL_SCHEMA):
        for purpose in (
                "approval_list", "approval_get",
                "approval_redeem_issue", "approval_redeem_rotate"):
            assert purpose in authority
        for field in (
                "requested_approval_revision",
                "requested_service_account_id", "requested_limit"):
            assert field in authority
        assert "int8send(requested_approval_revision)" in authority
        assert "uuid_send(requested_service_account_id)" in authority
        assert "int4send(requested_limit)" in authority


def test_the_online_redemption_surface_stays_closed_in_this_foundation():
    assert not (ROOT / "scripts" / "init_control_redeemer_role.sh").exists()
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "PG_CONTROL_REDEMPTION_DSN" not in env
    for route in (
            "/v1/org/admin/service-account-approvals",
            "service-account-approvals/{approval_id}/redeem"):
        assert route not in (ROOT / "pipeline" / "api" / "app.py").read_text(
            encoding="utf-8")
