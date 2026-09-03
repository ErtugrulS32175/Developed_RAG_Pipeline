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
    assert index_db.SCHEMA_VERSION == 14
    assert control_db.CONTROL_SCHEMA_VERSION == 7


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
    for schema, prefix in (
            (DATA_SCHEMA, "rag"), (CONTROL_SCHEMA, "control")):
        assert f"{prefix}_one_staged_assertion_key" in schema
        assert f"{prefix}_one_verify_only_assertion_key" in schema
        assert "verify_started_at timestamptz" in schema
        assert "interval '300 seconds'" in schema
        assert "target_key_fingerprint bytea NOT NULL" in schema
        assert "previous_key_version <> target_key_version" in schema
        assert "'staged', 'admitted', 'activated', 'completed'" in schema
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
    prune = consume.index(
        "DELETE FROM rag_control.control_service_account_assertion_nonces")
    insert = consume.index(
        "INSERT INTO rag_control.control_service_account_assertion_nonces")
    assert verify < prune < insert
    assert "LIMIT 128" in consume
    assert "FOR UPDATE SKIP LOCKED" in consume
    assert "control_assertion_nonces_expiry" in CONTROL_SCHEMA
    assert "ON CONFLICT (key_version, nonce) DO NOTHING" in consume
    assert "requested_expires_at <= now_epoch" in consume
    assert "requested_issued_at > now_epoch + 5" in consume
    assert "key.state IN ('active', 'verify_only')" in consume


def test_both_migrations_bind_pgcrypto_to_the_exact_public_extension_members():
    for schema in (DATA_SCHEMA, CONTROL_SCHEMA):
        prelude = schema.split("CREATE TABLE", 1)[0]
        assert "ext.extnamespace" not in prelude
        assert "extension_namespace <> 'public'::pg_catalog.regnamespace" in (
            prelude)
        assert "public.hmac(bytea,bytea,text)" in prelude
        assert "public.gen_random_bytes(integer)" in prelude
        assert "dependency.deptype = 'e'" in prelude
        assert ("dependency.refclassid =\n"
                "                  'pg_catalog.pg_extension'::regclass"
                in prelude)
        assert "MESSAGE = 'pgcrypto_namespace_refused'" in prelude


def test_rotation_ledgers_are_private_bound_and_single_live_authorities():
    for schema, prefix in (
            (DATA_SCHEMA, "rag"), (CONTROL_SCHEMA, "control")):
        assert f"{prefix}_one_live_assertion_rotation" in schema
        assert f"{prefix}_assertion_key_rotation_fk" in schema
        assert "DEFERRABLE INITIALLY DEFERRED" in schema
        assert "assertion_rotation_keys_unbound" in schema
        assert "AFTER INSERT OR UPDATE OR DELETE" in schema
        assert "assertion_rotation_tombstone_immutable" in schema
        assert "assertion_rotation_transition_refused" in schema
    runtime_script = (ROOT / "scripts" / "init_runtime_role.sh").read_text(
        encoding="utf-8")
    assert "REVOKE ALL ON rag_service_account_assertion_rotations" in (
        runtime_script)


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


def test_the_online_http_surface_stays_closed_after_redeemer_is_provisioned():
    assert (ROOT / "scripts" / "init_control_redeemer_role.sh").is_file()
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "PG_CONTROL_REDEMPTION_DSN=" in env
    for route in (
            "/v1/org/admin/service-account-approvals",
            "service-account-approvals/{approval_id}/redeem"):
        assert route not in (ROOT / "pipeline" / "api" / "app.py").read_text(
            encoding="utf-8")


def test_deployable_data_mints_are_purpose_specific_and_generic_is_private():
    runtime_script = (ROOT / "scripts" / "init_runtime_role.sh").read_text(
        encoding="utf-8")
    signatures = (
        "rag_mint_service_account_approval_list_assertion",
        "rag_mint_service_account_approval_get_assertion",
        "rag_mint_service_account_approval_redeem_issue_assertion",
        "rag_mint_service_account_approval_redeem_rotate_assertion",
    )
    flattened = DATA_SCHEMA.replace("\n", " ")
    for name in signatures:
        assert f"FUNCTION {name}(" in flattened
        assert f"{name}(" in runtime_script
    assert "GRANT EXECUTE ON FUNCTION rag_mint_service_account_assertion" not in (
        runtime_script)


def test_control_exposes_only_asserted_redemption_signatures():
    for name in (
            "control_asserted_list_redeemable_service_account_approvals",
            "control_asserted_get_redeemable_service_account_approval",
            "control_asserted_redeem_service_account_issue",
            "control_asserted_redeem_service_account_rotation"):
        body = CONTROL_SCHEMA.split(f"rag_control.{name}(", 1)[1].split(
            "CREATE OR REPLACE FUNCTION", 1)[0]
        assert "control_consume_service_account_assertion" in body
        assert "VOLATILE SECURITY DEFINER" in body
    flattened = " ".join(CONTROL_SCHEMA.split())
    for signature in (
            "control_list_redeemable_service_account_approvals( uuid, integer)",
            "control_redeem_service_account_issue( uuid, uuid, uuid, bigint, "
            "bytea, bigint, bytea, bytea, bytea)",
            "control_redeem_service_account_rotation( uuid, uuid, uuid, bigint, "
            "bytea, bigint, bytea, bytea, bytea)"):
        assert "DROP FUNCTION IF EXISTS rag_control." + signature in flattened
