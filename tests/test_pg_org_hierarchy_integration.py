"""Real PostgreSQL proof for directional organization visibility."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_TENANT_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_TENANT_PG_DSN is absent")

TENANT = uuid.UUID("82000000-0000-0000-0000-000000000001")
CEO = uuid.UUID("82000000-0000-0000-0000-000000000010")
MANAGER_A = uuid.UUID("82000000-0000-0000-0000-000000000011")
MANAGER_B = uuid.UUID("82000000-0000-0000-0000-000000000012")
LEAF_A = uuid.UUID("82000000-0000-0000-0000-000000000013")
LEAF_B = uuid.UUID("82000000-0000-0000-0000-000000000014")


@pytest.fixture(scope="module")
def org_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_org_" + uuid.uuid4().hex[:12]
    role = "ragtest_org_role_" + uuid.uuid4().hex[:12]
    password = "org-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cursor.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        connection = psycopg.connect(DSN, user=role, password=password)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
        connection.commit()
        db.init_schema(connection)
        db.set_tenant_context(connection, TENANT, service=True)
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO org_tenants (id, name) VALUES (%s, 'T')",
                           (TENANT,))
            for position_id, parent, title, kind, monitor, protected in (
                (CEO, None, "CEO", "root", True, True),
                (MANAGER_A, CEO, "Manager A", "manager", True, False),
                (MANAGER_B, CEO, "Manager B", "manager", True, False),
                (LEAF_A, MANAGER_A, "Leaf A", "member", False, False),
                (LEAF_B, MANAGER_B, "Leaf B", "member", False, False),
            ):
                cursor.execute(
                    "INSERT INTO org_positions "
                    "(id, tenant_id, parent_id, title, kind, "
                    "can_monitor_descendants, protected_from_monitoring) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (position_id, TENANT, parent, title, kind, monitor, protected))
            identities = {}
            for label, position in (("ceo", CEO), ("manager-a", MANAGER_A),
                                    ("manager-b", MANAGER_B),
                                    ("leaf-a", LEAF_A), ("leaf-b", LEAF_B)):
                identity_id = uuid.uuid4()
                identities[label] = identity_id
                cursor.execute(
                    "INSERT INTO org_identities (id, issuer, subject) "
                    "VALUES (%s, 'open-webui', %s)", (identity_id, label))
                cursor.execute(
                    "INSERT INTO org_memberships "
                    "(tenant_id, identity_id, position_id, display_label, state) "
                    "VALUES (%s, %s, %s, %s, 'active')",
                    (TENANT, identity_id, position, label))
        connection.commit()
        yield connection, identities
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cursor.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        finally:
            admin.close()


def test_ceo_sees_every_descendant_but_no_one_sees_the_ceo(org_database):
    connection, identities = org_database
    db.set_tenant_context(connection, TENANT)
    assert {row["display_label"] for row in
            db.visible_org_members(connection, identities["ceo"])} == {
                "manager-a", "manager-b", "leaf-a", "leaf-b"}
    for viewer in ("manager-a", "manager-b", "leaf-a", "leaf-b"):
        assert "ceo" not in {row["display_label"] for row in
                             db.visible_org_members(connection,
                                                    identities[viewer])}


def test_manager_sees_only_own_branch_and_leaf_sees_nobody(org_database):
    connection, identities = org_database
    assert [row["display_label"] for row in
            db.visible_org_members(connection, identities["manager-a"])] == [
                "leaf-a"]
    assert db.visible_org_members(connection, identities["leaf-a"]) == []


def test_a_denied_monitor_attempt_is_one_content_free_audit_event(org_database):
    """Persist a real, authenticated refusal without private request data."""
    connection, identities = org_database
    actor = identities["leaf-a"]
    subject = identities["ceo"]
    request_id = "denied-" + uuid.uuid4().hex[:24]
    db.set_tenant_context(connection, TENANT, actor_id=actor)

    event_id = db.record_org_decision(
        connection, actor_id=actor, subject_id=subject,
        action="monitor_view", reason_code="management_duty",
        allowed=False, request_id=request_id)

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, tenant_id, actor_id, subject_id, action, reason_code, "
            "decision, request_id FROM org_audit_events WHERE id = %s",
            (event_id,))
        row = cursor.fetchone()
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'org_audit_events' ORDER BY column_name")
        columns = {item[0] for item in cursor.fetchall()}

    assert row == (
        uuid.UUID(event_id), TENANT, actor, subject, "monitor_view",
        "management_duty", "denied", request_id,
    )
    assert columns == {
        "action", "actor_id", "created_at", "decision", "id", "reason_code",
        "request_id", "subject_id", "tenant_id",
    }
    for forbidden in (
            "question", "answer", "passage", "source_path", "token",
            "credential", "topology"):
        assert forbidden not in columns


def test_levels_are_derived_from_the_tree_not_client_claims(org_database):
    connection, identities = org_database
    assert db.org_context(connection, identities["ceo"])["level"] == 1
    assert db.org_context(connection, identities["manager-a"])["level"] == 2
    assert db.org_context(connection, identities["leaf-a"])["level"] == 3


def test_cycle_reparenting_is_refused_by_postgresql(org_database):
    connection, _identities = org_database
    with pytest.raises(Exception, match="org cycle refused"):
        with connection.cursor() as cursor:
            cursor.execute("UPDATE org_positions SET parent_id = %s, "
                           "kind = 'manager' WHERE id = %s", (LEAF_A, CEO))
    connection.rollback()


def test_the_complete_topology_is_replaced_once_under_version_control(
        org_database):
    connection, identities = org_database
    initial_version = db.org_topology(connection)["architecture_version"]
    positions = [
        {"id": CEO, "parent_id": None, "title": "Chief Executive",
         "kind": "root", "can_monitor_descendants": True,
         "protected_from_monitoring": True},
        {"id": MANAGER_A, "parent_id": CEO, "title": "Manager A",
         "kind": "manager", "can_monitor_descendants": True,
         "protected_from_monitoring": False},
        {"id": MANAGER_B, "parent_id": CEO, "title": "Manager B",
         "kind": "manager", "can_monitor_descendants": True,
         "protected_from_monitoring": False},
        {"id": LEAF_A, "parent_id": MANAGER_A, "title": "Leaf A",
         "kind": "member", "can_monitor_descendants": False,
         "protected_from_monitoring": False},
        {"id": LEAF_B, "parent_id": MANAGER_B, "title": "Leaf B",
         "kind": "member", "can_monitor_descendants": False,
         "protected_from_monitoring": False},
    ]
    members = [
        {"issuer": "open-webui", "subject": label,
         "position_id": position_id, "display_label": label,
         "app_role": "admin" if label == "ceo" else "reader",
         "state": "active"}
        for label, position_id in (
            ("ceo", CEO), ("manager-a", MANAGER_A),
            ("manager-b", MANAGER_B), ("leaf-a", LEAF_A),
            ("leaf-b", LEAF_B))
    ]

    result = db.replace_org_topology(
        connection, expected_version=initial_version, name="Renamed Tenant",
        positions=positions, members=members, actor_id=identities["ceo"],
        request_id=uuid.uuid4())

    assert result == {
        "architecture_version": initial_version + 1, "policy_epoch": 2}
    topology = db.org_topology(connection)
    assert topology["name"] == "Renamed Tenant"
    assert topology["architecture_version"] == initial_version + 1
    assert {row["subject"] for row in topology["members"]} == {
        "ceo", "manager-a", "manager-b", "leaf-a", "leaf-b"}

    with pytest.raises(db.OrgVersionConflict,
                       match="organizasyon mimarisi degisti"):
        db.replace_org_topology(
            connection, expected_version=initial_version, name="Stale Tenant",
            positions=positions, members=members,
            actor_id=identities["ceo"], request_id=uuid.uuid4())
    connection.rollback()
