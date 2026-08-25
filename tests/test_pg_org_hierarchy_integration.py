"""Real PostgreSQL proof for directional organization visibility."""
import os
import uuid
import threading
from queue import Queue

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
LEAF_C = uuid.UUID("82000000-0000-0000-0000-000000000015")


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
                if label == "ceo":
                    cursor.execute(
                        "INSERT INTO org_architects "
                        "(tenant_id, identity_id) VALUES (%s, %s)",
                        (TENANT, identity_id))
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


def test_list_org_audit_events_is_keyset_and_filterable(org_database):
    connection, identities = org_database
    actor = identities["leaf-a"]
    db.set_tenant_context(connection, TENANT, actor_id=actor)
    db.record_org_decision(
        connection, actor_id=actor, subject_id=identities["manager-a"],
        action="monitor_view", reason_code="management_duty", allowed=True,
        request_id="event-001")
    db.record_org_decision(
        connection, actor_id=actor, subject_id=identities["manager-b"],
        action="topology_read", reason_code="security_review", allowed=True,
        request_id="event-002")

    rows = db.list_org_audit_events(
        connection, limit=1, actions=["monitor_view", "topology_read"],
        decisions=["allowed"], reasons=["management_duty", "security_review"])
    assert len(rows) == 2

    cursor = (rows[0]["created_at"], rows[0]["id"])
    tail = db.list_org_audit_events(
        connection, limit=1,
        actions=["monitor_view", "topology_read"],
        before=cursor,
    )
    assert len(tail) <= 2
    assert all((row["created_at"], row["id"]) < cursor for row in tail)

    with pytest.raises(db.OrgAuditQueryRefused):
        db.list_org_audit_events(connection, limit=1, actions=["invalid_action"])


def test_membership_lifecycle_transitions_update_visibility_and_versions(
        org_database):
    connection, identities = org_database
    topology = db.org_topology(connection)
    architecture = topology["architecture_version"]
    policy = topology["policy_epoch"]

    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE org_memberships SET state = 'pending' "
            "WHERE identity_id = %s", (identities["leaf-a"],))
        connection.commit()

    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])
    activated = db.update_org_member(
        connection, actor_id=identities["ceo"],
        target_identity_id=identities["leaf-a"],
        expected_architecture_version=architecture,
        expected_policy_epoch=policy,
        state="active",
        request_id="request-001")

    assert activated["state"] == "active"
    assert activated["architecture_version"] == architecture + 1
    assert activated["policy_epoch"] == policy + 1
    assert activated["position_id"] == LEAF_A
    architecture = activated["architecture_version"]
    policy = activated["policy_epoch"]

    manager_visible = {
        row["identity_id"] for row in db.visible_org_members(connection, identities["manager-a"])
    }
    assert identities["leaf-a"] in manager_visible

    suspended = db.update_org_member(
        connection, actor_id=identities["ceo"],
        target_identity_id=identities["leaf-a"],
        expected_architecture_version=architecture,
        expected_policy_epoch=policy,
        state="suspended",
        request_id="request-002")

    assert suspended["state"] == "suspended"
    assert suspended["architecture_version"] == architecture + 1
    assert suspended["policy_epoch"] == policy + 1
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM org_audit_events "
            "WHERE subject_id = %s AND action = 'membership_change' "
            "AND request_id = 'request-002'",
            (identities["leaf-a"],))
        assert cursor.fetchone()[0] == 1
    architecture = suspended["architecture_version"]
    policy = suspended["policy_epoch"]

    manager_visible = {
        row["identity_id"] for row in db.visible_org_members(connection, identities["manager-a"])
    }
    assert identities["leaf-a"] not in manager_visible

    reactivated = db.update_org_member(
        connection, actor_id=identities["ceo"],
        target_identity_id=identities["leaf-a"],
        expected_architecture_version=architecture,
        expected_policy_epoch=policy,
        state="active",
        request_id="request-003")

    assert reactivated["state"] == "active"
    assert reactivated["architecture_version"] == architecture + 1
    assert reactivated["policy_epoch"] == policy + 1


def test_membership_role_and_position_update_uses_open_position(org_database):
    connection, identities = org_database
    topology = db.org_topology(connection)
    architecture = topology["architecture_version"]
    policy = topology["policy_epoch"]

    new_position = LEAF_C
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO org_positions "
            "(id, tenant_id, parent_id, title, kind, "
            "can_monitor_descendants, protected_from_monitoring) "
            "VALUES (%s, %s, %s, %s, %s, false, false)",
            (new_position, TENANT, MANAGER_A, "Leaf C", "member"))
    connection.commit()

    updated = db.update_org_member(
        connection, actor_id=identities["ceo"],
        target_identity_id=identities["leaf-b"],
        expected_architecture_version=architecture,
        expected_policy_epoch=policy,
        app_role="admin",
        position_id=new_position,
        request_id="req-role-pos")

    assert updated["app_role"] == "admin"
    assert updated["position_id"] == new_position
    assert updated["state"] == "active"


def test_membership_lifecycle_rejects_invalid_transition(org_database):
    connection, identities = org_database
    architecture = db.org_topology(connection)["architecture_version"]
    policy = db.org_topology(connection)["policy_epoch"]
    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE org_memberships SET state = 'active' "
            "WHERE identity_id = %s", (identities["leaf-b"],))
        connection.commit()

    with pytest.raises(db.OrgMembershipStateRefused,
                       match="uyelik durumu gecersiz"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=identities["leaf-b"],
            expected_architecture_version=architecture,
            expected_policy_epoch=policy,
            state="pending", request_id="req-invalid")


def test_membership_lifecycle_is_cross_tenant_safe(org_database):
    connection, identities = org_database
    architecture = db.org_topology(connection)["architecture_version"]
    policy = db.org_topology(connection)["policy_epoch"]

    db.set_tenant_context(connection, TENANT, service=True)
    with connection.cursor() as cursor:
        other_tenant = uuid.uuid4()
        foreign_identity = uuid.uuid4()
        foreign_position = uuid.uuid4()
        cursor.execute(
            "INSERT INTO org_tenants (id, name) VALUES (%s, 'Other')",
            (other_tenant,))
        cursor.execute(
            "INSERT INTO org_positions "
            "(id, tenant_id, parent_id, title, kind, "
            "can_monitor_descendants, protected_from_monitoring) "
            "VALUES (%s, %s, NULL, 'Foreign CEO', 'root', true, true)",
            (foreign_position, other_tenant))
        cursor.execute(
            "INSERT INTO org_identities (id, issuer, subject) "
            "VALUES (%s, 'open-webui', 'foreign-subject')",
            (foreign_identity,))
        cursor.execute(
            "INSERT INTO org_architects (tenant_id, identity_id) "
            "VALUES (%s, %s)",
            (other_tenant, foreign_identity))
    connection.commit()
    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])

    with pytest.raises(db.OrgIdentityConflict,
                       match="baska tenant'a aktif"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=foreign_identity,
            expected_architecture_version=architecture,
            expected_policy_epoch=policy,
            state="suspended",
            request_id="req-cross")


def test_actor_from_other_tenant_cannot_mutate_targeted_membership(
        org_database):
    connection, identities = org_database
    architecture = db.org_topology(connection)["architecture_version"]
    policy = db.org_topology(connection)["policy_epoch"]
    db.set_tenant_context(connection, TENANT, service=True)
    with connection.cursor() as cursor:
        other_tenant = uuid.uuid4()
        foreign_identity = uuid.uuid4()
        cursor.execute(
            "INSERT INTO org_tenants (id, name) VALUES (%s, 'Other')",
            (other_tenant,))
        cursor.execute(
            "INSERT INTO org_positions "
            "(id, tenant_id, parent_id, title, kind, "
            "can_monitor_descendants, protected_from_monitoring) "
            "VALUES (%s, %s, NULL, 'Foreign CEO', 'root', true, true)",
            (uuid.uuid4(), other_tenant))
        cursor.execute(
            "INSERT INTO org_identities (id, issuer, subject) "
            "VALUES (%s, 'open-webui', 'foreign-actor')",
            (foreign_identity,))
        cursor.execute(
            "INSERT INTO org_architects (tenant_id, identity_id) "
            "VALUES (%s, %s)",
            (other_tenant, foreign_identity))
    connection.commit()

    db.set_tenant_context(connection, TENANT, actor_id=foreign_identity)
    with pytest.raises(db.OrgIdentityConflict,
                       match="islemci mimari yetkisine sahip degil"):
        db.update_org_member(
            connection, actor_id=foreign_identity,
            target_identity_id=identities["leaf-a"],
            expected_architecture_version=architecture,
            expected_policy_epoch=policy,
            state="suspended",
            request_id="req-actor")


def test_membership_lifecycle_stale_request_fails_and_no_audit_event(org_database):
    connection, identities = org_database
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM org_audit_events WHERE actor_id = %s",
                       (identities["ceo"],))
        before = cursor.fetchone()[0]
    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])

    with pytest.raises(db.OrgMembershipStateRefused,
                       match="uyelik durumu gecersiz"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=identities["leaf-b"],
            expected_architecture_version=db.org_topology(connection)[
                "architecture_version"],
            expected_policy_epoch=db.org_topology(connection)["policy_epoch"],
            state="pending", request_id="req-fail")

    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM org_audit_events WHERE actor_id = %s",
                       (identities["ceo"],))
        after = cursor.fetchone()[0]
    assert after == before


def test_membership_stale_architecture_fences_subsequent_request(org_database):
    connection, identities = org_database
    architecture = db.org_topology(connection)["architecture_version"]
    policy = db.org_topology(connection)["policy_epoch"]
    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])
    db.update_org_member(
        connection, actor_id=identities["ceo"],
        target_identity_id=identities["leaf-a"],
        expected_architecture_version=architecture,
        expected_policy_epoch=policy,
        state="suspended", request_id="req-first")
    with pytest.raises(db.OrgVersionConflict,
                       match="organizasyon mimarisi degisti"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=identities["leaf-a"],
            expected_architecture_version=architecture,
            expected_policy_epoch=policy,
            state="active",
            request_id="req-stale")


def test_membership_lifecycle_concurrent_cas_race(org_database):
    connection, identities = org_database
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE org_memberships SET state = 'active' "
            "WHERE tenant_id = %s AND identity_id = %s",
            (TENANT, identities["leaf-b"]))
        connection.commit()
    architecture = db.org_topology(connection)["architecture_version"]
    policy = db.org_topology(connection)["policy_epoch"]

    with connection.cursor() as cursor:
        cursor.execute("SELECT current_schema()")
        schema = cursor.fetchone()[0]

    from psycopg import sql
    import psycopg

    def open_actor_connection():
        conn = psycopg.connect(DSN)
        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(schema)))
        db.set_tenant_context(conn, TENANT, actor_id=identities["ceo"])
        return conn

    gate = threading.Barrier(2)
    outcomes = Queue()

    def attempt(state_name, request_id):
        conn = open_actor_connection()
        try:
            gate.wait()
            row = db.update_org_member(
                conn, actor_id=identities["ceo"],
                target_identity_id=identities["leaf-b"],
                expected_architecture_version=architecture,
                expected_policy_epoch=policy,
                state=state_name,
                request_id=request_id)
            outcomes.put(("ok", row["state"]))
        except db.OrgVersionConflict as error:
            outcomes.put(("org_version_conflict", str(error)))
        except Exception as error:
            outcomes.put(("error", type(error).__name__))
        finally:
            conn.close()

    first = threading.Thread(target=attempt, args=("suspended", "req-cas-race-1"))
    second = threading.Thread(target=attempt, args=("suspended", "req-cas-race-2"))
    first.start()
    second.start()
    first.join()
    second.join()

    results = []
    while not outcomes.empty():
        results.append(outcomes.get())
    assert len(results) == 2
    ok = [item for item in results if item[0] == "ok"]
    conflict = [item for item in results
                if item[0] == "org_version_conflict"]
    failures = [item for item in results if item[0] == "error"]

    assert len(ok) == 1
    assert len(conflict) == 1
    assert not failures

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM org_audit_events "
            "WHERE tenant_id = %s AND actor_id = %s "
            "AND request_id IN ('req-cas-race-1', 'req-cas-race-2')",
            (TENANT, identities["ceo"]))
        assert cursor.fetchone()[0] == 1


def test_actor_parameter_cannot_impersonate_the_bound_architect(org_database):
    connection, identities = org_database
    topology = db.org_topology(connection)
    db.set_tenant_context(
        connection, TENANT, actor_id=identities["manager-a"])

    with pytest.raises(db.OrgIdentityConflict,
                       match="islemci mimari yetkisine sahip degil"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=identities["leaf-b"],
            expected_architecture_version=topology["architecture_version"],
            expected_policy_epoch=topology["policy_epoch"],
            app_role="editor", request_id="request-impersonation")


def test_membership_rejects_a_position_from_another_tenant(org_database):
    connection, identities = org_database
    topology = db.org_topology(connection)
    other_tenant = uuid.uuid4()
    foreign_position = uuid.uuid4()
    db.set_tenant_context(connection, TENANT, service=True)
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO org_tenants (id, name) VALUES (%s, 'Position Other')",
            (other_tenant,))
        cursor.execute(
            "INSERT INTO org_positions "
            "(id, tenant_id, parent_id, title, kind, "
            "can_monitor_descendants, protected_from_monitoring) "
            "VALUES (%s, %s, NULL, 'Foreign Position', 'root', true, true)",
            (foreign_position, other_tenant))
    connection.commit()
    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])

    with pytest.raises(db.OrgIdentityConflict,
                       match="pozisyon bu tenant'ta bulunamadi"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=identities["leaf-b"],
            expected_architecture_version=topology["architecture_version"],
            expected_policy_epoch=topology["policy_epoch"],
            position_id=foreign_position, request_id="request-foreign-position")


def test_membership_noop_does_not_advance_versions_or_write_audit(org_database):
    connection, identities = org_database
    db.set_tenant_context(connection, TENANT, actor_id=identities["ceo"])
    topology = db.org_topology(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT app_role FROM org_memberships "
            "WHERE tenant_id = %s AND identity_id = %s",
            (TENANT, identities["leaf-b"]))
        current_role = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM org_audit_events "
            "WHERE request_id = 'request-noop'")
        assert cursor.fetchone()[0] == 0

    with pytest.raises(db.OrgMembershipStateRefused,
                       match="uyelik degisikligi etkisiz"):
        db.update_org_member(
            connection, actor_id=identities["ceo"],
            target_identity_id=identities["leaf-b"],
            expected_architecture_version=topology["architecture_version"],
            expected_policy_epoch=topology["policy_epoch"],
            app_role=current_role, request_id="request-noop")

    after = db.org_topology(connection)
    assert after["architecture_version"] == topology["architecture_version"]
    assert after["policy_epoch"] == topology["policy_epoch"]
