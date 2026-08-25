"""Organization hierarchy policy and HTTP authorization boundaries."""
from datetime import datetime, timezone
from contextlib import contextmanager
import uuid

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api
from pipeline.api import auth, org_policy
from scripts import bootstrap_org


TENANT = uuid.UUID("81000000-0000-0000-0000-000000000001")
ARCHITECT = uuid.UUID("81000000-0000-0000-0000-000000000002")
MEMBER_IDENTITY = uuid.UUID("81000000-0000-0000-0000-000000000003")
ROOT = uuid.UUID("81000000-0000-0000-0000-000000000010")
MANAGER = uuid.UUID("81000000-0000-0000-0000-000000000011")
LEAF = uuid.UUID("81000000-0000-0000-0000-000000000012")


def _positions():
    return [
        {"id": LEAF, "parent_id": MANAGER, "title": "Leaf",
         "kind": "member", "can_monitor_descendants": False,
         "protected_from_monitoring": False},
        {"id": ROOT, "parent_id": None, "title": "CEO",
         "kind": "root", "can_monitor_descendants": True,
         "protected_from_monitoring": True},
        {"id": MANAGER, "parent_id": ROOT, "title": "Manager",
         "kind": "manager", "can_monitor_descendants": True,
         "protected_from_monitoring": False},
    ]


def _members():
    return [{
        "issuer": "open-webui", "subject": "leaf-user",
        "position_id": LEAF, "display_label": "Leaf User",
        "app_role": "reader", "state": "active",
    }]


def test_topology_is_parent_ordered_and_models_the_requested_visibility_shape():
    ordered, members = org_policy.ordered_topology(_positions(), _members())
    assert [item["id"] for item in ordered] == [ROOT, MANAGER, LEAF]
    assert ordered[0]["protected_from_monitoring"] is True
    assert ordered[-1]["can_monitor_descendants"] is False
    assert members == _members()


@pytest.mark.parametrize("mutator, message", [
    (lambda rows: rows.__setitem__(0, {**rows[0],
                                      "can_monitor_descendants": True}),
     "alt seviye"),
    (lambda rows: rows.__setitem__(2, {**rows[2], "parent_id": LEAF}),
     "dongulu"),
    (lambda rows: rows.append(dict(rows[0])), "benzersiz"),
    (lambda rows: rows.__setitem__(1, {
        **rows[1], "can_monitor_descendants": False}), "root tum alt"),
])
def test_invalid_or_privilege_widening_trees_are_refused(mutator, message):
    rows = _positions()
    mutator(rows)
    with pytest.raises(org_policy.TopologyRefused, match=message):
        org_policy.ordered_topology(rows, _members())


@pytest.fixture
def org_api(monkeypatch):
    current = {"principal": auth.Principal(
        TENANT, "org_architect", ARCHITECT, "openwebui",
        org_architect=True)}
    monkeypatch.setattr(api, "_request_principal",
                        lambda _request: current["principal"])

    @contextmanager
    def connection():
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    return TestClient(api.app), current


def test_architect_capability_alone_grants_no_document_or_model_read(org_api):
    client, _current = org_api
    assert client.get("/v1/models").status_code == 403
    assert client.get("/documents").status_code == 403


def test_user_learns_only_own_level_and_architecture_flag(org_api, monkeypatch):
    client, current = org_api
    current["principal"] = auth.Principal(
        TENANT, "reader", ARCHITECT, "openwebui", LEAF, False)
    monkeypatch.setattr(api.db, "org_context", lambda _conn, identity: {
        "identity_id": identity, "position_id": LEAF, "title": "Leaf",
        "kind": "member", "level": 3, "can_monitor_descendants": False,
        "protected_from_monitoring": False, "architecture_version": 7,
        "policy_epoch": 9, "display_label": "Leaf User",
        "app_role": "reader",
    })
    response = client.get("/v1/org/me")
    assert response.status_code == 200
    body = response.json()
    assert body["membership"]["level"] == 3
    assert body["membership"]["title"] == "Leaf"
    assert body["architecture_admin"] is False
    assert "parent_id" not in body["membership"]


def test_leaf_monitor_view_is_empty_and_audited_without_content(org_api,
                                                                 monkeypatch):
    client, current = org_api
    current["principal"] = auth.Principal(
        TENANT, "reader", ARCHITECT, "openwebui", LEAF, False)
    recorded = []
    monkeypatch.setattr(api.db, "visible_org_members",
                        lambda _conn, _identity: [])
    monkeypatch.setattr(api.db, "record_org_decision",
                        lambda _conn, **fields: recorded.append(fields))
    response = client.get(
        "/v1/org/visible-members?reason_code=management_duty")
    assert response.status_code == 200
    assert response.json() == {"members": []}
    assert recorded[0]["action"] == "monitor_view"
    assert recorded[0]["subject_id"] is None


def test_only_architect_can_read_or_replace_the_topology(org_api, monkeypatch):
    client, current = org_api
    topology = {"id": TENANT, "name": "Example", "architecture_version": 1,
                "policy_epoch": 1, "positions": [], "members": []}
    monkeypatch.setattr(api.db, "org_topology", lambda _conn: topology)
    monkeypatch.setattr(api.db, "record_org_decision", lambda *_a, **_k: None)
    assert client.get("/v1/org/admin/topology").status_code == 200

    current["principal"] = auth.Principal(
        TENANT, "admin", ARCHITECT, "openwebui", ROOT, False)
    assert client.get("/v1/org/admin/topology").status_code == 403


def test_an_authenticated_topology_denial_writes_one_closed_audit_event(
        org_api, monkeypatch):
    """An authenticated actor's refusal is evidence, without request content.

    The actor is a valid active OpenWebUI principal, but organization-admin
    presentation or document-admin capability must not impersonate the
    separately assigned architecture capability.  The refusal therefore has
    a stable actor and request id worth auditing; no topology body, position
    title, external subject or credential belongs in that event.
    """
    client, current = org_api
    current["principal"] = auth.Principal(
        TENANT, "admin", ARCHITECT, "openwebui", ROOT, False)
    recorded = []
    monkeypatch.setattr(api.db, "record_org_decision",
                        lambda _conn, **fields: recorded.append(fields))

    response = client.get("/v1/org/admin/topology")

    assert response.status_code == 403
    assert len(recorded) == 1
    assert set(recorded[0]) == {
        "actor_id", "subject_id", "action", "reason_code", "allowed",
        "request_id",
    }
    assert recorded[0] == {
        "actor_id": ARCHITECT,
        "subject_id": None,
        "action": "topology_read",
        "reason_code": "system_operation",
        "allowed": False,
        "request_id": response.headers["X-Request-ID"],
    }


def test_topology_replace_is_closed_and_optimistically_versioned(org_api,
                                                                   monkeypatch):
    client, _current = org_api
    captured = []
    monkeypatch.setattr(
        api.db, "replace_org_topology",
        lambda _conn, **fields: captured.append(fields) or {
            "architecture_version": 2, "policy_epoch": 2})
    body = {
        "expected_version": 1,
        "name": "Example",
        "positions": [{**row, "id": str(row["id"]),
                       "parent_id": (None if row["parent_id"] is None else
                                     str(row["parent_id"]))}
                      for row in _positions()],
        "members": [{**row, "position_id": str(row["position_id"])}
                    for row in _members()],
    }
    response = client.put("/v1/org/admin/topology", json=body)
    assert response.status_code == 200
    assert response.json() == {"architecture_version": 2, "policy_epoch": 2}
    assert [row["id"] for row in captured[0]["positions"]] == [
        ROOT, MANAGER, LEAF]

    broken = dict(body)
    broken["positions"] = [{**body["positions"][0],
                            "can_monitor_descendants": True}]
    assert client.put("/v1/org/admin/topology", json=broken).status_code == 422


def test_only_architect_can_read_audit_event_history(org_api, monkeypatch):
    client, current = org_api
    current["principal"] = auth.Principal(
        TENANT, "admin", ARCHITECT, "openwebui", ROOT, False)
    recorded = []
    monkeypatch.setattr(api.db, "record_org_decision",
                        lambda *_a, **fields: recorded.append(fields))
    assert client.get("/v1/org/admin/audit-events").status_code == 403
    assert recorded[0] == {
        "actor_id": ARCHITECT,
        "subject_id": None,
        "action": "events_view",
        "reason_code": "system_operation",
        "allowed": False,
        "request_id": recorded[0]["request_id"],
    }


def test_audit_events_returns_keyset_and_audits_events_view(org_api, monkeypatch):
    client, _current = org_api
    rows = [
        {
            "id": uuid.uuid4(),
            "action": "events_view",
            "reason_code": "system_operation",
            "decision": "allowed",
            "request_id": "req-a",
            "actor_id": ARCHITECT,
            "subject_id": None,
            "created_at": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
        {
            "id": uuid.uuid4(),
            "action": "topology_read",
            "reason_code": "system_operation",
            "decision": "allowed",
            "request_id": "req-b",
            "actor_id": ARCHITECT,
            "subject_id": None,
            "created_at": datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
        },
    ]
    captured = {}
    recorded = []

    def list_events(_conn, **fields):
        captured.update(fields)
        return rows

    monkeypatch.setattr(api.db, "list_org_audit_events", list_events)
    monkeypatch.setattr(api.db, "record_org_decision",
                        lambda *_a, **fields: recorded.append(fields))

    response = client.get(
        "/v1/org/admin/audit-events?limit=1&action=events_view"
        "&decision=allowed&reason_code=system_operation")

    assert response.status_code == 200
    body = response.json()
    assert body["events"][0]["event_id"] == str(rows[0]["id"])
    assert body["has_more"] is True
    assert body["next_cursor"]["before_id"] == str(rows[0]["id"])
    assert body["events"][0]["action"] == "events_view"
    assert body["events"][0]["actor_id"] == str(ARCHITECT)
    assert captured["actions"] == ["events_view"]
    assert captured["decisions"] == ["allowed"]
    assert captured["reasons"] == ["system_operation"]
    assert recorded[0] == {
        "actor_id": ARCHITECT,
        "subject_id": None,
        "action": "events_view",
        "reason_code": "system_operation",
        "allowed": True,
        "request_id": recorded[0]["request_id"],
    }


def test_audit_events_rejects_invalid_cursor(org_api):
    client, _current = org_api
    response = client.get("/v1/org/admin/audit-events?before_created_at=2026-01-01T12:00:00")
    assert response.status_code == 422


def test_membership_update_denial_is_closed_and_uses_topology_audit(org_api,
                                                                  monkeypatch):
    client, current = org_api
    current["principal"] = auth.Principal(
        TENANT, "admin", ARCHITECT, "openwebui", ROOT, False)
    recorded = []
    monkeypatch.setattr(api.db, "record_org_decision",
                        lambda *_a, **_k: recorded.append(_k))
    response = client.put(
        f"/v1/org/admin/members/{MEMBER_IDENTITY}",
        json={"expected_architecture_version": 1,
              "expected_policy_epoch": 1,
              "state": "suspended"})
    assert response.status_code == 403
    assert recorded[0] == {
        "actor_id": ARCHITECT,
        "subject_id": None,
        "action": "membership_change",
        "reason_code": "system_operation",
        "allowed": False,
        "request_id": recorded[0]["request_id"],
    }


def test_membership_update_requires_mutation_fields(org_api):
    client, _current = org_api
    response = client.put(
        f"/v1/org/admin/members/{MEMBER_IDENTITY}",
        json={"expected_architecture_version": 1,
              "expected_policy_epoch": 1})
    assert response.status_code == 422


def test_membership_update_calls_db_with_closed_request(org_api, monkeypatch):
    client, _current = org_api
    captured = []
    expected = {
        "identity_id": MEMBER_IDENTITY,
        "display_label": "Leaf User",
        "app_role": "editor",
        "state": "suspended",
        "position_id": LEAF,
        "architecture_version": 3,
        "policy_epoch": 4,
    }

    def mutate(conn, **fields):
        captured.append(fields)
        return expected

    monkeypatch.setattr(api.db, "update_org_member", mutate)
    body = {
        "expected_architecture_version": 2,
        "expected_policy_epoch": 1,
        "state": "suspended",
        "app_role": "editor",
        "position_id": str(LEAF),
    }
    response = client.put(
        f"/v1/org/admin/members/{MEMBER_IDENTITY}", json=body)
    assert response.status_code == 200
    assert response.json()["state"] == "suspended"
    assert response.json()["position_id"] == str(LEAF)
    assert captured[0] == {
        "actor_id": ARCHITECT,
        "target_identity_id": MEMBER_IDENTITY,
        "expected_architecture_version": 2,
        "expected_policy_epoch": 1,
        "state": "suspended",
        "app_role": "editor",
        "position_id": LEAF,
        "request_id": response.headers["X-Request-ID"],
    }


def test_membership_update_rejects_invalid_state_transition(org_api, monkeypatch):
    client, _current = org_api
    monkeypatch.setattr(
        api.db, "update_org_member",
        lambda _conn, **_kw: (_ for _ in ()).throw(
            api.db.OrgMembershipStateRefused("uyelik durumu gecersiz")))
    response = client.put(
        f"/v1/org/admin/members/{MEMBER_IDENTITY}",
        json={"expected_architecture_version": 1,
              "expected_policy_epoch": 1,
              "state": "pending"})
    assert response.status_code == 422


def test_membership_update_rejects_stale_arch_or_policy(org_api, monkeypatch):
    client, _current = org_api
    monkeypatch.setattr(
        api.db, "update_org_member",
        lambda _conn, **_kw: (_ for _ in ()).throw(
            api.db.OrgVersionConflict("organizasyon mimarisi degisti")))
    response = client.put(
        f"/v1/org/admin/members/{MEMBER_IDENTITY}",
        json={"expected_architecture_version": 99,
              "expected_policy_epoch": 1,
              "state": "active"})
    assert response.status_code == 409

    monkeypatch.setattr(
        api.db, "update_org_member",
        lambda _conn, **_kw: (_ for _ in ()).throw(
            api.db.OrgPolicyConflict("organizasyon politikasi degisti")))
    response = client.put(
        f"/v1/org/admin/members/{MEMBER_IDENTITY}",
        json={"expected_architecture_version": 1,
              "expected_policy_epoch": 99,
              "state": "active"})
    assert response.status_code == 409


def test_bootstrap_cli_prints_only_closed_ids(monkeypatch, capsys):
    closed = SimpleConnection()
    monkeypatch.setattr(bootstrap_org.db, "get_conn", lambda **_kw: closed)
    monkeypatch.setattr(
        bootstrap_org.db, "require_runtime_ready", lambda _conn: None)
    captured = []
    monkeypatch.setattr(
        bootstrap_org.db, "bootstrap_org_tenant",
        lambda _conn, **fields: captured.append(fields) or {
            "tenant_id": TENANT, "identity_id": ARCHITECT})
    bootstrap_org.main([
        "--tenant-id", str(TENANT), "--tenant-name", "Example",
        "--openwebui-subject", "private-subject",
    ])
    output = capsys.readouterr().out
    assert "private-subject" not in output
    assert str(TENANT) in output and str(ARCHITECT) in output
    assert closed.closed


class SimpleConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True
