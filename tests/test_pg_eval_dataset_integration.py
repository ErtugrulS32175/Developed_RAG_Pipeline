"""Real PostgreSQL proof for governed evaluation-dataset lifecycle and RLS."""
from concurrent.futures import ThreadPoolExecutor
import os
import uuid

import pytest

from pipeline.evaluation import datasets
from pipeline.index import db


DSN = os.getenv("RAGTEST_EVAL_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_EVAL_PG_DSN is absent")

TENANT_A = uuid.UUID("a1000000-0000-4000-8000-000000000001")
TENANT_B = uuid.UUID("b1000000-0000-4000-8000-000000000001")
ROOT_A = uuid.UUID("a1000000-0000-4000-8000-000000000010")
MANAGER_A = uuid.UUID("a1000000-0000-4000-8000-000000000011")
MANAGER_B = uuid.UUID("a1000000-0000-4000-8000-000000000012")
OWNER_POS = uuid.UUID("a1000000-0000-4000-8000-000000000013")
PROTECTED_POS = uuid.UUID("a1000000-0000-4000-8000-000000000014")
SUSPENDED_POS = uuid.UUID("a1000000-0000-4000-8000-000000000015")
ROOT_B = uuid.UUID("b1000000-0000-4000-8000-000000000010")

ROOT_ACTOR = uuid.UUID("a1000000-0000-4000-8000-000000000020")
MANAGER = uuid.UUID("a1000000-0000-4000-8000-000000000021")
PEER = uuid.UUID("a1000000-0000-4000-8000-000000000022")
OWNER = uuid.UUID("a1000000-0000-4000-8000-000000000023")
PROTECTED = uuid.UUID("a1000000-0000-4000-8000-000000000024")
SUSPENDED = uuid.UUID("a1000000-0000-4000-8000-000000000025")
OTHER_TENANT_ACTOR = uuid.UUID("b1000000-0000-4000-8000-000000000020")

CASE_A = "11111111-1111-4111-8111-111111111111"
CASE_B = "22222222-2222-4222-8222-222222222222"
PRIVATE_VALUES = {
    "q": "PRIVATE_EVAL_QUESTION_SENTINEL",
    "key": "PRIVATE_EVAL_KEY_SENTINEL",
    "answer": "PRIVATE_EVAL_ANSWER_SENTINEL",
}


def _cases():
    return [
        {"case_key": CASE_A, "q": PRIVATE_VALUES["q"],
         "key": PRIVATE_VALUES["key"], "answer": PRIVATE_VALUES["answer"],
         "pages": [7, 19], "type": "metin"},
        {"case_key": CASE_B, "q": "ikinci kurgu soru",
         "key": "ikinci kurgu anahtar", "answer": "ikinci kurgu cevap",
         "pages": [23], "type": "sayisal"},
    ]


def _set_context(conn, tenant, actor=None, *, service=False):
    db.set_tenant_context(conn, tenant, actor_id=actor, service=service)


@pytest.fixture
def eval_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_eval_" + uuid.uuid4().hex[:12]
    role = "ragtest_eval_role_" + uuid.uuid4().hex[:12]
    password = "eval-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cur.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        conn = psycopg.connect(DSN, user=role, password=password)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
        conn.commit()
        _set_context(conn, TENANT_A, service=True)
        db.init_schema(conn)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO org_tenants (id, name) VALUES (%s, %s)",
                [(TENANT_A, "Tenant A"), (TENANT_B, "Tenant B")])
            cur.executemany(
                "INSERT INTO org_positions "
                "(id, tenant_id, parent_id, title, kind, "
                "can_monitor_descendants, protected_from_monitoring) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (ROOT_A, TENANT_A, None, "Root A", "root", True, True),
                    (MANAGER_A, TENANT_A, ROOT_A, "Manager A", "manager",
                     True, False),
                    (MANAGER_B, TENANT_A, ROOT_A, "Manager B", "manager",
                     True, False),
                    (OWNER_POS, TENANT_A, MANAGER_A, "Owner", "member",
                     False, False),
                    (PROTECTED_POS, TENANT_A, ROOT_A, "Protected", "manager",
                     True, True),
                    (SUSPENDED_POS, TENANT_A, MANAGER_A, "Suspended", "member",
                     False, False),
                    (ROOT_B, TENANT_B, None, "Root B", "root", True, True),
                ])
            identities = [
                (ROOT_ACTOR, "root-a"), (MANAGER, "manager-a"),
                (PEER, "manager-b"), (OWNER, "owner"),
                (PROTECTED, "protected"), (SUSPENDED, "suspended"),
                (OTHER_TENANT_ACTOR, "tenant-b-owner"),
            ]
            cur.executemany(
                "INSERT INTO org_identities (id, issuer, subject) "
                "VALUES (%s, 'open-webui', %s)", identities)
            cur.executemany(
                "INSERT INTO org_memberships "
                "(tenant_id, identity_id, position_id, display_label, "
                "app_role, state) VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (TENANT_A, ROOT_ACTOR, ROOT_A, "Root", "admin", "active"),
                    (TENANT_A, MANAGER, MANAGER_A, "Manager", "admin", "active"),
                    (TENANT_A, PEER, MANAGER_B, "Peer", "admin", "active"),
                    (TENANT_A, OWNER, OWNER_POS, "Owner", "editor", "active"),
                    (TENANT_A, PROTECTED, PROTECTED_POS, "Protected", "editor",
                     "active"),
                    (TENANT_A, SUSPENDED, SUSPENDED_POS, "Suspended", "editor",
                     "active"),
                    (TENANT_B, OTHER_TENANT_ACTOR, ROOT_B, "Tenant B", "editor",
                     "active"),
                ])
        conn.commit()
        yield {
            "conn": conn, "admin": admin, "schema": schema,
            "role": role, "password": password,
        }
    finally:
        if conn is not None:
            conn.close()
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cur.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        finally:
            admin.close()


def _create(conn, actor, slug):
    _set_context(conn, TENANT_A, actor)
    return db.create_eval_dataset(
        conn, actor_id=actor, slug=slug, label=slug.replace("-", " ").title())


def _import(conn, actor, dataset):
    _set_context(conn, TENANT_A, actor)
    updated = db.replace_eval_cases(
        conn, actor_id=actor, dataset_id=dataset["id"],
        version_id=dataset["latest_version_id"], expected_revision=1,
        cases=_cases())
    assert updated["revision"] == 2
    assert updated["case_count"] == 2
    assert updated["state"] == "draft"
    assert updated["version_number"] == 1
    assert updated["content_sha256"] == datasets.version_digest(_cases())
    return updated


def _publish(conn, actor, dataset, *, revision=2, epoch=1, digest=None):
    _set_context(conn, TENANT_A, actor)
    return db.publish_eval_version(
        conn, actor_id=actor, dataset_id=dataset["id"],
        version_id=dataset["latest_version_id"],
        expected_revision=revision, expected_policy_epoch=epoch,
        expected_draft_sha256=(
            datasets.version_digest(_cases()) if digest is None else digest))


def test_owner_imports_and_publishes_one_digest_bound_version(eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "owner-lifecycle")
    _import(conn, OWNER, dataset)
    published = _publish(conn, OWNER, dataset)

    assert published["state"] == "published"
    assert published["case_count"] == 2
    assert published["content_sha256"] == datasets.version_digest(_cases())
    _set_context(conn, TENANT_A, OWNER)
    persisted = db.read_eval_cases(
        conn, actor_id=OWNER, dataset_id=dataset["id"],
        version_id=dataset["latest_version_id"])
    assert [{**row, "case_key": str(row["case_key"])}
            for row in persisted] == _cases()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state, case_count, encode(content_sha256, 'hex') "
            "FROM eval_dataset_versions WHERE id = %s",
            (dataset["latest_version_id"],))
        assert cur.fetchone() == (
            "published", 2, datasets.version_digest(_cases()))


def test_duplicate_slug_is_a_closed_domain_conflict(eval_database):
    conn = eval_database["conn"]
    first = _create(conn, OWNER, "one-owner-slug")
    with pytest.raises(db.EvalDatasetConflict):
        _create(conn, OWNER, "one-owner-slug")

    _set_context(conn, TENANT_A, OWNER)
    rows = db.list_eval_datasets(conn, actor_id=OWNER, limit=100)
    matching = [row for row in rows if row["slug"] == "one-owner-slug"]
    assert [row["id"] for row in matching] == [first["id"]]


def test_retire_requires_a_sealed_lifecycle_and_prevents_new_drafts(
        eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "retired-release")
    with pytest.raises(db.EvalDatasetStateRefused):
        db.retire_eval_dataset(
            conn, actor_id=OWNER, dataset_id=dataset["id"],
            expected_revision=1, expected_policy_epoch=1)

    _import(conn, OWNER, dataset)
    _publish(conn, OWNER, dataset)
    _set_context(conn, TENANT_A, OWNER)
    retired = db.retire_eval_dataset(
        conn, actor_id=OWNER, dataset_id=dataset["id"],
        expected_revision=2, expected_policy_epoch=1)
    assert retired["state"] == "retired"
    assert retired["revision"] == 3

    with pytest.raises(db.EvalDatasetStateRefused):
        db.create_eval_draft(
            conn, actor_id=OWNER, dataset_id=dataset["id"],
            expected_revision=3)
    _set_context(conn, TENANT_A, OWNER)
    rows = db.list_eval_datasets(conn, actor_id=OWNER, limit=100)
    row = next(item for item in rows if item["id"] == dataset["id"])
    assert row["state"] == "retired"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT event_type, base_revision, resulting_revision "
            "FROM eval_dataset_events WHERE dataset_id = %s "
            "ORDER BY created_at, id", (dataset["id"],))
        assert cur.fetchall()[-1] == ("dataset_retired", 2, 3)


def test_manager_reads_descendant_but_peer_and_protected_target_stay_hidden(
        eval_database):
    conn = eval_database["conn"]
    owner_dataset = _create(conn, OWNER, "visible-descendant")
    protected_dataset = _create(conn, PROTECTED, "protected-dataset")

    _set_context(conn, TENANT_A, MANAGER)
    visible = db.list_eval_datasets(conn, actor_id=MANAGER, limit=20)
    assert {row["id"] for row in visible} == {owner_dataset["id"]}
    assert db.read_eval_cases(
        conn, actor_id=MANAGER, dataset_id=owner_dataset["id"],
        version_id=owner_dataset["latest_version_id"]) == []

    _set_context(conn, TENANT_A, PEER)
    assert db.list_eval_datasets(conn, actor_id=PEER, limit=20) == []

    _set_context(conn, TENANT_A, ROOT_ACTOR)
    root_visible = db.list_eval_datasets(conn, actor_id=ROOT_ACTOR, limit=20)
    assert owner_dataset["id"] in {row["id"] for row in root_visible}
    assert protected_dataset["id"] not in {row["id"] for row in root_visible}

    _set_context(conn, TENANT_A, PROTECTED)
    own = db.list_eval_datasets(conn, actor_id=PROTECTED, limit=20)
    assert protected_dataset["id"] in {row["id"] for row in own}


def test_manager_can_curate_a_descendant_but_peer_cannot_mutate_it(
        eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "manager-read-only")
    _set_context(conn, TENANT_A, MANAGER)
    updated = db.replace_eval_cases(
        conn, actor_id=MANAGER, dataset_id=dataset["id"],
        version_id=dataset["latest_version_id"], expected_revision=1,
        cases=_cases())
    assert updated["revision"] == 2 and updated["case_count"] == 2

    _set_context(conn, TENANT_A, PEER)
    with pytest.raises(db.EvalDatasetAccessRefused):
        db.replace_eval_cases(
            conn, actor_id=PEER, dataset_id=dataset["id"],
            version_id=dataset["latest_version_id"], expected_revision=2,
            cases=_cases())
    # The refused public call rolls its transaction back. PostgreSQL therefore
    # restores the previously committed manager GUCs; bind the peer again so
    # this direct-SQL attack is judged under the intended identity.
    _set_context(conn, TENANT_A, PEER)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE eval_datasets SET label = 'forged' WHERE id = %s",
            (dataset["id"],))
        assert cur.rowcount == 0
    conn.rollback()


def test_suspension_revokes_owner_read_and_write_authority(eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, SUSPENDED, "suspension-fence")
    _set_context(conn, TENANT_A, service=True)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'suspended' "
            "WHERE tenant_id = %s AND identity_id = %s",
            (TENANT_A, SUSPENDED))
    conn.commit()

    _set_context(conn, TENANT_A, SUSPENDED)
    assert db.list_eval_datasets(conn, actor_id=SUSPENDED, limit=20) == []
    with pytest.raises(db.EvalDatasetAccessRefused):
        db.replace_eval_cases(
            conn, actor_id=SUSPENDED, dataset_id=dataset["id"],
            version_id=dataset["latest_version_id"], expected_revision=1,
            cases=_cases())


def test_cross_tenant_actor_cannot_read_or_forge_tenant_a_rows(eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "tenant-a-only")
    _set_context(conn, TENANT_B, OTHER_TENANT_ACTOR)
    assert db.list_eval_datasets(
        conn, actor_id=OTHER_TENANT_ACTOR, limit=20) == []
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM eval_datasets")
        assert cur.fetchone()[0] == 0
        with pytest.raises(Exception):
            cur.execute(
                "INSERT INTO eval_dataset_events "
                "(id, tenant_id, dataset_id, actor_id, event_type, "
                "resulting_revision) VALUES (%s, %s, %s, %s, "
                "'dataset_created', 1)",
                (uuid.uuid4(), TENANT_A, dataset["id"], OTHER_TENANT_ACTOR))
    conn.rollback()


def test_stale_revision_and_policy_epoch_write_nothing(eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "stale-publish")
    _import(conn, OWNER, dataset)
    with pytest.raises(db.EvalDatasetConflict):
        _publish(conn, OWNER, dataset, revision=3)
    with pytest.raises(db.EvalDatasetConflict):
        _publish(conn, OWNER, dataset, epoch=2)
    with pytest.raises(db.EvalDatasetConflict):
        _publish(conn, OWNER, dataset, digest="0" * 64)
    _set_context(conn, TENANT_A, OWNER)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state, revision FROM eval_dataset_versions WHERE id = %s",
            (dataset["latest_version_id"],))
        assert cur.fetchone() == ("draft", 2)
        cur.execute(
            "SELECT count(*) FROM eval_dataset_events "
            "WHERE version_id = %s AND event_type = 'version_published'",
            (dataset["latest_version_id"],))
        assert cur.fetchone()[0] == 0


def test_two_concurrent_publishers_create_exactly_one_publication(eval_database):
    import psycopg
    from psycopg import sql

    main = eval_database["conn"]
    dataset = _create(main, OWNER, "concurrent-publish")
    _import(main, OWNER, dataset)

    def attempt():
        conn = psycopg.connect(
            DSN, user=eval_database["role"], password=eval_database["password"])
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(eval_database["schema"])))
            conn.commit()
            try:
                return "ok", _publish(conn, OWNER, dataset)
            except (db.EvalDatasetConflict, db.EvalDatasetStateRefused):
                return "refused", None
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: attempt(), range(2)))
    assert sorted(status for status, _value in results) == ["ok", "refused"]

    _set_context(main, TENANT_A, OWNER)
    with main.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM eval_dataset_events "
            "WHERE version_id = %s AND event_type = 'version_published'",
            (dataset["latest_version_id"],))
        assert cur.fetchone()[0] == 1
        cur.execute(
            "SELECT current_version_id, revision FROM eval_datasets WHERE id = %s",
            (dataset["id"],))
        assert cur.fetchone() == (dataset["latest_version_id"], 2)


def _owner_level_attack(conn, statement, parameters):
    """Disable RLS transactionally so triggers, not policies, face the attack."""
    try:
        with pytest.raises(Exception, match="immutable"):
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE eval_dataset_versions DISABLE ROW LEVEL SECURITY")
                cur.execute("ALTER TABLE eval_cases DISABLE ROW LEVEL SECURITY")
                cur.execute(
                    "ALTER TABLE eval_dataset_events DISABLE ROW LEVEL SECURITY")
                cur.execute(statement, parameters)
    finally:
        # A missing trigger must fail the assertion without committing the
        # forged mutation or the transactional RLS-disable into the fixture.
        conn.rollback()


def test_sealed_version_cases_and_events_are_owner_level_immutable(eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "immutable-release")
    _import(conn, OWNER, dataset)
    _publish(conn, OWNER, dataset)
    version = dataset["latest_version_id"]
    draft = db.create_eval_draft(
        conn, actor_id=OWNER, dataset_id=dataset["id"],
        expected_revision=2)
    _set_context(conn, TENANT_A, OWNER)

    attacks = [
        ("UPDATE eval_dataset_versions SET case_count = 1 WHERE id = %s",
         (version,)),
        ("DELETE FROM eval_dataset_versions WHERE id = %s", (version,)),
        ("UPDATE eval_cases SET question = 'forged' WHERE version_id = %s",
         (version,)),
        ("UPDATE eval_cases SET version_id = %s "
         "WHERE version_id = %s AND ordinal = 1",
         (draft["id"], version)),
        ("DELETE FROM eval_cases WHERE version_id = %s", (version,)),
        ("INSERT INTO eval_cases "
         "(tenant_id, version_id, case_key, ordinal, question, document_key, "
         "expected_answer, pages, question_type, content_sha256) VALUES "
         "(%s, %s, %s, 3, 'q', 'k', 'a', ARRAY[1], 'metin', %s)",
         (TENANT_A, version, uuid.uuid4(), b"x" * 32)),
        ("UPDATE eval_dataset_events SET case_count = 1 "
         "WHERE version_id = %s AND event_type = 'version_published'",
         (version,)),
        ("DELETE FROM eval_dataset_events WHERE version_id = %s", (version,)),
    ]
    for statement, parameters in attacks:
        _owner_level_attack(conn, statement, parameters)


def test_event_rows_are_content_free_and_match_publication_digest(eval_database):
    conn = eval_database["conn"]
    dataset = _create(conn, OWNER, "content-free-events")
    _import(conn, OWNER, dataset)
    published = _publish(conn, OWNER, dataset)
    _set_context(conn, TENANT_A, OWNER)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'eval_dataset_events'")
        columns = {row[0] for row in cur.fetchall()}
        assert not ({"q", "question", "key", "document_key", "answer",
                     "expected_answer", "pages", "path", "source"} & columns)
        cur.execute(
            "SELECT event_type, case_count, encode(content_sha256, 'hex') "
            "FROM eval_dataset_events WHERE version_id = %s "
            "ORDER BY created_at, id", (dataset["latest_version_id"],))
        rows = cur.fetchall()
    rendered = repr(rows)
    assert all(value not in rendered for value in PRIVATE_VALUES.values())
    assert rows[-1] == (
        "version_published", 2, published["content_sha256"])
