"""One failed statement must cost one request, never the process.

The API used to cache a single module-level connection with no rollback and no
reconnect: any database error left it in a failed transaction and every later
request died until a restart. These tests lock the replacement contract at OUR
seam -- a connection is borrowed per request, returned on both exit paths, and
a failure in one request cannot leak state into the next. The pool's own
commit/rollback behaviour belongs to psycopg_pool; what is ours is that every
caller goes through it and holds nothing across requests.
"""
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone, tzinfo

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api


class FakePool:
    """Counts checkouts and returns, and hands out a fresh connection each
    time -- the properties a caching bug would violate."""

    def __init__(self):
        self.handed_out = []
        self.returned = 0

    @contextmanager
    def connection(self):
        conn = object()
        self.handed_out.append(conn)
        try:
            yield conn
        finally:
            self.returned += 1


@pytest.fixture
def pooled(monkeypatch):
    pool = FakePool()

    @contextmanager
    def publish_lock(_conn, _filename):
        # the session lock talks to a real cursor; these tests hand out
        # bare objects on purpose -- the lock's own contract is covered
        # in test_api_end_to_end
        yield

    monkeypatch.setattr(api.db, "get_pool", lambda: pool)
    monkeypatch.setattr(api.db, "init_schema", lambda conn: None)
    monkeypatch.setattr(api.db, "document_publish_lock", publish_lock)
    monkeypatch.setattr(api, "_schema_ready", False)
    return pool


def _headers():
    return {"Authorization": f"Bearer {api.API_KEY}"} if api.API_KEY else {}


def test_a_failed_request_does_not_poison_the_next(pooled, monkeypatch):
    calls = []

    def get_document(_conn, document_id):
        calls.append(document_id)
        if len(calls) == 1:
            raise RuntimeError("KURGU_VERITABANI_HATASI")
        return {"id": document_id, "filename": "kurgu.pdf",
                "file_type": "pdf", "status": "done"}

    monkeypatch.setattr(api.db, "get_document", get_document)
    client = TestClient(api.app, raise_server_exceptions=False)

    first = client.get("/documents/kurgu-belge-kimligi", headers=_headers())
    second = client.get("/documents/kurgu-belge-kimligi", headers=_headers())

    assert first.status_code == 500
    assert second.status_code == 200
    assert second.json()["status"] == "done"
    # the failing request's connection went back to the pool, and the second
    # request got its own -- nothing was cached across the failure
    assert len(pooled.handed_out) == 2
    assert pooled.handed_out[0] is not pooled.handed_out[1]
    assert pooled.returned == 2


def test_every_request_borrows_and_returns_its_own_connection(pooled, monkeypatch):
    monkeypatch.setattr(
        api.db, "get_document",
        lambda _conn, document_id: {"id": document_id, "filename": "kurgu.pdf",
                                    "file_type": "pdf", "status": "pending"})
    client = TestClient(api.app)

    for _ in range(3):
        assert client.get("/documents/kurgu-id", headers=_headers()).status_code == 200

    assert len(pooled.handed_out) == 3
    assert len(set(map(id, pooled.handed_out))) == 3
    assert pooled.returned == 3


def test_schema_init_runs_once_not_per_request(pooled, monkeypatch):
    ran = []
    monkeypatch.setattr(api.db, "init_schema", lambda conn: ran.append(1))
    monkeypatch.setattr(
        api.db, "get_document",
        lambda _conn, document_id: {"id": document_id, "filename": "kurgu.pdf",
                                    "file_type": "pdf", "status": "done"})
    client = TestClient(api.app)

    client.get("/documents/kurgu-id", headers=_headers())
    client.get("/documents/kurgu-id", headers=_headers())

    assert ran == [1]


def test_oversized_upload_is_refused_before_disk_and_database(
        pooled, monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "UPLOAD_MAX_BYTES", 10)
    upserts = []
    monkeypatch.setattr(api.db, "upsert_document",
                        lambda *a, **k: upserts.append(a) or "kurgu-id")

    response = TestClient(api.app).post(
        "/documents/upload", headers=_headers(),
        files={"file": ("kurgu.pdf", b"X" * 11, "application/pdf")})

    assert response.status_code == 413
    assert list(upload_dir.iterdir()) == []
    assert upserts == []
    # the cap refused the body without ever needing a connection
    assert pooled.handed_out == []


def test_upload_under_the_cap_still_works(pooled, monkeypatch, tmp_path):
    from pipeline.index import publication

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    # Package 3C: the endpoint publishes through the shared service, so
    # the destination and the two candidate seams live there now.
    monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "UPLOAD_MAX_BYTES", 10)
    monkeypatch.setattr(api.db, "lookup_document", lambda *a, **k: None)
    monkeypatch.setattr(
        api.db, "stage_candidate",
        lambda _conn, filename, *a, **k: ("kurgu-id", "kurgu-aday",
                                          filename))
    monkeypatch.setattr(api.db, "finalize_candidate_publication",
                        lambda *a, **k: True)

    @contextmanager
    def publish_lock(_conn, _filename):
        yield

    monkeypatch.setattr(api.db, "document_publish_lock", publish_lock)

    response = TestClient(api.app).post(
        "/documents/upload", headers=_headers(),
        files={"file": ("kurgu.pdf", b"X" * 10, "application/pdf")})

    assert response.status_code == 200
    assert (upload_dir / "kurgu.pdf").read_bytes() == b"X" * 10
    assert pooled.returned == 1


def test_lifespan_closes_the_pool_and_clears_the_global(monkeypatch):
    """Controlled shutdown must not depend on the OS reclaiming sockets."""
    from pipeline.index import db

    class ClosablePool:
        closed = False

        def close(self):
            self.closed = True

    fake = ClosablePool()
    monkeypatch.setattr(db, "_pool", fake)
    with TestClient(api.app):
        pass
    assert fake.closed
    assert db._pool is None


def test_retrieve_borrows_from_the_pool_per_query(monkeypatch):
    from pipeline.index import db
    from pipeline.retrieval import query

    pool = FakePool()
    seen = []
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    monkeypatch.setattr(query, "embed_dense", lambda q: [0.0])
    monkeypatch.setattr(query, "embed_sparse", lambda q: ([1], [1.0]))
    monkeypatch.setattr(
        db, "hybrid_search",
        lambda conn, *a, **k: seen.append(conn) or [])

    query.retrieve("kurgu soru")
    query.retrieve("kurgu soru")

    assert seen == pool.handed_out
    assert len(set(map(id, seen))) == 2
    assert pooled_returns_match(pool)


def pooled_returns_match(pool):
    return pool.returned == len(pool.handed_out)


# --- the inventory query's own contract ---------------------------------
#
# The listing endpoint is the first read that returns MANY rows, and the
# three properties below are the ones a page of rows can quietly get
# wrong: values spliced into the SQL, an order that is not total, and a
# "next page" flag derived from a second statement that disagrees with
# the first. They are checked at the cursor seam, because that is where
# the statement and its parameters are still separable.


class RecordingCursor:
    """Captures the statement and its parameters instead of running them."""

    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return [dict(row) for row in self._rows]


class RecordingConn:
    def __init__(self, rows=()):
        self.cur = RecordingCursor(rows)

    def cursor(self, row_factory=None):
        return self.cur


def _listing_row(**overrides):
    import uuid

    row = {
        "id": uuid.UUID("00000000-0000-0000-0000-0000000000aa"),
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "uploaded_at": f"{999:04d}-01-01T00:00:00+00:00",
        "status": "done",
        "status_note": None,
        "active_generation": 3,
    }
    row.update(overrides)
    return row


def test_the_listing_query_is_parameterized_and_totally_ordered():
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=20, offset=40)

    sql, params = conn.cur.executed[0]
    # the page bounds travel as PARAMETERS, never as text in the statement
    assert params == {"limit": 21, "offset": 40}
    assert "20" not in sql and "40" not in sql
    # `uploaded_at` alone is not unique: two rows sharing a clock tick can
    # swap places between pages, showing one twice and never showing the
    # other. The id tie-break makes the sequence total.
    assert "ORDER BY uploaded_at DESC, id DESC" in sql
    # no filter was supplied, so no filter appears ANYWHERE: not as a
    # clause in the statement and not as a key in the params dict
    assert "WHERE" not in sql
    assert "status" not in params and "file_type" not in params


def test_the_listing_query_asks_for_one_row_beyond_the_page():
    """`has_more` is answered by the SAME scan the page came from.

    A second COUNT over the table would be a different snapshot, and the
    flag could then contradict the rows next to it."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=1, offset=0)

    _sql, params = conn.cur.executed[0]
    assert params["limit"] == 2


def test_the_listing_query_never_selects_the_candidate_columns():
    """The projection is the gate: what is not selected cannot leak."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0)

    sql = conn.cur.executed[0][0]
    assert "content_sha256" not in sql
    assert "candidate_id" not in sql


def test_the_listing_returns_a_string_document_id():
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    rows = db.list_documents(conn, limit=5, offset=0)

    assert rows[0]["document_id"] == "00000000-0000-0000-0000-0000000000aa"
    assert "id" not in rows[0]
    assert rows[0]["active_generation"] == 3


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(0, 0), (-1, 0), (5, -1), (True, 0), (5, True), ("5", 0), (5, "0")],
)
def test_the_listing_refuses_a_page_it_cannot_size(limit, offset):
    """A page size is arithmetic, and `limit + 1` on an unchecked value is
    how OFFSET -1 reaches the server. The API rejects these in its own
    signature; this is the guard for every other caller."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    with pytest.raises(ValueError):
        db.list_documents(conn, limit=limit, offset=offset)
    assert conn.cur.executed == []


@pytest.mark.parametrize(
    ("kwargs", "clause", "key"),
    [
        ({"status": "kurgu-durum"}, "status = %(status)s", "status"),
        ({"file_type": "kurgu-tur"}, "file_type = %(file_type)s",
         "file_type"),
    ],
)
def test_a_single_filter_travels_only_as_a_parameter(kwargs, clause, key):
    """One filter alone: its static clause joins the statement, its VALUE
    never does -- it reaches the database only through the params dict.
    The other filter appears nowhere, in neither statement nor params."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, **kwargs)

    sql, params = conn.cur.executed[0]
    value = kwargs[key]
    assert clause in sql
    assert value not in sql
    assert params == {"limit": 6, "offset": 0, key: value}
    other = "file_type" if key == "status" else "status"
    assert f"{other} = " not in sql and other not in params
    # the filter narrows the scan BEFORE the page is cut from it, and the
    # total order survives filtering
    assert sql.index("WHERE") < sql.index("ORDER BY")
    assert "ORDER BY uploaded_at DESC, id DESC" in sql
    assert sql.index("ORDER BY") < sql.index("LIMIT")


def test_both_filters_combine_with_and_fully_parameterized():
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0,
                      status="kurgu-durum", file_type="kurgu-tur")

    sql, params = conn.cur.executed[0]
    assert "WHERE status = %(status)s AND file_type = %(file_type)s" in sql
    assert params == {"limit": 6, "offset": 0,
                      "status": "kurgu-durum", "file_type": "kurgu-tur"}
    assert "kurgu-durum" not in sql and "kurgu-tur" not in sql
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")


def test_a_quoting_hostile_filter_value_cannot_reach_the_statement_text():
    """The scenario an interpolated fragment would fall to: a value
    carrying quote syntax. It goes through exactly like any other value
    -- into the params dict, never into the SQL -- so the statement the
    server would prepare is the same closed text every call sends."""
    from pipeline.index import db

    hostile = "kurgu' OR uploaded_at > 'epoch"
    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, status=hostile)

    sql, params = conn.cur.executed[0]
    assert hostile not in sql and "'" not in sql
    assert params["status"] == hostile


@pytest.mark.parametrize(
    "kwargs",
    [{"status": ""}, {"file_type": ""}, {"status": 5},
     {"file_type": b"pdf"}, {"status": True}],
)
def test_the_listing_refuses_a_filter_it_cannot_read(kwargs):
    """The db seam mirrors the API's refusal for every OTHER caller: a
    filter that is not a non-empty string raises before any statement is
    built, so the refused call executes nothing at all."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    with pytest.raises(ValueError):
        db.list_documents(conn, limit=5, offset=0, **kwargs)
    assert conn.cur.executed == []


@pytest.mark.parametrize("key", ["status", "file_type"])
def test_a_long_filter_value_travels_as_a_parameter_like_any_other(key):
    """LENGTH IS NOT A SHAPE. Both columns are unbounded `text` in the
    schema, so a 128-character filter is an ordinary non-empty string:
    the seam accepts it, the clause it joins is the same static one, and
    the value goes to the database through the params dict without ever
    entering the statement text. The seam has never capped a filter's
    length, and this pins that -- the API layer's short-lived 64-character
    cap was the divergence, not this."""
    from pipeline.index import db

    long_value = "kurgu-" + "u" * 122
    assert len(long_value) == 128

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, **{key: long_value})

    sql, params = conn.cur.executed[0]
    assert f"{key} = %({key})s" in sql
    assert long_value not in sql
    # intact, not truncated at 64 or anywhere else
    assert params == {"limit": 6, "offset": 0, key: long_value}
    assert len(params[key]) == 128


# --- the upload window at the same seam ---------------------------------
#
# `documents.uploaded_at` is `timestamptz`, and psycopg adapts an aware
# datetime to it natively. The two properties a date filter can quietly
# get wrong are therefore: a timestamp RENDERED into the statement text
# (where the server would re-parse it under its own timezone setting),
# and a bound that is not an instant at all being compared as if it were.
# Both are checked here, at the cursor seam, where the statement and its
# parameters are still separable.

UTC = timezone.utc
INSTANT_ONCE = datetime(999, 1, 1, tzinfo=UTC)
INSTANT_SONRA = datetime(999, 4, 4, tzinfo=UTC)


class OffsetlessZone(tzinfo):
    """A tzinfo that is ATTACHED and still names no offset.

    Python calls such a value naive, and it is exactly as uncomparable as
    one carrying no tzinfo at all -- which is why the seam asks
    `utcoffset()` rather than `tzinfo is not None`. This class is what
    would slip through the weaker check."""

    def utcoffset(self, _dt):
        return None

    def dst(self, _dt):
        return None

    def tzname(self, _dt):
        return "KURGU"


@pytest.mark.parametrize(
    ("key", "clause"),
    [
        ("uploaded_after", "uploaded_at > %(uploaded_after)s"),
        ("uploaded_before", "uploaded_at < %(uploaded_before)s"),
    ],
)
def test_a_single_date_bound_travels_only_as_a_parameter(key, clause):
    """One bound alone: its static clause joins the statement -- with the
    EXCLUSIVE operator -- and its value never does, in any rendering. It
    reaches the database as the datetime OBJECT, through the params dict.
    The other bound appears nowhere at all."""
    from pipeline.index import db

    # a non-UTC offset on purpose: what travels is the instant, not the
    # spelling, and neither spelling may reach the statement text
    bound = datetime(999, 2, 2, 3, 0, tzinfo=timezone(timedelta(hours=3)))
    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, **{key: bound})

    sql, params = conn.cur.executed[0]
    assert clause in sql
    for rendering in (str(bound), repr(bound), bound.isoformat(),
                      str(bound.astimezone(UTC)),
                      bound.astimezone(UTC).isoformat(), "999", "0999"):
        assert rendering not in sql
    # the OBJECT itself, not a copy of its text
    assert params == {"limit": 6, "offset": 0, key: bound}
    assert params[key] is bound
    assert isinstance(params[key], datetime)
    other = "uploaded_before" if key == "uploaded_after" else "uploaded_after"
    assert other not in sql and other not in params
    # the window narrows the scan BEFORE the page is cut from it, and the
    # total order survives it
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert "ORDER BY uploaded_at DESC, id DESC" in sql
    # `limit + 1` is asked of the WINDOWED query
    assert params["limit"] == 6


def test_all_four_filters_combine_with_and_fully_parameterized():
    """Every filter this seam has, at once: four static clauses ANDed in
    the statement, four values in the params dict, none of them anywhere
    in the SQL text."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0,
                      status="kurgu-durum", file_type="kurgu-tur",
                      uploaded_after=INSTANT_ONCE,
                      uploaded_before=INSTANT_SONRA)

    sql, params = conn.cur.executed[0]
    assert ("WHERE status = %(status)s AND file_type = %(file_type)s "
            "AND uploaded_at > %(uploaded_after)s "
            "AND uploaded_at < %(uploaded_before)s " in sql)
    assert params == {"limit": 6, "offset": 0,
                      "status": "kurgu-durum", "file_type": "kurgu-tur",
                      "uploaded_after": INSTANT_ONCE,
                      "uploaded_before": INSTANT_SONRA}
    assert "kurgu-durum" not in sql and "kurgu-tur" not in sql
    for instant in (INSTANT_ONCE, INSTANT_SONRA):
        assert str(instant) not in sql
        assert instant.isoformat() not in sql
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")


def test_an_unsupplied_date_bound_appears_in_neither_statement_nor_params():
    """An absent bound adds no clause and no key -- checked next to a
    filter that IS supplied, so the statement provably still assembles."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, status="kurgu-durum")

    sql, params = conn.cur.executed[0]
    assert "uploaded_at >" not in sql and "uploaded_at <" not in sql
    assert "uploaded_after" not in sql and "uploaded_before" not in sql
    assert params == {"limit": 6, "offset": 0, "status": "kurgu-durum"}


@pytest.mark.parametrize(
    "kwargs",
    [
        # TEXT is refused, not parsed: turning a string into an instant
        # needs a timezone policy, that policy belongs to the layer
        # talking to the caller, and a second one here would be a second
        # answer to the same question
        {"uploaded_after": "0999-01-01T00:00:00Z"},
        {"uploaded_before": "0999-01-01T00:00:00+00:00"},
        # naive: nobody said which instant this is
        {"uploaded_after": datetime(999, 1, 1)},
        {"uploaded_before": datetime(999, 1, 1)},
        # tzinfo attached, no offset answered -- naive by every rule that
        # matters, and what `tzinfo is not None` would have admitted
        {"uploaded_after": datetime(999, 1, 1, tzinfo=OffsetlessZone())},
        {"uploaded_before": datetime(999, 1, 1, tzinfo=OffsetlessZone())},
        # a date is not a datetime, and a number is not either
        {"uploaded_after": date(999, 1, 1)},
        {"uploaded_before": 0},
    ],
)
def test_the_listing_refuses_a_date_bound_it_cannot_read(kwargs):
    """The seam takes None or an AWARE datetime and nothing else. The
    refusal happens before any statement is built, so the refused call
    executes nothing at all."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    with pytest.raises(ValueError):
        db.list_documents(conn, limit=5, offset=0, **kwargs)
    assert conn.cur.executed == []


@pytest.mark.parametrize(
    "kwargs",
    [
        # equal: both bounds are exclusive, so this window can hold no row
        {"uploaded_after": INSTANT_ONCE, "uploaded_before": INSTANT_ONCE},
        # the same instant carried by a different offset -- equality here
        # is between INSTANTS, so a differing spelling changes nothing
        {"uploaded_after": INSTANT_ONCE,
         "uploaded_before": INSTANT_ONCE.astimezone(
             timezone(timedelta(hours=-5)))},
        # reversed
        {"uploaded_after": INSTANT_SONRA, "uploaded_before": INSTANT_ONCE},
    ],
)
def test_the_listing_refuses_an_empty_or_reversed_window(kwargs):
    """A window that can match nothing is a mistake to report, not an
    empty page to hand back -- and it is reported before a statement is
    built, so the refused call executes nothing."""
    from pipeline.index import db

    # the premise of each case, stated rather than assumed
    assert kwargs["uploaded_after"] >= kwargs["uploaded_before"]

    conn = RecordingConn([_listing_row()])
    with pytest.raises(ValueError):
        db.list_documents(conn, limit=5, offset=0, **kwargs)
    assert conn.cur.executed == []


def test_a_window_whose_bounds_are_spelled_differently_is_still_one_window():
    """The positive half of the rule above: the same two instants written
    with different offsets are accepted when they really do bound a
    window, and each reaches the database as its own object."""
    from pipeline.index import db

    after = INSTANT_ONCE.astimezone(timezone(timedelta(hours=3)))
    before = INSTANT_SONRA.astimezone(timezone(timedelta(hours=-5)))
    assert after == INSTANT_ONCE and before == INSTANT_SONRA

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0,
                      uploaded_after=after, uploaded_before=before)

    sql, params = conn.cur.executed[0]
    assert ("WHERE uploaded_at > %(uploaded_after)s "
            "AND uploaded_at < %(uploaded_before)s " in sql)
    assert params == {"limit": 6, "offset": 0,
                      "uploaded_after": after, "uploaded_before": before}
