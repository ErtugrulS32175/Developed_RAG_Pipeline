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


# --- the filename search at the same seam --------------------------------
#
# MEASURED: no live PostgreSQL is reachable in this loop -- the local
# Docker daemon is down and CI declares no database service -- so the real
# server's `ILIKE ... ESCAPE` semantics execute NOWHERE here. Asserting on
# the statement text alone would therefore prove only that some string was
# assembled, never that the pattern MEANS what it must.
#
# `_ilike` is the missing half. It is an INDEPENDENT model of the SUBSET
# of the operator these fixtures exercise, written from the operator's
# rules and knowing nothing about the transform under test: the escape
# character consumes the next character literally, `%` is any run of
# characters, `_` is exactly one, the match is anchored, and ASCII letters
# compare case-insensitively. Feeding it the pattern the PRODUCTION seam
# actually built is a real round-trip -- the transform is proven THROUGH a
# LIKE interpreter, not against a hard-coded string.
#
# WHAT IT DOES NOT PROVE, and must not be read as proving: it is not
# PostgreSQL. Case folding here is Python's, under the process's own
# rules; the server's is decided by the column's collation, which this
# model neither knows nor consults. So the metacharacter, escape and
# anchoring behaviour it pins is trustworthy for the ASCII fixtures in
# this battery, and any claim about real collation or full Unicode case
# folding is NOT established here and waits on a live server.


def _ilike(value, pattern, escape="!"):
    """`value ILIKE pattern ESCAPE escape`, over the ASCII subset.

    Models the pattern language -- escape, `%`, `_`, anchoring -- plus
    ASCII case-insensitivity. It is NOT a PostgreSQL implementation: the
    server folds case by the column's collation, this folds it by
    Python's rules. See the note above the definition."""
    import re

    parts = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == escape:
            index += 1
            if index >= len(pattern):
                # the server's own error: a pattern may not end on a lone
                # escape character
                raise ValueError("LIKE deseni kacis karakteriyle bitemez")
            parts.append(re.escape(pattern[index]))
        elif char == "%":
            parts.append("(?s:.)*")
        elif char == "_":
            parts.append("(?s:.)")
        else:
            parts.append(re.escape(char))
        index += 1
    return re.fullmatch("".join(parts), value, re.IGNORECASE) is not None


def _pattern_sent(q):
    """The pattern the PRODUCTION seam puts in the params dict for `q`.

    Everything below reads the pattern from a real call rather than
    rebuilding it, so the substring wrapping is part of what is proven
    and not part of the test's own assumptions."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q=q)
    return conn.cur.executed[0][1]["filename_search"]


def _escape_last(value):
    """The order this contract REFUSES: `%` and `_` first, the escape
    character last -- which re-escapes the escape characters the earlier
    steps just inserted."""
    return value.replace("%", "!%").replace("_", "!_").replace("!", "!!")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("kurgu", "kurgu"),                 # nothing to escape
        ("%", "!%"),
        ("_", "!_"),
        ("!", "!!"),                        # the escape character itself
        ("!%_", "!!!%!_"),
        ("kurgu%100_rapor!", "kurgu!%100!_rapor!!"),
    ],
)
def test_the_search_escape_helper_is_pure_and_returns_a_new_value(
        value, expected):
    """One helper, no side effects: it reads its argument, returns a new
    value, and leaves the argument exactly as it was. Being directly
    callable is the point -- the transform is the part that can be got
    subtly wrong, so it is testable without a statement, a cursor or a
    connection anywhere near it."""
    from pipeline.index import db

    before = "".join(value)
    result = db.escape_like_pattern(value)

    assert result == expected
    assert value == before                  # the input was not mutated
    if expected != value:
        assert result is not value          # a NEW value came back


def test_a_single_exclamation_becomes_a_doubled_one():
    """The escape character is not exempt from escaping: a caller looking
    for `!` in a name means the character, and an unescaped one in the
    pattern would instead swallow whatever follows it."""
    from pipeline.index import db

    assert db.escape_like_pattern("!") == "!!"
    assert db.escape_like_pattern("kurgu!") == "kurgu!!"


def test_escaping_the_escape_character_last_would_break_a_literal_search():
    """THE ORDER TEST. Exclamation, then percent, then underscore -- and
    this fails if that changes.

    MEASURED: doing the escape character LAST double-escapes what the
    earlier steps inserted. `%` becomes `!%` and then `!!%`, which the
    server reads as a literal `!` followed by the WILDCARD -- so the
    search silently matches on `!` instead of `%`, or finds nothing at
    all. Both orders are run through the interpreter here, so the
    difference is a demonstrated match/no-match rather than two strings
    that merely differ."""
    from pipeline.index import db

    assert db.escape_like_pattern("%") == "!%"
    assert _escape_last("%") == "!!%"
    assert db.escape_like_pattern("%") != _escape_last("%")

    carries_percent = "kurgu-%100-rapor.pdf"
    carries_bang = "kurgu-!100-rapor.pdf"

    right = _pattern_sent("%")
    assert right == "%!%%"
    assert _ilike(carries_percent, right)
    assert not _ilike(carries_bang, right)

    # the refused order, through the same interpreter: it stops matching
    # the percent it was asked about and starts matching an exclamation
    # nobody searched for
    wrong = "%" + _escape_last("%") + "%"
    assert not _ilike(carries_percent, wrong)
    assert _ilike(carries_bang, wrong)


NAMES = [
    "kurgu-rapor.pdf",          # none of the three characters
    "kurgu-%100-rapor.pdf",     # a real percent
    "kurgu_100_rapor.pdf",      # a real underscore
    "kurgu-!100-rapor.pdf",     # a real exclamation
]


@pytest.mark.parametrize(
    ("q", "matching"),
    [
        ("%", ["kurgu-%100-rapor.pdf"]),
        ("_", ["kurgu_100_rapor.pdf"]),
        ("!", ["kurgu-!100-rapor.pdf"]),
    ],
)
def test_a_metacharacter_search_matches_only_that_literal_character(
        q, matching):
    """Proven THROUGH the interpreter, which is the only place the
    operator's semantics exist in this loop.

    `%` must not match every name, `_` must not match an arbitrary single
    character, and `!` -- the escape character -- must find itself. Each
    case is checked positively AND negatively against the same four
    names, so "matches only these" is measured rather than asserted."""
    pattern = _pattern_sent(q)

    matched = [name for name in NAMES if _ilike(name, pattern)]

    assert matched == matching
    assert len(matched) < len(NAMES)        # it did not match everything


def test_a_combined_metacharacter_value_is_matched_character_for_character():
    """All three at once, in one value: the pattern is the literal run
    and nothing else -- not the same characters in another order, and not
    a wildcard standing in for any of them."""
    pattern = _pattern_sent("!%_")

    assert pattern == "%!!!%!_%"
    assert _ilike("kurgu-!%_-rapor.pdf", pattern)
    assert not _ilike("kurgu-%_!-rapor.pdf", pattern)      # another order
    assert not _ilike("kurgu-!x_-rapor.pdf", pattern)      # `%` as wildcard
    assert not _ilike("kurgu-rapor.pdf", pattern)


def test_a_search_that_already_looks_escaped_stays_literal():
    """`!%` is what an escaped percent looks like in a PATTERN, but as
    INPUT it is two ordinary characters. It must not be read as an escape
    the caller wrote, and it must not be un-escaped: the value is data,
    the pattern is ours."""
    pattern = _pattern_sent("!%")

    assert pattern == "%!!!%%"
    assert _ilike("kurgu-!%-rapor.pdf", pattern)
    assert not _ilike("kurgu-%-rapor.pdf", pattern)        # the `!` is real
    assert not _ilike("kurgu-!-rapor.pdf", pattern)        # so is the `%`


def test_a_search_travels_only_as_a_parameter():
    """The clause is static text: the column, the operator and the escape
    character are written in the code. The VALUE reaches the database
    only as the transformed pattern in the params dict, and the raw one
    appears nowhere in the statement."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q="Kurgu-Rapor")

    sql, params = conn.cur.executed[0]
    assert "filename ILIKE %(filename_search)s ESCAPE '!'" in sql
    assert "Kurgu-Rapor" not in sql
    assert params == {"limit": 6, "offset": 0,
                      "filename_search": "%Kurgu-Rapor%"}
    # the search narrows the scan BEFORE the page is cut from it, the
    # total order survives it, and `limit + 1` is asked of the SEARCHED
    # query
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert "ORDER BY uploaded_at DESC, id DESC" in sql
    assert params["limit"] == 6
    # the projection is untouched by a search
    assert "content_sha256" not in sql and "candidate_id" not in sql


def test_the_escape_character_and_the_clause_are_pinned_separately():
    """The clause is ONE literal string, so it no longer derives from the
    constant the transform uses -- and two things that must agree but do
    not derive from each other can drift. Each is therefore pinned on its
    own: the constant the escaping applies, and the exact clause text the
    server is asked to read it with. If either is edited alone, this
    fails."""
    from pipeline.index import db

    assert db.DOCUMENT_SEARCH_ESCAPE == "!"

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q="kurgu")

    sql, _ = conn.cur.executed[0]
    assert "filename ILIKE %(filename_search)s ESCAPE '!'" in sql


@pytest.mark.parametrize(
    "hostile",
    [
        "kurgu' OR uploaded_at > 'epoch",   # quote syntax
        "kurgu\\ters",                      # a backslash
        "kurgu'; DROP TABLE documents; --",
    ],
)
def test_a_hostile_search_value_cannot_reach_the_statement_text(hostile):
    """Parameterization stops injection; escaping fixes pattern meaning.
    This is the first half: whatever the value carries, the statement the
    server would prepare is the same closed text every call sends.

    The BACKSLASH case is the one worth naming -- the escape authority
    here is the exclamation mark and nothing else, so a backslash is an
    ordinary character that is neither doubled nor consumed."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q=hostile)

    sql, params = conn.cur.executed[0]
    assert hostile not in sql
    assert "'" not in sql.replace("ESCAPE '!'", "")
    # What this test pins is CARRIAGE, not the escape algorithm: the
    # value reaches the database as a parameter and nothing else. So the
    # expectation is built through the production helper rather than
    # assuming the raw text survives -- one of these values contains an
    # underscore, which IS a LIKE metacharacter and is escaped exactly as
    # the dedicated escape tests require. The backslash is not: the
    # escape authority here is the exclamation mark and nothing else.
    assert params["filename_search"] == "%" + db.escape_like_pattern(
        hostile) + "%"
    assert _ilike("kurgu" + hostile + ".pdf", params["filename_search"])


def test_a_non_ascii_search_travels_safely_through_the_params_dict():
    """`filename` is `text` and the API already accepts non-ASCII names,
    so a search for one is an ordinary search: it is not transformed, not
    normalised and not refused -- it reaches the database as a value.

    THAT CARRIAGE IS THE WHOLE CLAIM. How the server would match this
    value is decided by the column's collation and its Unicode case
    folding, neither of which runs here; the round-trip below differs
    only in ASCII case, so it exercises carriage and anchoring rather
    than Unicode folding. Non-ASCII matching semantics wait on a live
    server."""
    from pipeline.index import db

    q = "belge-özet-Şubat"
    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q=q)

    sql, params = conn.cur.executed[0]
    assert q not in sql
    assert params["filename_search"] == "%belge-özet-Şubat%"
    assert _ilike("KURGU belge-özet-Şubat raporu.pdf",
                  params["filename_search"])


def test_a_long_search_value_is_a_valid_search():
    """LENGTH IS NOT A SHAPE HERE EITHER. MEASURED: there is no length
    authority for `filename` anywhere -- the column is unbounded `text`,
    the upload validator has no length check, and UPLOAD_MAX_BYTES is a
    body cap. So no maximum is invented for `q`."""
    from pipeline.index import db

    long_value = "kurgu-" + "u" * 250
    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q=long_value)

    _sql, params = conn.cur.executed[0]
    assert params["filename_search"] == "%" + long_value + "%"
    assert len(params["filename_search"]) == 258


@pytest.mark.parametrize(
    "q",
    ["", 5, b"kurgu", True, ["kurgu"], 0.5],
)
def test_the_listing_refuses_a_search_it_cannot_read(q):
    """The seam takes None or a NON-EMPTY STRING and nothing else, and it
    refuses before any statement is built -- so the refused call executes
    nothing at all. That matters more here than for the equality filters:
    the value is about to be TRANSFORMED into a pattern, and a transform
    on a value nobody checked is how a non-string reaches `.replace`.

    The refusal is pinned to THIS gate by its message, so a test that
    passes because some earlier check (the page bounds, the window)
    happened to fire is not mistaken for this one -- and the control at
    the end runs the identical call with a well-formed search, which
    reaches the cursor exactly once."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    with pytest.raises(ValueError, match="^q "):
        db.list_documents(conn, limit=5, offset=0, q=q)
    # nothing was built and nothing was executed: zero statements
    assert conn.cur.executed == []

    control = RecordingConn([_listing_row()])
    db.list_documents(control, limit=5, offset=0, q="kurgu")
    assert len(control.cur.executed) == 1


def test_an_unsupplied_search_appears_in_neither_statement_nor_params():
    """No `q` means no filename clause and no filename key -- checked
    next to a filter that IS supplied, so the statement provably still
    assembles."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, status="kurgu-durum")

    sql, params = conn.cur.executed[0]
    assert "ILIKE" not in sql and "ESCAPE" not in sql
    assert "filename_search" not in sql
    assert params == {"limit": 6, "offset": 0, "status": "kurgu-durum"}


@pytest.mark.parametrize(
    ("kwargs", "clause"),
    [
        ({"status": "kurgu-durum"}, "status = %(status)s"),
        ({"file_type": "kurgu-tur"}, "file_type = %(file_type)s"),
        ({"uploaded_after": INSTANT_ONCE}, "uploaded_at > %(uploaded_after)s"),
        ({"uploaded_before": INSTANT_SONRA},
         "uploaded_at < %(uploaded_before)s"),
    ],
)
def test_a_search_ands_with_each_existing_filter(kwargs, clause):
    """One existing filter at a time, each ANDed with the search: both
    clauses are in the statement, both values are in the params dict, and
    neither value is in the SQL."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0, q="kurgu", **kwargs)

    sql, params = conn.cur.executed[0]
    assert clause + " AND filename ILIKE %(filename_search)s ESCAPE '!'" in sql
    assert params["filename_search"] == "%kurgu%"
    for key, value in kwargs.items():
        assert params[key] == value
        assert str(value) not in sql
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")


def test_all_five_filters_combine_with_and_fully_parameterized():
    """Every filter this seam has, at once: five static clauses ANDed in
    the statement, five values in the params dict, none of them anywhere
    in the SQL text. The date bounds keep their EXCLUSIVE operators next
    to the search, so the boundary behaviour is unchanged by it."""
    from pipeline.index import db

    conn = RecordingConn([_listing_row()])
    db.list_documents(conn, limit=5, offset=0,
                      status="kurgu-durum", file_type="kurgu-tur",
                      uploaded_after=INSTANT_ONCE,
                      uploaded_before=INSTANT_SONRA,
                      q="kurgu%rapor")

    sql, params = conn.cur.executed[0]
    assert ("WHERE status = %(status)s AND file_type = %(file_type)s "
            "AND uploaded_at > %(uploaded_after)s "
            "AND uploaded_at < %(uploaded_before)s "
            "AND filename ILIKE %(filename_search)s ESCAPE '!' " in sql)
    assert params == {"limit": 6, "offset": 0,
                      "status": "kurgu-durum", "file_type": "kurgu-tur",
                      "uploaded_after": INSTANT_ONCE,
                      "uploaded_before": INSTANT_SONRA,
                      "filename_search": "%kurgu!%rapor%"}
    assert "kurgu-durum" not in sql and "kurgu-tur" not in sql
    assert "kurgu%rapor" not in sql
    for instant in (INSTANT_ONCE, INSTANT_SONRA):
        assert str(instant) not in sql
        assert instant.isoformat() not in sql
    assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")


# --- the retrieval scope's own contract ----------------------------------
#
# `hybrid_search` runs TWO statements over one code-owned WHERE clause and
# fuses their rankings with RRF. Fusion cannot tell which ranking a row
# arrived on, so a scope that reached only one statement would let the
# other one carry an out-of-scope candidate into the fused result -- and
# then into the reranker, the context and a citation. These tests are at
# the cursor seam, where the statement and its parameters are still
# separable, and the fake cursor APPLIES the scope the way the server would
# so both directions can be proven: what the scope keeps out, and that it
# really would have come in without it.

IC_BELGE = "11111111-1111-1111-1111-111111111111"
DIS_BELGE = "22222222-2222-2222-2222-222222222222"
YOK_BELGE = "33333333-3333-3333-3333-333333333333"


def _chunk_row(chunk_id, document_id, filename):
    return {
        "id": chunk_id,
        "type": "text",
        "text": f"{filename} icindeki kurgu pasaj.",
        "source_tag": "kurgu",
        "page": 7,
        "headings": [],
        "table_data": None,
        "document_id": document_id,
        "filename": filename,
    }


CORPUS = [
    _chunk_row("ic-1", IC_BELGE, "kapsam-icinde.pdf"),
    _chunk_row("dis-1", DIS_BELGE, "kapsam-disinda.pdf"),
    # a legacy chunk with no document row at all: reachable as it has always
    # been, but it belongs to no NAMED document, so a scope must exclude it
    _chunk_row("eski-1", None, None),
]


class HybridCursor:
    """Records both statements AND answers them from a tiny corpus.

    Recording alone proves the clause was sent; answering proves what the
    clause does. The scope is applied here exactly as the server would
    apply it -- from the statement text and the parameter that came with
    it -- so a test can assert the row that stayed out really would have
    come back without the clause.
    """

    def __init__(self, rows, scope_clause):
        self._rows = rows
        self._scope_clause = scope_clause
        self._result = []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        rows = list(self._rows)
        if self._scope_clause in sql:
            scope = set(params[0])
            rows = [row for row in rows if row["document_id"] in scope]
        self._result = rows

    def fetchall(self):
        return [dict(row) for row in self._result]


class HybridConn:
    def __init__(self, rows, scope_clause):
        self.cur = HybridCursor(rows, scope_clause)

    def cursor(self, row_factory=None):
        return self.cur


def _hybrid(rows=CORPUS, **kwargs):
    """Run one hybrid search over a fake corpus; return (rows, cursor)."""
    from pipeline.index import db

    conn = HybridConn(rows, db.DOCUMENT_SCOPE_CLAUSE)
    found = db.hybrid_search(conn, [0.0], [1], [1.0], top_k=5, **kwargs)
    return found, conn.cur


def test_an_unscoped_hybrid_search_sends_no_clause_and_no_parameter():
    """Today's behaviour, unchanged: absent means absent in BOTH the
    statement and the parameter tuple, not a scope that happens to be
    empty."""
    from pipeline.index import db

    _found, cur = _hybrid()

    assert len(cur.executed) == 2
    for sql, params in cur.executed:
        assert db.DOCUMENT_SCOPE_CLAUSE not in sql
        assert "document_id = ANY" not in sql
        # only the ranking vector and the page size travel
        assert len(params) == 2


def test_both_hybrid_statements_carry_the_SAME_scope_clause():
    """One clause, written once, sent twice. If the dense statement were
    scoped and the sparse one were not, a candidate from outside the scope
    would enter the fusion through the sparse ranking and nothing later
    could tell it apart from a legitimate one."""
    from pipeline.index import db

    _found, cur = _hybrid(document_ids=(IC_BELGE,))

    dense_sql, dense_params = cur.executed[0]
    sparse_sql, sparse_params = cur.executed[1]
    assert db.DOCUMENT_SCOPE_CLAUSE in dense_sql
    assert db.DOCUMENT_SCOPE_CLAUSE in sparse_sql
    assert dense_sql.count(db.DOCUMENT_SCOPE_CLAUSE) == 1
    assert sparse_sql.count(db.DOCUMENT_SCOPE_CLAUSE) == 1
    # the same scope, bound the same way, on both
    assert dense_params[0] == [IC_BELGE]
    assert sparse_params[0] == [IC_BELGE]
    # and the two statements differ ONLY in how they rank
    assert (dense_sql.split("ORDER BY")[0]
            == sparse_sql.split("ORDER BY")[0])


def test_the_scope_clause_is_static_text_and_the_identifiers_are_values():
    """No identifier is ever interpolated. The proof is a value carrying
    quote syntax: it travels as a parameter like any other value, so the
    statement the server prepares is the same closed text every call
    sends."""
    from pipeline.index import db

    hostile = "11111111-1111-1111-1111-111111111111' OR '1'='1"
    _found, cur = _hybrid(document_ids=(hostile,))

    for sql, params in cur.executed:
        assert hostile not in sql
        assert IC_BELGE not in sql
        assert params[0] == [hostile]
    # the clause the code owns, spelled out with a placeholder and nothing
    # else -- pinned here rather than derived from the statement it is in
    assert db.DOCUMENT_SCOPE_CLAUSE == "c.document_id = ANY(%s::uuid[])"


def test_the_scope_narrows_the_query_not_the_candidates_it_returned():
    """Inside the statement, ahead of the cut. A scope applied after LIMIT
    would answer a scoped question with whatever survived an UNSCOPED
    top-k, so a document that really holds the answer would come back
    empty whenever other documents filled the pool first."""
    _found, cur = _hybrid(document_ids=(IC_BELGE,))

    for sql, _params in cur.executed:
        assert sql.index("WHERE") < sql.index("ORDER BY") < sql.index("LIMIT")


def test_a_document_outside_the_scope_would_have_matched_without_it():
    """The negative and the positive in one test. Without the scope the
    other document's passage comes back; with it, it does not -- so the
    scoped result is evidence about the clause rather than about a corpus
    that happened to hold one document."""
    unscoped, _ = _hybrid()
    scoped, _ = _hybrid(document_ids=(IC_BELGE,))

    assert {row["filename"] for row in unscoped} == {
        "kapsam-icinde.pdf", "kapsam-disinda.pdf", None}
    assert [row["filename"] for row in scoped] == ["kapsam-icinde.pdf"]


def test_several_identifiers_return_exactly_that_set():
    scoped, _ = _hybrid(document_ids=(IC_BELGE, DIS_BELGE))

    assert {row["filename"] for row in scoped} == {
        "kapsam-icinde.pdf", "kapsam-disinda.pdf"}
    # the legacy row belongs to no named document and is therefore outside
    # every scope, even one naming every document there is
    assert None not in {row["filename"] for row in scoped}


def test_an_unknown_identifier_yields_an_empty_scope_not_the_corpus():
    from pipeline.index import db

    scoped, cur = _hybrid(document_ids=(YOK_BELGE,))

    assert scoped == []
    # it did not fall back: both statements were still scoped
    for sql, params in cur.executed:
        assert db.DOCUMENT_SCOPE_CLAUSE in sql
        assert params[0] == [YOK_BELGE]


def test_a_mixed_scope_is_scoped_to_the_known_identifier_alone():
    scoped, _ = _hybrid(document_ids=(IC_BELGE, YOK_BELGE))

    assert [row["filename"] for row in scoped] == ["kapsam-icinde.pdf"]


def test_the_generation_filter_is_parenthesised_so_the_scope_binds_to_all():
    """The existing filter is an OR. ANDing the scope onto a bare `a OR b`
    would bind to `b` alone and leave every legacy NULL-document row
    reachable from outside the requested scope -- which is why the row is
    in the corpus above and why it is asserted twice."""
    unscoped, _ = _hybrid()
    scoped, cur = _hybrid(document_ids=(IC_BELGE,))

    assert any(row["document_id"] is None for row in unscoped)
    assert all(row["document_id"] is not None for row in scoped)
    for sql, _params in cur.executed:
        assert "WHERE (c.document_id IS NULL OR d.id IS NOT NULL) AND " in sql


# --- resolving identifiers to filenames ----------------------------------
#
# The LlamaIndex index carries `{page, type, filename}` and no identifier,
# so a scope stated in ids can only reach it as names. Resolution is a
# SELECT against the same `documents` table everything else keys on, which
# is what keeps the id the one authority for what a document is.


class NameCursor:
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
        return list(self._rows)


class NameConn:
    def __init__(self, rows=()):
        self.cur = NameCursor(rows)

    def cursor(self, row_factory=None):
        return self.cur


def test_the_filename_resolution_is_a_parameterised_select_only_query():
    from pipeline.index import db

    conn = NameConn([("kapsam-icinde.pdf",)])
    db.filenames_for_documents(conn, (IC_BELGE, DIS_BELGE))

    sql, params = conn.cur.executed[0]
    assert sql.startswith("SELECT filename FROM documents")
    assert "= ANY(%s::uuid[])" in sql
    # the whole set travels as ONE array parameter; neither identifier is
    # anywhere in the statement
    assert params == ([IC_BELGE, DIS_BELGE],)
    assert IC_BELGE not in sql and DIS_BELGE not in sql
    upper = sql.upper()
    for verb in ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE", "DROP"):
        assert verb not in upper


def test_resolved_filenames_come_back_deduplicated_and_ordered():
    """Two ids resolving to one name produce ONE filter value, so a
    repetition cannot become a repeated filter further down."""
    from pipeline.index import db

    conn = NameConn([("b.pdf",), ("a.pdf",), ("b.pdf",)])

    assert db.filenames_for_documents(conn, (IC_BELGE, DIS_BELGE)) == [
        "a.pdf", "b.pdf"]


def test_an_unresolvable_identifier_resolves_to_no_name_at_all():
    """An empty list is an EMPTY SCOPE, never "no scope" -- the caller is
    the one that must not widen, and it cannot widen what it never got."""
    from pipeline.index import db

    conn = NameConn([])

    assert db.filenames_for_documents(conn, (YOK_BELGE,)) == []
