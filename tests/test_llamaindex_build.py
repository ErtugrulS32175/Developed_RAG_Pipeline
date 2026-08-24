"""build_index is a SHADOW build with an atomic swap, never an append and
never a drop-first.

Round 17 found the append (two builds doubled the rows); round 18 found
the cure worse than honest: dropping the live table BEFORE building meant
a failed build destroyed the old comparison index and delivered nothing.
The build now lands in a shadow table, verifies, and swaps in one
transaction -- a failed build leaves the previous snapshot serving. These
tests pin the ordering without needing the optional package installed.
"""
import sys
import types

import pytest


TENANT = "00000000-0000-0000-0000-000000000001"
IC_BELGE = "11111111-1111-1111-1111-111111111111"
DIS_BELGE = "22222222-2222-2222-2222-222222222222"
YOK_BELGE = "33333333-3333-3333-3333-333333333333"
IC_SURUM = "44444444-4444-4444-4444-444444444444"
DIS_SURUM = "55555555-5555-5555-5555-555555555555"
CHUNK = "66666666-6666-6666-6666-666666666666"
IC_KEY = f"{TENANT}:{IC_BELGE}:{IC_SURUM}:7"
DIS_KEY = f"{TENANT}:{DIS_BELGE}:{DIS_SURUM}:9"
ESKI_KEY = f"{TENANT}:{IC_BELGE}:{DIS_SURUM}:7"


@pytest.fixture
def fake_llama(monkeypatch):
    events = []
    core = types.ModuleType("llama_index.core")

    class Document:
        def __init__(self, text=None, metadata=None):
            self.text = text
            # kept so a test can read what the index would actually carry:
            # the scope route depends on that metadata and on nothing else
            self.metadata = metadata

    class StorageContext:
        @staticmethod
        def from_defaults(vector_store=None):
            return object()

    class VectorStoreIndex:
        @staticmethod
        def from_documents(docs, storage_context=None, show_progress=None):
            events.append(("yaz", len(docs)))
            for doc in docs:
                events.append(("meta", doc.metadata))

    core.Document = Document
    core.StorageContext = StorageContext
    core.VectorStoreIndex = VectorStoreIndex
    monkeypatch.setitem(sys.modules, "llama_index",
                        types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", core)
    return events, core


def _wire_db(monkeypatch, events, shadow_count=1):
    from pipeline.index import db
    from pipeline.retrieval import rag_llamaindex

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            self._last = str(sql)
            events.append(("sql", self._last))

        def fetchall(self):
            return [(
                "kurgu metin", 1, "text", "kurgu.pdf", CHUNK, IC_KEY)]

        def fetchone(self):
            return (shadow_count,)

        def executemany(self, sql, params):
            events.append(("many", str(sql) + " " + repr(tuple(params))))

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            events.append(("commit", None))

        def close(self):
            events.append(("kapat", None))

    monkeypatch.setattr(rag_llamaindex, "_configure_models", lambda: None)
    monkeypatch.setattr(rag_llamaindex, "_store",
                        lambda table_name=None: object())
    def get_conn(**kwargs):
        events.append(("baglam", kwargs))
        return Conn()

    monkeypatch.setattr(db, "get_conn", get_conn)
    return rag_llamaindex


def test_build_lands_in_a_shadow_and_swaps_after_the_write(
        monkeypatch, fake_llama):
    events, _core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events)

    rag_llamaindex.build_index()

    sqls = [text for kind, text in events if kind == "sql"]
    target = f"data_{rag_llamaindex.TABLE}"
    shadow = f"data_{rag_llamaindex.TABLE}_kurulum"
    # first statement clears a DEAD SHADOW, never the live table
    assert "DROP TABLE IF EXISTS" in sqls[0] and shadow in sqls[0]
    assert target not in sqls[0].replace(shadow, "")
    # the live table is touched only AFTER the write succeeded
    write_at = events.index(("yaz", 1))
    target_drop_at = next(
        i for i, (kind, text) in enumerate(events)
        if kind == "sql" and "DROP TABLE" in text
        and shadow not in text and target in text)
    rename_at = next(
        i for i, (kind, text) in enumerate(events)
        if kind == "sql" and "RENAME TO" in text)
    assert write_at < target_drop_at < rename_at
    assert shadow in events[rename_at][1] and target in events[rename_at][1]
    # and the connection's whole life is bounded
    assert events[-1] == ("kapat", None)
    assert ("baglam", {"service": True}) in events


def test_a_failed_write_leaves_the_old_index_untouched(
        monkeypatch, fake_llama):
    events, core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events)

    def broken(docs, storage_context=None, show_progress=None):
        raise RuntimeError("kurgu yazma hatasi")

    core.VectorStoreIndex.from_documents = broken
    with pytest.raises(RuntimeError, match="kurgu yazma"):
        rag_llamaindex.build_index()

    target = f"data_{rag_llamaindex.TABLE}"
    shadow = f"data_{rag_llamaindex.TABLE}_kurulum"
    for kind, text in events:
        if kind == "sql" and shadow not in str(text):
            assert target not in str(text)   # the live table was never named
    assert ("kapat", None) in events


def test_an_empty_shadow_refuses_the_swap(monkeypatch, fake_llama):
    events, _core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events, shadow_count=0)

    with pytest.raises(RuntimeError, match="golge tablo bos"):
        rag_llamaindex.build_index()

    assert not any("RENAME TO" in text for kind, text in events
                   if kind == "sql")


def test_the_copy_reads_every_retained_ready_version_and_safe_legacy(
        monkeypatch, fake_llama):
    """Rollback needs old ready builds in the snapshot, never staged rows."""
    events, _core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events)

    rag_llamaindex.build_index()

    select = next(text for kind, text in events
                  if kind == "sql" and text.startswith("SELECT c.text"))
    assert "FROM document_version_builds b" in select
    assert "c.version_id = b.version_id" in select
    assert "c.generation = b.generation" in select
    assert "UNION ALL" in select
    assert "c.version_id IS NULL" in select
    assert "d.active_version_id IS NULL" in select
    assert "c.generation = d.active_generation" in select
    assert "d.archived_at IS NULL" not in select
    assert "c.version_id::text || ':' || c.generation::text" in select
    assert "':legacy:'" in select


def test_the_index_metadata_carries_tenant_qualified_scope_authority(
        monkeypatch, fake_llama):
    """A global snapshot never treats a tenant-local filename as authority."""
    events, _core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events)

    rag_llamaindex.build_index()

    metadata = [meta for kind, meta in events if kind == "meta"]
    assert metadata == [{
        "page": 1,
        "type": "text",
        "filename": "kurgu.pdf",
        "chunk_id": CHUNK,
        "scope_key": IC_KEY,
    }]
    for meta in metadata:
        assert set(meta) == {
            "page", "type", "filename", "chunk_id", "scope_key"}


def test_vector_and_scope_manifest_swap_under_one_exclusive_lock(
        monkeypatch, fake_llama):
    events, _core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events)

    rag_llamaindex.build_index()

    scope_shadow = rag_llamaindex._scope_table(
        rag_llamaindex.TABLE + "_kurulum")
    scope_live = rag_llamaindex._scope_table()
    insert = next(text for kind, text in events if kind == "many")
    assert scope_shadow in insert and IC_KEY in insert
    lock_at = next(
        index for index, (kind, text) in enumerate(events)
        if kind == "sql" and "pg_advisory_xact_lock(hashtext" in text)
    first_live_drop = next(
        index for index, (kind, text) in enumerate(events)
        if kind == "sql" and "DROP TABLE" in text
        and (f"data_{rag_llamaindex.TABLE}" in text or scope_live in text)
        and "kurulum" not in text
    )
    renames = [
        text for kind, text in events
        if kind == "sql" and "RENAME TO" in text
    ]
    assert lock_at < first_live_drop
    assert len(renames) == 2
    assert any(scope_shadow in text and scope_live in text for text in renames)


# --- scoping the alternative engine --------------------------------------
#
# The route is fixed by the measurement above: resolve the identifiers to
# tenant-qualified keys through `documents`, then filter on that metadata AT
# CONSTRUCTION. These tests use the same fake-module strategy as the build
# tests -- no optional package is installed, and nothing is skipped or
# xfailed.

@pytest.fixture
def fake_filters(monkeypatch):
    """The metadata-filter API, faked at the seam the engine imports it."""
    module = types.ModuleType("llama_index.core.vector_stores.types")

    class FilterOperator:
        IN = "in"

    class MetadataFilter:
        def __init__(self, key=None, value=None, operator=None):
            self.key = key
            self.value = value
            self.operator = operator

    class MetadataFilters:
        def __init__(self, filters=None):
            self.filters = list(filters or [])

    module.FilterOperator = FilterOperator
    module.MetadataFilter = MetadataFilter
    module.MetadataFilters = MetadataFilters
    package = types.ModuleType("llama_index.core.vector_stores")
    package.types = module
    monkeypatch.setitem(sys.modules, "llama_index",
                        types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core",
                        types.ModuleType("llama_index.core"))
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores", package)
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores.types",
                        module)
    return module


class FakeNode:
    def __init__(self, text, filename, page=1, scope_key=None):
        self.text = text
        self.metadata = {
            "page": page,
            "type": "text",
            "filename": filename,
            "scope_key": filename if scope_key is None else scope_key,
        }

    def get_content(self):
        return self.text


class FakeRetriever:
    def __init__(self, nodes, filters=None):
        self.nodes = nodes
        self.filters = filters
        self.asked = []

    def retrieve(self, question):
        self.asked.append(question)
        return self.nodes


class FakeIndex:
    """Records how every retriever was CONSTRUCTED, which is the thing
    under test: a filter that arrives here reached the query."""

    def __init__(self, nodes, keep_filters=True):
        self.nodes = nodes
        self.keep_filters = keep_filters
        self.constructed = []
        self.retrievers = []

    def as_retriever(self, **kwargs):
        self.constructed.append(kwargs)
        retriever = FakeRetriever(
            self.nodes,
            kwargs.get("filters") if self.keep_filters else None)
        self.retrievers.append(retriever)
        return retriever


class Mode:
    HYBRID = "hybrid"


def _wire_scope(monkeypatch, index=None, names=(), nodes=None,
                covered_names=None):
    """The engine with its index and its database seam replaced.

    The REAL `db.lock_retrieval_scope_keys` runs against a recording cursor,
    so the resolution statement and its lifecycle lock are production code.
    """
    from contextlib import contextmanager

    from pipeline.index import db
    from pipeline.retrieval import rag_llamaindex

    class Statements(list):
        authority_open = False

    statements = Statements()

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, sql, params=None):
            self._last = str(sql)
            statements.append((self._last, params))

        def fetchall(self):
            selected = (names if covered_names is None
                        or "kapsam" not in self._last.lower()
                        else covered_names)
            return [(name,) for name in selected]

    class Conn:
        def cursor(self, row_factory=None):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            statements.append(("ROLLBACK", None))

    class Pool:
        @contextmanager
        def connection(self):
            statements.authority_open = True
            try:
                yield Conn()
            finally:
                statements.authority_open = False

    if index is None:
        index = FakeIndex(nodes if nodes is not None else [])
    monkeypatch.setattr(db, "get_pool", lambda: Pool())
    monkeypatch.setattr(db, "set_tenant_context", lambda *_a, **_k: None)
    monkeypatch.setattr(db, "clear_tenant_context", lambda *_a, **_k: None)
    monkeypatch.setattr(db, "current_execution_actor", lambda: None)
    monkeypatch.setattr(db, "begin_retrieval_snapshot", lambda _conn: None)
    monkeypatch.setattr(db, "retrieval_policy_epoch", lambda _conn: 1)
    monkeypatch.setattr(
        db, "resolve_document_scope",
        lambda _conn, *, document_ids=None, collection_ids=None, tags=None:
        tuple(document_ids or ()))
    monkeypatch.setattr(
        db, "active_document_ids",
        lambda _conn: (() if not names else (IC_BELGE,)))
    monkeypatch.setattr(rag_llamaindex, "_index", lambda: (index, Mode))
    return rag_llamaindex, index, statements


def test_a_scoped_query_is_filtered_at_construction_not_afterwards(
        monkeypatch, fake_filters):
    """THE WHOLE ROUTE, in one test.

    The identifiers are resolved to scope keys through `documents`, and the
    resulting metadata filter is handed to the retriever WHEN IT IS BUILT.
    The returned nodes are then checked against the same live authority.
    """
    nodes = [FakeNode("kapsam icindeki pasaj", "kapsam-icinde.pdf",
                      scope_key=IC_KEY)]
    rag_llamaindex, index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY,), nodes=nodes)

    chunks = rag_llamaindex.retrieve("kurgu soru",
                                     document_ids=(IC_BELGE,))

    # the filter reached CONSTRUCTION
    filters = index.constructed[0]["filters"]
    assert [(f.key, f.value, f.operator) for f in filters.filters] == [
        ("scope_key", [IC_KEY], fake_filters.FilterOperator.IN)]
    # ... and the retriever kept the object it was handed
    assert index.retrievers[0].filters is filters
    assert index.retrievers[0].asked == ["kurgu soru"]
    assert [chunk["filename"] for chunk in chunks] == ["kapsam-icinde.pdf"]
    # resolution went through `documents`, as a parameterised SELECT
    assert len(statements) == 4
    assert "pg_advisory_xact_lock_shared" in statements[0][0]
    sql, params = statements[1]
    assert sql.startswith("SELECT tenant_id::text || ':' || id::text")
    assert params == ([IC_BELGE],)
    assert IC_BELGE not in sql
    assert "kapsam" in statements[2][0].lower()
    assert statements[2][1] == ([IC_KEY],)
    assert statements[3] == ("ROLLBACK", None)


def test_the_resolved_scope_keys_are_exact_filter_values_only(
        monkeypatch, fake_filters):
    """A scope key is a value, never text assembled into a statement."""
    rag_llamaindex, index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY, DIS_KEY))

    rag_llamaindex.retrieve("kurgu soru",
                            document_ids=(IC_BELGE, DIS_BELGE))

    (metadata_filter,) = index.constructed[0]["filters"].filters
    # The resolution seam returns names deduplicated AND sorted, which is
    # what makes the dedup deterministic; the db battery pins the same
    # contract directly. The order is stated here rather than assumed from
    # the order the fixture happened to declare its names in.
    assert metadata_filter.value == sorted(
        [IC_KEY, DIS_KEY])
    assert metadata_filter.operator == fake_filters.FilterOperator.IN
    for value in metadata_filter.value:
        assert value.count(":") == 3
    # and no resolved key was ever put into statement text either
    for sql, _params in statements:
        assert IC_KEY not in sql and DIS_KEY not in sql


def test_a_returned_node_outside_the_live_scope_fails_closed(
        monkeypatch, fake_filters):
    """The same document's old version cannot bypass the live filter."""
    nodes = [FakeNode(
        "ayni belgenin eski surum pasaji",
        "ayni-ad.pdf",
        scope_key=ESKI_KEY)]
    rag_llamaindex, index, _statements = _wire_scope(
        monkeypatch,
        names=(IC_KEY,),
        nodes=nodes)

    with pytest.raises(RuntimeError, match="yetki kapsami disinda"):
        rag_llamaindex.retrieve("kurgu soru", document_ids=(IC_BELGE,))
    assert index.retrievers[0].asked == ["kurgu soru"]


def test_an_unscoped_query_uses_live_scope_keys_not_the_stale_snapshot(
        monkeypatch, fake_filters):
    """Archive can change after build, so every query carries live authority."""
    rag_llamaindex, index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY,),
        nodes=[FakeNode("pasaj", "kurgu.pdf", scope_key=IC_KEY)])

    rag_llamaindex.retrieve("kurgu soru")

    filters = index.constructed[0]["filters"]
    assert filters.filters[0].value == [IC_KEY]
    assert index.retrievers[0].filters is filters
    assert "pg_advisory_xact_lock_shared" in statements[0][0]
    assert "COALESCE(active_version_id::text, 'legacy')" in statements[1][0]
    assert statements[1][1] == ([IC_BELGE],)
    assert "scope_key FROM" in statements[2][0]
    assert statements[3] == ("ROLLBACK", None)


def test_no_active_document_means_no_snapshot_query(monkeypatch, fake_filters):
    rag_llamaindex, index, statements = _wire_scope(
        monkeypatch, names=(), nodes=[FakeNode("eski pasaj", "arsiv.pdf")])

    assert rag_llamaindex.retrieve("kurgu soru") == []
    assert index.constructed == []
    assert statements == [("ROLLBACK", None)]


def test_lifecycle_lock_is_held_until_snapshot_nodes_are_projected(
        monkeypatch, fake_filters):
    rag_llamaindex, _index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY,),
        nodes=[FakeNode("pasaj", "kurgu.pdf", scope_key=IC_KEY)])
    projected_while_locked = []
    original = rag_llamaindex._as_chunks

    def project(nodes, allowed_scope_keys):
        projected_while_locked.append(statements.authority_open)
        return original(nodes, allowed_scope_keys)

    monkeypatch.setattr(rag_llamaindex, "_as_chunks", project)

    rag_llamaindex.retrieve("kurgu soru")

    assert projected_while_locked == [True]
    assert statements.authority_open is False


def test_an_unknown_identifier_never_widens_the_query(
        monkeypatch, fake_filters):
    """Nothing resolved, so the scope is EMPTY -- and an empty scope is
    answered with nothing. Querying the index unfiltered here is the one
    mistake that turns a narrowing request into the whole corpus."""
    rag_llamaindex, index, _statements = _wire_scope(
        monkeypatch, names=(), nodes=[FakeNode("pasaj", "kurgu.pdf")])

    assert rag_llamaindex.retrieve("kurgu soru",
                                   document_ids=(YOK_BELGE,)) == []
    assert index.constructed == []


def test_a_scope_fails_closed_when_the_filter_api_is_absent(monkeypatch):
    """No `fake_filters` fixture here: the metadata-filter API cannot be
    imported at all. The engine refuses; it does not answer the wider
    question and label the answer scoped."""
    rag_llamaindex, index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY,),
        nodes=[FakeNode("pasaj", "kurgu.pdf", scope_key=IC_KEY)])
    monkeypatch.setitem(sys.modules, "llama_index",
                        types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core",
                        types.ModuleType("llama_index.core"))
    monkeypatch.delitem(sys.modules, "llama_index.core.vector_stores",
                        raising=False)
    monkeypatch.delitem(sys.modules, "llama_index.core.vector_stores.types",
                        raising=False)

    with pytest.raises(RuntimeError, match="kapsamli sorgu reddedildi"):
        rag_llamaindex.retrieve("kurgu soru", document_ids=(IC_BELGE,))

    # refused BEFORE anything that looks like answering
    assert index.constructed == []
    assert statements[-1] == ("ROLLBACK", None)
    assert all(not statement.startswith("SELECT c.")
               for statement, _params in statements)


def test_a_retriever_that_drops_the_filter_fails_closed(
        monkeypatch, fake_filters):
    """A keyword a retriever silently ignores is the failure this engine
    must not survive: the query would run unscoped and the result would be
    published as though it had been scoped."""
    index = FakeIndex(
        [FakeNode("pasaj", "kurgu.pdf", scope_key=IC_KEY)],
        keep_filters=False)
    rag_llamaindex, index, _statements = _wire_scope(
        monkeypatch, index=index, names=(IC_KEY,))

    with pytest.raises(RuntimeError, match="kapsamli sorgu reddedildi"):
        rag_llamaindex.retrieve("kurgu soru", document_ids=(IC_BELGE,))

    # the retriever was built, but it was never ASKED anything
    assert index.retrievers[0].asked == []


def test_the_checked_answer_forwards_the_scope_to_retrieval(
        monkeypatch, fake_filters):
    """The public path carries the scope; an unscoped one still calls
    `retrieve` with the single argument it has always been given."""
    from pipeline.retrieval import rag_llamaindex

    seen = []
    monkeypatch.setattr(rag_llamaindex, "retrieve",
                        lambda question, **kwargs: seen.append(kwargs) or [])
    monkeypatch.setattr(
        "pipeline.generation.answer.generate_structured",
        lambda _question, _context: '{"dayanak": [], "cevap": '
                                    '"Bu bilgi mevcut belgelerde bulunamadi."}')

    rag_llamaindex.answer_checked("kurgu soru")
    rag_llamaindex.answer_checked("kurgu soru", document_ids=(IC_BELGE,))

    assert seen == [{}, {"document_ids": (IC_BELGE,)}]


def test_checked_answer_refuses_a_runtime_top_k_that_disagrees_with_the_plan(
        monkeypatch):
    from pipeline.retrieval import planner, rag_llamaindex

    monkeypatch.setattr(rag_llamaindex, "TOP_K", planner.TOP_K + 1)
    monkeypatch.setattr(
        rag_llamaindex, "retrieve",
        lambda *_args, **_kwargs: pytest.fail("retrieval was called"))

    with pytest.raises(planner.PlannerError,
                       match="^planner_runtime_policy_mismatch$"):
        rag_llamaindex.answer_checked("kurgu soru")


def test_scoping_rebuilds_nothing(monkeypatch, fake_filters):
    """No index is rebuilt and no schema is touched: the only statement a
    scoped query runs is the SELECT that resolves the names."""
    rag_llamaindex, _index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY,),
        nodes=[FakeNode("pasaj", "kapsam-icinde.pdf", scope_key=IC_KEY)])

    rag_llamaindex.retrieve("kurgu soru", document_ids=(IC_BELGE,))

    assert len(statements) == 4
    for statement, _params in statements:
        upper = statement.upper()
        for verb in ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE",
                     "DROP", "RENAME"):
            assert verb not in upper


def test_an_active_version_missing_from_the_snapshot_refuses_before_query(
        monkeypatch, fake_filters):
    """A false empty result must not hide a stale snapshot."""
    rag_llamaindex, index, statements = _wire_scope(
        monkeypatch, names=(IC_KEY,), covered_names=(),
        nodes=[FakeNode("eski pasaj", "kurgu.pdf", scope_key=IC_KEY)])

    with pytest.raises(RuntimeError, match="anlik goruntusu"):
        rag_llamaindex.retrieve("kurgu soru", document_ids=(IC_BELGE,))

    assert index.constructed == []
    assert "kapsam" in statements[2][0].lower()
    assert statements[-1] == ("ROLLBACK", None)


def test_the_table_name_is_normalised_to_lower_case(monkeypatch):
    """SQLAlchemy folds unquoted identifiers to lower case while the
    maintenance statements quote exactly -- a mixed-case configuration
    would reset one table and write another."""
    import importlib

    monkeypatch.setenv("LLAMAINDEX_TABLE", "KurguKarisikTablo")
    from pipeline.retrieval import rag_llamaindex

    reloaded = importlib.reload(rag_llamaindex)
    try:
        assert reloaded.TABLE == "kurgukarisiktablo"
    finally:
        monkeypatch.delenv("LLAMAINDEX_TABLE")
        importlib.reload(rag_llamaindex)
