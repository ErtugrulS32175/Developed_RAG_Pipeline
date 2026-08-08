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


@pytest.fixture
def fake_llama(monkeypatch):
    events = []
    core = types.ModuleType("llama_index.core")

    class Document:
        def __init__(self, text=None, metadata=None):
            self.text = text

    class StorageContext:
        @staticmethod
        def from_defaults(vector_store=None):
            return object()

    class VectorStoreIndex:
        @staticmethod
        def from_documents(docs, storage_context=None, show_progress=None):
            events.append(("yaz", len(docs)))

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
            return [("kurgu metin", 1, "text", "kurgu.pdf")]

        def fetchone(self):
            return (shadow_count,)

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
    monkeypatch.setattr(db, "get_conn", lambda: Conn())
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


def test_the_copy_reads_only_the_active_generation(monkeypatch, fake_llama):
    """The swap must not loosen the round-16 parity fix: the SELECT keeps
    the same active-generation filter as the native engine."""
    events, _core = fake_llama
    rag_llamaindex = _wire_db(monkeypatch, events)

    rag_llamaindex.build_index()

    select = next(text for kind, text in events
                  if kind == "sql" and text.startswith("SELECT c.text"))
    assert "c.generation = d.active_generation" in select
    assert "c.document_id IS NULL OR d.id IS NOT NULL" in select


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
