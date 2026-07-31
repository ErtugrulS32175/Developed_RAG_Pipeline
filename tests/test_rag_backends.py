"""The question-answering engine is selectable, the way the table engine is.

Adding an alternative must not put the working one at risk: the default keeps
behaving exactly as before, and an engine whose package is absent fails with a
message instead of breaking anything that does not use it.
"""
import pytest

from pipeline.retrieval import rag_backends


def test_the_default_is_the_engine_built_here():
    assert rag_backends.RAG_BACKEND == "native"


def test_both_engines_are_registered():
    assert set(rag_backends.BACKENDS) == {"native", "llamaindex"}


def test_an_unknown_engine_is_refused_by_name():
    with pytest.raises(ValueError) as e:
        rag_backends.get("yokboyle")
    assert "yokboyle" in str(e.value)


def test_the_native_engine_resolves_without_optional_packages():
    """Whatever else is registered, the default path must not depend on it."""
    retrieve, answer = rag_backends.get("native")
    assert callable(retrieve) and callable(answer)


def test_a_missing_optional_engine_says_how_to_install_it(monkeypatch):
    """Selecting an engine that is not installed should explain itself, not
    surface an ImportError from somewhere deep in a dependency."""
    from pipeline.retrieval import rag_llamaindex

    def no_package():
        raise RuntimeError(rag_llamaindex._MISSING)

    monkeypatch.setattr(rag_llamaindex, "_require", no_package)
    with pytest.raises(RuntimeError) as e:
        rag_llamaindex.retrieve("soru")
    assert "requirements-llamaindex.txt" in str(e.value)


def test_selecting_an_engine_routes_to_it(monkeypatch):
    monkeypatch.setitem(rag_backends.BACKENDS, "sahte",
                        lambda: (lambda q, k=15: [{"text": "x"}], lambda q: "cevap"))
    assert rag_backends.answer("soru", backend="sahte") == "cevap"
    assert rag_backends.retrieve("soru", backend="sahte") == [{"text": "x"}]


def test_the_alternative_engine_reuses_this_project_s_prompt():
    """Both engines must answer through the same context builder and prompt.
    Otherwise a difference in answer quality could be a difference in prompting,
    and the comparison would measure the wrong thing."""
    import inspect

    from pipeline.retrieval import rag_llamaindex

    src = inspect.getsource(rag_llamaindex.answer)
    assert "build_context" in src and "generate" in src
