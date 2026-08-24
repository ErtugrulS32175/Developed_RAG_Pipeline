"""The question-answering engine is selectable, the way the table engine is.

Adding an alternative must not put the working one at risk: the default keeps
behaving exactly as before, and an engine whose package is absent fails with a
message instead of breaking anything that does not use it.
"""
from contextlib import contextmanager

import pytest

from pipeline.retrieval import rag_backends
from pipeline.validation.rag.answer_guard import ANSWERED, GuardResult


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

    @contextmanager
    def authority(*_args, **_kwargs):
        yield (["scope-key"], ["document-id"], 1, "all_visible")

    monkeypatch.setattr(rag_llamaindex, "_require", no_package)
    monkeypatch.setattr(rag_llamaindex, "_lifecycle_scope_keys", authority)
    with pytest.raises(RuntimeError) as e:
        rag_llamaindex.retrieve("soru")
    assert "requirements-llamaindex.txt" in str(e.value)


def test_selecting_an_engine_routes_to_it(monkeypatch):
    checked = GuardResult(ANSWERED, "kontrol edilmis cevap", ())
    monkeypatch.setitem(
        rag_backends.BACKENDS,
        "sahte",
        lambda: (
            lambda q, k=15: [{"text": "x"}],
            lambda q: "cevap",
            lambda q: checked,
        ),
    )
    assert rag_backends.answer("soru", backend="sahte") == "cevap"
    assert rag_backends.retrieve("soru", backend="sahte") == [{"text": "x"}]
    assert rag_backends.answer_checked("soru", backend="sahte") is checked


def test_a_legacy_two_callable_backend_keeps_its_plain_path(monkeypatch):
    monkeypatch.setitem(
        rag_backends.BACKENDS,
        "eski",
        lambda: (
            lambda q, k=15: [{"text": "x"}],
            lambda q: "cevap",
        ),
    )

    assert rag_backends.answer("soru", backend="eski") == "cevap"
    assert rag_backends.retrieve("soru", backend="eski") == [{"text": "x"}]
    with pytest.raises(RuntimeError):
        rag_backends.answer_checked("soru", backend="eski")


SCOPE = ("11111111-1111-1111-1111-111111111111",
         "22222222-2222-2222-2222-222222222222")


def _register(monkeypatch, name, checked):
    monkeypatch.setitem(
        rag_backends.BACKENDS,
        name,
        lambda: (lambda q, k=15: [{"text": "x"}], lambda q: "cevap", checked),
    )


def test_an_unscoped_checked_call_still_passes_exactly_one_argument(
        monkeypatch):
    """The seam widened without moving: an engine written before the scope
    existed declares one parameter, and an unscoped question must still be
    the call it declared. Anything else would break every backend -- and
    every replaced seam in the tests that are not editable here -- on the
    day a scope became merely POSSIBLE."""
    result = GuardResult(ANSWERED, "kontrol edilmis cevap", ())
    seen = []

    def only_the_question(question):
        seen.append(question)
        return result

    _register(monkeypatch, "dar", only_the_question)

    assert rag_backends.answer_checked("soru", backend="dar") is result
    assert seen == ["soru"]


def test_a_scope_reaches_the_engine_as_a_keyword(monkeypatch):
    result = GuardResult(ANSWERED, "kontrol edilmis cevap", ())
    seen = []

    def scoped(question, *, document_ids=None):
        seen.append((question, document_ids))
        return result

    _register(monkeypatch, "kapsamli", scoped)

    assert rag_backends.answer_checked(
        "soru", backend="kapsamli", document_ids=SCOPE) is result
    assert seen == [("soru", SCOPE)]


def test_no_scope_means_no_keyword_at_all(monkeypatch):
    """Absent is absent: the engine is not handed `document_ids=None`, it is
    handed nothing, so "unscoped" is the same call it has always been rather
    than a scope that happens to be empty."""
    result = GuardResult(ANSWERED, "kontrol edilmis cevap", ())
    seen = []

    def scoped(question, **kwargs):
        seen.append(kwargs)
        return result

    _register(monkeypatch, "anahtarsiz", scoped)

    rag_backends.answer_checked("soru", backend="anahtarsiz")

    assert seen == [{}]


def test_the_arity_check_still_counts_callables_not_signatures(monkeypatch):
    """Two or three, as before. The registry validates HOW MANY callables a
    backend supplies -- it never inspects what they accept, which is what
    lets a one-argument engine and a scope-aware one coexist."""
    monkeypatch.setitem(
        rag_backends.BACKENDS, "dortlu",
        lambda: (lambda q: None, lambda q: None, lambda q: None,
                 lambda q: None))
    monkeypatch.setitem(
        rag_backends.BACKENDS, "callable_degil",
        lambda: (lambda q: None, "cagirilamaz"))

    with pytest.raises(TypeError):
        rag_backends.get("dortlu")
    with pytest.raises(TypeError):
        rag_backends.get("callable_degil")


def test_a_legacy_two_callable_backend_is_untouched_by_a_scope(monkeypatch):
    """A scope cannot smuggle a two-callable backend onto the checked path:
    the refusal is the same RuntimeError it has always been."""
    monkeypatch.setitem(
        rag_backends.BACKENDS,
        "eski_kapsam",
        lambda: (lambda q, k=15: [{"text": "x"}], lambda q: "cevap"),
    )

    with pytest.raises(RuntimeError):
        rag_backends.answer_checked("soru", backend="eski_kapsam",
                                    document_ids=SCOPE)
    assert rag_backends.answer("soru", backend="eski_kapsam") == "cevap"


def test_every_widened_seam_takes_the_scope_keyword_only_with_a_default():
    """One shape at every hop. Keyword-only with a default is what keeps
    every existing POSITIONAL call site meaning what it meant -- the
    evaluation harness passes `top_k` positionally to one of these."""
    import inspect

    from pipeline.index import db
    from pipeline.retrieval import query, rag_llamaindex

    seams = (
        rag_backends.answer_checked,
        query.ask_checked,
        query.retrieve,
        rag_llamaindex.answer_checked,
        rag_llamaindex.retrieve,
        db.hybrid_search,
    )
    for seam in seams:
        parameter = inspect.signature(seam).parameters["document_ids"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, seam
        assert parameter.default is None, seam


def test_the_widened_seams_did_not_reorder_what_came_before():
    """The new parameter was ADDED, not inserted: every parameter a caller
    could already pass positionally is still in the position it was in."""
    import inspect

    from pipeline.index import db
    from pipeline.retrieval import query, rag_llamaindex

    def positional(fn):
        return [
            name for name, parameter in inspect.signature(fn).parameters.items()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]

    assert positional(db.hybrid_search) == [
        "conn", "dense_vec", "sparse_indices", "sparse_values", "top_k",
        "rrf_k"]
    assert positional(query.retrieve) == ["query", "top_k"]
    assert positional(query.ask_checked) == ["question"]
    assert positional(rag_llamaindex.retrieve) == ["question", "top_k"]
    assert positional(rag_llamaindex.answer_checked) == ["question"]
    assert positional(rag_backends.answer_checked) == ["question", "backend"]


def test_the_alternative_engine_reuses_this_project_s_prompt():
    """Both engines must answer through the same context builder and prompt.
    Otherwise a difference in answer quality could be a difference in prompting,
    and the comparison would measure the wrong thing."""
    import inspect

    from pipeline.retrieval import rag_llamaindex

    src = inspect.getsource(rag_llamaindex.answer)
    assert "build_context" in src and "generate" in src
    checked_src = inspect.getsource(rag_llamaindex.answer_checked)
    assert "generate_structured" in checked_src
    assert "validate_structured" in checked_src
