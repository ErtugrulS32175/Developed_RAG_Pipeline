"""The optional Ragas path must fail locally, before rented GPU time starts."""
import asyncio
import json
import re
import sys
import types
from types import SimpleNamespace

import pytest

from eval.answer import ragas_check


def _clear_eval_embedding(monkeypatch):
    for name in (
        "EVAL_EMBED_API_URL",
        "EVAL_EMBED_MODEL_NAME",
        "EVAL_EMBED_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_similarity_embedding_is_labelled_correlated(monkeypatch):
    from pipeline.index import embeddings

    _clear_eval_embedding(monkeypatch)
    monkeypatch.setattr(embeddings, "EMBED_API_URL", "http://embed.test/v1/embeddings")
    monkeypatch.setattr(embeddings, "EMBED_MODEL_NAME", "kurgu-uretim-gomusu")

    settings = ragas_check._embedding_settings()

    assert settings["url"] == "http://embed.test/v1/embeddings"
    assert settings["model"] == "kurgu-uretim-gomusu"
    assert settings["mode"] == "production_embedding_correlated"


def test_eval_embedding_override_requires_url_and_model_together(monkeypatch):
    _clear_eval_embedding(monkeypatch)
    monkeypatch.setenv("EVAL_EMBED_API_URL", "http://audit.test/v1/embeddings")

    with pytest.raises(ValueError, match="birlikte"):
        ragas_check._embedding_settings()


def test_explicit_eval_embedding_is_reported_without_claiming_independence(
        monkeypatch):
    _clear_eval_embedding(monkeypatch)
    monkeypatch.setenv("EVAL_EMBED_API_URL", "http://audit.test/v1/embeddings")
    monkeypatch.setenv("EVAL_EMBED_MODEL_NAME", "kurgu-denetim-gomusu")

    settings = ragas_check._embedding_settings()

    assert settings["model"] == "kurgu-denetim-gomusu"
    assert settings["mode"] == "explicit_eval_model"


def test_clients_read_llm_config_from_generation_not_retrieval(monkeypatch):
    from pipeline.generation import answer as generation
    from pipeline.index import embeddings

    _clear_eval_embedding(monkeypatch)
    monkeypatch.setattr(
        generation,
        "LLM_API_URL",
        "http://generation.test/v1/chat/completions",
    )
    monkeypatch.setattr(generation, "LLM_MODEL_NAME", "kurgu-yanit-modeli")
    monkeypatch.setattr(generation, "LLM_API_KEY", "kurgu-anahtar")
    monkeypatch.setattr(
        embeddings,
        "EMBED_API_URL",
        "http://embed.test/v1/embeddings",
    )
    monkeypatch.setattr(embeddings, "EMBED_MODEL_NAME", "kurgu-gomu-modeli")

    calls = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai)

    clients = ragas_check._clients()

    assert calls == [
        {
            "base_url": "http://generation.test/v1",
            "api_key": "kurgu-anahtar",
        },
        {
            "base_url": "http://embed.test/v1",
            "api_key": "not-needed",
        },
    ]
    assert clients["llm_model"] == "kurgu-yanit-modeli"
    assert clients["embedding_model"] == "kurgu-gomu-modeli"


def test_metric_builder_executes_lazy_imports_and_builds_only_the_plan(
        monkeypatch, tmp_path):
    calls = {}

    class DiskCacheBackend:
        def __init__(self, cache_dir):
            calls["cache_dir"] = cache_dir

    def llm_factory(model, **kwargs):
        calls["llm"] = (model, kwargs)
        return "KURGU_LLM"

    def embedding_factory(provider, **kwargs):
        calls["embedding"] = (provider, kwargs)
        return "KURGU_EMBEDDING"

    class Faithfulness:
        def __init__(self, llm):
            self.llm = llm

    class SemanticSimilarity:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    ragas = types.ModuleType("ragas")
    ragas.__path__ = []
    ragas.__version__ = "kurgu-surum"
    cache = types.ModuleType("ragas.cache")
    cache.DiskCacheBackend = DiskCacheBackend
    llms = types.ModuleType("ragas.llms")
    llms.llm_factory = llm_factory
    embeddings_package = types.ModuleType("ragas.embeddings")
    embeddings_package.__path__ = []
    embeddings_base = types.ModuleType("ragas.embeddings.base")
    embeddings_base.embedding_factory = embedding_factory
    metrics_package = types.ModuleType("ragas.metrics")
    metrics_package.__path__ = []
    collections = types.ModuleType("ragas.metrics.collections")
    collections.Faithfulness = Faithfulness
    collections.SemanticSimilarity = SemanticSimilarity

    for name, module in {
        "ragas": ragas,
        "ragas.cache": cache,
        "ragas.llms": llms,
        "ragas.embeddings": embeddings_package,
        "ragas.embeddings.base": embeddings_base,
        "ragas.metrics": metrics_package,
        "ragas.metrics.collections": collections,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    fake_clients = {
        "llm_client": object(),
        "llm_model": "kurgu-yanit-modeli",
        "embedding_client": object(),
        "embedding_model": "kurgu-gomu-modeli",
        "embedding_mode": "explicit_eval_model",
    }
    monkeypatch.setattr(ragas_check, "_clients", lambda: fake_clients)
    monkeypatch.setattr(ragas_check, "CACHE_DIR", tmp_path / "cache")

    metrics, metadata = ragas_check.build_metrics()

    assert set(metrics) == {"faithfulness", "answer_similarity"}
    assert metrics["faithfulness"].llm == "KURGU_LLM"
    assert metrics["answer_similarity"].embeddings == "KURGU_EMBEDDING"
    assert metadata["similarity_threshold"] is None
    assert metadata["diagnostic_only"] is True
    assert metadata["embedding_mode"] == "explicit_eval_model"
    assert metadata["ragas_version"] == "kurgu-surum"
    assert metadata["metric_classes"] == {
        "faithfulness": "Faithfulness",
        "answer_similarity": "SemanticSimilarity",
    }
    assert re.fullmatch(r"[0-9a-f]{12}", metadata["evaluation_config_id"])
    assert metadata["evaluation_config_id"] == (
        ragas_check.evaluation_config_id(metadata)
    )
    assert "FactualCorrectness" not in repr(metrics)
    assert "ContextPrecision" not in repr(metrics)


def _metadata(**changes):
    value = {
        "ragas_version": "kurgu-surum",
        "metric_classes": {
            "faithfulness": "Faithfulness",
            "answer_similarity": "SemanticSimilarity",
        },
        "llm_model": "kurgu-yanit-modeli",
        "embedding_model": "kurgu-gomu-modeli",
        "embedding_mode": "production_embedding_correlated",
        "similarity_threshold": None,
        "diagnostic_only": True,
    }
    value.update(changes)
    value["evaluation_config_id"] = ragas_check.evaluation_config_id(value)
    return value


def test_evaluation_config_id_is_stable_and_changes_with_model_identity():
    first = _metadata()
    reordered = dict(reversed(list(first.items())))
    different = _metadata(
        embedding_model="kurgu-farkli-gomu-modeli",
        embedding_mode="explicit_eval_model",
    )

    assert ragas_check.evaluation_config_id(first) == (
        ragas_check.evaluation_config_id(reordered)
    )
    assert first["evaluation_config_id"] != different["evaluation_config_id"]


def test_result_path_separates_mode_config_sample_and_run_tag(tmp_path):
    metadata = _metadata()
    audit_metadata = _metadata(
        embedding_model="kurgu-denetim-gomusu",
        embedding_mode="explicit_eval_model",
    )

    full = ragas_check.result_path(tmp_path, "kurgu", metadata)
    audit = ragas_check.result_path(tmp_path, "kurgu", audit_metadata)
    smoke = ragas_check.result_path(
        tmp_path,
        "kurgu",
        metadata,
        limit=5,
        run_tag="tekrar-1",
    )

    assert metadata["embedding_mode"] in full.name
    assert metadata["evaluation_config_id"] in full.name
    assert audit_metadata["embedding_mode"] in audit.name
    assert full != audit
    assert full != smoke
    assert smoke.name.endswith("_limit5_tekrar-1.json")
    with pytest.raises(ValueError, match="run tag"):
        ragas_check.result_path(
            tmp_path,
            "kurgu",
            metadata,
            run_tag="../uygunsuz",
        )


def test_existing_result_is_refused_and_never_replaced(tmp_path):
    path = tmp_path / "kurgu.json"
    path.write_text('{"ilk": true}', encoding="utf-8")

    with pytest.raises(SystemExit, match="uzerine yazilmadi"):
        ragas_check.refuse_existing_result(path)
    with pytest.raises(SystemExit, match="uzerine yazilmadi"):
        ragas_check.write_new_result(path, {"ikinci": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ilk": True}


def test_main_refuses_an_existing_result_before_any_metric_call(
        monkeypatch, tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    question_dir = tmp_path / "data" / "rag_eval"
    in_dir.mkdir()
    out_dir.mkdir()
    question_dir.mkdir(parents=True)
    (in_dir / "rag_answers_kurgu-seti.json").write_text(
        json.dumps({
            "sorular": [{
                "soru": "Kurgu soru?",
                "cevap": "Kurgu cevap.",
                "cevap_dogru": True,
                "baglam": "Kurgu baglam.",
            }],
        }),
        encoding="utf-8",
    )
    (question_dir / "kurgu-seti.json").write_text(
        json.dumps([{"q": "Kurgu soru?", "answer": "Kurgu cevap."}]),
        encoding="utf-8",
    )

    metadata = _metadata()
    existing = ragas_check.result_path(out_dir, "kurgu-seti", metadata)
    existing.write_text('{"ilk": true}', encoding="utf-8")
    calls = []

    async def must_not_score(*_args, **_kwargs):
        calls.append("score")
        raise AssertionError("metrik cagrilmamali")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ragas_check,
        "build_metrics",
        lambda: ({"faithfulness": object()}, metadata),
    )
    monkeypatch.setattr(ragas_check, "score_row", must_not_score)
    monkeypatch.setattr(sys, "argv", [
        "ragas_check",
        "--set",
        "kurgu-seti",
        "--in-dir",
        str(in_dir),
        "--out-dir",
        str(out_dir),
    ])

    with pytest.raises(SystemExit, match="uzerine yazilmadi"):
        ragas_check.main()

    assert calls == []
    assert json.loads(existing.read_text(encoding="utf-8")) == {"ilk": True}


def test_score_row_routes_each_metric_only_the_inputs_it_needs():
    calls = {}

    class Metric:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        async def ascore(self, **kwargs):
            calls[self.name] = kwargs
            return SimpleNamespace(value=self.value)

    row = {
        "soru": "Kurgu soru nedir?",
        "cevap": "Kurgu cevaptir.",
        "baglam": "Birinci kurgu pasaj.\n\n---\n\nIkinci kurgu pasaj.",
    }
    metrics = {
        "faithfulness": Metric("faithfulness", 0.75),
        "answer_similarity": Metric("answer_similarity", 0.625),
    }

    scored = asyncio.run(
        ragas_check.score_row(metrics, row, "Kurgu referans cevabi.")
    )

    assert calls["faithfulness"] == {
        "user_input": row["soru"],
        "response": row["cevap"],
        "retrieved_contexts": [
            "Birinci kurgu pasaj.",
            "Ikinci kurgu pasaj.",
        ],
    }
    assert calls["answer_similarity"] == {
        "response": row["cevap"],
        "reference": "Kurgu referans cevabi.",
    }
    assert scored["faithfulness"] == 0.75
    assert scored["answer_similarity"] == 0.625
    assert scored["faithfulness_saniye"] >= 0
    assert scored["answer_similarity_saniye"] >= 0


def test_a_metric_failure_is_recorded_without_aborting_the_other_metric():
    class BrokenMetric:
        async def ascore(self, **_kwargs):
            raise RuntimeError("kurgu metrik hatasi")

    class WorkingMetric:
        async def ascore(self, **_kwargs):
            return SimpleNamespace(value=0.8)

    row = {"soru": "Kurgu?", "cevap": "Kurgu.", "baglam": "Kurgu pasaj."}
    scored = asyncio.run(ragas_check.score_row(
        {
            "faithfulness": BrokenMetric(),
            "answer_similarity": WorkingMetric(),
        },
        row,
        "Kurgu referans.",
    ))

    assert scored["faithfulness"] is None
    assert scored["faithfulness_hata"].startswith("RuntimeError")
    assert scored["answer_similarity"] == 0.8
