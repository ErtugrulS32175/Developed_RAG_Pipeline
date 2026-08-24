"""Closed, content-free evidence about one checked retrieval run."""
from dataclasses import dataclass
import re
import time
import uuid


TRACE_VERSION = 2
PLANNER_POLICY_VERSION = 1
QUERY_CLASSES = frozenset({"factual"})
RETRIEVAL_MODES = frozenset({"hybrid_balanced"})
FALLBACKS = frozenset({"none"})
SCOPE_KINDS = frozenset({
    "all_visible", "explicit_documents", "metadata_filters",
    "intersection", "empty",
})
V1_TOP_K = 15
V1_CANDIDATE_LIMIT = 60
BACKENDS = frozenset({"native", "llamaindex"})
STAGES = frozenset({
    "plan", "retrieve", "rerank", "context", "generate", "validate",
})
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class TraceStage:
    name: str
    duration_ms: int

    def __post_init__(self):
        if self.name not in STAGES:
            raise ValueError("unknown retrieval trace stage")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("trace duration must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    trace_version: int
    trace_id: str
    backend: str
    planner_policy_version: int
    query_class: str
    retrieval_mode: str
    fallback: str
    scope_kind: str
    policy_epoch: int
    top_k: int
    candidate_limit: int
    scope_document_count: int | None
    retrieved_count: int
    reranked_count: int | None
    context_passage_count: int
    context_utf8_bytes: int
    stages: tuple[TraceStage, ...]

    def __post_init__(self):
        if (type(self.trace_version) is not int
                or self.trace_version != TRACE_VERSION):
            raise ValueError("unknown retrieval trace version")
        if type(self.trace_id) is not str or not _TRACE_ID.fullmatch(
                self.trace_id):
            raise ValueError("invalid retrieval trace id")
        if type(self.backend) is not str or self.backend not in BACKENDS:
            raise ValueError("unknown retrieval trace backend")
        if (type(self.planner_policy_version) is not int
                or self.planner_policy_version != PLANNER_POLICY_VERSION):
            raise ValueError("unknown planner policy version")
        if (type(self.query_class) is not str
                or self.query_class not in QUERY_CLASSES):
            raise ValueError("unknown query class")
        if (type(self.retrieval_mode) is not str
                or self.retrieval_mode not in RETRIEVAL_MODES):
            raise ValueError("unknown retrieval mode")
        if type(self.fallback) is not str or self.fallback not in FALLBACKS:
            raise ValueError("unknown retrieval fallback")
        if (type(self.scope_kind) is not str
                or self.scope_kind not in SCOPE_KINDS):
            raise ValueError("unknown retrieval scope kind")
        if type(self.policy_epoch) is not int or self.policy_epoch < 1:
            raise ValueError("invalid retrieval policy epoch")
        if type(self.top_k) is not int or self.top_k != V1_TOP_K:
            raise ValueError("invalid retrieval top k")
        if (type(self.candidate_limit) is not int
                or self.candidate_limit != V1_CANDIDATE_LIMIT):
            raise ValueError("invalid retrieval candidate limit")
        counts = (self.retrieved_count, self.context_passage_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("trace counts must be non-negative integers")
        if (self.scope_document_count is not None
                and (type(self.scope_document_count) is not int
                     or self.scope_document_count < 0)):
            raise ValueError("invalid trace scope count")
        if (self.scope_kind == "empty"
                and self.scope_document_count != 0):
            raise ValueError("empty scope must have zero documents")
        if (self.scope_kind == "all_visible"
                and self.scope_document_count is not None
                and self.scope_document_count < 1):
            raise ValueError("visible scope must be absent or non-empty")
        if (self.scope_kind not in {"all_visible", "empty"}
                and (self.scope_document_count is None
                     or self.scope_document_count < 1)):
            raise ValueError("named scope must be non-empty")
        if (self.reranked_count is not None
                and (type(self.reranked_count) is not int
                     or self.reranked_count < 0)):
            raise ValueError("invalid reranked count")
        if (self.scope_kind == "empty"
                and (self.retrieved_count != 0
                     or self.reranked_count not in {None, 0}
                     or self.context_passage_count != 0)):
            raise ValueError("empty scope cannot produce retrieval evidence")
        if (type(self.context_utf8_bytes) is not int
                or self.context_utf8_bytes < 0):
            raise ValueError("invalid context byte count")
        if ((self.context_passage_count == 0)
                != (self.context_utf8_bytes == 0)):
            raise ValueError("context count and bytes disagree")
        if type(self.stages) is not tuple or not self.stages:
            raise TypeError("trace stages must be a non-empty tuple")
        if any(not isinstance(stage, TraceStage) for stage in self.stages):
            raise TypeError("trace stages contain an invalid value")
        names = tuple(stage.name for stage in self.stages)
        expected = (
            ("plan", "retrieve", "rerank", "context", "generate", "validate")
            if self.backend == "native"
            else ("plan", "retrieve", "context", "generate", "validate")
        )
        if names != expected:
            raise ValueError("trace stages are incomplete or out of order")
        if self.backend == "native" and self.reranked_count is None:
            raise ValueError("native trace requires a reranked count")
        if self.backend == "llamaindex" and self.reranked_count is not None:
            raise ValueError("llamaindex trace cannot claim a rerank stage")
        if (self.reranked_count is not None
                and self.reranked_count > self.retrieved_count):
            raise ValueError("reranked count exceeds retrieved count")
        if self.retrieved_count > self.top_k:
            raise ValueError("retrieved count exceeds top k")
        available = (self.retrieved_count if self.reranked_count is None
                     else self.reranked_count)
        if self.context_passage_count > available:
            raise ValueError("context count exceeds available passages")

    def public(self) -> dict:
        """Return the only JSON shape allowed to cross the API boundary."""
        return {
            "trace_version": self.trace_version,
            "trace_id": self.trace_id,
            "backend": self.backend,
            "planner_policy_version": self.planner_policy_version,
            "query_class": self.query_class,
            "retrieval_mode": self.retrieval_mode,
            "fallback": self.fallback,
            "scope_kind": self.scope_kind,
            "policy_epoch": self.policy_epoch,
            "top_k": self.top_k,
            "candidate_limit": self.candidate_limit,
            "scope_document_count": self.scope_document_count,
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "context_passage_count": self.context_passage_count,
            "context_utf8_bytes": self.context_utf8_bytes,
            "stages_ms": {stage.name: stage.duration_ms
                          for stage in self.stages},
        }


def clock() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def new_trace(*, backend, planner_policy_version, query_class,
              retrieval_mode, fallback, scope_kind, policy_epoch, top_k,
              candidate_limit, scope_document_count, retrieved_count,
              reranked_count, context_passage_count, context_utf8_bytes,
              stages):
    return RetrievalTrace(
        trace_version=TRACE_VERSION,
        trace_id=uuid.uuid4().hex,
        backend=backend,
        planner_policy_version=planner_policy_version,
        query_class=query_class,
        retrieval_mode=retrieval_mode,
        fallback=fallback,
        scope_kind=scope_kind,
        policy_epoch=policy_epoch,
        top_k=top_k,
        candidate_limit=candidate_limit,
        scope_document_count=scope_document_count,
        retrieved_count=retrieved_count,
        reranked_count=reranked_count,
        context_passage_count=context_passage_count,
        context_utf8_bytes=context_utf8_bytes,
        stages=tuple(stages),
    )
