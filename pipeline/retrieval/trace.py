"""Closed, content-free evidence about one checked retrieval run."""
from dataclasses import dataclass
import re
import time
import uuid


BACKENDS = frozenset({"native", "llamaindex"})
STAGES = frozenset({"retrieve", "rerank", "context", "generate", "validate"})
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
    trace_id: str
    backend: str
    scope_document_count: int | None
    retrieved_count: int
    reranked_count: int | None
    context_passage_count: int
    stages: tuple[TraceStage, ...]

    def __post_init__(self):
        if not isinstance(self.trace_id, str) or not _TRACE_ID.fullmatch(
                self.trace_id):
            raise ValueError("invalid retrieval trace id")
        if self.backend not in BACKENDS:
            raise ValueError("unknown retrieval trace backend")
        counts = (self.retrieved_count, self.context_passage_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("trace counts must be non-negative integers")
        if (self.scope_document_count is not None
                and (type(self.scope_document_count) is not int
                     or self.scope_document_count < 0)):
            raise ValueError("invalid trace scope count")
        if (self.reranked_count is not None
                and (type(self.reranked_count) is not int
                     or self.reranked_count < 0)):
            raise ValueError("invalid reranked count")
        if type(self.stages) is not tuple or not self.stages:
            raise TypeError("trace stages must be a non-empty tuple")
        if any(not isinstance(stage, TraceStage) for stage in self.stages):
            raise TypeError("trace stages contain an invalid value")
        names = tuple(stage.name for stage in self.stages)
        expected = (
            ("retrieve", "rerank", "context", "generate", "validate")
            if self.backend == "native"
            else ("retrieve", "context", "generate", "validate")
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
        available = (self.retrieved_count if self.reranked_count is None
                     else self.reranked_count)
        if self.context_passage_count > available:
            raise ValueError("context count exceeds available passages")

    def public(self) -> dict:
        """Return the only JSON shape allowed to cross the API boundary."""
        return {
            "trace_id": self.trace_id,
            "backend": self.backend,
            "scope_document_count": self.scope_document_count,
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "context_passage_count": self.context_passage_count,
            "stages_ms": {stage.name: stage.duration_ms
                          for stage in self.stages},
        }


def clock() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def new_trace(*, backend, scope_document_count, retrieved_count,
              reranked_count, context_passage_count, stages):
    return RetrievalTrace(
        trace_id=uuid.uuid4().hex,
        backend=backend,
        scope_document_count=scope_document_count,
        retrieved_count=retrieved_count,
        reranked_count=reranked_count,
        context_passage_count=context_passage_count,
        stages=tuple(stages),
    )
