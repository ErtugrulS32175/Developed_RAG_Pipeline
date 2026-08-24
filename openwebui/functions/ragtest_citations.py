"""
title: RAGTest Evidence Citations
author: RAGTest
version: 1.0.0
required_open_webui_version: 0.11.0
"""
from __future__ import annotations

import re

from pydantic import BaseModel


_REF = re.compile(r"^[A-Za-z0-9_-]{43}$")
_SOURCES = frozenset({"model", "derived"})
_BASE_KEYS = frozenset({"evidence_ref", "document_name", "page", "source"})
_FEEDBACK_KEYS = _BASE_KEYS | {"feedback_ref"}
_DOCUMENT_HINT = "Kanıt içeriğini görmek için Kanıtı göster eylemini kullanın."


def _closed_citation(value):
    """Return a UI-safe citation or ``None`` for any open/malformed shape."""
    if (not isinstance(value, dict)
            or frozenset(value) not in {_BASE_KEYS, _FEEDBACK_KEYS}):
        return None
    evidence_ref = value.get("evidence_ref")
    document_name = value.get("document_name")
    page = value.get("page")
    source = value.get("source")
    feedback_ref = value.get("feedback_ref")
    if (not isinstance(evidence_ref, str)
            or _REF.fullmatch(evidence_ref) is None):
        return None
    if (not isinstance(document_name, str)
            or not document_name.strip()
            or len(document_name) > 500
            or any(ord(char) < 32 for char in document_name)):
        return None
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return None
    if source not in _SOURCES:
        return None
    if ("feedback_ref" in value
            and (not isinstance(feedback_ref, str)
                 or _REF.fullmatch(feedback_ref) is None)):
        return None
    return evidence_ref, document_name, page, feedback_ref


def source_event(value, *, include_feedback=True):
    """Build Open WebUI's documented source-event shape without passage data."""
    citation = _closed_citation(value)
    if citation is None:
        return None
    evidence_ref, document_name, page, feedback_ref = citation
    metadata = {
        "source": document_name,
        "name": document_name,
        "page": page,
    }
    if include_feedback and feedback_ref is not None:
        metadata["ragtest_feedback_ref"] = feedback_ref
    return {
        "type": "source",
        "data": {
            "source": {"name": document_name, "id": evidence_ref},
            "document": [_DOCUMENT_HINT],
            "metadata": [metadata],
        },
    }


class Filter:
    """Translate RAGTest's checked metadata into persisted OWUI sources."""

    class Valves(BaseModel):
        priority: int = 0

    def __init__(self):
        self.valves = self.Valves()
        self._seen = {}

    @staticmethod
    def _stream_key(metadata):
        metadata = metadata if isinstance(metadata, dict) else {}
        return (str(metadata.get("chat_id") or ""),
                str(metadata.get("message_id") or ""))

    async def stream(self, event: dict, __event_emitter__=None,
                     __metadata__=None) -> dict:
        """Emit each checked source once; never alter or drop provider chunks."""
        if not isinstance(event, dict):
            return event
        key = self._stream_key(__metadata__)
        seen = self._seen.setdefault(key, set())
        citations = event.get("rag_citations")
        if (event.get("rag_status") == "answered"
                and isinstance(citations, list)
                and __event_emitter__ is not None):
            feedback_refs = {
                value.get("feedback_ref")
                for value in citations
                if isinstance(value, dict)
                and _REF.fullmatch(str(value.get("feedback_ref") or ""))
            }
            include_feedback = len(feedback_refs) <= 1
            for value in citations:
                source = source_event(
                    value, include_feedback=include_feedback)
                if source is None:
                    continue
                evidence_ref = source["data"]["source"]["id"]
                if evidence_ref in seen:
                    continue
                await __event_emitter__(source)
                seen.add(evidence_ref)
        terminal = event.get("done") is True or any(
            isinstance(choice, dict) and choice.get("finish_reason") is not None
            for choice in (event.get("choices") or [])
        )
        if terminal:
            self._seen.pop(key, None)
        return event
