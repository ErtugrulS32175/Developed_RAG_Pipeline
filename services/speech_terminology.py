"""Validated, deployment-owned terminology for the local speech service.

The registry may be large, but a request never sends the whole registry to
Whisper.  A deterministic selector compiles one closed context into a small
token-bounded hotword pack.  This module has no network, database or tenant
authority; today the context is selected by deployment configuration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Callable

SCHEMA_VERSION = 1
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_TERMS = 10_000
MAX_TERM_CHARS = 96
MAX_ALIASES = 8
MAX_CONTEXTS_PER_TERM = 16
MAX_PRIORITY = 1_000
DEFAULT_CONTEXT = "default"
DEFAULT_MAX_HOTWORD_TOKENS = 96
DEFAULT_MAX_HOTWORD_PHRASES = 16

_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_ROOT_KEYS = frozenset({
    "schema_version", "profile_id", "revision", "language", "terms",
})
_TERM_KEYS = frozenset({"canonical", "aliases", "contexts", "priority"})


class TerminologyConfigError(Exception):
    """Closed configuration refusal; source data is never in the message."""


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Term:
    canonical: str
    aliases: tuple[str, ...]
    contexts: frozenset[str]
    priority: int


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class HotwordPack:
    text: str = dataclasses.field(repr=False)
    phrase_count: int
    token_count: int


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Registry:
    profile_id: str
    revision: str
    language: str
    sha256: str
    terms: tuple[Term, ...] = dataclasses.field(repr=False)
    contexts: frozenset[str]

    def __repr__(self) -> str:
        return (f"Registry(profile_id={self.profile_id!r}, "
                f"revision={self.revision!r}, language={self.language!r}, "
                f"sha256={self.sha256!r}, term_count={len(self.terms)})")

    def require_context(self, context: str) -> None:
        if type(context) is not str or context not in self.contexts:
            raise TerminologyConfigError(
                "SPEECH_TERMINOLOGY_CONTEXT is not supported")

    def select(
            self, context: str, count_tokens: Callable[[str], int], *,
            max_tokens: int = DEFAULT_MAX_HOTWORD_TOKENS,
            max_phrases: int = DEFAULT_MAX_HOTWORD_PHRASES) -> HotwordPack:
        """Compile a deterministic pack without returning registry contents.

        Terms are ordered by deployment-owned priority and then by their
        normalized canonical spelling.  A phrase that would cross the token
        budget is skipped; later shorter phrases may still fit.
        """
        self.require_context(context)
        if (type(max_tokens) is not int or not 1 <= max_tokens <= 1024
                or type(max_phrases) is not int
                or not 1 <= max_phrases <= 256):
            raise TerminologyConfigError("hotword budget is invalid")

        phrases: list[str] = []
        token_count = 0
        ordered = sorted(
            (term for term in self.terms if context in term.contexts),
            key=lambda term: (-term.priority, _normal(term.canonical)),
        )
        for term in ordered:
            # Alias order is curated inside the immutable registry revision.
            variants = (term.canonical,) + term.aliases
            for phrase in variants:
                if len(phrases) >= max_phrases:
                    break
                candidate = ", ".join((*phrases, phrase))
                measured = count_tokens(candidate)
                if type(measured) is not int or measured < 1:
                    raise TerminologyConfigError(
                        "speech tokenizer returned an invalid count")
                if measured <= max_tokens:
                    phrases.append(phrase)
                    token_count = measured
        if not phrases:
            raise TerminologyConfigError(
                "terminology context has no phrase within the budget")
        return HotwordPack(", ".join(phrases), len(phrases), token_count)


def _normal(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _closed_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise TerminologyConfigError(
                "terminology registry contains a duplicate key")
        value[key] = item
    return value


def _identifier(value, field: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise TerminologyConfigError(f"terminology {field} is invalid")
    return value


def _text(value, field: str) -> str:
    if (type(value) is not str or value != value.strip()
            or not value or len(value) > MAX_TERM_CHARS
            or "," in value
            or any(unicodedata.category(char).startswith("C")
                   for char in value)):
        raise TerminologyConfigError(f"terminology {field} is invalid")
    return value


def _string_list(value, field: str, maximum: int, *, identifiers=False):
    if type(value) is not list or not 1 <= len(value) <= maximum:
        raise TerminologyConfigError(f"terminology {field} is invalid")
    convert = _identifier if identifiers else _text
    converted = tuple(convert(item, field) for item in value)
    if len({_normal(item) for item in converted}) != len(converted):
        raise TerminologyConfigError(f"terminology {field} is duplicated")
    return converted


def _term(value, seen: set[str]) -> Term:
    if type(value) is not dict or set(value) != _TERM_KEYS:
        raise TerminologyConfigError("terminology term shape is invalid")
    canonical = _text(value["canonical"], "canonical")
    aliases_value = value["aliases"]
    if type(aliases_value) is not list or len(aliases_value) > MAX_ALIASES:
        raise TerminologyConfigError("terminology aliases is invalid")
    aliases = tuple(_text(item, "alias") for item in aliases_value)
    if len({_normal(item) for item in aliases}) != len(aliases):
        raise TerminologyConfigError("terminology aliases is duplicated")
    contexts = frozenset(_string_list(
        value["contexts"], "context", MAX_CONTEXTS_PER_TERM,
        identifiers=True))
    priority = value["priority"]
    if (type(priority) is not int
            or not 0 <= priority <= MAX_PRIORITY):
        raise TerminologyConfigError("terminology priority is invalid")
    for phrase in (canonical, *aliases):
        key = _normal(phrase)
        if key in seen:
            raise TerminologyConfigError("terminology phrase is duplicated")
        seen.add(key)
    return Term(canonical, aliases, contexts, priority)


def load_registry(path: str | Path) -> Registry:
    """Load one regular, bounded UTF-8 JSON snapshot and validate it closed."""
    source = Path(path)
    try:
        info = source.lstat()
    except OSError:
        raise TerminologyConfigError("terminology registry is unavailable") \
            from None
    reparse = bool(getattr(info, "st_file_attributes", 0) & 0x400)
    if source.is_symlink() or reparse or not stat.S_ISREG(info.st_mode):
        raise TerminologyConfigError(
            "terminology registry must be a regular file")
    if not 1 <= info.st_size <= MAX_REGISTRY_BYTES:
        raise TerminologyConfigError("terminology registry size is invalid")
    try:
        raw = source.read_bytes()
    except OSError:
        raise TerminologyConfigError("terminology registry is unavailable") \
            from None
    if len(raw) != info.st_size:
        raise TerminologyConfigError("terminology registry changed while read")
    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_closed_object)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TerminologyConfigError("terminology registry is invalid JSON") \
            from None
    if type(document) is not dict or set(document) != _ROOT_KEYS:
        raise TerminologyConfigError("terminology registry shape is invalid")
    if type(document["schema_version"]) is not int \
            or document["schema_version"] != SCHEMA_VERSION:
        raise TerminologyConfigError("terminology schema version is invalid")
    profile_id = _identifier(document["profile_id"], "profile_id")
    revision = _identifier(document["revision"], "revision")
    if document["language"] != "tr":
        raise TerminologyConfigError("terminology language is invalid")
    values = document["terms"]
    if type(values) is not list or not 1 <= len(values) <= MAX_TERMS:
        raise TerminologyConfigError("terminology term count is invalid")
    seen: set[str] = set()
    terms = tuple(_term(value, seen) for value in values)
    contexts = frozenset(
        context for term in terms for context in term.contexts)
    if DEFAULT_CONTEXT not in contexts:
        raise TerminologyConfigError(
            "terminology registry has no default context")
    return Registry(
        profile_id=profile_id,
        revision=revision,
        language="tr",
        sha256=hashlib.sha256(raw).hexdigest(),
        terms=terms,
        contexts=contexts,
    )


def main(argv=None) -> int:
    """Validate a deployment snapshot without printing its terms or path."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("speech_terminology: exactly one registry file is required",
              file=sys.stderr)
        return 2
    try:
        registry = load_registry(arguments[0])
    except TerminologyConfigError as exc:
        print(f"speech_terminology: config_error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "profile_id": registry.profile_id,
        "revision": registry.revision,
        "language": registry.language,
        "sha256": registry.sha256,
        "term_count": len(registry.terms),
        "context_count": len(registry.contexts),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
