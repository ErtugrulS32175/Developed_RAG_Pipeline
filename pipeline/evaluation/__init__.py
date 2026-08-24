"""Closed evaluation-dataset contracts.

The package deliberately contains no storage or network code.  It turns an
untrusted dataset document into immutable, canonically digestible values that
later persistence layers can bind to their own tenant and lifecycle records.
"""

from .datasets import (
    CASE_KEYS,
    CASE_TYPES,
    VERSION_KEYS,
    EvalCase,
    EvalDatasetError,
    EvalDatasetVersion,
    canonical_case_bytes,
    canonical_version_bytes,
    case_digest,
    case_sha256,
    load_version_json,
    new_case_key,
    normalize_cases,
    project_legacy,
    project_legacy_case,
    validate_case,
    validate_version,
    version_digest,
    version_sha256,
)

__all__ = (
    "CASE_KEYS",
    "CASE_TYPES",
    "VERSION_KEYS",
    "EvalCase",
    "EvalDatasetError",
    "EvalDatasetVersion",
    "canonical_case_bytes",
    "canonical_version_bytes",
    "case_digest",
    "case_sha256",
    "load_version_json",
    "new_case_key",
    "normalize_cases",
    "project_legacy",
    "project_legacy_case",
    "validate_case",
    "validate_version",
    "version_digest",
    "version_sha256",
)
