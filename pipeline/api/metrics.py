"""Bounded, content-free process metrics in Prometheus text format."""
from __future__ import annotations

import threading
from collections import defaultdict


_LOCK = threading.Lock()
_REQUESTS = defaultdict(int)
_DURATION_MS = defaultdict(float)
_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH"})


def observe(method, route, status, duration_ms):
    """Record only code-owned route templates, never raw URLs or query data."""
    if method not in _ALLOWED_METHODS:
        method = "OTHER"
    if (not isinstance(route, str) or not route.startswith("/")
            or "?" in route or len(route) > 200):
        route = "unmatched"
    status_class = f"{int(status) // 100}xx" if 100 <= int(status) <= 599 else "5xx"
    key = (method, route, status_class)
    with _LOCK:
        _REQUESTS[key] += 1
        _DURATION_MS[key] += max(0.0, float(duration_ms))


def render():
    lines = [
        "# HELP rag_http_requests_total Completed HTTP requests.",
        "# TYPE rag_http_requests_total counter",
        "# HELP rag_http_request_duration_ms_sum Total request duration.",
        "# TYPE rag_http_request_duration_ms_sum counter",
    ]
    with _LOCK:
        snapshot = [
            (key, _REQUESTS[key], _DURATION_MS[key])
            for key in sorted(_REQUESTS)
        ]
    for (method, route, status_class), count, duration in snapshot:
        labels = (f'method="{method}",route="{route}",'
                  f'status_class="{status_class}"')
        lines.append(f"rag_http_requests_total{{{labels}}} {count}")
        lines.append(
            f"rag_http_request_duration_ms_sum{{{labels}}} {duration:.3f}")
    return "\n".join(lines) + "\n"


def reset_for_tests():
    with _LOCK:
        _REQUESTS.clear()
        _DURATION_MS.clear()
