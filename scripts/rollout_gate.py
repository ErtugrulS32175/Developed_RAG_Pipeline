"""Fail-closed deployment gate joining offline quality and live readiness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


ROLLOUT_GATE_VERSION = 1
MAX_REPORT_BYTES = 1024 * 1024
_READY_CHECKS = {"veritabani", "sema", "embedding"}


def _quality_passed(path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (type(report) is dict
            and set(report) == {
                "quality_gate_version", "passed", "sets", "failures"}
            and type(report.get("quality_gate_version")) is int
            and report.get("quality_gate_version") == 1
            and report.get("passed") is True
            and type(report.get("sets")) is dict
            and type(report.get("failures")) is list
            and not report["failures"])


def _ready(url) -> bool:
    try:
        response = requests.get(url, timeout=5)
        body = response.json()
    except (requests.RequestException, ValueError):
        return False
    return (response.status_code == 200 and type(body) is dict
            and set(body) == {"status", "kontroller"}
            and body.get("status") == "ready"
            and type(body.get("kontroller")) is dict
            and set(body["kontroller"]) == _READY_CHECKS
            and all(value is True for value in body["kontroller"].values()))


def evaluate(quality_report, ready_url) -> dict[str, object]:
    failures = []
    if not _quality_passed(quality_report):
        failures.append("quality_gate_failed")
    if not _ready(ready_url):
        failures.append("readiness_failed")
    return {"rollout_gate_version": ROLLOUT_GATE_VERSION,
            "passed": not failures, "failures": failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--ready-url", default="http://127.0.0.1:8000/ready")
    args = parser.parse_args(argv)
    result = evaluate(args.quality_report, args.ready_url)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
