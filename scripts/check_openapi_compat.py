"""Conservative backwards-compatibility gate for the version-one API floor."""
import argparse
import json
from pathlib import Path


HTTP_METHODS = frozenset({
    "get", "put", "post", "delete", "options", "head", "patch", "trace",
})
OPERATION_CONTRACT_KEYS = (
    "parameters", "requestBody", "responses", "security",
)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def incompatibilities(baseline, current):
    failures = []
    if baseline.get("openapi") != current.get("openapi"):
        failures.append("OpenAPI dialect changed")
    for section in ("securitySchemes", "schemas"):
        before = baseline.get("components", {}).get(section, {})
        after = current.get("components", {}).get(section, {})
        for name, contract in before.items():
            if name not in after:
                failures.append(f"component removed: {section}/{name}")
            elif after[name] != contract:
                failures.append(f"component changed: {section}/{name}")
    for path, before_item in baseline.get("paths", {}).items():
        after_item = current.get("paths", {}).get(path)
        if after_item is None:
            failures.append(f"path removed: {path}")
            continue
        for method, before_operation in before_item.items():
            if method not in HTTP_METHODS:
                continue
            after_operation = after_item.get(method)
            if after_operation is None:
                failures.append(f"operation removed: {method.upper()} {path}")
                continue
            for key in OPERATION_CONTRACT_KEYS:
                if before_operation.get(key) != after_operation.get(key):
                    failures.append(
                        f"operation {key} changed: {method.upper()} {path}")
    return tuple(failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    failures = incompatibilities(_load(args.baseline), _load(args.current))
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit("OpenAPI backwards-compatibility gate failed")
    print("OpenAPI remains backwards compatible with the version-one floor")


if __name__ == "__main__":
    main()
