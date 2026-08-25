"""Export or verify the deterministic OpenAPI document without repo writes."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def openapi_document():
    root = Path(tempfile.gettempdir()) / "ragtest-openapi-contract"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ALLOW_INSECURE_LOCAL", "1")
    os.environ.setdefault("API_BIND_HOST", "127.0.0.1")
    os.environ.setdefault(
        "RAG_DB_CONTEXT_SECRET",
        "openapi-contract-only-context-key-with-more-than-32-bytes",
    )
    os.environ.setdefault("UPLOAD_DIR", str(root / "uploads"))
    os.environ.setdefault("OUTPUT_DIR", str(root / "output"))
    os.environ.setdefault("EXPORT_DIR", str(root / "export"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Route registration only needs endpoint signatures. Importing the real
    # ingest runtime here would initialize its tokenizer and turn a deterministic
    # schema check into a model-cache/network dependency on clean CI runners.
    # The placeholder is process-local, is never invoked, and is removed again
    # after the application module has been imported.
    module_name = "pipeline.index.ingest"
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = types.ModuleType(module_name)
    try:
        from pipeline.api.app import app
        return app.openapi()
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def canonical_bytes(document):
    return (json.dumps(
        document, ensure_ascii=False, sort_keys=True, indent=2,
    ) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = canonical_bytes(openapi_document())
    if args.check:
        actual = args.path.read_bytes()
        if actual != expected:
            raise SystemExit("checked OpenAPI snapshot is stale")
        print("OpenAPI snapshot is current")
        return
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_bytes(expected)


if __name__ == "__main__":
    main()
