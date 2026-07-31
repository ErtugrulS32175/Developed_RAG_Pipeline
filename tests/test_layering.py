"""The product must not depend on its own test tooling.

`pipeline/` is what ships -- the API, the ingest path, the answer guard.
`eval/` is how we measure it. Dependencies run one way: eval imports pipeline.

This got broken once, quietly, by putting the answer guard in pipeline while
its Turkish number handling still lived in eval. Two things go wrong when that
happens. A deployment carrying only pipeline/ fails at import. And, worse, the
scorer and the guard end up sharing code they want to pull in opposite
directions -- the scorer tuned to be generous so it does not call a correct
answer wrong, the guard tuned to be strict so an unsupported figure cannot pass.
Tuning either one then silently changes the other, and neither side's tests
would notice.
"""
import ast
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"


def _imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_no_pipeline_module_imports_eval_or_tests():
    offenders = []
    for path in sorted(PIPELINE.glob("*.py")):
        for module in _imported_modules(path):
            root = module.split(".")[0]
            if root in ("eval", "tests"):
                offenders.append(f"{path.name} -> {module}")
    assert offenders == [], (
        "uretim kodu olcum aracina bagimli hale gelmis: " + ", ".join(offenders))
