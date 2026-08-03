"""Executable ``python -m`` targets must survive package moves."""
import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("pipeline", "eval", "scripts", "services")


def _module_targets(path):
    """(line, module-or-None) for command lists containing ``-m``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        values = [
            item.value if isinstance(item, ast.Constant) else None
            for item in node.elts
        ]
        for index, value in enumerate(values[:-1]):
            if value == "-m":
                module = values[index + 1]
                yield node.lineno, module if isinstance(module, str) else None


def test_executable_module_paths_resolve():
    targets = []
    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            targets.extend((path, line, module)
                           for line, module in _module_targets(path))

    assert targets, "en az bir calistirilabilir -m hedefi bekleniyordu"
    dynamic = [f"{path.name}:{line}" for path, line, module in targets
               if module is None]
    assert dynamic == [], (
        "statik olarak dogrulanamayan -m hedefleri: " + ", ".join(dynamic)
    )
    missing = [
        f"{path.name}:{line} -> {module}"
        for path, line, module in targets
        if importlib.util.find_spec(module) is None
    ]
    assert missing == [], "cozulemeyen -m hedefleri: " + ", ".join(missing)
