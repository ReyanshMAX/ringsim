"""ringsim.proc must not import geometry/solver/dataset/validation (CLAUDE.md)."""
import ast
from pathlib import Path

import ringsim.proc

FORBIDDEN = {"geometry", "solver", "dataset", "validation", "devices", "cli"}
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "__future__", "dataclasses", "typing", "functools"}


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield "ringsim.proc"     # relative import stays inside proc
            elif node.module:
                yield node.module


def test_no_forbidden_imports():
    root = Path(ringsim.proc.__file__).parent
    files = sorted(root.glob("*.py"))
    assert files
    for f in files:
        for mod in _imports(f):
            parts = mod.split(".")
            if parts[0] == "ringsim":
                assert parts[:2] == ["ringsim", "proc"], f"{f.name} imports {mod}"
                assert not (set(parts) & FORBIDDEN), f"{f.name} imports {mod}"
            else:
                assert parts[0] in ALLOWED_THIRD_PARTY, f"{f.name} imports {mod}"
