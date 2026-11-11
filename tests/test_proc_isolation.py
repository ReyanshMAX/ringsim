"""ringsim.proc imports only the standard library, numpy, scipy and itself (D-008, D-045)."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import ringsim.proc

PROC_DIR = Path(ringsim.proc.__file__).parent
ALLOWED_THIRD_PARTY = {"numpy", "scipy"}
FORBIDDEN = ["kwave", "torch", "smplx", "trimesh", "h5py", "yaml", "matplotlib", "manifold3d"]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import"
            yield node.module


def test_static_imports():
    files = sorted(PROC_DIR.rglob("*.py"))
    assert files
    bad = []
    for f in files:
        for mod in _imports(f):
            top = mod.split(".")[0]
            ok = (top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY
                  or mod == "ringsim.proc" or mod.startswith("ringsim.proc."))
            if not ok:
                bad.append(f"{f.name}: {mod}")
    assert not bad, bad


def test_fresh_subprocess_pulls_no_sim_dependency():
    code = ("import sys, json, ringsim.proc; "
            f"print(json.dumps([m for m in {FORBIDDEN!r} if m in sys.modules]))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
