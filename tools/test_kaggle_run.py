#!/usr/bin/env python3
"""Offline test of tools/kaggle_run.py against a fake `kaggle` CLI. No network to Kaggle, no GPU.

  python tools/test_kaggle_run.py

Builds a throwaway repo with a one-module package, a fake `kaggle` that mimics the CLI calls the runner
makes and executes the rendered kernel script locally (fake /kaggle tree, fake nvidia-smi), then checks
the round trip: wheel -> dataset -> kernel -> output -> data/shards. This proves the plumbing, not Kaggle.
Rerun it after any change to kaggle_run.py.
"""
import json, os, shutil, stat, subprocess, sys, tempfile, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAKE_KAGGLE = r'''#!__PY__
import json, os, shutil, subprocess, sys
from pathlib import Path
S = Path(os.environ["FAKE_STATE"]); S.mkdir(exist_ok=True)
a = sys.argv[1:]
def arg(flag): return a[a.index(flag) + 1]
if a[:2] == ["kernels", "list"]:
    print("ref,title\nfakeuser/x,x"); sys.exit(0)
if a[:2] == ["datasets", "status"]:
    if not (S / "dataset").exists(): print("404 - Not Found", file=sys.stderr); sys.exit(1)
    print("ready"); sys.exit(0)
if a[:2] == ["datasets", "create"] or a[:2] == ["datasets", "version"]:
    shutil.rmtree(S / "dataset", ignore_errors=True); shutil.copytree(arg("-p"), S / "dataset"); sys.exit(0)
if a[:2] == ["kernels", "push"]:
    k = Path(arg("-p")); meta = json.loads((k / "kernel-metadata.json").read_text())
    assert meta["kernel_type"] == "script" and meta["enable_gpu"] and "--accelerator" in a and arg("--accelerator") == "NvidiaTeslaT4"
    root = S / "kroot"; shutil.rmtree(root, ignore_errors=True)
    (root / "input").mkdir(parents=True); (root / "working").mkdir()
    if meta["dataset_sources"]: shutil.copytree(S / "dataset", root / "input" / "ringsim-code")
    env = dict(os.environ, KAGGLE_ROOT=str(root), PATH=os.environ["FAKE_BIN"] + os.pathsep + os.path.dirname(os.environ["FAKE_PY"]) + os.pathsep + os.environ["PATH"])
    r = subprocess.run([os.environ["FAKE_PY"], str(k / "run_job.py")], env=env, capture_output=True, text=True)
    (root / "working" / "kernel.log").write_text(r.stdout + r.stderr)
    (S / "status").write_text("COMPLETE" if r.returncode == 0 else "ERROR"); sys.exit(0)
if a[:2] == ["kernels", "status"]:
    print(f'fakeuser/x has status "KernelWorkerStatus.{(S / "status").read_text()}"'); sys.exit(0)
if a[:2] == ["kernels", "output"]:
    out = Path(arg("-p")); out.mkdir(parents=True, exist_ok=True)
    ign = shutil.ignore_patterns("run_meta.json") if os.environ.get("FAKE_NO_META") else None
    shutil.copytree(S / "kroot" / "working", out, dirs_exist_ok=True, ignore=ign); sys.exit(0)
print("fake kaggle: unhandled", a, file=sys.stderr); sys.exit(9)
'''


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    td = Path(tempfile.mkdtemp(prefix="ksim-"))
    repo, bin_, state = td / "repo", td / "bin", td / "state"
    (repo / "tools").mkdir(parents=True); (repo / "src" / "pkgx").mkdir(parents=True); bin_.mkdir(); (repo / "kaggle").mkdir()
    shutil.copy2(HERE / "kaggle_run.py", repo / "tools" / "kaggle_run.py")
    (repo / "kaggle" / "config.toml").write_text('[kaggle]\nusername = "fakeuser"\npoll_s = 0\n')
    (repo / "src" / "pkgx" / "__init__.py").write_text("X = 1\n")
    (repo / "pyproject.toml").write_text(textwrap.dedent('''
        [build-system]
        requires = ["setuptools>=69"]
        build-backend = "setuptools.build_meta"
        [project]
        name = "pkgx"
        version = "0.0.0"
        [tool.setuptools.packages.find]
        where = ["src"]
    '''))
    (repo / "jobs").mkdir(); (repo / "jobs" / "t_job.yaml").write_text("job_id: t_job\n")
    venv = td / "venv"; run([sys.executable, "-m", "venv", str(venv)])
    py = str(venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python"))
    WIN = os.name == "nt"
    fake_py = bin_ / "fake_kaggle.py"; fake_py.write_text(FAKE_KAGGLE.replace("__PY__", sys.executable))
    if WIN:    # Windows cannot execute shebang scripts: use .cmd shims
        fk = bin_ / "kaggle.cmd"; fk.write_text(f'@"{sys.executable}" "{fake_py}" %*\r\n')
    else:
        fk = bin_ / "kaggle"; shutil.copy2(fake_py, fk); fk.chmod(fk.stat().st_mode | stat.S_IEXEC)

    def set_gpu(name):
        if WIN:
            (bin_ / "nvidia-smi.cmd").write_text(f"@echo {name}\r\n")
        else:
            smi = bin_ / "nvidia-smi"; smi.write_text(f"#!/bin/sh\necho '{name}'\n"); smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    set_gpu("Tesla T4")
    env = dict(os.environ, KAGGLE_BIN=str(fk), FAKE_STATE=str(state), FAKE_BIN=str(bin_), FAKE_PY=py,
               TMPDIR=str(td), TEMP=str(td), TMP=str(td))
    fails = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if not cond else ""))
        if not cond: fails.append(name)

    R = [sys.executable, str(repo / "tools" / "kaggle_run.py")]
    r = run(R + ["run", "jobs/t_job.yaml", "--cmd", "python -c \"import pkgx,sys;open(sys.argv[1],'w').write('h5')\" {out}/t_job.h5"], cwd=repo, env=env)
    check("run exits 0", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
    meta = json.loads((repo / "data/kaggle/t_job/out/run_meta.json").read_text()) if (repo / "data/kaggle/t_job/out/run_meta.json").exists() else {}
    check("kernel meta rc 0 and build id matched", meta.get("rc") == 0 and meta.get("got_build_id") == meta.get("build_id"), str(meta))
    check("shard copied to data/shards", (repo / "data/shards/t_job.h5").exists())

    (repo / "src" / "pkgx" / "__init__.py").write_text("X = 2\n")       # dirty change -> new build id, same flow
    r2 = run(R + ["run", "jobs/t_job.yaml", "--cmd", "python -c \"import pkgx;assert pkgx.X==2\""], cwd=repo, env=env)
    check("uncommitted change ships and runs", r2.returncode == 0, r2.stdout[-1500:] + r2.stderr[-1500:])

    r3 = run(R + ["run", "jobs/t_job.yaml", "--cmd", "false"], cwd=repo, env=env)
    check("failing job command -> exit 1", r3.returncode == 1)

    set_gpu("Tesla P100-PCIE-16GB")
    r4 = run(R + ["run", "jobs/t_job.yaml", "--cmd", "true"], cwd=repo, env=env)
    check("P100 -> exit 3", r4.returncode == 3, r4.stdout[-800:] + r4.stderr[-800:])
    set_gpu("Tesla T4")

    (repo / "jobs" / "renamed.yaml").write_text("job_id: t_job\n")
    r6 = run(R + ["run", "jobs/renamed.yaml", "--cmd", "true"], cwd=repo, env=env)
    check("job_id != file name -> exit 2", r6.returncode == 2, r6.stdout[-800:] + r6.stderr[-800:])

    r7 = run(R + ["run", "jobs/t_job.yaml", "--cmd", "python -c \"print(1)\""], cwd=repo, env=dict(env, FAKE_NO_META="1"))
    check("COMPLETE without run_meta.json -> exit 1", r7.returncode == 1 and "NO_RUN_META" in r7.stdout,
          r7.stdout[-800:] + r7.stderr[-800:])

    lock = td / "ringsim-kaggle.lock"
    check("lock released after runs", not lock.exists())

    r5 = run(R + ["--dry-run", "run", "jobs/t_job.yaml"], cwd=repo, env=env)
    check("dry-run exits 0 and calls nothing", r5.returncode == 0 and "DRY: " in r5.stdout, r5.stdout[-800:] + r5.stderr[-800:])

    k = (repo / "data/kaggle/t_job/kernel/run_job.py")
    check("rendered kernel compiles", k.exists() and run([sys.executable, "-m", "py_compile", str(k)]).returncode == 0)

    r8 = run(R + ["--dry-run", "bench"], cwd=repo, env=env)
    kb = repo / "data/kaggle/bench/kernel/run_job.py"
    check("dry-run bench renders a compiling kernel", r8.returncode == 0 and kb.exists()
          and run([sys.executable, "-m", "py_compile", str(kb)]).returncode == 0, r8.stdout[-800:] + r8.stderr[-800:])

    shutil.rmtree(td, ignore_errors=True)
    print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
