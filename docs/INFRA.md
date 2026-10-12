# INFRA.md — environment, local runs, Kaggle kernels

## Overview

Two execution targets: local CPU (tests, tiny grids, processing, validation) and
Kaggle T4 script kernels (all production sims, D-016). One kernel run = one JobSpec
= one shard.

## Non-goals

- No Docker, cloud VMs, Colab, or job queue.
- No automatic retry or kernel chaining in v1; jobs are pushed by hand via the CLI.

## Environment

```toml
# pyproject.toml (dependencies section)
requires-python = "TBD(Q-010)"     # ==<Kaggle minor>.*  e.g. "==3.11.*"
dependencies = [
  "k-wave-python==TBD(Q-010)",
  "numpy>=1.26", "scipy>=1.11", "h5py>=3.10", "trimesh>=4.0",
  "smplx>=0.1.28", "torch>=2.1",          # smplx needs torch; CPU build is fine
  "pyyaml>=6", "matplotlib>=3.8", "rtree>=1.2",   # rtree for trimesh.contains
  "manifold3d>=2.4",                               # trimesh boolean engine (ring mesh)
]
[project.scripts]
ringsim = "ringsim.cli:main"
```

Pin the exact k-wave-python version AND the Python minor version that ran in the passing
T4 smoke test (Q-010). Local development uses a separate venv on that same minor
(`uv venv --python 3.X`), never the system Python 3.13. `ringsim kaggle push` checks
the kernel log's `python --version` against the pin and fails loudly on mismatch. `ringsim.proc`
needs only numpy/scipy; keep it importable without the rest installed
(`pip install -e .[proc]` extra with numpy, scipy only).

## Data that is not in git

```
data/models/mano/MANO_RIGHT.pkl     Q-008, manual download
data/poses/{grab,ih26m}/*.npz       Q-008, produced by scripts/convert_*.py
data/shards/*.h5                    sim outputs
data/captures/<session>/            real phone captures
```
On Kaggle, MANO and pose `.npz` files come from a **private Kaggle dataset**
`ringsim-assets` (uploaded with `kaggle datasets version`), mounted at
`/kaggle/input/ringsim-assets/`. Never put MANO files in a public dataset or the repo
(license). Both Kaggle datasets stay private.

## Kaggle kernel

```
kaggle/
  kernel-metadata.template.json
  run_job.py
```

```json
{
  "id": "<kaggle-username>/ringsim-{job_id}",
  "title": "ringsim {job_id}",
  "code_file": "run_job.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<kaggle-username>/ringsim-assets", "<kaggle-username>/ringsim-code"]
}
```

Code reaches the kernel as a wheel in a second private dataset `ringsim-code`, not via
`pip install git+https://github.com/...` (D-034). That works whether the GitHub repo is
public or private and puts no GitHub token in a kernel.

`ringsim kaggle push` does, in order:
1. refuse if the working tree is dirty;
2. `python -m build --wheel` with version `0.0.0+g<short hash>` (setuptools-scm or a
   generated `_version.py`);
3. `kaggle datasets version -p dist/ -m "<short hash>" --dir-mode skip` on `ringsim-code`
   (previous wheels deleted from `dist/` first, so the dataset holds exactly one);
4. render `run_job.py` and `kernel-metadata.json`, then `kaggle kernels push`.

`run_job.py` (rendered per job, job YAML embedded as a string):
```python
import glob, subprocess, sys
whl = glob.glob("/kaggle/input/ringsim-code/*.whl")
assert len(whl) == 1 and "+g{git_hash}" in whl[0], whl       # exact commit
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", whl[0]])
import os; os.environ["RINGSIM_DATA"] = "/kaggle/input/ringsim-assets"
from ringsim.cli import main
main(["sim", "--job", "/kaggle/working/job.yaml", "--gpu",
      "--out-dir", "/kaggle/working/shards"])
```

The kernel prints `python --version` and `pip show k-wave-python` first, so every log
records both versions (Q-010 check). GPU type: verify `nvidia-smi` in the
kernel log reports a T4; abort with a clear error if it is a P100 (D-016).

Accelerator selection and the weekly GPU quota are set in Kaggle's UI/account limits;
check the current quota before planning bulk runs (Q-007).

## Local runs

```
ringsim sim --job jobs/p1_free.yaml            # CPU; only for dims <= 96^3
pytest -q
```
`sim` without `--gpu` refuses grids larger than 96³ (excluding PML) with an error
pointing to `ringsim kaggle push`.

## Env vars

| Var | Read by | Required | Meaning |
|---|---|---|---|
| `RINGSIM_DATA` | geometry, dataset | no (default `./data`) | root for models/, poses/ |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | `kaggle` CLI | for push/pull | or `~/.kaggle/kaggle.json` |
| `OBSIDIAN_VAULT` | `scripts/obsidian_sync.py` | for vault sync | `C:\Users\reyan\Desktop\knowledge-base` |

## Notes

- Kernels have an execution time limit; `ShardWriter` flushes every 10 frames so a
  killed kernel still yields a usable partial shard (DATASET.md). Size jobs from the
  measured per-frame `wall_s`, targeting ≤70% of the limit.
