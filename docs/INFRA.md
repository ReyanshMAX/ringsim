# INFRA.md — environment, local runs, Kaggle runs

## Overview

Two execution targets: local (tests, tiny grids, processing, validation) and Kaggle T4 script
kernels (all production sims, D-016). One kernel run = one JobSpec = one shard. Kaggle runs are
driven by `tools/kaggle_run.py`, which agents call themselves (D-040); the user never pushes or
downloads anything by hand.

## Non-goals

- No Docker, cloud VMs, Colab, or job queue.
- No automatic kernel chaining. One command, one job; batches are a shell loop in `scripts/`.
- The runner does not live in the `ringsim` package: it must work before the package exists (D-038).

## Environment

```toml
# pyproject.toml (Phase 1 creates it)
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ringsim"
version = "0.0.0"
requires-python = "==3.13.*"                       # Kaggle image Python 3.13.15 (D-035)
dependencies = [
  "k-wave-python==0.6.2",                          # D-035
  "numpy>=1.26", "scipy>=1.11", "h5py>=3.10", "trimesh>=4.0",
  "smplx>=0.1.28", "torch>=2.1",                   # smplx needs torch; CPU build is fine
  "pyyaml>=6", "matplotlib>=3.8", "rtree>=1.2",    # rtree for trimesh.contains
  "manifold3d>=2.4",                               # trimesh boolean engine (ring mesh)
]
[project.optional-dependencies]
proc = ["numpy>=1.26", "scipy>=1.11"]              # ringsim.proc needs only these
dev = ["pytest>=8"]
[project.scripts]
ringsim = "ringsim.cli:main"
[tool.setuptools.packages.find]
where = ["src"]
[tool.setuptools.package-data]
"ringsim.devices" = ["*.yaml"]
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs k-Wave on CPU; run with -m slow"]
addopts = "-m 'not slow'"
```

Pins come from the 2026-10-14 T4 smoke run (D-035): Python 3.13.15, k-wave-python 0.6.2. Local
development uses a uv venv on Python 3.13 (`uv venv --python 3.13`). `ringsim.proc` stays importable
with only the `proc` extra installed.

## Data that is not in git

```
data/models/mano/MANO_RIGHT.pkl     Q-008, manual download
data/poses/{grab,ih26m}/*.npz       Q-008, produced by scripts/convert_*.py
data/shards/*.h5                    sim outputs
data/captures/<session>/            real phone captures
data/kaggle/<job_id>/               runner scratch: wheel, rendered kernel, pulled output
```

On Kaggle, MANO and pose `.npz` files come from the private dataset `ringsim-assets`, mounted under
`/kaggle/input/`. Never put MANO files in a public dataset or the repo (license). Both datasets stay private.

## Kaggle runs

One-time setup, done by the user: `pip install kaggle` and `kaggle auth login` (the current CLI uses
OAuth, not `kaggle.json`). Settings live in `kaggle/config.toml`.

```
python tools/kaggle_run.py doctor                    CLI, auth, python, lock
python tools/kaggle_run.py selftest                  smoke kernel: T4 check, versions, tiny k-Wave sim on CPU+GPU
python tools/kaggle_run.py run jobs/<job>.yaml       full round trip (below)
python tools/kaggle_run.py run jobs/<job>.yaml --cmd "pytest -m slow tests/test_x.py"   any command instead of `ringsim sim`
python tools/kaggle_run.py run ... --no-wait         push and return; later: status <job_id>, pull <job_id>
python tools/kaggle_run.py --dry-run run ...         build and render, call kaggle for nothing
python tools/test_kaggle_run.py                      offline test against a fake CLI; rerun after editing the runner
```

`run` does, in order:
1. take the lock `<tmp>/ringsim-kaggle.lock` (one push at a time across seats and worktrees);
2. hash `src/**` + `pyproject.toml` → **BUILD_ID** (12 hex). Uncommitted changes are fine: the hash is of
   content, so a seat can test work in progress;
3. `pip wheel` the package into `data/kaggle/<job_id>/code/` next to a `BUILD_ID` file, then
   `kaggle datasets create` (first time) or `version` on the private dataset `ringsim-code`; wait until `ready`;
4. render `run_job.py` (job YAML and command embedded) + `kernel-metadata.json`; `kaggle kernels push --accelerator NvidiaTeslaT4`;
5. release the lock once the kernel reports RUNNING (inputs are mounted by then), poll every 30 s;
6. `kaggle kernels output` into `data/kaggle/<job_id>/out/`, copy `*.h5` to `data/shards/`, print the log tail and `run_meta.json`.

The kernel script, in order: print Python and `nvidia-smi`; exit 3 if the GPU is not a T4 (D-016); find
`BUILD_ID` under `/kaggle/input`, exit 4 `BUILD_ID_MISMATCH` if it differs from the expected one (the runner
retries twice after 45 s; Kaggle can mount a stale version); `pip install` the wheel, then again with
`--force-reinstall --no-deps` (same version string, new code); set `RINGSIM_DATA` if the assets dataset is
mounted; run the command with `{job}` and `{out}` filled in; write `run_meta.json` (rc, why, gpu, versions,
wall_s, build id) to `/kaggle/working`. Everything in `/kaggle/working` comes back.

Exit codes: 0 ok · 1 job or tooling failed · 2 usage/config · 3 wrong GPU · 4 timeout or stale/missing code
dataset · 5 lock held by another run. On any nonzero code read the printed log tail before retrying; do not
retry blindly, quota is limited (Q-007).

Notes:
- Kernel slug is `ringsim-<job_id>`; reusing a `job_id` overwrites that kernel's latest run. Use new ids for new work.
- "GPU T4 x2" exposes two T4s; v1 uses one. Parallel sims on both are a Phase 5 option (Q-007).
- Kernels have an execution time limit; `ShardWriter` flushes every 10 frames so a killed kernel still yields a
  usable partial shard (DATASET.md). Size jobs from measured `wall_s`, targeting ≤70% of the limit.
- Check the account's weekly GPU quota in Kaggle before planning bulk runs (Q-007).
- Notebook-type kernels fail under papermill (stripped kernelspec); script kernels work (v1 finding). Do not change `kernel_type`.

## Local runs

```
ringsim sim --job jobs/p1_free.yaml            # CPU; only for dims <= 96^3
pytest -q
```
`sim` without `--gpu` refuses grids larger than 96³ (excluding PML) with an error pointing to the runner.
The k-Wave OMP binary (v1.4.1, shipped by k-wave-python 0.6.2) needs `libhdf5_serial_hl.so.100` on Ubuntu 24.04
(`apt-get install libhdf5-hl-100t64`). Whether CPU k-Wave runs natively on the user's Windows machine is Q-018.

## Env vars

| Var | Read by | Required | Meaning |
|---|---|---|---|
| `RINGSIM_DATA` | geometry, dataset | no (default `./data`) | root for models/, poses/ |
| `KAGGLE_USERNAME` | `tools/kaggle_run.py` | no | else `[kaggle] username`, else detected from the CLI login |
| `KAGGLE_BIN` | `tools/kaggle_run.py` | no | path to the `kaggle` executable (tests use a fake) |
| `OBSIDIAN_VAULT` | `scripts/obsidian_sync.py` | for vault sync | `C:\Users\reyan\Desktop\knowledge-base` |
