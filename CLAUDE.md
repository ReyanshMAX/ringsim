# ring-sim — Claude Code Entry Point

## What this is

Acoustic simulation pipeline for a science-fair project: a smart ring that estimates
wrist-relative 3D hand pose (20 joints) from inaudible FMCW chirp echoes. This repo
generates labeled training data by simulating echoes off MANO hand meshes in k-Wave,
then converting them to C-FMCW echo profiles (original + differential) with the exact
processing code the real hardware will use. Model training, per-user calibration and
ring firmware are out of scope (separate suites).

Single user, batch-compute workload. The central constraint: **sim and real data pass
through one shared package (`ringsim.proc`)**, and every device (validation phone, ring)
is a config object run through the same pipeline. The pipeline is validated against an
iPhone 17 Pro before the ring exists; the iOS capture app for that lives here too.

The repo was reset on 2026-11-10: specs only, code rebuilt from them. The previous
attempt is branch `archive/v1` (read-only reference, D-039).

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.13 (D-035), local venv via uv | sim, processing, dataset, validation |
| Solver | k-wave-python 0.6.2 (D-035), density smoothed (D-036) | 3D k-space pseudospectral, CUDA binary |
| Hand model | MANO via `smplx` | right hand, license-gated files |
| Geometry | `trimesh` | meshes, voxelization |
| Signal | `numpy`, `scipy` | `ringsim.proc` depends on nothing else |
| Storage | HDF5 via `h5py` | one shard per sim run |
| Compute | Kaggle T4 via `tools/kaggle_run.py`; local CPU for small grids | P100 unusable |
| Capture app | Swift 5 / SwiftUI / AVAudioEngine, XcodeGen | GitHub Actions macOS build, sideloaded |

## Repo structure

```
ring-sim/
  CLAUDE.md  STATUS.md  BUILD.md  DECISIONS.md  OPEN_QUESTIONS.md
  docs/                      spec docs (routing table below)
  tools/kaggle_run.py        run a job on a Kaggle T4 and pull the result (exists; docs/INFRA.md)
  tools/test_kaggle_run.py   offline test for it (fake CLI)
  kaggle/config.toml         runner settings
  scripts/obsidian_sync.py   mirrors specs into the Obsidian vault (exists)
  pyproject.toml             Phase 1 creates it (contents in docs/INFRA.md)
  src/ringsim/               devices/ geometry/ solver/ proc/ dataset/ validation/ cli.py
  capture-app/               iOS app (Phase 2)
  .github/workflows/ios.yml  unsigned IPA build (Phase 2)
  tests/
  data/                      gitignored: models/, poses/, shards/, captures/, kaggle/
```

## Where to look

| Working on | Read |
|---|---|
| current state, what to do next | STATUS.md |
| build order, acceptance criteria | BUILD.md |
| why something is the way it is | DECISIONS.md |
| anything unresolved | OPEN_QUESTIONS.md |
| module boundaries, data flow, CLI, JobSpec | docs/ARCHITECTURE.md |
| phone / ring / test configs | docs/DEVICES.md |
| poses, MANO, scenes, voxelization | docs/GEOMETRY.md |
| k-Wave setup, grid, pulse, convergence | docs/SOLVER.md |
| chirps, IR->rx, C-FMCW, differential, temperature | docs/PROCESSING.md |
| HDF5 schema, labels, randomization | docs/DATASET.md |
| phone rig, capture protocols, metrics | docs/VALIDATION.md |
| iOS capture app + CI build | docs/CAPTURE_APP.md |
| Kaggle runs, env pins, local runs | docs/INFRA.md |
| Obsidian log, overview, spec mirror | docs/OBSIDIAN.md |

## Standing rules

1. **Read STATUS.md first.** It is the current state of the build. Do not infer
   progress from the codebase.
2. **Specs are source of truth.** If the implementation must deviate from a spec
   doc, update that doc in the same change. A stale spec is worse than no spec.
3. **Never invent unspecified behavior.** If a requirement is missing, ambiguous,
   or contradicts another doc, stop and ask. Add it to OPEN_QUESTIONS.md. Do not
   pick a reasonable-sounding default and proceed.
4. **Do not relitigate DECISIONS.md.** Those choices are settled with reasons
   recorded. Reopen one only if new information directly invalidates the stated
   reason, and say which reason and why.

## Project-specific rules

- `ringsim.proc` never imports from `geometry`, `solver`, `dataset` or `validation`.
  It is the code the real hardware pipeline imports.
- A physical parameter tagged `TBD(Q-xxx)` in a config raises at load time. Never
  replace it with a guessed number.
- **Run Kaggle yourself.** Never ask the user to push, poll or download anything:
  `python tools/kaggle_run.py run jobs/<job>.yaml` does all of it (docs/INFRA.md). If it
  reports an auth or CLI problem, run `python tools/kaggle_run.py doctor`, then tell the
  user the one thing that is missing.
- **Several seats at once:** STATUS.md and the Obsidian vault are written by the lead
  seat only; other seats report in their `docs/HANDOFF.md` section and the room. Kaggle
  runs are started by the backend seat (the runner also takes a lock, so two pushes
  cannot collide).
- **Vault:** the lead ends every session that changed anything by following
  docs/OBSIDIAN.md (log entry, overview status, `scripts/obsidian_sync.py`).
- Do not overclaim in docstrings, READMEs, plots or vault notes. This builds on
  Ring-a-Pose and ActSonic (Cornell SciFi Lab); cite them where their methods are reused.
- Measurements go into STATUS.md only from a command's output, with the command named.
- Pieces of `archive/v1` may be copied only if they pass this suite's tests and match
  these specs (D-039).

Start every session: read STATUS.md, then the doc(s) for the task.
End every session: update STATUS.md (lead) or your HANDOFF.md section (other seats).
