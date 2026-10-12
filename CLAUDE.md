# ring-sim — Claude Code Entry Point

## What this is

Acoustic simulation pipeline for a science-fair project: a smart ring that estimates
wrist-relative 3D hand pose (20 joints) from inaudible FMCW chirp echoes. This repo
generates labeled training data by simulating echoes off MANO hand meshes in k-Wave,
then converting them into C-FMCW echo profiles (original + differential) using the
exact processing code the real hardware will use. Model training, per-user
calibration, and ring firmware are out of scope (separate suites).

Single user, single developer, batch-compute workload. The central constraint:
**sim and real data must pass through one shared processing package (`ringsim.proc`)**,
and every device (validation phone, ring) is a config object run through the same
pipeline. The pipeline is validated against an iPhone 17 Pro before the ring exists.
The iOS capture app used for that validation also lives here.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python = Kaggle's minor version (Q-010), local venv via uv | sim, processing, dataset, validation |
| Solver | k-wave-python (pinned, see Q-010) | 3D k-space pseudospectral, CUDA binary |
| Hand model | MANO via `smplx` | right hand, license-gated model files |
| Geometry | `trimesh` | mesh ops, voxelization |
| Signal | `numpy`, `scipy` | `ringsim.proc` depends on nothing else |
| Storage | HDF5 via `h5py` | one shard per sim run |
| Compute | Kaggle T4 script kernels via `kaggle` CLI; local CPU for small grids | P100 unusable |
| Capture app | Swift 5 / SwiftUI / AVAudioEngine, XcodeGen | built on GitHub Actions macOS runner, sideloaded |

## Repo structure

```
ring-sim/
  CLAUDE.md  STATUS.md  BUILD.md  DECISIONS.md  OPEN_QUESTIONS.md
  docs/                      spec docs (see routing table)
  pyproject.toml
  src/ringsim/
    devices/                 DeviceConfig + YAML configs (iphone17pro, ring, test_pair)
    geometry/                pose sources, MANO mesh, scene assembly, voxelization
    solver/                  grid sizing, pulse, k-Wave runner
    proc/                    SHARED processing: chirp synth, IR->rx, align, C-FMCW, differential
    dataset/                 HDF5 schema, writer, reader, domain randomization
    validation/              real-capture loading, sim-vs-real metrics, plots
    cli.py                   `ringsim` entry point
  kaggle/                    kernel template + run_job.py
  scripts/obsidian_sync.py   mirrors specs into the Obsidian vault
  capture-app/               iOS app (project.yml, Sources/, Resources/tx/)
  .github/workflows/ios.yml  unsigned IPA build
  tests/
  data/                      gitignored: models/, poses/, shards/, captures/
```

## Where to look

| Working on | Read |
|---|---|
| current state, what to do next | STATUS.md |
| build order, acceptance criteria | BUILD.md |
| why something is the way it is | DECISIONS.md |
| anything unresolved | OPEN_QUESTIONS.md |
| module boundaries, data flow, CLI | docs/ARCHITECTURE.md |
| phone / ring / test configs | docs/DEVICES.md |
| poses, MANO, scenes, voxelization | docs/GEOMETRY.md |
| k-Wave setup, grid, pulse, convergence | docs/SOLVER.md |
| chirps, IR->rx, C-FMCW, differential | docs/PROCESSING.md |
| HDF5 schema, labels, randomization | docs/DATASET.md |
| phone rig, capture protocol, metrics | docs/VALIDATION.md |
| iOS capture app + CI build | docs/CAPTURE_APP.md |
| Kaggle kernels, env pinning, local runs | docs/INFRA.md |
| Obsidian work log, overview, spec mirror | docs/OBSIDIAN.md |

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

- `ringsim.proc` must never import from `ringsim.geometry`, `solver`, or `dataset`.
  It is the code the real hardware pipeline imports.
- Any physical parameter tagged `TBD(Q-xxx)` in a config raises at load time. Never
  replace it with a guessed number.
- **End every session that changed anything by updating the Obsidian vault**
  (docs/OBSIDIAN.md): append to `isef-log.md`, update `isef-overview.md` status if a
  phase moved, add project-level decisions, then run `scripts/obsidian_sync.py`.
- Do not overclaim in docstrings, READMEs, plots, or vault notes: this builds on Ring-a-Pose and
  ActSonic (Cornell SciFi Lab); cite them where their methods are reused.

Start every session: read STATUS.md, then the doc(s) for the task.
End every session: update STATUS.md, then the vault (docs/OBSIDIAN.md).
