# Status

**Last updated:** 2026-11-11
**Current phase:** 2 of 5 — Phone + flat plate (Phase 1 done 2026-11-11)
**Next action:** start Phase 2 (BUILD.md): capture app per docs/CAPTURE_APP.md (seat 5) and `ringsim export-tx`;
then Protocol A step 0 (which tx channel drives only the bottom speaker, D-032).

---

## Done

- Repo reset to specs on 2026-11-10; v1 code kept on branch `archive/v1` (D-039). At reset v1 had Phase 1's
  local code (`ringsim.proc`, devices, free-field medium, CPU `run_ir`, shard writer/reader) and 28 passing
  tests; none of it had run on Kaggle. Facts learned there are folded into the spec docs.
- Spec docs reworked 2026-11-10 (INFRA, BUILD, ARCHITECTURE, DECISIONS D-038..D-040, vault single-writer rule).
- Spec audit 2026-11-10, all fixes agreed with the user and written into the specs:
  - motion within a frame: IR snapshots interpolated along the chirp (D-041 supersedes D-005; Q-019, Q-020)
  - reference-normalized profiles against a free-air recording (D-042; Q-021)
  - medium findings: D-043 pose conventions, D-044 sim window sized for hot air, D-045 numpy/scipy-only base install,
    D-046 any device sample rate at read time, D-047 clamp box + sim background + fitted hand shape; Kaggle runner
    fixes (D-038 amended)
  - high findings: D-048 contact and covered sensors, D-049 ring size fixed per sequence, D-050 convergence on
    referenced profiles + ring gate, D-051 forearm length measured before deciding; Phase 1 benchmark
  - kept as a proposal only (user decision): near-field phantoms + ring pilot before bulk generation (Q-022)
- **Phase 0 — Kaggle runner on a real T4** (2026-11-10). All criteria met:
  - `python tools/test_kaggle_run.py` → ALL PASS on Windows (12 checks)
  - `python tools/kaggle_run.py doctor` → only the expected pyproject/src FAIL
  - `python tools/kaggle_run.py selftest` → exit 0; T4 × 2, Python 3.13.16, k-wave-python 0.6.2, CPU and GPU sims finite
  - Runner fixes found by running it for real: username from `kaggle config view` (fresh account), UTF-8 forced for the
    CLI (Windows log download), 3-attempt output download (transient SSL error), fresh k-Wave objects per call in the
    selftest (k-wave-python modifies them in place), output dir emptied before each pull, licence text dropped from
    `run_meta.json`
  - Deviations: none from BUILD.md; D-035 re-measured (Python 3.13.16, still within the pin)
- Q-018 answered → D-052: k-Wave runs on the Windows machine but ~50× slower than Kaggle's CPU; all k-Wave runs go
  through the runner.
- Q-024 resolved 2026-11-11 → D-053: a solid touching a grid face goes NaN, so the forearm is cut flat 3 cells inside
  the grid and `build_medium` / `run_ir` refuse any solid within 3 cells of a face. Measured end echo of a blunt cut:
  −50 dB vs the arm's own echo.

- **Phase 1 — Free-field pipeline, local and Kaggle** (2026-11-11, round 1 of the multi-seat setup: lead + sim, proc,
  dataset/validation seats). All 12 criteria met; evidence inline on each BUILD.md box:
  - `python -m pytest -q -rs` → 115 passed, 2 deselected (slow), 0 skipped at 609b02a
  - CPU run: `python tools/kaggle_run.py run jobs/p1_free.yaml --cmd "ringsim sim --job {job} --out-dir {out}"` → exit 0,
    BUILD_ID c12f529749d8; shard attrs vs DATASET.md: none missing, none extra; `ref_ir` present; reader raw and ref
  - T4 run: `python tools/kaggle_run.py run jobs/p1_free.yaml` → exit 0, BUILD_ID fb85a88049c9; CPU vs T4 IR 1.331e-05
  - monopole (`ringsim profile` / `freefield_check`, raw gates, D-057): peak bin 3 vs ideal 3, rel L2 5.21 %; 44.1 kHz
    override (D-058) 4.74 %; arrival −0.888 sim samples
  - slow test: `python tools/kaggle_run.py run jobs/p1_tempscale.yaml --cmd "pytest -q -s -m slow tests/test_temperature_scaling.py"`
    → 2 passed, err_scaled 0.0455 % (< 0.5 %), err_unscaled 4.78 %, |g| 1.01786 vs 1/α 1.01780
  - Spec gaps found while building, each decided by the user and written into the docs (D-058 reworded by the lead to match its implementation): D-054 (slow tests on Kaggle,
    revised the same day), D-055 (DeviceConfig from a shard), D-056 (`ref_ir` always its own call), D-057 (monopole
    gates on raw), D-058 (44.1 kHz override), D-059 (JobSpec), D-060 (free-field API), D-061 (Medium.occ +
    port_covered), D-062 (CPU limit), D-063 (`wall_s` scope), D-064 (motion-test window), D-065 ("hold"), D-066
    (run-time failures). The slow-test fixture had bar 2's faces on grid points (v1 too); bar 2 moved 1 mm, PROCESSING.md.
  - Deviations: none found; in the per-sha reviews each behaviour change came with its spec-doc edit in the same commit.
  - Tooling: `scripts/private-commit.sh` mktemp fix (Git Bash); claude-multiplayer room server patched locally for
    Windows paths (outside this repo).

## In progress

- **Phase 2 — Phone + flat plate** (not started)

## Next up

1. Phase 2 — Phone + flat plate (Q-005, Q-017, Q-020 resolved inside the phase)
2. Phase 3 — Static hand + convergence (needs Q-008; Q-026 to decide)
3. Phase 4 — Sequences + hand validation (Q-019 resolved inside; Q-025 to decide)
4. Phase 5 — Ring + randomization + bulk generation (needs Q-001, Q-002, Q-014, Q-021; Q-022 to decide)

## Blocked

None.

## Measurements

| Quantity | Value | Source |
|---|---|---|
| T4 selftest, 48³ (+PML 10) × 300 steps, air | GPU 1.74 s, CPU (Kaggle) 10.47 s; outputs agree to < 1e-6 relative | `python tools/kaggle_run.py selftest` 2026-11-10 |
| Fixed cost per k-Wave call (T4) | 1.05 s wall (25³ + PML, 915 steps; k-Wave's own total 0.80 s) | `python tools/kaggle_run.py bench` 2026-11-11 |
| T4, ring-sized grid dx 2.0 mm (115×70×50 + PML = 0.85 M cells, 915 steps) | 3.18 s/snapshot → 1131 snapshots/GPU-hour | same |
| T4, dx 1.5 mm (148×88×60 + PML = 1.45 M cells, 1219 steps) | 6.25 s/snapshot → 576/GPU-hour | same |
| T4, dx 1.0 mm (220×130×92 + PML = 4.03 M cells, 1753 steps) | 21.67 s/snapshot → 166/GPU-hour | same |
| Both T4s at once, dx 2.0 mm | 4 snapshots in 11.0 s incl. process start-up → 1311/hour for the pair (1.16× one GPU) | same |
| Forearm end echo (D-053), bar 74 mm from sensor | blunt cut −50 dB vs the bar's own echo (−53 dB vs direct); fades ≤ 4 dB better; pyramid taper 13 dB worse | T4 kernel `ringsim-diag3` (scratch script), 2026-11-11 |
| Local CPU k-Wave (Windows) | 45³ incl. PML × 100 steps: 50.3 s | scratch run of k-wave-python 0.6.2, 2026-11-11 |
| wall_s/frame T4 (p1_free: test_pair free field, dx 2 mm, 61×12×12 + PML 10, n_t 1433) | 2.31 s per snapshot (kspaceFirstOrder3D only, D-063); + 1.23 s for the sequence's `ref_ir` call | `python tools/kaggle_run.py run jobs/p1_free.yaml` 2026-11-11, BUILD_ID fb85a88049c9, shard `/seq/free/wall_s` |
| wall_s/frame CPU, Kaggle (p1_free, same grid) | 7.69 s per snapshot; + 7.67 s for `ref_ir` | `python tools/kaggle_run.py run jobs/p1_free.yaml --cmd "ringsim sim --job {job} --out-dir {out}"` 2026-11-11, BUILD_ID c12f529749d8 |
| p1_free CPU vs T4 IR | 1.331e-05 relative L2 | `validation.freefield.compare_ir(cpu, t4, "free")` 2026-11-11 (seats 2 and 4) |
| Temperature scaling (D-037), 64³ fixture, 343 → 337 m/s, Kaggle CPU | scaled vs direct 0.0455 % over 17–27 kHz (unscaled 4.78 %); \|g\| 1.01786 vs 1/α 1.01780 | `python tools/kaggle_run.py run jobs/p1_tempscale.yaml --cmd "pytest -q -s -m slow tests/test_temperature_scaling.py"` 2026-11-11, BUILD_ID fb85a88049c9 |
| static noise floor n | — | Phase 2 |
| chosen dx | — | Phase 3 |
| frames / T4 GPU-hour (ring) | — | Phase 5 |

Benchmark scope: k-Wave call only (HDF5 write, binary, read back) on an estimated ring domain (220×130×90 mm); a smoothed
solid block off the grid faces. Not included: MANO, voxelization, extra snapshots per frame (D-041), the per-sequence
reference snapshot (D-042). Frames/GPU-hour = snapshots/GPU-hour ÷ snapshots per frame.

v1 numbers (2026-10-14, superseded by the rows above for planning): CPU p1_free 3.69 s (61×12×12 + PML, 1270 steps);
T4 64³ × 1200 steps 2.7 s; T4 124³ × 1300 steps 13.0 s.

## Deviations from spec

None in the rebuild (Phase 1 closed with none). (v1's deviations are already written into PROCESSING.md, DATASET.md, DEVICES.md, BUILD.md.)

---

## Update protocol

The lead seat writes this file. Other seats put their update in their `docs/HANDOFF.md` section.
Write when:
- a phase completes → move to Done with date and deviations
- an acceptance criterion completes → check its box
- work stops mid-phase → set **Next action** to the specific next step (function or criterion), not an area
- something blocks → add to Blocked with its Q-ID
- implementation diverges from a spec doc → log under Deviations AND update that doc (standing rule 2)
- a number is measured → fill the Measurements table with the command that produced it; never estimate into it
- after any of the above, at session end → vault log entry + `python scripts/obsidian_sync.py` (docs/OBSIDIAN.md)
