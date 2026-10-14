# Status

**Last updated:** 2026-10-14
**Current phase:** 1 of 5 — Free-field pipeline, local and Kaggle
**Next action:** User reviews the Phase 1 local code. Then implement `ringsim kaggle push/pull` + `kaggle/run_job.py` + `kaggle/kernel-metadata.template.json` (INFRA.md) and run criterion 7 (`ringsim kaggle push --job jobs/p1_free.yaml`). Nothing has been pushed to Kaggle.

---

## Done

- Phase 1 local criteria (2026-10-14): pyproject + uv venv (Python 3.13.16, k-wave-python 0.6.2), package skeleton, all of `ringsim.proc` + tests, `load_device` + test_pair/iphone17pro/ring YAMLs, free-field `build_medium`, CPU `run_ir`, `ShardWriter`, reader, `ringsim sim`/`profile`, `jobs/p1_free.yaml`.
- Spec suite generated (2026-10-11).
- Vault synced: 2026-10-12 and 2026-10-14 log entries, overview status, spec mirror (2026-10-14).
- Temperature moved to read time by exact time-scaling, verified numerically (D-037, 2026-10-14).
- Kaggle T4 smoke run (2026-10-14): versions pinned (D-035), density smoothing chosen (D-036), k-Wave output layout `[n_t, n_sensor]` confirmed.
- Obsidian vault set up: overview, log, decisions, spec mirror under `03-competitions/isef-ring-pose/` (2026-10-11).
- Spec audit fixes: forearm instead of wrist cap (D-031), single-speaker phone output (D-032), differential convergence criterion (D-033), wheel-based Kaggle install (D-034), Python version pinned to Kaggle (2026-10-12).
- Phone geometry, ring signal/label conventions, and sideload path filled from Apple drawings, Ring-a-Pose paper, and lapse-app (2026-10-11).

## In progress

- **Phase 1 — Free-field pipeline**
  - [x] `pytest -q` passes incl. proc tests + isolation test (28 passed, 2026-10-14)
  - [x] Slow temperature-scaling test passes on CPU (D-037): 0.035% scaled vs 5.67% unscaled, |g| = 1.0179 = 1/α (2026-10-14)
  - [x] `load_device`: phone loads with Q-005 warnings; ring raises naming Q-001 (and Q-002)
  - [x] CPU free-field shard written with full attrs (`data/shards/p1_free.h5`, git b1b3f68)
  - [x] Profile matches ideal monopole (criterion revised, BUILD.md): envelope peak bin 4 vs reference 3, rel L2 5.2%
  - [x] IR arrival matches 0.100/c + PULSE_DELAY within 2 samples (414.93 vs 416.47 µs, −0.89 samples)
  - [ ] Kernel log versions + wheel hash match pins
  - [ ] T4 kernel shard matches CPU shard (1e-3 rel L2)
  - [ ] Per-frame wall_s (CPU, T4) recorded below — CPU done, T4 pending

## Next up

1. Phase 2 — Phone + flat plate (Q-005 resolved inside the phase)
2. Phase 3 — Static hand + convergence (needs Q-008)
3. Phase 4 — Sequences + hand validation
4. Phase 5 — Ring + randomization + bulk generation (needs Q-001, Q-002, Q-014)

## Blocked

- Vault update for 2026-10-14 (Phase 1 local session): `scripts/obsidian_sync.py` exited 1, `OBSIDIAN_VAULT` unset in the cloud container. Next local session: append the 2026-10-14 log entry from git history (commits 2e07f14..HEAD), update `isef-overview.md` status (Phase 1 local criteria done, Kaggle pending), run the sync. No project-level decisions to add (all changes this session are implementation-level).

## Measurements

| Quantity | Value | Source |
|---|---|---|
| wall_s/frame CPU (p1_free) | 3.69 s (3 consecutive runs: 3.46 / 3.61 / 3.69; earlier dev runs 4.40, 6.44); grid 61×12×12 + PML, 1270 steps; 4-vCPU Xeon 2.1 GHz cloud container, OMP binary | `ringsim sim --job jobs/p1_free.yaml`, shard `wall_s`, 2026-10-14 |
| wall_s/frame T4 (p1_free) | — | Phase 1 |
| T4 smoke: 64³ × 1200 steps, smoothed plate | 2.7 s | smoke run 2026-10-14 |
| T4 smoke: 124³ × 1300 steps, smoothed plate | 13.0 s | smoke run 2026-10-14 |
| static noise floor n | — | Phase 2 |
| chosen dx | — | Phase 3 |
| frames / T4 GPU-hour (ring) | — | Phase 5 |

## Deviations from spec

All agreed with the user 2026-10-14; spec docs updated in the same change.
- `align_direct_path(stream, tx, band, fs, search_frames=2)`: takes band/fs (needed for the bandpass) and picks the earliest local max of the xcorr *envelope*, not of |xcorr| (PROCESSING.md).
- `rx_from_ir` passes a long Kaiser FIR to `resample_poly`; the default filter droops 21% at 22 kHz and failed the spec's own 3% test (PROCESSING.md).
- `cfmcw_profiles` correlation is linear (samples past the frame end = 0) (PROCESSING.md).
- `ConfigIncomplete` lists every TBD, so `ring` names Q-001 and Q-002 (DEVICES.md).
- `iphone17pro.yaml` and `ring.yaml` added in Phase 1 (verbatim from DEVICES.md) for the load_device criterion.
- Free/plate shards omit hand datasets; reader returns `labels=None`; `failed_reasons` dataset added (DATASET.md).
- Phase 1 profile criterion replaced: match an ideal monopole instead of "bin 0, nothing else > 10%" (unwindowed chirp sidelobes 26–29%; k-Wave mass source ≈ 90° phase puts the peak at bin 3) (BUILD.md).
- Temperature-scaling gain is 1/α at equal dt, not α² (PROCESSING.md).
- `reader.load_sequence(snr_db=...)` raises until Q-016 is answered.
- Local CPU k-Wave needs the system package `libhdf5-hl-100t64` (INFRA.md).

---

## Update protocol

Write to this file when:
- a phase completes → move to Done with date and deviations
- an acceptance criterion completes → check its box
- work stops mid-phase → set **Next action** to the specific next step (function or criterion), not an area
- something blocks → add to Blocked with its Q-ID
- implementation diverges from a spec doc → log under Deviations AND update that doc (standing rule 2)
- a number is measured → fill the Measurements table; never estimate into it
- after any of the above, at session end → vault log entry + `python scripts/obsidian_sync.py` (docs/OBSIDIAN.md)
