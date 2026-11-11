# Status

**Last updated:** 2026-11-10
**Current phase:** 0 of 5 — Kaggle runner proven on a real T4
**Next action:** `python tools/kaggle_run.py doctor`, then `python tools/kaggle_run.py selftest`. Fix whatever
breaks in `tools/kaggle_run.py`, rerun `python tools/test_kaggle_run.py`. Then start Phase 1 (BUILD.md).

---

## Done

- Repo reset to specs on 2026-11-10; v1 code kept on branch `archive/v1` (D-039). At reset v1 had Phase 1's
  local code (`ringsim.proc`, devices, free-field medium, CPU `run_ir`, shard writer/reader) and 28 passing
  tests; none of it had run on Kaggle. Facts learned there are folded into the spec docs.
- `tools/kaggle_run.py` written; passes `tools/test_kaggle_run.py` against a fake CLI (2026-11-10).
  Not yet run against real Kaggle.
- Spec docs reworked 2026-11-10 (INFRA, BUILD, ARCHITECTURE, DECISIONS D-038..D-040, Q-018, vault single-writer rule).

## In progress

- **Phase 0 — Kaggle runner**
  - [ ] `doctor` passes on the user's machine
  - [ ] `selftest` completes: log shows T4, Python and k-wave-python versions match D-035, tiny k-Wave sim finite on CPU and GPU
  - [ ] `selftest` result recorded under Measurements

## Next up

1. Phase 1 — Free-field pipeline, local and Kaggle
2. Phase 2 — Phone + flat plate (Q-005 resolved inside the phase)
3. Phase 3 — Static hand + convergence (needs Q-008)
4. Phase 4 — Sequences + hand validation
5. Phase 5 — Ring + randomization + bulk generation (needs Q-001, Q-002, Q-014)

## Blocked

- Q-018: whether local CPU k-Wave runs on the user's Windows machine. Until answered, Phase 1 local criteria that
  need k-Wave on CPU may be run through `kaggle_run.py` instead.

## Measurements

Carried over from v1 (measured 2026-10-14; re-measure when the rebuilt code exists, do not cite as current).

| Quantity | Value | Source |
|---|---|---|
| wall_s/frame CPU (p1_free) | 3.69 s; grid 61×12×12 + PML, 1270 steps; 4-vCPU Xeon 2.1 GHz cloud container, OMP binary | v1 `ringsim sim --job jobs/p1_free.yaml` |
| T4: 64³ × 1200 steps, smoothed plate | 2.7 s | v1 smoke kernel |
| T4: 124³ × 1300 steps, smoothed plate | 13.0 s | v1 smoke kernel |
| wall_s/frame T4 (p1_free) | — | Phase 1 |
| static noise floor n | — | Phase 2 |
| chosen dx | — | Phase 3 |
| frames / T4 GPU-hour (ring) | — | Phase 5 |

## Deviations from spec

None yet in the rebuild. (v1's deviations are already written into PROCESSING.md, DATASET.md, DEVICES.md, BUILD.md.)

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
