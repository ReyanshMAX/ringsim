# Status

**Last updated:** 2026-10-11
**Current phase:** 1 of 5 — Free-field pipeline, local and Kaggle
**Next action:** Resolve Q-010 (k-wave-python and Python versions from the passing T4 smoke-test kernel log), then create `pyproject.toml` and the package skeleton per CLAUDE.md repo structure.

---

## Done

- Spec suite generated (2026-10-11).
- Obsidian vault set up: overview, log, decisions, spec mirror under `03-competitions/isef-ring-pose/` (2026-10-11).
- Spec audit fixes: forearm instead of wrist cap (D-031), single-speaker phone output (D-032), differential convergence criterion (D-033), wheel-based Kaggle install (D-034), Python version pinned to Kaggle (2026-10-12).
- Phone geometry, ring signal/label conventions, and sideload path filled from Apple drawings, Ring-a-Pose paper, and lapse-app (2026-10-11).

## In progress

- **Phase 1 — Free-field pipeline**
  - [ ] `pytest -q` passes incl. proc tests + isolation test
  - [ ] `load_device`: phone loads with Q-005 warnings; ring raises naming Q-001
  - [ ] CPU free-field shard written with full attrs
  - [ ] Direct-path peak at bin 0, nothing else > 10%
  - [ ] IR arrival matches 0.100/c + PULSE_DELAY within 2 samples
  - [ ] Kernel log versions + wheel hash match pins
  - [ ] T4 kernel shard matches CPU shard (1e-3 rel L2)
  - [ ] Per-frame wall_s (CPU, T4) recorded below

## Next up

1. Phase 2 — Phone + flat plate (Q-005 resolved inside the phase)
2. Phase 3 — Static hand + convergence (needs Q-008)
3. Phase 4 — Sequences + hand validation
4. Phase 5 — Ring + randomization + bulk generation (needs Q-001, Q-002, Q-014)

## Blocked

- **Q-010** k-wave-python + Python versions — blocks Phase 1 start
- **Vault sync pending (2026-10-12):** computer unreachable at session end. Next local session: append `_vault_pending/isef-log-2026-10-12.md` to `isef-log.md`, run `scripts/obsidian_sync.py`, delete the pending file.

## Measurements

| Quantity | Value | Source |
|---|---|---|
| wall_s/frame CPU (p1_free) | — | Phase 1 |
| wall_s/frame T4 (p1_free) | — | Phase 1 |
| static noise floor n | — | Phase 2 |
| chosen dx | — | Phase 3 |
| frames / T4 GPU-hour (ring) | — | Phase 5 |

## Deviations from spec

None.

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
