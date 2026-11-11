# HANDOFF — seat reports, newest round first

The lead (seat 1) owns this file's header and lane table. Every seat writes ONLY its own section of the current
round, edited from main's copy (`git show main:docs/HANDOFF.md`). If `scripts/private-commit.sh` refuses this file,
re-read main's copy, put back only your own section and commit again; never hand-merge another seat's section.
STATUS.md and the Obsidian vault are written by the lead only, from these sections.

## Lanes

A lane is a set of files one seat owns and nobody else edits. Need a change in someone else's file: post to its owner.

| Seat | Lane | Owns | Keeps current |
|---|---|---|---|
| 1 LEAD | plan, contracts, review | STATUS.md, DECISIONS.md, OPEN_QUESTIONS.md, BUILD.md checkboxes, CLAUDE.md, docs/ARCHITECTURE.md, docs/OBSIDIAN.md, docs/HANDOFF.md (header + lane table), scripts/*.sh, scripts/obsidian_sync.py, .gitignore, .gitattributes, nextsessions/NEXT-PROMPTS.md, the Obsidian vault. No feature code. | ARCHITECTURE |
| 2 SIM | devices, geometry, solver, CLI, Kaggle | pyproject.toml, src/ringsim/__init__.py, src/ringsim/cli.py, src/ringsim/devices/** (incl. yaml configs), src/ringsim/geometry/**, src/ringsim/solver/**, jobs/*.yaml, tools/kaggle_run.py, tools/test_kaggle_run.py, kaggle/config.toml, tests/test_devices*.py, tests/test_geometry*.py, tests/test_solver*.py, tests/test_cli*.py | DEVICES, GEOMETRY, SOLVER, INFRA |
| 3 PROC | ringsim.proc (shared with the hardware) | src/ringsim/proc/**, tests/test_proc_*.py (incl. test_proc_isolation.py), tests/test_temperature_scaling.py (slow; seat 2 runs it on Kaggle) | PROCESSING |
| 4 DATASET AND VALIDATION | shards, reader, free-field check, test harness | src/ringsim/dataset/** (incl. jobspec.py, Q-030), src/ringsim/validation/**, tests/conftest.py, every test file not owned above | DATASET, VALIDATION |
| 5 SPECIALIST | OFF this round | Phase 2 candidate: capture-app/**, .github/workflows/ios.yml | CAPTURE_APP |

Bootstrap: seat 2's skeleton (9f682af) created every subpackage's empty `__init__.py`. From that sha on,
`proc/__init__.py` is seat 3's and `dataset/__init__.py`, `validation/__init__.py` are seat 4's.

One owner each: Kaggle runs and the `ringsim-code` upload = seat 2 · full `pytest -q` before round end = seat 4 ·
BUILD.md boxes, STATUS.md, vault = seat 1 · push = the user · a global install = the seat that needs it, after the
user's yes in that seat's own chat.

---

# Round 1 — Phase 1 (2026-11-11)

Goal: every Phase 1 acceptance criterion in BUILD.md checked, each with the command that proved it.
Room: `#ringsim`. Main at round start: ea6dfc5.

## Seat 1 LEAD

(written at round end)

## Seat 2 SIM

## Seat 3 PROC

## Seat 4 DATASET AND VALIDATION

**On main (all in my lane):** 6e9942f conftest · 4ab63ef JobSpec + ShardWriter · 95e5267 writer tolerance + DATASET.md ·
17f3187 reader · 4888d46 freefield + tests · 2751496 reader ref stream exactly 51 frames · this section (the commit that carries it).

**Evidence**
- Full suite, worktree at main 0e66bec (round end): `python -m pytest -q -rs` -> 115 passed, 2 deselected in 18.47s, 0 skipped
  (the 2 deselected are the slow temperature tests, green on Kaggle per seat 2). Earlier: 103 passed at 2751496, 112 at 0a1c2ba, 115 at 609b02a.
- CPU shard (Kaggle CPU, build_id c12f529749d8, tree 9fec03d), my own run on a copy: attr dump vs DATASET.md static scene:
  file attrs / seq attrs / seq datasets missing: none, extra: none.
- `freefield_check` 48 kHz: raw peak bin 3 vs 3, rel_l2 0.0521, gain phase 0.120 rad, arrival -0.888 sim samples, pass True;
  ref (reported): peak 0/0, rel_l2 2.85e-05. 44.1 kHz override (D-058): raw 3/3, rel_l2 0.0474, arrival -0.888, pass True.
- T4 shard (build_id fb85a88049c9, tree 609b02a), my run on copies: `compare_ir(cpu, t4, "free")` = 1.331e-05 (< 1e-3);
  T4 freefield_check pass True, raw peak 3 vs 3, rel_l2 0.0521, arrival -0.888 samples, attr dump none missing / none extra,
  wall_s [2.3135] s. CPU and T4 trees differ in src only in cli.py (D-066
  wrapper) and proc/ir.py (read-time int check); geometry/ and solver/ identical, so the IR comparison is valid.

**What I built**
- `dataset/jobspec.py`: JobSpec, load_job, job_from_dict, job_yaml, ConfigIncomplete. Only `dx_mm` -> `dx_m` is converted; the
  nested blocks (placement, poses, motion, dr) stay in the YAML's units. When GEOMETRY.md types them they must convert to SI at
  load (ARCHITECTURE Notes).
- `dataset/writer.py`: ShardWriter per DATASET.md. Added (lead-approved): `write_ref_ir`, `write_snapshot(..., port_covered)`.
  Hand schedules raise NotImplementedError (D-025 label frame, Phase 3). No NaN check in the writer: D-066 puts it in the CLI.
- `dataset/reader.py`: load_sequence (ref/raw, temp_c, dev_override rules, partial-shard rule, SequenceFailed), list_sequences,
  SeqSample, `profiles_from_arrays` (shared with validation). snr_db not None -> NotImplementedError naming Q-016.
  The reference and the data are each aligned on their own direct-path offset (Q-026, a Phase 3 question).
- `validation/freefield.py`: freefield_check, compare_ir, shard_has_ref_ir, monopole_ir. Readings the lead confirmed: window
  rule = override window <= shard window; complex gain on analytic signals; arrival = argmax |hilbert(ir)| vs (d/c + PULSE_DELAY).
- `randomize.py`: nothing written (lead ruling; Phase 5).
- Tests: test_dataset_writer (9), test_dataset_reader (13), test_validation_freefield (7), conftest fixtures.

**Open / next**
- Phase 2 (not started): plate sequences (writer already takes schedule=None for `plate`), real-capture side of validation.
- Q-016 (noise model), Q-025 (--frames), Q-026 (reference offset).
- Worktree note: files I write on Windows get CRLF; `git diff --ignore-space-at-eol` is empty against main.

## Seat 5 SPECIALIST (off)
