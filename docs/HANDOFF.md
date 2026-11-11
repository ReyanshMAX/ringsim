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

Shas on main (all through `scripts/private-commit.sh`; the first one through a scratch copy of the script with only the
mktemp fix, before 8db4701 fixed main's copy):
- 9f682af skeleton: pyproject.toml verbatim from INFRA, empty `__init__.py` x7, cli.py with all 6 subcommands
- 4ef88b3 devices: DeviceConfig + load_device / load_device_yaml (D-055) / device_yaml, 3 YAMLs verbatim; DEVICES.md
- b0d2b68 geometry (free_scene, build_medium with D-053 guard, D-036 smoothing, Medium.occ / port_covered D-061,
  save_medium_png) + solver (constants, pulse, grid, simulate, run_ir, wall_s D-063); SOLVER.md, GEOMETRY.md
- aadaacf runner: tests/ + pyproject.toml embedded as a tar in the kernel, `[sim,dev]` install (D-054 revised); INFRA.md
- 9fec03d CLI: `ringsim sim` / `ringsim profile` for scene free (D-056, D-060, D-062); medium_time_axis; SOLVER.md
- 609b02a CLI run-time failure check (D-066: run_error / nonfinite_ir); jobs/p1_tempscale.yaml; GEOMETRY.md

Evidence (command → output):
- `ringsim --help` → lists sim, profile, export-tx, real, validate, converge
- `pytest -q tests/test_devices.py` → 16 passed in 0.94s (4ef88b3)
- `pytest -q tests/test_solver.py tests/test_geometry.py tests/test_devices.py tests/test_cli_skeleton.py` → 47 passed in 5.53s
- `python tools/test_kaggle_run.py` → ALL PASS (15 checks) (aadaacf)
- full `pytest -q` on main 609b02a → 115 passed, 2 deselected in 12.69s
- `python tools/kaggle_run.py doctor` → all ok
- `python tools/kaggle_run.py --dry-run run jobs/p1_free.yaml --cmd "ringsim sim --job {job} --out-dir {out}"` → build
  c12f529749d8 (9fec03d), DRY calls only
- CPU: `python tools/kaggle_run.py run jobs/p1_free.yaml --cmd "ringsim sim --job {job} --out-dir {out}"` (tree 9fec03d)
  → EXIT=0, rc 0, BUILD_ID c12f529749d8 = got_build_id, Python 3.13.16, k-Wave-python 0.6.2, kernel wall_s 79.4.
  Log: "p1_free: grid (61, 12, 12), dx 2.00 mm, n_t 1433, k 12, CPU", "free snapshot 0: wall_s 7.69",
  "free ref_ir: wall_s 7.67"
- T4: `python tools/kaggle_run.py run jobs/p1_free.yaml` (tree 609b02a) → EXIT=0, rc 0, BUILD_ID fb85a88049c9 =
  got_build_id, Tesla T4, kernel wall_s 68.6; shard wall_s [2.314] s, ref_ir call 1.23 s
- CPU and T4 trees differ in src only in cli.py (D-066 wrapper) and proc/ir.py (read-time int check); geometry/ and
  solver/ are identical, so the IR comparison is valid
- `compare_ir(cpu_h5, t4_h5, "free")` → 1.331e-05 (< 1e-3)
- `freefield_check(cpu_h5, "free")` → pass True; raw peak bin 3 vs 3, rel_l2 0.0521, gain phase 0.120 rad; ref rel_l2
  2.85e-05; arrival_err -0.888 sim samples. 44.1 kHz override (D-058) → pass True, raw rel_l2 0.0474, ref 7.57e-05.
  T4 shard → pass True, raw 3 vs 3, rel_l2 0.0521, arrival -0.888
- `ringsim profile --shard <cpu_h5> --seq free --out profile.png` → rc 0, JSON pass True
- slow test: `python tools/kaggle_run.py run jobs/p1_tempscale.yaml --cmd "pytest -q -s -m slow tests/test_temperature_scaling.py"`
  (tree 609b02a, contains 871ff4a) → EXIT=0, rc 0, BUILD_ID fb85a88049c9 = got_build_id, kernel wall_s 147.3;
  "alpha=0.98251 n_t=1275 fs_sim=576000 err_scaled=0.0455% err_unscaled=4.7786% |g|=1.01786 1/alpha=1.01780";
  "2 passed, 12 warnings in 76.61s". The 12 warnings, all known and harmless: k-wave-python 0.6.2 internal
  `DeprecationWarning: Call to deprecated function ... time_reversal_boundary_data / time_rev (Use TimeReversal class
  instead) -- Deprecated since version 0.4.1`, raised inside kwave/kWaveSimulation.py lines 60, 208 (x4), 659, 667, 1187,
  2 per k-Wave call; we never use time reversal. Not the kspaceFirstOrder3D FutureWarning (that one is silenced, SOLVER.md).

Deviations from spec: none. Shapes added beyond the first spec text were each posted and ruled first (D-055 device_yaml,
D-061 Medium.occ/port_covered, simulate, free_scene, medium_time_axis, D-066); each spec doc changed in the same commit.

Open in my lane:
- Q-025: `--frames` on static scenes is refused until Phase 3/4 defines frame ranges.
- The uv venv has no pip; run tools/kaggle_run.py and tools/test_kaggle_run.py with the system `py -3.13`.

next: Phase 2 (BUILD.md) in my lane: BoxBody + plateau voxelization, multi-hole ports, scene plate + placement,
Q-017 on/off-grid box test, Q-020 sub-cell plate-step run on T4, plate sims for `validate plate`.

## Seat 3 PROC

Shas on main (lane files only): 5626762 physics + ChirpSpec/chirp · 8128abf ir + profile + required tests ·
99da2d7 slow temperature-scaling test · 44b8235 review fixes (isolation subprocess also forbids ringsim.geometry /
solver / dataset / validation / devices; tx_frame and export_sweep_wav refusals) · 0a1c2ba non-integer sample rates
refused, synthetic-bins test on signed peaks · 871ff4a fixture bar 2 at c + (24, -6, -26) mm (user).

Done: every PROCESSING.md signature. physics (C_SIM, TEMP_RANGE_C, speed_of_sound, C_MIN, C_MAX, temperature_alpha);
chirp (ChirpSpec, synth_chirp, tx_frame, export_tx_wav, export_sweep_wav); ir (rx_from_ir, temperature_scale,
snapshot_weights, rx_stream); profile (align_direct_path, frame_stream, cfmcw_profiles, free_air_reference,
cfmcw_profiles_ref, magnitude, differential). Helpers private (_band_mask, _bandpass_sos, _resample_fir, _ratio).
Imports: stdlib, numpy, scipy, ringsim.proc only.

Evidence (`pytest -q <file>`, worktree at 0a1c2ba; unchanged since except the slow test's fixture):
- tests/test_proc_chirp.py: 8 passed in 1.10s
- tests/test_proc_ir.py: 11 passed in 2.07s
- tests/test_proc_isolation.py: 2 passed in 3.27s
- tests/test_proc_motion.py: 3 passed in 5.21s
- tests/test_proc_profile.py: 3 passed in 1.04s
- tests/test_proc_reference.py: 3 passed in 1.21s
- tests/test_temperature_scaling.py (slow, seat 2 on Kaggle, tree 609b02a, BUILD_ID fb85a88049c9):
  `python tools/kaggle_run.py run jobs/p1_tempscale.yaml --cmd "pytest -q -s -m slow tests/test_temperature_scaling.py"`
  -> EXIT 0; "alpha=0.98251 n_t=1275 fs_sim=576000 err_scaled=0.0455% err_unscaled=4.7786% |g|=1.01786
  1/alpha=1.01780"; "2 passed, 12 warnings in 76.61s" (< 0.5 %; |g|·α − 1 = 6e-5 < 1 %)
- Full suite (seat 4, main 0e66bec): 115 passed, 2 deselected, 0 skipped.
Measured by those tests: motion differential error 0.0163 (frame-start, v 0.1, < 0.05), 0.0644 (T/2, v 0.3, < 0.10),
hold 1.3647 (> 0.5); reference max rel L2 over 50 device responses 5.19e-3 (< 1e-2), self-reference 3.59e-3 (< 1e-2);
rx_from_ir in-band |Y/X| 0.997-1.000 at 50/48/44.1 kHz (< 3 %); signed profile peaks at bins 1 and 19 (0 and 20, ±1).
End to end (seats 2 + 4, not my tests): the proc chain on the real CPU and T4 p1_free shards passes freefield_check
(raw bin 3 vs 3, rel L2 0.0521; 44.1 kHz override 0.0474).

Spec changes, PROCESSING.md, each in the same commit as its code: snapshot_weights "hold" = D-065 (8128abf);
the rx_from_ir test pads tx with 100 zeros each side (lead confirmed; unpadded in-band ratio 0.91-1.04) (8128abf);
export_sweep_wav dur_s < 20 ms raises ValueError (44b8235); rx_from_ir / rx_stream raise on a non-integer fs_sim or
fs_device: grid.time_axis gives 1/dt = 799999.9999999999 at dx 1.5 mm / 50 kHz, which int() truncated (0a1c2ba);
temperature fixture bar 2 moved 1 mm per axis and c defined as 65 mm (871ff4a).
Decisions used: D-064 (motion test: ring timing, window "none"), D-065 (hold). Deviations from spec: none left open.

Open: Q-002 (ring chirp window and Ring-a-Pose details; test_proc_motion revisits it); Q-026 (reader aligns the ref
stream on its own offset; a data/ref offset difference shifts referenced profiles by that many lags; Phase 3).
Next: Phase 2 proc support (export-tx / sweep WAVs, real 48 kHz captures through proc) and call-site reviews.

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
