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
| 3 PROC | ringsim.proc (shared with the hardware) | src/ringsim/proc/**, tests/test_proc_*.py (incl. test_proc_isolation.py), tests/test_temperature_scaling.py (slow; seat 2 runs it on Kaggle), src/ringsim/validation/subcell.py + tests/test_validation_subcell.py (D-082, from round 2) | PROCESSING |
| 4 DATASET AND VALIDATION | shards, reader, free-field check, test harness | src/ringsim/dataset/** (incl. jobspec.py, Q-030), src/ringsim/validation/**, tests/conftest.py, every test file not owned above | DATASET, VALIDATION |
| 5 SPECIALIST | iOS capture app (from round 2) | capture-app/**, .github/workflows/ios.yml, docs/CAPTURE_APP.md | CAPTURE_APP |

Bootstrap: seat 2's skeleton (9f682af) created every subpackage's empty `__init__.py`. From that sha on,
`proc/__init__.py` is seat 3's and `dataset/__init__.py`, `validation/__init__.py` are seat 4's.

One owner each: Kaggle runs and the `ringsim-code` upload = seat 2 · full `pytest -q` before round end = seat 4 ·
BUILD.md boxes, STATUS.md, vault = seat 1 · push = the user · a global install = the seat that needs it, after the
user's yes in that seat's own chat.

---

# Round 2 — Phase 2 (2026-11-11)

Goal: every Phase 2 acceptance criterion in BUILD.md checked, each with the command (or, for [device] and protocol
items, the user's report) that proved it. Room: `#ringsim`. Main at round start: a071e9e.
Decided with the user at round start: D-067..D-073 (occupancy, sweep, plate-metric shapes, real-capture details, plate
scene, Q-020 method, capture-app details). New: Q-027 (rig measurements: plate thickness, clamp box).

## Seat 1 LEAD

## Seat 2 SIM

## Seat 3 PROC

Shas on main (all through `scripts/private-commit.sh`):
- d17a703 `tests/test_proc_real.py`: synthetic real-capture call path (float32 48 kHz WAV, own random start offset per recording)
- 28d66a0 PROCESSING.md Notes: real-capture call sequence (per-recording align, REF frames, frame-0 skip, alignment limits)
- c4dc235 `magnitude()` docs: also feeds the Protocol B peak-bin metric, never the model
- 3d6ff42 D-086 `align_direct_path(stream, tx, band, fs, search_window_s=0.25)`; PROCESSING.md signature, Notes and required tests; `tests/test_proc_align.py`
- 0b0e896 `validation/subcell.py` + `tests/test_validation_subcell.py` (Q-020 analysis, D-072 / D-076 / D-082; seat 2's draft, fixed)
- 3ca680e `subcell.envelope_lag` zero-pads before the Hilbert transform (as align does)
- this commit: HANDOFF section + PASTE-SEAT3-2026-11-11.txt

Evidence (`pytest -q <file>`, worktree at main 60b547a):
- tests/test_proc_chirp.py: 8 passed in 1.34s
- tests/test_proc_ir.py: 11 passed in 2.10s
- tests/test_proc_isolation.py: 2 passed in 3.56s
- tests/test_proc_motion.py: 3 passed in 2.48s
- tests/test_proc_profile.py: 3 passed in 1.23s
- tests/test_proc_reference.py: 3 passed in 2.17s
- tests/test_proc_real.py: 8 passed in 2.07s
- tests/test_proc_align.py: 19 passed in 1.66s
- tests/test_validation_subcell.py: 12 passed in 13.67s
- tests/test_temperature_scaling.py (slow, seat 2 on Kaggle): unchanged since round 1 (BUILD_ID fb85a88049c9)
- Full suite on my worktree before the plate-test fix: 229 passed, 2 deselected, 1 failed (`test_truncated_take_raises_naming_it`,
  seat 4's file, caused by D-086 making `align_direct_path` raise; seat 4 fixed it in 60b547a). Seat 4 runs the last full suite.

Measured (synthetic, `tests/test_proc_real.py`, `-s`):
- Protocol B peak bins, background-subtracted, referenced: 80 / 110 / 140 / 170 / 200 mm -> 22 / 31 / 39 / 48 / 56 (ideal 22.4 /
  30.8 / 39.2 / 47.6 / 56.0); slope 283.3 bins/m vs 2fs/c 279.9; same with frame 0 kept or dropped
- REF from two free-air takes with offsets 140 and 58: rel L2 4.40e-04
- a 5-sample offset error between data and REF moves the peak from bin 31 to bin 6
- frame 0 of a take that starts with silence: peak bin right, rel L2 0.437 to the steady-state profile (D-070)
- alignment vs a plate echo of 0.3-3 x the direct path: stable up to 0.6x, +1 sample at 1.0-1.5x, locks onto the echo at 3x (Q-028)
- `export_tx_wav` / `export_sweep_wav` output: float32 mono 48 kHz, 600 and 96000 samples, peak 1.0; the WAVs committed in
  capture-app/Resources/tx are byte-identical (sha256 216756b8f43e43e3..., ca688080ed1bec67...)
- sweep spectrum 16-23.9 kHz flat within 2.2 dB, 23.9-24 kHz -107 dB: no defect at f1 = Nyquist (lead's Q1 ruling)
- D-086: first chirp at 0, 200, 500, 587, 600, 650, 777, 1000, 1500, 2400, 6000, 11980 samples -> offset = start + 13 (the
  causal band-pass delay); 12000, 12100, 13800, 24000 and noise-only streams raise ValueError. The old code returned -587, -32,
  -141, -436, -568, -1457, -2399, -4955, -23992 relative to start for 587..24000
- `align_direct_path` contrast threshold ALIGN_MIN_PEAK_TO_MEDIAN = 8: noise-only streams 3.3-4.2 (50 draws), a chirp at 0 dB
  per-sample SNR 14-15, at -20 dB 3.5 (refused)
- subcell `envelope_lag`: |lag - tau| <= 1e-4 sim samples over sub-sample phases at delays 20 / 200 / 466 us (fs_sim 571.7 kHz); seat
  2's draft read 5 us as 15.6 us because it cut the cross-correlation at lag 0 before the Hilbert transform

Reviews posted (no defects unless noted): c95284d, bb3c857 (seat 4); b5d5c37 (seat 2); 04a67f9 (seat 4: free_air_<n> must sort
on the integer, D-079 noise floor and drift inputs, D-083 length check); 6e847dc (WAVs byte-identical to proc's writers); c64a4db.

Spec changes (PROCESSING.md, each in the same commit as its code): D-086 signature, Notes and required tests; `magnitude` wording;
the real-capture call sequence under Notes.
Decisions used: D-042, D-046, D-064, D-065, D-068, D-070, D-079, D-082, D-083, D-086.

Open:
- Q-028: watch for peak bin 3 at 80 / 110 mm in the first real `validate plate` report (direct-path residue, not a proc bug).
- Q-026 (reader offsets, Phase 3) and Q-002 (Ring-a-Pose details) need a USER ANSWER.
- D-086's 250 ms margin is unchecked against a real take: the first device take's leading silence (WAV) and `input_latency_s` /
  `output_latency_s` (JSON) settle it.
- subcell.py has not run on the 11 T4 shards (seat 2); a failing slope or offset is a finding about D-036 / voxelization, not
  about the estimator (the fake-k-Wave tests recover +/-0.9 mm shifts to 0.02 mm).
Next: Phase 2 device criteria as the user's captures arrive (Protocol A, Protocol B, CAPTURE_APP acceptance); subcell numbers.

## Seat 4 DATASET AND VALIDATION

Shas on main (all through `scripts/private-commit.sh`):
- cbe4a7a `validation/real.py` load_capture (.wav only) + `tests/test_dataset_plate.py` (static plate shard, S = 1; writer/reader needed no code change)
- c95284d `validation/sweep.py` (response, phase_residual), `validation/metrics.py` (plate_metrics, noise_floor, free_air_drift), `load_session`; VALIDATION.md
- bb3c857 load_capture adds `meta["session"]` (session.json beside the take, None if absent)
- 74b4c37 metrics docstrings (D-079: noise floor not background-subtracted; drift frames >= n_avg + 1)
- 04a67f9 `validation/plate.py` plate_inputs / plate_report (D-070, D-078..D-080); take names in VALIDATION.md
- e8104a7 D-083: take with < 3 frames after alignment raises naming it; report key `noise_floor_basis` "pooled over takes"
- 77432a7 `validation/protocol_a.py` (+ `sweep.noise_floor_response`, no alignment, mean power, D-089), rgrille / mirrored (D-087), report.md; load_capture accepts null tx_file / tx_sha256 (D-088); VALIDATION.md
- 60b547a plate.py re-raises align_direct_path's ValueError naming the take (`from err`); plate test starts up to 3000 samples (D-086)
- this commit: this section + `nextsessions/PASTE-SEAT4-2026-11-11.txt`

Evidence (command -> output):
- `pytest -q -rs` on main a071e9e (job 0) -> 115 passed, 2 deselected, 0 skipped
- `pytest -q` on main 1c15710 -> 197 passed, 2 deselected; on 89cc68e + mine -> 206 passed; on 60b547a -> 240 passed, 2 deselected, 6 warnings in 155.41s (0 skipped)
- my lane files (`pytest -q tests/test_dataset_reader.py tests/test_dataset_writer.py tests/test_dataset_plate.py tests/test_validation_freefield.py tests/test_validation_real.py tests/test_validation_sweep.py tests/test_validation_metrics.py tests/test_validation_plate.py tests/test_validation_protocol_a.py`) -> 63 passed in 88.85s
- not run: the full suite after 94a00ed (docs and PASTE files only since 60b547a, plus seat 3's 3ca680e which was already in the 60b547a run)

Reviews done (read-only, no compiler for Swift): 6e847dc (4 findings, all fixed: mic permission 1c15710, UILaunchScreen + SWIFT_VERSION 446d7fb, fail() order 1c15710), 1c15710, 446d7fb, 22660b7 (no defects; one device risk: a no-playback take has no player, so if it records empty connect inputNode to a muted mixer), 89cc68e `ringsim real` (calls load_capture / noise_floor as signed).

Decisions used: D-068, D-070, D-074, D-078, D-079, D-080, D-083, D-086, D-087, D-088, D-089.

Open items:
- Q-028: with the plate echo as strong as the direct path (80 / 110 mm at gain 1 in the synthetic rig) alignment moves a sample against the background take and leaves a direct-path residue at bin 3, which metric 1 (peak beyond bin 2) picks, in sim and real alike. My plate test uses echo gain 0.3 for this reason.
- Everything here ran on synthetic WAVs and synthetic shards. Real behaviour of sweep alignment, plate_report and protocol_a is unknown until the user's captures arrive.
- Protocol A cannot test a no-playback floor on real data until the app build with "none (no playback)" is installed.
- The 6 warnings in the full suite come from other seats' tests and were not looked at.

## Seat 5 SPECIALIST (capture app)

Shas on main (all through `scripts/private-commit.sh`, lane files only):
- 6e847dc capture-app/ (project.yml, Info.plist, Sources: RingCaptureApp, ContentView, AudioRig, Recording,
  DebugLog), Resources/tx/ WAVs, .github/workflows/ios.yml, CAPTURE_APP.md (D-073, D-074 rules)
- 1c15710 mic permission at launch (take refused unless granted); abort + delete on AVAudioEngineConfigurationChange
  and media-services reset; file released before delete in fail()
- 446d7fb D-084 `UILaunchScreen: {}` + `SWIFT_VERSION: "5.0"` (project.yml and the CAPTURE_APP block together);
  D-085 measurement-off recipe in CAPTURE_APP "Acceptance"
- 22660b7 D-088 tx file "none (no playback)": no player, `tx_file` / `tx_sha256` null, schema_version 1, 18 keys

Evidence (command → output):
- `py -3.13 lanecheck.py <wt>` (scratch; extracts the CAPTURE_APP yaml blocks) → project.yml and ios.yml identical to
  the spec blocks, yaml.safe_load ok, Info.plist holds every project.yml info property, LANECHECK PASS (at 22660b7)
- `ringsim export-tx --device iphone17pro --out capture-app/Resources/tx/` → phone_tx.wav 48000 float32 (600,) peak
  1.0, sha256 216756b8f43e43e3…; sweep_16k_24k_2s.wav 48000 float32 (96000,) peak 1.0, sha256 ca688080ed1bec67…;
  `git show main:<wav> | sha256sum` gives the same (seat 3 checked them byte-identical to proc's writers)
- metadata keys = seat 4's `META_KEYS` in validation/real.py (18, cbe4a7a)
- CI: NOT RUN. No Swift compiler here; the app has never compiled. `git ls-remote origin refs/heads/main` →
  a071e9e at round end: the user had not pushed, so `gh run list --workflow ios.yml` → 404 (workflow not on the
  default branch yet). Reviews without a compiler: seat 4 (findings fixed in 1c15710, 446d7fb).

Decisions used: D-073, D-074, D-080 (take names; duplicate `_2` stems raise), D-084, D-085, D-086 (leading silence
handled in proc, app unchanged), D-087 (Protocol A labels), D-088.

Open in my lane:
- CAPTURE_APP acceptance (BUILD.md Phase 2 box 1): CI green → [device] steps (sent to the lead, held until green) →
  user's accept_10s take → `load_capture` length 480000 + `ringsim real` direct_peak_bin 0 → leading-silence length
  vs the D-086 0.25 s window → D-085 measurement-off branch (only after the acceptance take).
- Not verified anywhere yet: XcodeGen's flat copy of Resources/tx into the bundle root (the app lists bundle-root
  WAVs), AVAudioEngine start with no player (D-088 take), Swift 5 compile of the notification / tap closures.

next: CAPTURE_APP acceptance, starting with the CI result after the user's push.

---

# Round 1 — Phase 1 (2026-11-11)

Goal: every Phase 1 acceptance criterion in BUILD.md checked, each with the command that proved it.
Room: `#ringsim`. Main at round start: ea6dfc5.

## Seat 1 LEAD

Round result: **Phase 1 done**; all 12 BUILD.md Phase 1 boxes checked, each with its command and output inline (last: ef926ef).

- Shas: 8db4701 (this file, bench box, `private-commit.sh` mktemp fix for Git Bash) · a419a35, 22af4bc, 214ff1c, 486a3c7
  (D-054..D-066, Q-025, Q-026, ARCHITECTURE contracts) · 7a62b44, d2403ce, 0e66bec, ef926ef (boxes, STATUS).
- Decided with the user this round: D-054 (slow tests: tests tar embedded in the kernel script; first form superseded
  the same day), D-055 `load_device_yaml`, D-056 `ref_ir` always its own call, D-057 monopole gates on raw, D-058 the
  44.1 kHz override (window rule reworded to match the reader), D-059 JobSpec in dataset/, D-060 free-field API and
  seq_id "free", D-061 Medium.occ + port_covered, D-062 CPU limit by total cells, D-063 `wall_s` = kspaceFirstOrder3D,
  D-064 motion-test window none, D-065 "hold" = zero-order hold, D-066 run-time failures and `nonfinite_ir`. Spec fix:
  bar 2 of the slow-test fixture moved 1 mm (PROCESSING.md).
- Open, for later phases: Q-025 (`--frames` meaning, Phase 3/4), Q-026 (sim reference offset; ask the user before
  Phase 3).
- Process: the user talks only to the lead. Every seat question goes to the lead in `#ringsim`, and answers come back
  tagged "USER ANSWER". This is in the canonical RULES of every round-2 PASTE file. I standardized the RULES block
  across all five files (seat 4's text + "the capture app is seat 5's").
- Seat 5 launched for round 2: capture app (`nextsessions/PASTE-SEAT5-2026-11-11.txt`).
- Outside the repo: claude-multiplayer's `server.ts` / `host.ts` patched locally (`fileURLToPath` for the broker path;
  `new URL(...).pathname` gave "/C:/..." on Windows). A `git pull` there undoes it.
- Vault: log entry for 2026-11-11, overview status (Phase 2 of 5); `python scripts/obsidian_sync.py` exit 0.

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
