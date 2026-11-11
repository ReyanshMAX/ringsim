# Build Plan

Phases are vertical slices. Each one runs end to end when complete.

On finishing a phase: verify every acceptance criterion, then update STATUS.md and
the Obsidian vault (docs/OBSIDIAN.md) before starting the next phase. When stopping
mid-phase, set STATUS.md **Next action** to the exact next criterion or function.
Kaggle runs are done by the agent with `tools/kaggle_run.py` (docs/INFRA.md), never handed to the user.

---

## Phase 0 — Kaggle runner on a real T4

**Scope**
- `tools/kaggle_run.py`, `kaggle/config.toml`, `tools/test_kaggle_run.py` already exist (D-038). This phase
  proves them on real Kaggle and fixes what breaks.

**Non-goals for this phase**
- No ringsim code. `selftest` ships its own script and needs no wheel.

**Acceptance criteria**
- [x] `python tools/test_kaggle_run.py` prints `ALL PASS`. (2026-11-10, Windows)
- [x] `python tools/kaggle_run.py doctor` has no FAIL rows except the pyproject/src row. (2026-11-10)
- [x] `python tools/kaggle_run.py selftest` exits 0 on a T4; `run_meta.json` shows `gpu` containing `T4`, Python 3.13.x,
      k-wave-python 0.6.2, and `ksim.cpu` and `ksim.gpu` both `finite: true`. (2026-11-10)
- [x] Python and k-wave-python versions recorded in STATUS.md match D-035, or D-035 is amended with the new values.

**Depends on:** none (user has run `kaggle auth login` once)

---

## Phase 1 — Free-field pipeline, local and Kaggle

**Scope**
- Package skeleton, `pyproject.toml` (pins per D-035, contents in INFRA.md), `ringsim` CLI entry point.
- `devices/test_pair.yaml` + `load_device` with TBD rejection; `iphone17pro.yaml` and `ring.yaml` verbatim from DEVICES.md.
- `scene: free` (no solids), `build_medium`, `run_ir`, `ShardWriter`, `reader.load_sequence`.
- All of `ringsim.proc` with its required tests.
- `jobs/p1_free.yaml` runs on a Kaggle T4 through `tools/kaggle_run.py`.

**Non-goals for this phase**
- No MANO, no phone geometry, no randomization, no capture app.

**Interface contracts established**
```
src/ringsim/devices/config.py      load_device(name) -> DeviceConfig
src/ringsim/geometry/voxelize.py   build_medium(scene, dx, c, margin_m) -> Medium
src/ringsim/solver/kwave_runner.py run_ir(medium, range_window_m, fs_device, gpu) -> SimResult
src/ringsim/proc/*                 all signatures in docs/PROCESSING.md
src/ringsim/dataset/writer.py      ShardWriter
src/ringsim/proc/physics.py        C_SIM, C_MIN, C_MAX, TEMP_RANGE_C, speed_of_sound, temperature_alpha
src/ringsim/dataset/reader.py      load_sequence(path, seq_id, snr_db, temp_c, rng) -> SeqSample
```

**Acceptance criteria**
- [ ] `pytest -q` passes, including all `test_proc_*` tests (incl. `test_proc_motion`, D-041, and
      `test_proc_reference`, D-042) and the isolation test.
- [ ] Every sequence in the p1_free shard has `ref_ir`; `load_sequence` works in both `mode="ref"` and `mode="raw"`.
- [ ] The p1_free shard (simulated for 48 kHz) read with `dev_override` at 44.1 kHz (600-sample chirp, 17–21 kHz) passes
      the same monopole check as the 48 kHz read (D-046).
- [ ] Isolation test passes, incl. the fresh-subprocess check that `import ringsim.proc` pulls in no sim dependency (D-045).
- [ ] `pytest -m slow tests/test_temperature_scaling.py` passes (D-037 check; through the runner with `--cmd`, D-052).
- [ ] `load_device("iphone17pro")` loads, dropping `bottom_mic1` and `front_mic` with warnings naming Q-005; `load_device("ring")` raises `ConfigIncomplete` listing Q-001 and Q-002.
- [ ] `ringsim sim --job jobs/p1_free.yaml` (test_pair, 100 mm spacing, dx 2 mm, 1 frame), CPU, run through the runner with
      `--cmd` (D-052), completes and writes a shard with every attr in DATASET.md.
- [ ] `ringsim profile` on that shard matches an ideal monopole (H = j·2πf·e^{−2πifτ}, τ = 0.100/c) run through the same proc path: envelope peak within 1 bin of the reference's, rel L2 < 10% after one complex gain (`validation/freefield.py`). The unwindowed 4 kHz chirp has 26–29% range sidelobes and k-Wave's mass source makes the direct path a time derivative (≈90° phase), so the peak sits near bin 3, not 0 (v1 finding).
- [ ] Recorded IR arrival time matches `0.100/c + PULSE_DELAY` within 2 sim samples.
- [ ] `python tools/kaggle_run.py run jobs/p1_free.yaml` exits 0 on a T4 with a matching `BUILD_ID` (INFRA.md) and writes `data/shards/p1_free.h5`; that shard's IR matches the CPU shard to 1e-3 relative L2.
- [ ] Per-frame `wall_s` for CPU and T4 written to STATUS.md.
- [ ] `python tools/kaggle_run.py bench` result in STATUS.md Measurements (ring-sized grids, 1 and 2 T4s; run early, it needs
      no package code). Re-run with measured scene sizes once Phase 3 builds real hand scenes, and again if the ring gate or
      the forearm check (Phase 5) changes dx or the domain.

**Depends on:** Phase 0

---

## Phase 2 — Phone + flat plate, sim vs real

**Scope**
- `iphone17pro.yaml` as written in DEVICES.md (from Apple's dimensional drawings); `BoxBody` incl. plateau; multi-hole ports.
- `scene: plate` + plate placement. Q-017 (box faces on grid points) resolved here.
- Capture app per CAPTURE_APP.md, CI unsigned IPA, sideload (D-027).
- `ringsim export-tx`, `ringsim real`, `validation/sweep.py`, `ringsim validate plate`.
- Protocols A and B (VALIDATION.md).

**Non-goals for this phase**
- No hand, no MANO, no sequences, no randomization.

**Interface contracts established**
```
src/ringsim/validation/real.py     load_capture(dir_or_wav) -> (stream: np.ndarray, meta: dict)
src/ringsim/validation/sweep.py    response(sweep_wav, capture_wavs) -> (freqs, mag_db)
src/ringsim/validation/metrics.py  plate_metrics(sim_profiles, real_profiles, distances) -> dict
capture-app/                       metadata JSON schema v1
```

**Acceptance criteria**
- [ ] All CAPTURE_APP.md acceptance items pass.
- [ ] Protocol A step 0: a tx channel drives only the bottom speaker (≥ 20 dB drop with the bottom grille covered, < 3 dB with the earpiece covered); speaker side confirmed or ports mirrored; D-032 channel recorded.
- [ ] Protocol A run; hole-cover test identifies the `bottom` mic; D-023 confirmed or amended; Q-005 deleted.
- [ ] Bottom-speaker response in 18–22 kHz passes the Protocol A pass condition.
- [ ] `ringsim validate plate` report exists; peak bin error ≤ 1 at all 5 distances.
- [ ] Plate report includes the signed-shape metric (referenced and raw) and the free-air drift between session start and end;
      Protocol A report includes the residual phase.
- [ ] `clamp_box_mm` measured and in `session.json`; plate sims and the sim background include it; sim profiles are
      background-subtracted like real ones (D-047).
- [ ] Range linearity slope within 5% of 2·fs/c.
- [ ] Static noise floor `n` computed and written to STATUS.md (reported in Phase 3, not a threshold).
- [ ] Q-017 resolved: plate and phone-body voxelization gives identical results with box faces on and off grid points (test).
- [ ] Q-020 resolved: sub-cell motion check on T4 through the runner. Plate stepped away from the phone in 10 steps of
      0.2 mm at dx = 2 mm; the plate-echo delay read from each IR changes monotonically with position, with slope within
      10% of 2/c, and sits within 0.25 bin (~0.9 mm) of a rigid plate's delay 2d/c (the offset is reported in mm: the
      density smoothing of D-036 can move the effective surface). D-036 re-measured with the new occupancy (echo/direct
      vs rigid prediction) and DECISIONS updated.

**Depends on:** Phase 1, Q-005 and Q-020 (both resolved inside this phase)

---

## Phase 3 — Single static hand + grid convergence

**Scope**
- MANO loading, `hand_mesh`, forearm attachment (D-031), phone-frame placement (hand beyond bottom edge), rejection rule.
- One real-pose converter (`scripts/convert_ih26m.py` preferred, D-043) and `scripts/build_pose_pca.py`; `random_walk`
  needs the PCA they produce (static use only here).
- Medium PNG slices; `ringsim converge`.
- Private Kaggle dataset `ringsim-assets` created with MANO files; `use_assets = true` in `kaggle/config.toml`.

**Non-goals for this phase**
- No sequences, no differential validation, no real pose datasets, no ring.

**Interface contracts established**
```
src/ringsim/geometry/mano_hand.py  hand_mesh(pose48, betas) -> (Trimesh, joints[21,3])
src/ringsim/geometry/forearm.py    attach_forearm(hand, j21) -> Trimesh
src/ringsim/geometry/poses.py      random_walk(n_frames, fps, seed, betas) -> PoseSequence
src/ringsim/geometry/scene.py      build_scene(dev, pose48, betas, placement) -> Scene
```

**Acceptance criteria**
- [ ] Converter `--check 50` passes (< 0.1 mm per vertex vs the source's own MANO settings); `pose_pca.npz` built.
- [ ] `attach_forearm(hand_mesh(...))` is watertight for 100 random poses, and has no faces in the wrist plane other than the stitched loop (test: no flat cap).
- [ ] For a flat-hand frame, the forearm is cut flat 3 cells inside the grid face it crosses, and no solid voxel lies
      within 3 cells of any face (guard, D-053); both visible in the medium PNG. A test scene with a solid at a face is
      refused by `build_medium` and by `run_ir` before k-Wave is called.
- [ ] Forearm end-echo check (D-051, D-053): forearm-only scene (flat-hand forearm, no hand), domain A as `build_medium`
      makes it (forearm cut 3 cells inside the face), domain B extended 150 mm further with the forearm continuing (cut
      the same way at its own face). Their raw IRs agree within 2 %
      relative L2 over delays shorter than the round trip from the ports to domain A's cut: the cut adds no echo.
- [ ] Contact handling (D-048): 100 random poses plus the 5 adducted / fist poses of `jobs/converge_poses.yaml` voxelize with
      no failure, and no air voxel is enclosed by hand voxels (connected-component test on the unsmoothed occupancy;
      catches self-intersection holes).
- [ ] Medium PNG for a flat-hand frame shows the hand and phone in the expected positions (checked by the user, noted in STATUS.md).
- [ ] `ringsim converge` runs on T4 through the runner for dx ∈ {2.0, 1.5, 1.0} plus the 0.75 mm reference check, on the random and adducted subsets, and writes `reports/converge/report.md` with differential e(dx) on referenced profiles (D-050), separation s, raw-profile gain per dx, static error, real noise floor n, the sub-cell motion case, and wall time per dx (SOLVER.md).
- [ ] dx chosen per SOLVER.md rule; DECISIONS.md updated; Q-006 deleted.

**Depends on:** Phase 2, Q-008 (MANO and one real pose dataset)

---

## Phase 4 — Hand sequences, differential profiles, phone validation

**Scope**
- `load_sequence` for at least one real source (GRAB or InterHand2.6M) + `random_walk` sequences.
- `snapshot_schedule` (D-041); per-snapshot shard writing.
- Open/close synthetic sequence generator for Protocol C.
- `ringsim validate hand`; Protocol C captures.

**Non-goals for this phase**
- No ring config, no domain randomization, no bulk generation.

**Interface contracts established**
```
src/ringsim/geometry/poses.py      load_sequence(source, fps_out, n_frames, seed) -> PoseSequence
src/ringsim/validation/metrics.py  hand_metrics(sim: SeqSample, real_takes: list) -> dict
scripts/convert_grab.py | convert_ih26m.py  → data/poses/<src>/<id>.npz (GEOMETRY.md format; ih26m from Phase 3)
```

**Acceptance criteria**
- [ ] A converted real sequence loads, resamples to the phone frame rate, and simulates 120 frames on T4 into one shard.
- [ ] Protocol C: 2 sessions × 5 takes captured; `hand_mm` measured and `scripts/fit_betas.py` betas stored.
- [ ] `ringsim validate hand` report: sim r (fitted betas and betas = 0), real-vs-real r, temporal 2 Hz check for both.
- [ ] Q-009 threshold set from the real-vs-real ceiling; sim r meets it. Q-009 deleted.
- [ ] Protocol C stroke asymmetry reported for real, sim `cubic` and sim `hold` (VALIDATION.md metric 4).
- [ ] Speed distribution of the pose sources (max MANO-vertex displacement per frame, in the device frame) reported for
      the real source and `random_walk`, with the snapshots-per-frame it implies.
- [ ] Q-019 resolved: `snapshot_disp_mm` chosen from `test_proc_motion` swept over thresholds, the stroke-asymmetry
      result, and the snapshot cost; DECISIONS updated; Q-019 deleted.

**Depends on:** Phase 3, Q-008, Q-019 (resolved inside this phase)

---

## Phase 5 — Ring config, randomization, sharded generation

**Scope**
- `ring.yaml` (Q-001, Q-002), `RingBody` mesh (D-024), ring-frame placement, labels (D-025).
- DR v1 (DATASET.md), `scripts/make_jobs.py` (70/30 real/walk), throughput benchmark.
- First bulk batch of shards.

**Non-goals for this phase**
- No DR v2 (transducer response, reflectors, sleeve). No training.

**Interface contracts established**
```
src/ringsim/dataset/labels.py      MANO_TO_MP20, canonical_frame(j21), labels_canonical(j21)
src/ringsim/dataset/randomize.py   sample_sequence_dr(job, rng), add_noise(...)
scripts/make_jobs.py               --n-seq N --real-frac 0.7 --out jobs/batch_<k>/
```

**Acceptance criteria**
- [ ] `load_device("ring")` succeeds (all TBDs resolved).
- [ ] Ring gate passed before any bulk job (SOLVER.md "Ring gate", D-050): report in `reports/converge_ring/report.md`;
      ring dx recorded in DECISIONS.md.
- [ ] Forearm-length check before any bulk job (D-051): ring config, 5 poses (2 random, flat, fist, wrist-facing curl),
      domain cut at wrist + 10 mm vs extended along the forearm until no forearm point is within the range window of the
      ports. Report the referenced-profile and differential difference against the ring gate's 0.25·s yardstick. If it
      exceeds it: extended domain becomes the rule, wrist-bend DR is added (Q-023), DECISIONS updated, benchmark numbers
      redone for the bigger domain. Otherwise D-031's domain stands.
- [ ] `proc.cfmcw_profiles` parity with Ring-a-Pose verified per Q-002 resolution (test against their reference output or published figure values).
- [ ] Medium PNG for a ring frame shows ring around the specified phalanx; every ring-band voxel is solid (skin touching
      the band is contact, D-048).
- [ ] Ring size is fixed per sequence (D-049): the ring mesh in ring frame is identical (vertex for vertex) across all
      snapshots of a sequence, for 10 random-walk sequences.
- [ ] Label test: `labels_canonical` on the MANO flat-hand pose puts index MCP on +x and the palm normal on +z; 100 random poses all give `hand_scale_m` in 0.06–0.12 m.
- [ ] Throughput benchmark (frames/GPU-hour on T4) in STATUS.md; dataset size set; Q-007 deleted.
- [ ] First batch generated; `load_sequence` with `snr_db` returns labels [T,20,3] and profiles for every shard (smoke script over all shards).
- [ ] First-batch report: failed-sequence rate by reason, and the fraction of snapshots with `port_covered` (D-048).

**Depends on:** Phase 4, Q-001, Q-002, Q-014, Q-016
