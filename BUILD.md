# Build Plan

Phases are vertical slices. Each one runs end to end when complete.

On finishing a phase: verify every acceptance criterion, then update STATUS.md and
the Obsidian vault (log entry + overview status + sync, docs/OBSIDIAN.md) before
starting the next phase. When stopping mid-phase, set STATUS.md
**Next action** to the exact next criterion or function.

---

## Phase 1 — Free-field pipeline, local and Kaggle

**Scope**
- Package skeleton, `pyproject.toml` (pins per D-035), CLI entry point.
- `devices/test_pair.yaml` + `load_device` with TBD rejection.
- `scene: free` (no solids), `build_medium`, `run_ir`, `ShardWriter`, `reader.load_sequence`.
- All of `ringsim.proc` with its required tests.
- `ringsim kaggle push/pull` for this job, shipping code as a wheel in the private `ringsim-code` dataset (D-034).

**Non-goals for this phase**
- No MANO, no phone geometry, no randomization, no capture app.

**Interface contracts established**
```
src/ringsim/devices/config.py      load_device(name) -> DeviceConfig
src/ringsim/geometry/voxelize.py   build_medium(scene, dx, c, margin_m) -> Medium
src/ringsim/solver/kwave_runner.py run_ir(medium, range_window_m, fs_device, gpu) -> SimResult
src/ringsim/proc/*                 all signatures in docs/PROCESSING.md
src/ringsim/dataset/writer.py      ShardWriter
src/ringsim/proc/physics.py        C_SIM, C_MIN, TEMP_RANGE_C, speed_of_sound, temperature_alpha
src/ringsim/dataset/reader.py      load_sequence(path, seq_id, snr_db, temp_c, rng) -> SeqSample
```

**Acceptance criteria**
- [ ] `pytest -q` passes, including all `test_proc_*` tests and the isolation test.
- [ ] `pytest -m slow tests/test_temperature_scaling.py` passes on CPU (D-037 check).
- [ ] `load_device("iphone17pro")` loads, dropping `bottom_mic1` and `front_mic` with warnings naming Q-005; `load_device("ring")` raises `ConfigIncomplete` naming Q-001.
- [ ] `ringsim sim --job jobs/p1_free.yaml` (test_pair, 100 mm spacing, dx 2 mm, 1 frame) completes on CPU and writes `data/shards/p1_free.h5` with every attr in DATASET.md.
- [ ] `ringsim profile` on that shard matches an ideal monopole (H = j·2πf·e^{−2πifτ}, τ = 0.100/c) run through the same proc path: envelope peak within 1 bin of the reference's, rel L2 < 10% after one complex gain (`validation/freefield.py`). Revised 2026-10-14: the unwindowed 4 kHz chirp has 26–29% range sidelobes, and k-Wave's mass source makes the direct path a time derivative (≈90° phase), which puts the peak at bin 3, not 0.
- [ ] Recorded IR arrival time matches `0.100/c + PULSE_DELAY` within 2 sim samples.
- [ ] Kernel log shows `python --version` and `pip show k-wave-python` matching the pins, and the wheel's `+g<hash>` matching HEAD.
- [ ] `ringsim kaggle push --job jobs/p1_free.yaml` runs on a T4 (kernel log shows T4), `kaggle pull` retrieves a shard whose IR matches the CPU shard to 1e-3 relative L2.
- [ ] Per-frame `wall_s` for CPU and T4 written to STATUS.md.

**Depends on:** none (Q-010 resolved → D-035)

---

## Phase 2 — Phone + flat plate, sim vs real

**Scope**
- `iphone17pro.yaml` as written in DEVICES.md (from Apple's dimensional drawings); `BoxBody` incl. plateau; multi-hole ports.
- `scene: plate` + plate placement.
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
- [ ] Range linearity slope within 5% of 2·fs/c.
- [ ] Static noise floor `n` computed and written to STATUS.md (reported in Phase 3, not a threshold).

**Depends on:** Phase 1, Q-005 (resolved inside this phase)

---

## Phase 3 — Single static hand + grid convergence

**Scope**
- MANO loading, `hand_mesh`, forearm attachment (D-031), phone-frame placement (hand beyond bottom edge), rejection rule.
- `random_walk` (static use only here).
- Medium PNG slices; `ringsim converge`.

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
- [ ] `attach_forearm(hand_mesh(...))` is watertight for 100 random poses, and has no faces in the wrist plane other than the stitched loop (test: no flat cap).
- [ ] For a flat-hand frame, forearm voxels touch the grid face (assert in `build_medium`), visible in the medium PNG.
- [ ] Medium PNG for a flat-hand frame shows the hand and phone in the expected positions (checked by the user, noted in STATUS.md).
- [ ] `ringsim converge` runs on T4 for dx ∈ {2.0, 1.5, 1.0} plus the 0.75 mm reference check, on the random and adducted subsets, and writes `reports/converge/report.md` with differential e(dx), separation s, static error, real noise floor n, and wall time per dx (SOLVER.md).
- [ ] dx chosen per SOLVER.md rule; DECISIONS.md updated; Q-006 deleted.

**Depends on:** Phase 2, Q-008 (MANO)

---

## Phase 4 — Hand sequences, differential profiles, phone validation

**Scope**
- `load_sequence` for at least one real source (GRAB or InterHand2.6M) + `random_walk` sequences.
- Open/close synthetic sequence generator for Protocol C.
- `ringsim validate hand`; Protocol C captures.

**Non-goals for this phase**
- No ring config, no domain randomization, no bulk generation.

**Interface contracts established**
```
src/ringsim/geometry/poses.py      load_sequence(source, fps_out, n_frames, seed) -> PoseSequence
src/ringsim/validation/metrics.py  hand_metrics(sim: SeqSample, real_takes: list) -> dict
scripts/convert_grab.py | convert_ih26m.py  → data/poses/<src>/<id>.npz (GEOMETRY.md format)
```

**Acceptance criteria**
- [ ] A converted real sequence loads, resamples to the phone frame rate, and simulates 120 frames on T4 into one shard.
- [ ] Protocol C: 2 sessions × 5 takes captured.
- [ ] `ringsim validate hand` report: sim r, real-vs-real r, temporal 2 Hz check for both.
- [ ] Q-009 threshold set from the real-vs-real ceiling; sim r meets it. Q-009 deleted.

**Depends on:** Phase 3, Q-008

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
- [ ] `proc.cfmcw_profiles` parity with Ring-a-Pose verified per Q-002 resolution (test against their reference output or published figure values).
- [ ] Medium PNG for a ring frame shows ring around the specified phalanx; ring wall not penetrated.
- [ ] Label test: `labels_canonical` on the MANO flat-hand pose puts index MCP on +x and the palm normal on +z; 100 random poses all give `hand_scale_m` in 0.06–0.12 m.
- [ ] Throughput benchmark (frames/GPU-hour on T4) in STATUS.md; dataset size set; Q-007 deleted.
- [ ] First batch generated; `load_sequence` with `snr_db` returns labels [T,20,3] and profiles for every shard (smoke script over all shards).

**Depends on:** Phase 4, Q-001, Q-002, Q-014
