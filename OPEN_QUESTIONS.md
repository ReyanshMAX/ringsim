# Open Questions

Unresolved. Do not resolve these silently — ask, then move the answer to
DECISIONS.md (or the relevant config) and delete the entry here. Config fields
that depend on an entry are written as `TBD(Q-xxx)` and raise at load time.

Resolved 2026-11-11 and removed: Q-024 (forearm end, solids at the grid face → D-053), Q-017 (box faces on grid
points → D-067).
Resolved 2026-11-10 and removed: Q-018 (→ D-052).
Resolved 2026-10-14 and removed: Q-010 (→ D-035), Q-015 (→ D-036).
Resolved 2026-10-11 and removed: Q-003 (labels → D-025), Q-004 (phone geometry →
DEVICES.md from Apple drawings), Q-012 (phone chirp → D-022), Q-013 (sideload → D-027).

## Q-001: Ring placement details not stated in Ring-a-Pose

Known from the paper: right middle finger, width 11 mm, wall 2 mm, sensor section
3.58 mm, 3/4-circle band, speaker + mic side by side across the finger 5.4 mm apart,
facing the palm, speaker's inner edge centred on the finger.
Still unknown:
- which phalanx and where along it (`phalanx`, `along_bone_frac`); paper photos suggest
  the base of the finger but the text does not say
- angular extent of the thick sensor section (`sensor_section_arc_deg`)
- where the 1/4 gap is (`gap_center_deg`)
- speaker housing width (sets the speaker centre offset), and which side (radial/ulnar)
  the mic sits on → `points_mm` for both ports
- **Blocking:** Phase 5
- **Options:** ask the Cornell authors (already in contact); USound UT-P2019 datasheet for
  speaker width; Fig. 2 photos as a fallback, logged as an estimate in DECISIONS.md
- **Depends on it:** `ring.yaml`, ring mesh, port voxels

## Q-002: Ring-a-Pose processing details

Known: §3.4 (see PROCESSING.md "Ring-a-Pose parity"). Unknown: chirp window/taper and
sweep direction, causal vs zero-phase band-pass, signed vs magnitude correlation in echo
frames, whether t0 is re-detected per frame or fixed per session.
- **Blocking:** Phase 5 parity criterion. Applies to the raw profiles only: the referenced profiles
  used as sim model input (D-042) do not depend on these choices except the sweep direction.
- **Options:** their code if released; ask authors; otherwise keep current choices and
  log them as decisions with "unverified vs Ring-a-Pose" in the reason
- **Depends on it:** `ring.yaml` chirp window, `proc.cfmcw_profiles`

## Q-005: Phone transducer identification

Three unknowns, all settled by Protocol A (VALIDATION.md):
- which stereo channel drives only the bottom speaker, and whether the earpiece can be
  silenced at all (step 0, D-032)
- whether the speaker grille is on the side the drawing mapping puts it (step 0). Drawing re-read
  2026-11-13 (sheet 1, detail B, rendered): the bottom view uses screen-view x (third-angle projection;
  the camera lenses behind the edge sit at x_draw ≈ 31–47 and 50–66, matching the rear view's 14.37 / 32.36
  mirrored), so the left-hand mapping matches the drawing; step 0 still confirms it acoustically
- which physical mic is iOS's `bottom` data source; config assumes bottom mic 2 (D-023).
  The right-hand port group (x_draw 47.43, 49.69, 51.94, 54.20, 56.45) holds bottom mic 1; the
  "BOTTOM MIC 1" leader has two arrowheads, at the holes x_draw 49.69 and 51.94 (re-read 2026-11-13),
  so mic 1 is one or both of those two; which, if it matters, by single-hole covering (step 2)
- **Blocking:** Phase 2 plate sims
- **Depends on it:** `iphone17pro.yaml` ports, capture-app `txChannel`

## Q-006: Grid spacing

1.0 vs 1.5 vs 2.0 mm.
- **Blocking:** Phase 4 onward (Phase 3 resolves it)
- **Options:** convergence study on differential profiles (SOLVER.md, D-033)
- **Depends on it:** throughput, dataset size, Kaggle quota use

## Q-007: Dataset size target

- **Blocking:** no — needed before Phase 5 bulk generation
- **Inputs:** `python tools/kaggle_run.py bench` (ring-sized grids on T4, one and two GPUs;
  STATUS.md Measurements) gives snapshots per GPU-hour now, before any package code. GPU budget is
  the user's call later: if results look promising, more compute will be found, so the bench
  result is not compared against a quota here (user decision 2026-11-10).
- **Options:** Phase 5 throughput × available GPU hours; both T4s of "GPU T4 x2" in
  parallel (INFRA.md) doubles throughput; reference point: Ring-a-Pose
  trained user-independent models on ~18 h of data from 16 hands (§6.3.1), ≈ 5.4 M
  frames at 83.3 frames/s
- **Depends on it:** job sharding plan

## Q-008: Pose data and model licenses

MANO model files (MANO_RIGHT.pkl), GRAB, InterHand2.6M access.
- **Blocking:** Phase 3 (MANO), Phase 4 (sequences)
- **Options:** register for each; if a dataset is unavailable, raise the random-walk
  share and log it as a decision
- **Depends on it:** `geometry.poses` sources

## Q-009: Hand-sequence sim-vs-real acceptance threshold

- **Blocking:** Phase 4 sign-off
- **Options:** set from real-vs-real repeatability across takes/sessions
- **Depends on it:** Phase 4 acceptance


## Q-011: Transducer frequency response

Phone response measured in Phase 2. Ring parts (UT-P2019, ICS-43434) unmeasured.
- **Blocking:** no
- **Depends on it:** DR v2

## Q-014: Ring rotation randomization range

Ring-a-Pose does not report how far the ring rotates on the finger between wears.
- **Blocking:** Phase 5 jobs
- **Options:** ask authors; measure on yourself once the ring exists; temporary value
  logged as a decision
- **Depends on it:** `dr.ring_rot_deg`

## Q-016: Reference power for read-time noise SNR

DATASET.md: AWGN SNR is "relative to direct-path power in band", but the window the
direct-path power is measured over (samples around t0? whole direct-only stream?) is
not defined. `reader.load_sequence(snr_db=...)` raises NotImplementedError until set.
- **Blocking:** Phase 5 (load-time DR); Phase 2 if plate validation needs noise
- **Options:** power of the band-passed stream in a window of ±(main-lobe width) around t0
  in a direct-path-only sim; or power of the profile at bin 0
- **Depends on it:** `dataset.randomize.add_noise` callers

## Q-019: Snapshot spacing threshold `snapshot_disp_mm` (D-041)

Max MANO-vertex displacement allowed between consecutive IR snapshots. Known (scratch check
2026-11-10, analytic point reflectors, not k-Wave): ≤ 1.8 mm gives ≤ 7 % differential-profile
error up to 0.59 m/s; 1.2 mm gives 2 %. Unknown: how much error matters downstream, and the cost,
which depends on how fast the pose sources move.
- **Blocking:** Phase 4 sequence jobs (`motion.snapshot_disp_mm` is TBD and raises)
- **Options:** start from 1.8 mm; tighten if Protocol C stroke asymmetry disagrees; loosen only with
  evidence that the model does not care
- **Depends on it:** snapshots per frame, frames per GPU-hour (Q-007)

## Q-020: Voxel occupancy that follows sub-cell motion

Binary `mesh.contains` occupancy does not change when a surface moves less than about half a cell.
At median finger speed a surface moves ~0.2 mm per frame (dx = 2 mm), so sim differentials for slow
motion are quantized, and snapshot interpolation (D-041) cannot fix snapshots that jump in whole
cells. D-036's smoothing was calibrated on binary occupancy.
- **Blocking:** Phase 3 (moving sequences, convergence differentials); resolved inside Phase 2
- **Options:** inside-fraction from supersampling each cell (e.g. 4³ points); signed distance through
  a smooth step about one cell wide; analytic fractions for boxes (would also settle Q-017).
  Separately: how a fraction maps to ρ (linear in ρ gives 84× air at 10 % fill; linear in log ρ or
  in impedance are alternatives), and how the port-voxel check reads a fraction. D-036 only checked
  reflection strength; the smoothed density is already ~20× air one cell outside the surface, so
  the effective reflecting surface may sit ½–1 cell out. The Phase 2 check measures that offset.
- **Partly answered 2026-11-11:** boxes use analytic fractions, linear ρ, σ = 0.5 on top (D-067); the check's method
  is D-072. Still open: the method for hand meshes (Phase 3), and whether the Phase 2 T4 check passes.
- **T4 check 2026-11-12: FAIL with the linear map** (BUILD_ID 318bc88a27f6, STATUS Measurements): the echo delay falls
  ~0.4–1.0 µs per 0.2 mm step for 8 steps instead of rising 1.166 µs, then jumps; the double bounce agrees, so it is
  the medium, not the estimator. Reading (seats 2, 3; not yet tested): with ρ linear in frac a cell at 10 % fill is
  ~84× air (|R| ≈ 0.98), so the surface reflects at the cell boundary and moves in cell steps. Next: choose the
  fraction → ρ map (log ρ, a smooth step, or other; a 1D screening model is the first filter), then re-run the same jobs.
- **1D screening 2026-11-12 (seat 3, scratch model, not k-Wave; plane waves, no 3D edges or dispersion):** with σ 0.5 on
  ρ, linear ρ, log ρ, frac², frac³ and a √Z blend all fail; log ρ with the smoothing applied to log ρ passes at the jobs'
  lattice phase (slope −9 %, monotonic, offset 0.21 bin) but meets the 10 % slope gate at only 4 of 8 lattice phases
  (slope ±0.2, offsets ≤ 0.22 bin and monotonic at all 8). So the leading candidate is "log ρ, smoothed in log ρ", and the
  1.8 mm span (about one cell) limits the slope gate; more steps or a second lattice phase are the lever. Smoothing
  log ρ changes D-036's mechanism, so D-036 needs its own re-check with it.
- **Depends on it:** `geometry.voxelize.build_medium`, D-036, convergence study

## Q-021: Device-residual randomization after referencing (D-042)

The free-air reference cancels the device response common to all paths. Left over: transducer
directivity (echoes arrive from other directions than the direct path), paths only the real device
has (sound through the ring's plastic body), drift between the reference and the data.
- **Blocking:** Phase 5 read-time DR (`device residual` row in DATASET.md is TBD and raises)
- **Options:** size from Protocol A residual phase, Protocol B signed-shape r, free-air drift
  between session start and end, and the ring checks; a zero range is allowed only if those show
  the residual is below the static noise floor
- **Depends on it:** `dataset.randomize`

## Q-022: Near-field and ring checks before bulk generation (PROPOSAL, not decided)

Problem: the phone tests check echoes from a hand ~10 cm from a still phone. On the ring the speaker
is millimetres from skin, moves with the finger, and is often covered by it. Nothing checks that
regime against real measurements until the ring hardware exists, which is planned last, after the
expensive bulk dataset. If the sim is wrong there, the bulk data is wasted compute.
Proposal (audit 2026-11-10; the user asked to keep it as a proposal only):
- Protocol D, near-field phantoms, on the phone (would sit in Phase 3): wooden dowels Ø18 mm
  (finger-sized) at 0, 5, 10, 20, 40 mm from the bottom-edge ports; a pair 2 mm apart (adjacent
  fingers); one pressed over the mic hole (covered port). Dowels are exact cylinders, so the sim has
  no hand-model uncertainty. Compare referenced signed shape, peak bin, and the differential for a
  2 mm move, sim vs real; pass thresholds from real-vs-real repeatability.
- Method change: split Phase 5 into a pilot batch, then a minimal ring prototype (3D-printed band +
  the two transducers on the Teensy rig) captured and compared with the sim of that ring, then bulk
  generation. Replaces "hardware last" with "minimal ring before bulk, full hardware later".
- D-006 would get a note: an identical code path is necessary for phone results to transfer to the
  ring, not sufficient.
- **Blocking:** nothing yet; decide before Phase 5 bulk generation
- **Depends on it:** BUILD Phases 3 and 5, VALIDATION.md, the project method order

## Q-023: Wrist-bend randomization ranges (only if D-051 extends the forearm domain)

MANO has no wrist joint relative to the forearm; the forearm is extruded straight along the hand.
If the Phase 5 forearm-length check shows the forearm matters, wrist flexion/extension and
radial/ulnar deviation become per-sequence DR (forearm prism rotated at the wrist loop).
- **Blocking:** Phase 5, only if D-051's check fails
- **Options:** ranges from InterHand2.6M / GRAB global orientation relative to the arm if available;
  Ring-a-Pose's tested orientations (§6.3.4) as the extremes
- **Depends on it:** `dataset.randomize`, GEOMETRY forearm

## Q-025: `ringsim sim --frames a:b` semantics

ARCHITECTURE lists `--frames a:b` but no doc says what a partial frame range writes (which snapshots, the seq attrs,
one shard or several), and static scenes (`free`, `plate`) have one snapshot for all frames. Until answered,
`--frames` raises (raised by seat 2 2026-11-11; lead ruling: refuse, no default).
- **Blocking:** Phase 3/4 (frame ranges of hand sequences, splitting long jobs across kernels)
- **Options:** simulate snapshots of frames a..b-1 (+ the next frame start) into the same job's shard; or a separate
  shard per range, merged by a script
- **Depends on it:** `cli.py`, `ShardWriter` partial-sequence rules (DATASET.md)

## Q-026: Alignment offset of the sim free-air reference

`load_sequence` aligns the data stream and the `ref_ir` stream separately (`align_direct_path` on each). The direct path
of `cfmcw_profiles_ref` sits exactly at lag 0 only when the two offsets are equal; a difference of Δ samples shifts every
referenced profile by Δ lags. In the sim both streams start at t = 0, so the data offset could serve both; on the
phone the reference is a separate recording and must be aligned on its own (PROCESSING.md). Raised by seat 3 2026-11-11.
- **Blocking:** Phase 3 (a hand echo near the direct path can move the data's alignment); not Phase 1 (free scene:
  both streams identical)
- **Options:** keep separate offsets (same procedure as real); or the sim reader uses the data offset for the reference
- **Depends on it:** `dataset.reader`, Protocol C sim-vs-real comparison

## Q-027: Phone-rig measurements (plate thickness, clamp box)

The user's physical rig, measured once: the plate board's thickness (VALIDATION.md gives only "≥ 5 mm"; the sim uses
the real board's, D-071) and `clamp_box_mm` (outer bounding box of the clamp, ±1 mm, device-frame mm, D-047). Until
then `jobs/p2_plate_*.yaml` carry `TBD(Q-027)` for the plate thickness and `rig_boxes`, which raise at load.
- **Blocking:** Phase 2 real-comparison plate sims and the sim background (not the Q-020 check, which runs clamp out)
- **Options:** none; a measurement by the user, written to `session.json` (clamp) and the plate jobs
- **Depends on it:** `jobs/p2_plate_*.yaml`, `session.json`

## Q-028: Alignment shift when the plate echo is as strong as the direct path

`align_direct_path` takes the earliest envelope max >= 0.5·max. Synthetic checks (seat 3, d17a703; seat 4, 04a67f9):
the offset is stable for echoes up to 0.6× the direct path, shifts by +1 sample at >= 1.0×, and locks onto the echo
at 3×. A one-sample shift between a plate take and the background leaves a direct-path residue at bin 3 after
background subtraction, which metric 1 (peak beyond bin 2) can pick instead of the plate echo, in sim and real alike.
On the phone the plate echo should be well under the direct path, so it may not occur.
- **Blocking:** nothing yet; watch for peak bin 3 at 80 / 110 mm in the first real `validate plate` report
- **Options:** align background and plate takes on a shared direct-path template; exclude bins 0..3 from the
  peak search; sub-sample alignment
- **Depends on it:** `validation/plate.py`, the Phase 2 peak-bin criterion

## Q-029: Rig arms thicker than a cell (phone arm, plate support)

VALIDATION.md leaves the rod out of every phone sim because it is "thinner than a cell" (dx = 2 mm), and says nothing
about what holds the plate. The user's rig (not built yet, 2026-11-14): an 18 in ruler on edge running straight back along +y from a printed
holder on the phone's top 25 mm, ~6 cm of it clamped to the desk, desk edge ≥ 37.5 cm from the phone (D-091);
the plate would sit on a second arm
coming in along the speaker axis from beyond the plate, hidden behind it. The range window is 30 cm one-way
(`range_window_m` = 84·343/(2·48000) = 0.300 m, DEVICES.md), so the phone arm from ~150 mm to 300 mm from the
bottom-edge pair, and the plate dowel from the plate out to 300 mm, are inside it.
- **Blocking:** the Phase 2 plate sims and the sim background (same as Q-027), once the rig is built
- **Options:** phone arm: fold its first segment into `clamp_box_mm`; add a second rig box; or leave it unmodelled
  with a stated reason (behind the phone body) and the VALIDATION.md rod line reworded. Plate support: add it to the
  plate jobs as a box behind the plate; or leave it unmodelled (shadowed by the 100 mm plate) and remove it with the
  plate for the background takes so sim and real match (plate vs no plate)
- **Depends on it:** VALIDATION.md Rig, `session.json` `clamp_box_mm`, `jobs/p2_plate_*.yaml` `rig_boxes`
