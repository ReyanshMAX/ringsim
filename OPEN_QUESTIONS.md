# Open Questions

Unresolved. Do not resolve these silently — ask, then move the answer to
DECISIONS.md (or the relevant config) and delete the entry here. Config fields
that depend on an entry are written as `TBD(Q-xxx)` and raise at load time.

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
- **Blocking:** Phase 5 parity criterion
- **Options:** their code if released; ask authors; otherwise keep current choices and
  log them as decisions with "unverified vs Ring-a-Pose" in the reason
- **Depends on it:** `ring.yaml` chirp window, `proc.cfmcw_profiles`

## Q-005: Phone transducer identification

Three unknowns, all settled by Protocol A (VALIDATION.md):
- which stereo channel drives only the bottom speaker, and whether the earpiece can be
  silenced at all (step 0, D-032)
- whether the speaker grille is on the side the drawing mapping puts it (step 0)
- which physical mic is iOS's `bottom` data source; config assumes bottom mic 2 (D-023).
  The right-hand port group (x_draw 47.43–56.45) holds bottom mic 1; the drawing's
  leader to it is ambiguous between holes (step 2)
- **Blocking:** Phase 2 plate sims
- **Depends on it:** `iphone17pro.yaml` ports, capture-app `txChannel`

## Q-006: Grid spacing

1.0 vs 1.5 vs 2.0 mm.
- **Blocking:** Phase 4 onward (Phase 3 resolves it)
- **Options:** convergence study on differential profiles (SOLVER.md, D-033)
- **Depends on it:** throughput, dataset size, Kaggle quota use

## Q-007: Dataset size target

- **Blocking:** no — needed before Phase 5 bulk generation
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
