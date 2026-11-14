# VALIDATION.md — phone rig, capture protocols, sim-vs-real metrics

## Overview

Proves the sim reproduces real echo structure on an iPhone 17 Pro (D-007) before
any ring data is generated. Three protocols: transducer sweep (Q-005, Q-011), flat
plate at known distances (Phase 2), hand open/close (Phase 4). Real captures are
processed with the same `ringsim.proc` code as sim (D-008).

## Non-goals

- Not a pose-accuracy evaluation (no model exists here).
- No ring-hardware validation (later suite, once the ring is built).
- No claims of sim fidelity beyond what these metrics show; report them as measured.
- No near-field or ring-on-finger check yet. These protocols test echoes from targets ≥ 8 cm
  from a still phone; a near-field phantom protocol and a ring pilot before bulk generation are
  proposed in Q-022 (not adopted). Until then, do not claim the sim is validated for the ring.

## Rig

- Phone held in a clamp on a thin rod, free air, ≥40 cm from table, walls, body (D-018).
  The clamp grips only the top 25 mm of the phone (y_dev ≥ 50.0 mm), as far from the bottom-edge
  pair as possible. Its outer bounding box is measured once (±1 mm) and stored in `session.json` as
  `clamp_box_mm: {center: [x,y,z], size: [w,l,t]}` in device-frame mm; every phone sim, including
  the device-only reference, adds it as a rigid box (D-047). Reason: its echo (~25 cm extra path)
  lies inside both the 30 cm range window and the free-air reference gate (D-042). The rod is
  thinner than a cell and is not modelled.
- Active pair is the bottom speaker + bottom mic 2 (D-023): the bottom edge faces the
  target, screen facing up, phone horizontal.
- Plate: rigid flat board 100 × 100 × ≥5 mm (wood/acrylic), centred on the active
  speaker port's normal axis. Distance measured port-to-plate with a ruler, ±2 mm.
- Quiet room; record room temperature (°C) per session. Sims run at 343 m/s as always; the
  session temperature is passed to the reader as `temp_c` (D-037).
- Phone media volume at maximum for every take, never changed mid-session (D-090). Case off, Airplane mode
  and Do Not Disturb on, Bluetooth off, USB-C port empty.

## Capture file layout

```
data/captures/<session>/
  session.json          {"date","temp_c","phone":"iphone17pro","route","input_source",
                         "rig":"bottom_edge","clamp_box_mm","hand_mm","betas","notes"}
                        hand_mm / betas: Protocol C only (see there)
  <take>.wav            float32 mono 48 kHz (from capture app). Plate session stems (D-080): plate_<mm>mm_t<k>,
                        background_t<k> (no plate), free_air_<n>; any other stem raises. `ringsim validate plate`
                        orders free_air by integer n: lowest = session start, highest = end.
  <take>.json           capture-app metadata (CAPTURE_APP.md schema)
  free_air_<n>.wav      ≥ 5 s, active pair, nothing within 40 cm (D-018 rig, no target):
                        the session's reference for referenced profiles (D-042). One at the start
                        and one at the end of every session; the two are compared to check drift.
                        Drift number (D-079): both recordings aligned on their own offset, both referenced with the START
                        reference, only frames with index >= n_avg + 1 (51) of each; rel L2 of the two mean profiles.
```

## Protocol A — transducer sweep + mic identification (Phase 2, resolves Q-005, informs Q-011)

0. Speaker isolation and side check (D-032). Play the sweep with tx channel `left`, then
   `right`, 2 takes each, recording with the `bottom` mic. For each channel, repeat once
   with tape over the earpiece slot and once with tape over the left bottom grille
   (x_draw 15.40–22.16).
   - The channel whose in-band level drops ≥ 20 dB when the bottom grille is covered and
     < 3 dB when the earpiece is covered drives only the bottom speaker. Store it as
     `txChannel`. If neither channel meets both conditions, the earpiece cannot be
     silenced: stop and log it as blocked in STATUS.md.
   - If covering the RIGHT grille is what silences it instead, the speaker is on the
     other side from the drawing mapping: mirror x for the bottom ports in
     `iphone17pro.yaml` and amend DECISIONS.md D-023.
1. Bottom speaker (channel from step 0) → iOS `bottom` mic data source: play `sweep_16k_24k_2s.wav`, no target,
   3 takes. Output: magnitude and phase response 16–24 kHz (`validation/sweep.py`: deconvolve
   with the known sweep, H = R·conj(S)/(|S|²+1e-3·max|S|²), each take aligned on its own direct-path delay
   (xcorr envelope, parabolic sub-sample) before the complex average). Report the phase after removing its best-fit
   straight line (the part a delay cannot explain) over 18–22 kHz; informs Q-021.
2. Hole-cover test: repeat one take with tape over hole x_draw 24.42 (mic 2), then one
   with tape over the right-hand port group (mic 1 candidates). The take whose in-band
   level drops > 10 dB identifies the physical mic behind iOS's `bottom` data source.
   If it is mic 1, find the exact hole by covering single holes; update `iphone17pro.yaml`
   and DECISIONS.md D-023.
3. Pass condition for using the band: mean response 18–22 kHz at least 20 dB above the
   no-playback noise floor. If it fails, record it and stop; do not add gain stages.

Analysis (`validation/protocol_a.py`, D-087..D-089): session dir `data/captures/<date>_protA/`, labels
`protA0_<left|right>_<open|ear|grille|rgrille>_t<k>` (rgrille only when neither channel qualifies with the left grille taped),
`protA1_sweep_t<k>`, `protA2_<mic2|mic1grp>_t<k>`, `protA3_noplay_t<k>`; every take 10 s (D-090). Level = mean over 18–22 kHz of the aligned,
complex-averaged response in dB. Step 3 floor: no alignment (a no-playback take has no sweep): the recording cut to the
sweep length through the same regularised estimator, mean power, dB (`sweep.noise_floor_response`). Output: a dict and `report.md`.

## Protocol B — flat plate (Phase 2)

Distances: 80, 110, 140, 170, 200 mm. Per distance: 3 takes × 10 s of looped tx frames.
Also 3 takes with no plate (background).
Sim: `scene: plate` job per distance, 1 static frame (profile is static), read with the
session's `temp_c`. Plus one phone-only job (no plate, same clamp box): the sim background.
Sim plate profiles are background-subtracted with the sim background exactly as real plate
profiles are with the no-plate takes (D-047).

Metrics (`ringsim validate plate`):
1. **Peak bin error**: in background-subtracted mean profile, bin of the plate echo
   peak (largest |magnitude| beyond bin 2). Pass: |bin_sim − bin_real| ≤ 1 at every distance.
   Gated on the referenced profiles (D-074); the raw-profile result is reported alongside with its own flag.
2. **Range linearity**: fit bin vs distance for real; slope within 5% of 2·fs/c.
3. **Amplitude trend**: log peak magnitude vs log distance, sim and real slopes reported
   side by side (no pass threshold; informs Q-011).
4. **Static noise floor** `n` (reported next to the SOLVER convergence result; not a threshold): from no-plate takes,
   median over frames of `||P_k − mean P|| / ||mean P||`. Profiles are referenced with the session REF, NOT
   background-subtracted (subtracting makes mean P ≈ 0), frame 0 of each take skipped (D-079).
5. **Signed shape** (checks D-042; metrics 1–3 ignore phase): referenced profiles, background-subtracted,
   bins within ±10 of the plate peak. Fit one sub-bin delay (|δ| ≤ 1 bin; absorbs the ±2 mm ruler error)
   to the sim profile, then Pearson r between sim and real per distance. Report, no threshold; informs Q-021.
   Also report r for raw signed profiles, to show what referencing changes.

## Protocol C — hand open/close (Phase 4)

Hand ~100 mm beyond the bottom edge, palm facing the edge, fingers pointing up, wrist
resting on a fixed support so the wrist position is repeatable. Open↔close at ~1 Hz paced by a
metronome, 20 s per take, 5 takes per session, 2 sessions on different days.
Hand shape (D-047): echo positions depend on hand size, so the sim uses the subject's hand. Measure
once, in mm: hand length (wrist crease to middle fingertip), palm width (across the index and little
MCP knuckles), middle-finger circumference at the middle of the proximal phalanx; store as `hand_mm`.
`scripts/fit_betas.py` solves MANO betas 0–2 (others 0) so the flat-hand mesh matches the three
(length: wrist joint to middle-tip vertex; width: index-MCP to little-MCP joint distance plus the
mesh's skin thickness there; circumference: perimeter of the mesh cross-section at that point) and
stores `betas`. Also simulate with betas = 0 and report both, to show how much hand shape matters.

Sim: MANO sequence alternating flat hand ↔ fist (slerp, 1 Hz), fitted betas, same placement, read
with the session's `temp_c`; `hand_rot_deg` puts the fingers along +z, palm toward the phone, so the attached
forearm (D-031) runs −z like the real arm. The wrist support and table are not modelled:
they are static and cancel in the differential metric, so compare only differential
profiles in this protocol, never original profiles. Simulate one 1 s cycle (80 frames at 80 frames/s, D-022) with the
snapshot schedule (D-041) and tile the snapshots 20× (snapshot times shifted by 1 s per copy) to
build the 20 s stream; the real motion is periodic, and 1600 unique frames buy nothing.

Metrics (`ringsim validate hand`):
1. **Differential energy profile**: `E[j] = mean_t diff[t,j]^2`, normalized to sum 1.
   Pearson r between sim and real E. Threshold: Q-009.
2. **Real-vs-real ceiling**: same r across real takes/sessions. Report sim r alongside;
   Q-009 threshold is set relative to this ceiling.
3. **Temporal**: dominant frequency of `sum_j diff[t,j]^2` over time; sim and real must
   both show the 2 Hz component (two events per open/close cycle). Report, no threshold.
4. **Stroke asymmetry** (checks the motion model, D-041): opening moves the fingers one way
   relative to the phone and closing the other, so Doppler shifts their echoes in opposite
   directions. Split each take at the minima of `sum_j diff[t,j]^2` into strokes, group
   alternating strokes into A and B, and compute the energy-weighted mean bin of `E` for each
   group. Report |mean_A − mean_B| for real, for sim (`interp="cubic"`), and for sim with
   `interp="hold"` (frame-frozen, the old D-005 model) for contrast. Report, no threshold;
   informs Q-019.

## Report format

`reports/<name>/report.md` + PNGs: mean profiles (sim vs real overlaid), differential
heatmaps (time × bin) side by side, metrics table with pass/fail. Every number in the
report is computed by the CLI, never typed by hand.

## Notes

- The real room contributes echoes beyond the range window; all comparisons crop to
  `n_range_bins` (D-021).
- Absolute amplitude between sim and real is not comparable (unknown transducer gain).
  Compare shapes and positions only.
