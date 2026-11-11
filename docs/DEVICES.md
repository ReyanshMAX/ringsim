# DEVICES.md — DeviceConfig schema and the three device configs

## Overview

Every sensing device is data, not code (D-006). The same pipeline runs `test_pair`
(Phase 1), `iphone17pro` (validation, Phases 2–4), and `ring` (dataset, Phase 5).
Configs live in `src/ringsim/devices/<name>.yaml`; `load_device()` validates them.

## Non-goals

- No auto-download of geometry. Numbers are entered by hand with a source comment.
- No device-specific branches in pipeline code. If a device needs special handling,
  extend the schema.
- No rounded corners, buttons, lens bumps, or port acoustics (Notes).

## Schema

```python
@dataclass(frozen=True)
class Port:
    name: str
    kind: Literal["speaker", "mic"]
    points_mm: tuple[tuple[float, float, float], ...]   # 1+ holes; all driven/summed in phase
    normal: tuple[float, float, float]                  # outward unit vector

@dataclass(frozen=True)
class BoxBody:
    kind: Literal["box"]
    size_mm: tuple[float, float, float]                 # x width, y length, z thickness
    plateau_size_mm: tuple[float, float, float] | None  # camera plateau on rear face
    plateau_center_mm: tuple[float, float, float] | None

@dataclass(frozen=True)
class RingBody:
    kind: Literal["ring"]
    # inner diameter is NOT configured: derived per frame from the MANO finger (D-024)
    clearance_mm: float             # radial gap between finger surface and ring inner wall
    wall_mm: float                  # band thickness away from the sensor section
    sensor_section_mm: float        # thickness at the speaker/mic section
    sensor_section_arc_deg: float   # angular extent of the thick section, centred on palmar
    width_mm: float                 # along the bone
    arc_deg: float                  # 360 = closed band; Ring-a-Pose = 270
    gap_center_deg: float           # angle of the open gap, 0 = dorsal, +90 = radial side
    finger: Literal["thumb","index","middle","ring","little"]
    phalanx: Literal["proximal","middle"]
    along_bone_frac: float          # nominal; jittered by DR (DATASET.md)

@dataclass(frozen=True)
class NoBody:
    kind: Literal["none"]

@dataclass(frozen=True)
class DeviceConfig:
    name: str
    fs_hz: int
    chirp: ChirpSpec                # from ringsim.proc (PROCESSING.md)
    n_range_bins: int               # profile length; range_window_m = n*C_SIM/(2*fs)
    ports: tuple[Port, ...]
    active_pair: tuple[str, str]
    body: BoxBody | RingBody | NoBody
    frame: Literal["device", "ring"]
```

Loader rules:
- Any value written `TBD(Q-xxx)` → one `ConfigIncomplete("Q-xxx: <field>; Q-yyy: <field>; ...")`
  listing every TBD in the file (so `ring` names both Q-001 and Q-002), except inside a
  port that is not in `active_pair`: that port is dropped with a warning naming the Q-ID.
- `active_pair` names one speaker and one mic in `ports`.
- Box port points lie on the box surface within 0.5 mm.
- Chirp band inside the solver pulse band [16, 28] kHz (SOLVER.md).
- `round(chirp.frame_period_s * fs_hz)` is an integer to 1e-9.
- `range_window_m` property = `n_range_bins * 343.0 / (2 * fs_hz)`.

Device frame (phone): origin at body centroid, +x toward the right edge when looking
at the screen, +y toward the top edge, +z out of the screen.
Ring frame: origin at ring centre, +z along the bone toward the fingertip, +y dorsal
(back of hand), +x completes right-handed. Palmar = −y.

## test_pair.yaml (Phase 1)

```yaml
name: test_pair
fs_hz: 48000
chirp: {f0_hz: 18000, f1_hz: 22000, duration_s: 0.0125, frame_period_s: 0.0125, window: none}
n_range_bins: 84
body: {kind: none}
frame: device
ports:
  - {name: spk, kind: speaker, points_mm: [[0, 0, 0]],   normal: [0, 0, 1]}
  - {name: mic, kind: mic,     points_mm: [[100, 0, 0]], normal: [0, 0, 1]}
active_pair: [spk, mic]
```

## iphone17pro.yaml (Phases 2–4)

Source: Apple, *iPhone 17 Pro Dimensional Drawings* (2025-09-09), sheets 1 and 4.
Drawing datums: x from the left edge (screen view), y from the top edge.
Conversion: `x_dev = x_draw − 35.925`, `y_dev = 75.005 − y_draw`.

```yaml
name: iphone17pro
fs_hz: 48000
chirp:                       # D-015 band; D-022 timing
  f0_hz: 18000
  f1_hz: 22000
  duration_s: 0.0125         # 600 samples @ 48 kHz
  frame_period_s: 0.0125     # continuous FMCW, as Ring-a-Pose
  window: none
n_range_bins: 84             # 0.30 m one-way
frame: device
body:
  kind: box
  size_mm: [71.85, 150.01, 8.75]        # sheet 1: product width / length / thickness
  plateau_size_mm: [71.85, 58.01, 2.55] # sheet 4 plateau edge at y_draw 58.01; sheet 1 back plate→plateau 2.55
  plateau_center_mm: [0.0, 46.0, -5.65]
ports:
  # Bottom edge, sheet 1 detail B: 4X speaker ports at x_draw 15.40, 17.65, 19.91, 22.16;
  # bottom mic 2 at 24.42; 10X Ø1.06 ports at mid-thickness (z_draw 4.375).
  - name: bottom_spk
    kind: speaker
    points_mm: [[-20.525, -75.005, 0], [-18.275, -75.005, 0], [-16.015, -75.005, 0], [-13.765, -75.005, 0]]
    normal: [0, -1, 0]
  - name: bottom_mic2
    kind: mic
    points_mm: [[-11.505, -75.005, 0]]
    normal: [0, -1, 0]
  - name: bottom_mic1        # right-hand port group x_draw 47.43–56.45; exact hole per Q-005
    kind: mic
    points_mm: TBD(Q-005)
    normal: [0, -1, 0]
  # Top: receiver + front mic share one slot, sheet 1 detail A: length 27.38 centred at
  # x_draw 35.92, at y_draw 0.70–1.25. Front-mic position within the slot is not dimensioned.
  - name: earpiece
    kind: speaker
    points_mm: [[0.0, 74.03, 4.375]]
    normal: [0, 0, 1]
  - name: front_mic
    kind: mic
    points_mm: TBD(Q-005)
    normal: [0, 0, 1]
active_pair: [bottom_spk, bottom_mic2]  # D-023
```

## ring.yaml (Phase 5)

Source: Yu et al., *Ring-a-Pose*, IMWUT 8(4) 2024, §3.2–3.4, §5.2, §6.2.

```yaml
name: ring
fs_hz: 50000                   # §3.3
chirp:
  f0_hz: 20000                 # §3.3: 20–24 kHz
  f1_hz: 24000
  duration_s: 0.012            # 600 samples @ 50 kHz
  frame_period_s: 0.012        # continuous FMCW, 83.3 frames/s
  window: TBD(Q-002)           # not stated in paper
n_range_bins: 54               # §3.4.2: 54 px from direct path = 18.52 cm
frame: ring
body:
  kind: ring
  clearance_mm: 0.0            # sized to the finger (yarn + sliding knot, §3.2)
  wall_mm: 2.0                 # §3.2
  sensor_section_mm: 3.58      # §3.2
  sensor_section_arc_deg: TBD(Q-001)
  width_mm: 11.0               # §3.2
  arc_deg: 270                 # §3.2: 3/4 circle
  gap_center_deg: TBD(Q-001)
  finger: middle               # §6.2.1 main study, right hand
  phalanx: TBD(Q-001)
  along_bone_frac: TBD(Q-001)
ports:
  # Facing the palm, side by side across the finger, 5.4 mm apart (§3.2); "inner edge of
  # the speaker" centred on the finger (§5.2.2). Angular placement is computed from these
  # at scene build time: speaker centre offset by speaker_half_width toward side S.
  - {name: spk, kind: speaker, points_mm: TBD(Q-001), normal: [0, -1, 0]}
  - {name: mic, kind: mic,     points_mm: TBD(Q-001), normal: [0, -1, 0]}
active_pair: [spk, mic]
```

## Notes

- Phone corner radius (~R11.8, sheet 4), buttons, and rear lens rings (+1.88 mm past
  the plateau) are omitted: each is ≤1 cell at 2 mm and far from the bottom-edge pair.
  Revisit only if Phase 2 plate error exceeds 1 bin.
- Multi-hole ports: the speaker's 4 holes are 4 source voxels driven with the same pulse.
  If two holes fall in one voxel at the chosen dx, the voxel is used once.
- Ring parts (USound UT-P2019 speaker, TDK ICS-43434 mic) are recorded for the ring
  hardware suite; their responses are not modelled (Q-011).
- `fs_hz: 50000` for the ring is Ring-a-Pose's rate. If the Teensy build ends up at another rate,
  existing shards are read with `dev_override` (DATASET.md, D-046); no re-simulation is needed as
  long as the new chirp band stays in [16, 28] kHz and the range window fits the sim window.
