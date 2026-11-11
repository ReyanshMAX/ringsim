# GEOMETRY.md — poses, MANO mesh, scene assembly, voxelization

## Overview

Turns a pose sequence into per-frame solid geometry and labels. Produces a `Scene`
(meshes + ports + labels) and then a `Medium` (density grid) for the solver.

## Non-goals

- No left hand, sleeve, or body meshes (DR v2, D-013). The forearm IS modelled, as a
  plain extrusion, only so the hand does not end in a flat wrist cap (D-031).
- No soft-tissue deformation beyond MANO's blend shapes.
- No collision checking between hand and device beyond the rejection rule below.

## Pose sources (D-011)

```python
# src/ringsim/geometry/poses.py
SOURCES = {
  "ih26m": "data/poses/ih26m/<seq_id>.npz",     # MANO fits, native fps. Preferred: free-hand gestures (D-043)
  "grab":  "data/poses/grab/<seq_id>.npz",      # MANO right-hand params, native fps. Secondary: grasping
  "walk":  None,                                # generated
}
# Normalized on-disk format for real sequences (produced by scripts/convert_*.py, D-043):
#   pose:  float32 [T, 48]   axis-angle; first 3 = global orient (zeroed on load); joints 1-15 are
#                            FULL rotations with the MANO mean pose already included, i.e. what
#                            smplx expects with flat_hand_mean=True (the setting hand_mesh uses).
#                            A source stored relative to the mean (flat_hand_mean=False) gets
#                            hands_mean added by its converter.
#   betas: float32 [10]
#   fps:   float32 scalar
#   seg:   int32 [T]         contiguous-segment id; frames dropped by the converter split segments
# Converter checks (scripts/convert_<src>.py --check 50): 50 random frames posed with the source
# dataset's own MANO settings and with hand_mesh on the converted file agree to < 0.1 mm per vertex.
# GRAB: drop frames where any MANO vertex is within 5 mm of the object surface (GRAB ships object
# meshes and poses); objects are not simulated, and a hand closed around missing air is not a
# free-hand pose.
def load_sequence(source: str, fps_out: float, n_frames: int, seed: int) -> PoseSequence
    # 1. low-pass each joint's rotation at 8 Hz (zero-phase 4th-order Butterworth on the unit
    #    quaternions, signs made continuous frame to frame, renormalized): removes per-frame fitting
    #    jitter, which would otherwise become fake motion in the differential profiles
    # 2. resample to fps_out with slerp per joint (scipy.spatial.transform.Slerp)
    # 3. take a random window of n_frames + 1 poses inside one segment (seed); zero global orient
    #    (wrist-relative, scope decision)
def random_walk(n_frames: int, fps: float, seed: int, betas: np.ndarray) -> PoseSequence
    # keyframes every 0.5 s: coefficients c_i ~ N(0, std_i^2) clipped to ±2.5 std_i in the PCA of
    # the converted real poses (data/poses/pose_pca.npz: mean, components, per-component std of the
    # 45 finger-joint axis-angle values over all converted frames; scripts/build_pose_pca.py);
    # pose = mean + c @ components; slerp between keyframes. Returns n_frames + 1 poses.
    # Raises if pose_pca.npz is missing (D-043): N(0, I) in MANO's own PCA basis is only meaningful if
    # its components are scaled to unit variance, which this repo has not checked, and a spread
    # measured on the same real data the 70 % real share comes from keeps the two consistent.
```

Both return `n_frames + 1` poses: pose f is the pose at the start of frame f, and the last one is
the pose at the end of the last frame. It is simulated as the end snapshot (D-041) and has no label.

Mixing: a dataset job lists sources explicitly; the 70/30 real/walk ratio is enforced
by the job generator (`scripts/make_jobs.py`), not by the loader. Dataset access: Q-008.

## MANO

```python
# src/ringsim/geometry/mano_hand.py
# Model: data/models/mano/MANO_RIGHT.pkl (license-gated, Q-008). Load once per process.
import smplx
_MODEL = smplx.create("data/models", model_type="mano", is_rhand=True,
                      use_pca=False, flat_hand_mean=True)
def hand_mesh(pose48, betas) -> tuple[trimesh.Trimesh, np.ndarray]:
    # vertices 778, faces 1538, metres; joints = 16 MANO joints + 5 fingertip vertices
    # fingertip vertex ids: thumb 744, index 320, middle 443, ring 554, little 671
    # output joints [21,3] in order: wrist, MANO joints 1..15, then 5 tips
```

The raw MANO mesh is open at the wrist. Never close it with a flat cap: a cap is a
strong reflector that real hands do not have, about 10 cm from a middle-finger ring and
inside the 18.5 cm range window (D-031). Attach a forearm instead:

```python
# src/ringsim/geometry/forearm.py
FOREARM_LEN_M = 0.40            # long enough to leave any domain; build_medium cuts it 3 cells inside the grid face (D-053)
def attach_forearm(hand: trimesh.Trimesh, j21: np.ndarray) -> trimesh.Trimesh:
    # 1. boundary loop = the single open edge loop of the MANO mesh (the wrist opening)
    # 2. axis = unit(j21[0] - mean(j21[[1, 4, 7, 10]]))   # wrist minus mean of the 4 finger MCPs
    # 3. extrude the loop along axis by FOREARM_LEN_M (straight prism, no taper)
    # 4. stitch loop to prism, cap only the FAR end; assert result.is_watertight
    # returns hand+forearm as one mesh
```

Open issue (D-051, measured before deciding): the real forearm stays inside the range window for
roughly 8 cm past the wrist, but the domain below keeps only ~10 mm of it. Two checks settle it:
the Phase 3 end-echo check (the cut near the grid face makes no echo of its own) and the Phase 5
forearm-length check (does keeping the forearm out to the range limit change ring profiles by more
than the convergence yardstick?). If it does, the domain grows along the forearm and wrist bend is
added as sequence-level DR (ranges: Q-023).

How the forearm ends (D-053). A solid touching a grid face goes NaN in k-Wave, whatever the PML
settings (T4 diagnostics 2026-11-10), so the forearm cannot run into the PML. `build_medium`
computes the domain bbox from everything except the forearm (DOMAIN rule in Voxelization), then
cuts the forearm flat with the plane 3 cells inside the grid face it crosses (`trimesh` plane slice
with cap, before voxelizing). The flat end faces away from the hand and the sensors, so its echo is
weak: −50 dB below the arm's own echo in the 2026-11-11 measurement (D-053). Density fades gain
≤ 4 dB and are not used; tapers are worse and must not be used. If the forearm does not reach the
cut plane, the sequence fails with reason `forearm_inside_domain` (rejection rule below).

## Labels (D-012, D-025)

Ring-a-Pose labels are MediaPipe Hands keypoints 1–20 (wrist 0 subtracted), with the
palm plane rotated to a reference and hand size normalized by wrist→little-MCP length
(§3.6.1, Fig. 8). This repo writes the same 20 keypoints in a fixed canonical frame and
stores the scale separately, so the training suite can apply identical normalization.

```python
# src/ringsim/dataset/labels.py
# Index into the 21-joint array from hand_mesh (wrist, MANO 1..15, tips T,I,M,R,L),
# in MediaPipe order 1..20:
#   thumb CMC,MCP,IP,TIP | index MCP,PIP,DIP,TIP | middle ... | ring ... | little ...
MANO_TO_MP20 = np.array([13,14,15,16,  1,2,3,17,  4,5,6,18,  10,11,12,19,  7,8,9,20])
MP_WRIST, MP_INDEX_MCP, MP_LITTLE_MCP = 0, 5, 17      # MediaPipe ids (after prepending wrist)

def canonical_frame(j21: np.ndarray) -> np.ndarray:   # returns R [3,3], rows = axes
    a = j21[1]  - j21[0]                                # wrist → index MCP (MANO 1)
    b = j21[7]  - j21[0]                                # wrist → little MCP (MANO 7)
    x = a / |a|;  n = cross(a, b) / |cross(a, b)|;  y = cross(n, x)
    return stack([x, y, n])

def labels_canonical(j21: np.ndarray) -> tuple[np.ndarray, float]:
    R = canonical_frame(j21)
    lab = (j21[MANO_TO_MP20] - j21[0]) @ R.T            # [20,3] metres
    scale = |j21[7] - j21[0]|                           # hand_scale_m
    return lab, scale
```

MANO joints are kinematic-skeleton centres; MediaPipe landmarks are image-fitted
surface-ish points. The mapping is semantic, not geometric: expect a systematic per-joint
offset between sim labels and real MediaPipe labels. The training suite measures it;
this repo does not correct it.

## Placement

```python
@dataclass
class Placement:
    hand_offset_mm: tuple[float,float,float] | None   # device frame: wrist position (phone)
    hand_rot_deg: tuple[float,float,float] | None     # XYZ Euler, applied after MANO
    plate: dict | None                                # {size_mm: [w,h,t], distance_mm: d}
    ring_rot_deg: float = 0.0                         # rotation about finger bone (DR)
    ring_shift_mm: float = 0.0                        # shift along bone (DR, D-026)
    rig_boxes: tuple[dict, ...] = ()                  # rigid boxes in device frame, mm: {center, size};
                                                      # the phone clamp (VALIDATION.md rig, D-047)
```

Rig boxes are added to the scene as solids in every snapshot, including the device-only reference
snapshot, and are voxelized like the phone body (box rule, Q-017).

Phone frame (`frame: device`): hand mesh rotated by `hand_rot_deg`, then translated
so the wrist joint sits at `hand_offset_mm`. Plate: centred on the active speaker
port's centroid normal axis at `distance_mm` from the port, face perpendicular to the normal.

Phone with the bottom pair (D-023): the hand sits beyond the bottom edge (−y side);
`hand_offset_mm` is measured from the device centroid, so a hand 100 mm from the edge
has y ≈ −175.

Ring frame (`frame: ring`): compute the bone transform of `body.finger/phalanx` from
MANO joints (bone axis = parent joint → child joint, dorsal = MANO palm normal negated),
place ring centre at `along_bone_frac` along the bone plus the DR `ring_shift_mm`,
rotate by `ring_rot_deg` about the bone axis, then express the whole hand mesh in ring
frame.

Ring mesh (D-024, D-049), built ONCE per sequence, in ring frame:
1. Pose the hand flat (MANO zero finger pose) with the sequence's betas, place the ring frame on
   it as above (same `along_bone_frac` and `ring_shift_mm`), slice the hand mesh with the plane
   z = 0 and keep the cross-section loop around the origin (the finger).
2. `r_in = max distance from origin to that loop + clearance_mm`.
3. Band = annulus r_in → r_in + wall_mm, width `width_mm`, swept over `arc_deg` with
   the gap centred at `gap_center_deg`; the sensor section (`sensor_section_arc_deg`,
   centred on palmar −y) uses thickness `sensor_section_mm`. Build with
   `trimesh.creation` primitives + boolean union (manifold3d engine).
4. Port points from `ring.yaml` are placed on the outer surface of the sensor section.
5. Every snapshot of the sequence reuses this mesh unchanged: the ring is rigid and rides on the
   bone, like a real ring. Skin that bulges into it when the finger bends is contact (D-048).
   The device-only reference snapshot (D-042) uses the same mesh.
Labels are always in the canonical wrist frame, independent of scene frame.

## Snapshot schedule (D-041)

Each k-Wave run is a static snapshot. A moving sequence is simulated at the snapshot times this
function picks, and `proc.rx_stream` interpolates between them along the chirp.

```python
# src/ringsim/geometry/scene.py
SNAP_MAX_PER_FRAME = 4
@dataclass
class Schedule:
    snap_t: np.ndarray        # [S] seconds, sorted
    snap_pose: np.ndarray     # [S, 48]
    frame_snap: np.ndarray    # [T] index of each frame's start snapshot
    undersampled: np.ndarray  # [U] frames that hit SNAP_MAX_PER_FRAME
def snapshot_schedule(dev, seq: PoseSequence, placement, disp_max_m: float) -> Schedule
    # T = len(seq.mano_pose) - 1 frames, frame period P = dev.chirp.frame_period_s.
    # for each frame f:
    #   d_f = max over the 778 MANO vertices of |v(pose[f+1]) - v(pose[f])|, both expressed in the
    #         scene frame (ring frame for the ring, so the palm moving relative to the ring counts)
    #   M_f = ceil(d_f / disp_max_m), clipped to [1, SNAP_MAX_PER_FRAME]; if clipped, f → undersampled
    #   snapshots at f*P + m*P/M_f for m = 0..M_f-1; poses for m > 0 by per-joint slerp
    #   between pose[f] and pose[f+1] (same Slerp as load_sequence), betas unchanged
    # plus the end snapshot at T*P with pose[T].
    # Static scenes (free, plate) and single static hand frames: one snapshot at 0.
```

Reference scene (D-042): every sequence also gets one device-only snapshot, `build_scene` with no
hand mesh, no forearm and no plate: the phone body (phone) or the sequence's ring mesh (ring),
same dx and grid origin rule. Its IR is stored as `ref_ir`.

`disp_max_m` comes from the job's `motion.snapshot_disp_mm` (Q-019). Sequence-level DR parameters
(betas, ring rotation and shift) are the same for every snapshot in a sequence.

Contact and occlusion (D-048). Fists and fingers pressed together are real poses (9 of
Ring-a-Pose's 20 occlude the ring) and are simulated, never rejected:
- Hand overlapping the ring band is contact: occupancy is the union of all solids, so skin pressed
  against the ring is solid against solid, with no air gap. No penetration rejection.
- Port openings stay air: the port voxel (Voxelization step 5) is forced to air after voxelization
  and smoothing, even when skin covers it, like the real port's small opening. A snapshot whose port
  voxel was inside a solid before forcing is flagged in the shard's `port_covered`. At dx = 2 mm this
  opening is a 2 mm air pocket, larger than reality; the ring convergence gate (SOLVER.md) covers it.

Rejection rule (per sequence, before any simulation): build the scene for every snapshot of the
schedule and reject the sequence if a hand vertex lies inside the phone body (a placement error),
the forearm does not reach its cut plane (`forearm_inside_domain`), or any solid voxel lies within
3 cells of a grid face (`solid_near_grid_face`, D-053). Resample the sequence's DR
parameters up to 5 times, then fail the whole sequence. A run-time failure in any snapshot also
fails the whole sequence; the reader refuses failed sequences and the job generator over-provisions
(DATASET.md). No sequence ever has holes.

## Voxelization

```python
# src/ringsim/geometry/voxelize.py
RHO_AIR = 1.2; RHO_SOLID = 1000.0     # D-004
def build_medium(scene, dx, c, margin_m) -> Medium:
    # 1. bbox = union of scene mesh bounds and port positions, padded by margin_m,
    #    EXCLUDING the forearm part of the hand mesh (vertices beyond the wrist plane);
    #    the forearm is voxelized only where it falls inside that bbox (D-031)
    #    margin_m from SOLVER.md domain rule (air beyond solids is not needed:
    #    PML absorbs outgoing waves, and only solids reflect)
    # 2. grid dims via solver.grid.fast_dims(); grid centres x_i = origin + i*dx
    # 3. occupancy: for each mesh, candidate points inside mesh AABB only,
    #    inside test = generalized winding number >= 0.5 (libigl fast_winding_number; abi3 wheels
    #    exist for Python 3.13 on Windows and Linux, checked 2026-11-10), not mesh.contains: MANO
    #    self-intersects in fists, and a ray-crossing test counts a point inside two overlapping parts
    #    as outside, leaving air holes inside the hand (D-048). OR across meshes.
    #    Requirement (D-041): occupancy must change continuously when a surface moves by less than
    #    one cell. Binary contains() does not: at median finger speed a surface moves ~0.2 mm per
    #    frame, and at dx = 2 mm most frames change no voxel at all, so the sim's slow-motion
    #    differentials come out quantized. The method and the matching D-036 re-check are Q-020.
    #    Until Q-020 is resolved, binary occupancy is allowed for static scenes only (free, plate,
    #    single static hand frames), never for moving sequences.
    # 3b. guard (D-053): no solid voxel within FACE_GUARD_CELLS = 3 of any of the six grid faces,
    #     else the sequence fails with `solid_near_grid_face`. A solid at a face goes NaN in k-Wave.
    # 4. rho = where(occ, RHO_SOLID, RHO_AIR) as float32, then
    #    rho = scipy.ndimage.gaussian_filter(rho, RHO_SMOOTH_CELLS)   (D-036)
    #    RHO_SMOOTH_CELLS = 0.5. A sharp jump goes NaN at CFL 0.3 and under-reflects by
    #    ~38% when stable; smoothed gives 94-95% of a rigid reflector on CPU and T4.
    #    Port voxels are checked against the UNSMOOTHED occupancy (step 5).
    # 5. port voxel = round((port.pos + dx*port.normal - origin)/dx); record covered = occ[port]
    #    (before forcing), then set rho[port] = RHO_AIR (D-048). Never assert, never shift a port.
```

## Notes

- The inside test runs on the CPU while the GPU waits; restrict to mesh AABB and batch 1e6 points
  per call. If voxelization exceeds 20% of per-snapshot sim time on Kaggle, log it in STATUS.md
  as a performance issue, do not switch libraries without a DECISIONS entry.
- The forearm direction is fixed by the hand (wrist → MCPs), so in the phone hand rig
  it must match the real arm: VALIDATION.md Protocol C sets `hand_rot_deg` so the
  forearm points away from the phone, as in the real rig.
- Save a 3-slice PNG of `rho` (xy/xz/yz through the mic voxel) for the first frame
  of every job: `data/shards/<job_id>_medium.png`. This is the cheapest geometry check.
