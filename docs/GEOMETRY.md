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
  "grab":  "data/poses/grab/<seq_id>.npz",      # MANO right-hand params, native fps
  "ih26m": "data/poses/ih26m/<seq_id>.npz",     # MANO fits, native fps
  "walk":  None,                                # generated
}
# Normalized on-disk format for real sequences (produced by scripts/convert_*.py):
#   pose:  float32 [T, 48]   axis-angle; first 3 = global orient (zeroed on load)
#   betas: float32 [10]
#   fps:   float32 scalar
def load_sequence(source: str, fps_out: float, n_frames: int, seed: int) -> PoseSequence
    # resample to fps_out with slerp per joint (scipy.spatial.transform.Slerp),
    # take a random window of n_frames (seed); zero global orient (wrist-relative, scope decision)
def random_walk(n_frames: int, fps: float, seed: int, betas: np.ndarray) -> PoseSequence
    # keyframes every 0.5 s: sample 45-D MANO PCA coefficients ~ N(0, I) clipped to ±2.5σ,
    # map to axis-angle via MANO PCA basis; slerp between keyframes.
```

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
FOREARM_LEN_M = 0.40            # long enough to leave any domain; clipped at the grid face
def attach_forearm(hand: trimesh.Trimesh, j21: np.ndarray) -> trimesh.Trimesh:
    # 1. boundary loop = the single open edge loop of the MANO mesh (the wrist opening)
    # 2. axis = unit(j21[0] - mean(j21[[1, 4, 7, 10]]))   # wrist minus mean of the 4 finger MCPs
    # 3. extrude the loop along axis by FOREARM_LEN_M (straight prism, no taper)
    # 4. stitch loop to prism, cap only the FAR end; assert result.is_watertight
    # returns hand+forearm as one mesh
```

`build_medium` computes the domain bbox from everything except the forearm (DOMAIN rule
in Voxelization), so the forearm runs off the grid face. With `pml_inside=False` k-Wave
extends edge medium values into the PML, so the arm continues into the absorbing layer
and never produces an end echo. `build_medium` asserts that forearm voxels touch the
grid face; if not, the frame fails with reason `forearm_inside_domain`.

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
```

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

Ring mesh (D-024):
1. Slice the hand mesh with the plane z = 0 (ring frame); keep the cross-section loop
   around the origin (the finger).
2. `r_in = max distance from origin to that loop + clearance_mm`.
3. Band = annulus r_in → r_in + wall_mm, width `width_mm`, swept over `arc_deg` with
   the gap centred at `gap_center_deg`; the sensor section (`sensor_section_arc_deg`,
   centred on palmar −y) uses thickness `sensor_section_mm`. Build with
   `trimesh.creation` primitives + boolean union (manifold3d engine).
4. Port points from `ring.yaml` are placed on the outer surface of the sensor section.
Labels are always in the canonical wrist frame, independent of scene frame.

Rejection rule: if any hand vertex lies inside the device body (phone) or penetrates
the ring wall by > dx, resample the frame's DR params up to 5 times, then fail the frame
and record it in the shard's `failed_frames`.

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
    #    mesh.contains(points) (ray test); OR across meshes
    # 4. rho = where(occ, RHO_SOLID, RHO_AIR) as float32
    # 5. port voxel = round((port.pos + dx*port.normal - origin)/dx); assert rho[port]==RHO_AIR
```

## Notes

- `mesh.contains` is slow; restrict to mesh AABB and batch 1e6 points per call.
  If voxelization exceeds 20% of per-frame sim time on Kaggle, log it in STATUS.md
  as a performance issue, do not switch libraries without a DECISIONS entry.
- The forearm direction is fixed by the hand (wrist → MCPs), so in the phone hand rig
  it must match the real arm: VALIDATION.md Protocol C sets `hand_rot_deg` so the
  forearm points away from the phone, as in the real rig.
- Save a 3-slice PNG of `rho` (xy/xz/yz through the mic voxel) for the first frame
  of every job: `data/shards/<job_id>_medium.png`. This is the cheapest geometry check.
