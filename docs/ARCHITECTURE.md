# ARCHITECTURE.md — pipeline stages, module boundaries, CLI

## Overview

Batch pipeline: pose → hand mesh → scene (hand + device body + optional targets) →
voxel medium → k-Wave run with pulse at speaker port → pressure at mic port (IR) →
HDF5 shard. Processing (IR → received stream → C-FMCW profiles) happens on read,
through `ringsim.proc`, which real captures also go through.

## Non-goals

- No model training, dataloaders for PyTorch, or calibration routine.
- No job scheduler beyond "one Kaggle kernel = one job = one shard".
- No caching layer, database, or web UI.
- No moving geometry inside one k-Wave run. Motion within a frame is handled by interpolating
  static IR snapshots along the chirp (D-041).

## Data flow

```
JobSpec (YAML) ─┐
                ▼
geometry.poses.load_sequence ─► PoseSequence (T frames + 1 end pose of MANO params)
geometry.scene.snapshot_schedule ─► snapshot times + poses (every frame start, extra ones
                                    where the hand moves fast, end snapshot; D-041)
                ▼ per snapshot
geometry.mano_hand.hand_mesh ─► trimesh.Trimesh (device frame, metres)
geometry.scene.build_scene ───► Scene (meshes + ports)
geometry.voxelize.build_medium ─► Medium (rho grid, c scalar) + port indices
solver.kwave_runner.run_ir ───► IR float32 [n_t_sim] @ fs_sim (mic holes averaged)
                ▼
dataset.writer.ShardWriter ───► data/shards/<job_id>.h5
                ▼ (later, anywhere)
dataset.reader.load_sequence ─► IR snapshots + snapshot times + labels (temperature applied here, D-037)
proc.rx_stream ─► rx stream @ device fs (snapshot responses interpolated along the chirp, D-041)
dataset.randomize.add_noise (load-time DR)
proc.align_direct_path ─► proc.frame_stream ─► proc.cfmcw_profiles_ref ─► proc.differential
                                (reference: proc.free_air_reference of the device-only snapshot, D-042;
                                 proc.cfmcw_profiles for raw Ring-a-Pose-parity profiles)
```

Real capture path joins at `proc.align_direct_path` (validation/real.py loads WAV); its reference
comes from the session's free-air capture.

## Module interfaces

```python
# src/ringsim/devices/config.py
# Port, BoxBody, RingBody, DeviceConfig: full definitions in DEVICES.md
# ChirpSpec: defined in ringsim.proc.chirp (PROCESSING.md)
def load_device(name: str) -> DeviceConfig  # raises on any TBD(Q-xxx); reads devices/<name>.yaml, calls load_device_yaml
def load_device_yaml(text: str) -> DeviceConfig  # same rules; the reader rebuilds a shard's device from attrs (D-055)

# src/ringsim/geometry/poses.py
@dataclass
class PoseSequence:
    source: str                             # "grab:<id>", "ih26m:<id>", "walk:<seed>"
    fps: float
    mano_pose: np.ndarray                   # [T+1, 48] axis-angle, global orient = zeros; row T = end pose
    betas: np.ndarray                       # [10]
def load_sequence(source: str, fps_out: float, n_frames: int, seed: int) -> PoseSequence
def random_walk(n_frames: int, fps: float, seed: int, betas: np.ndarray) -> PoseSequence

# src/ringsim/geometry/mano_hand.py
def hand_mesh(pose48: np.ndarray, betas: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray]
    # returns mesh in MANO frame (metres) and joints [21,3]

# src/ringsim/geometry/scene.py
@dataclass
class Scene:
    meshes: list[trimesh.Trimesh]           # all solids, device frame, metres
    speaker: Port; mic: Port
    joints21: np.ndarray                    # [21,3] scene frame, metres
    labels: np.ndarray                      # [20,3] canonical wrist frame, metres (D-025)
    hand_scale_m: float                     # |wrist → little MCP| (Ring-a-Pose size normalizer)
def build_scene(dev: DeviceConfig, pose48, betas, placement: "Placement") -> Scene
def snapshot_schedule(dev, seq: PoseSequence, placement, disp_max_m: float) -> "Schedule"   # GEOMETRY.md

# src/ringsim/geometry/voxelize.py
@dataclass
class Medium:
    rho: np.ndarray                         # float32 [Nx,Ny,Nz]
    c: float
    dx: float
    origin_m: np.ndarray                    # world coords of voxel (0,0,0)
    src_idx: list[tuple[int,int,int]]       # one per speaker hole (deduplicated)
    mic_idx: list[tuple[int,int,int]]       # one per mic hole; recorded pressures are averaged
def build_medium(scene: Scene, dx: float, c: float, margin_m: float) -> Medium

# src/ringsim/solver/kwave_runner.py
@dataclass
class SimResult:
    ir: np.ndarray                          # float32 [n_t] pressure at mic (pulse response)
    pulse: np.ndarray                       # float32 [n_t] source signal
    fs_sim: float                           # = k * device fs (SOLVER.md time_axis)
    wall_s: float
def run_ir(medium: Medium, range_window_m: float, fs_device: int, gpu: bool) -> SimResult

# src/ringsim/proc/*  — see PROCESSING.md for full signatures
# src/ringsim/dataset/* — see DATASET.md
# src/ringsim/dataset/jobspec.py (D-059)
@dataclass
class JobSpec: ...                          # the JobSpec YAML below, mm converted to SI at load
def load_job(path: str) -> JobSpec          # raises on any TBD(Q-xxx)

# src/ringsim/validation/freefield.py (Phase 1, D-057, D-060); seq_id for scene free is "free"
def freefield_check(path: str, seq_id: str, dev_override: DeviceConfig | None = None) -> dict
    # ideal monopole H = j·2πf·e^{−2πifτ}, τ = 0.100/c, through the same proc path; gates on mode="raw",
    # reports mode="ref"; keys: peak bins, rel L2, gain phase, arrival-time error, pass flags
def compare_ir(path_a: str, path_b: str, seq_id: str) -> float   # relative L2 of the two shards' IRs
```

## CLI

```
ringsim sim      --job jobs/<job>.yaml [--gpu] [--frames a:b] [--out-dir DIR]
                                                                 # writes <DIR, default data/shards>/<job_id>.h5
ringsim profile  --shard <h5> --seq <id> --out <png>            # plot profiles + differential; prints
                                                                 # freefield_check as JSON (D-060)
ringsim export-tx --device iphone17pro --out capture-app/Resources/tx/
ringsim real     --capture data/captures/<name>.wav --device iphone17pro --out <png>
ringsim validate plate|hand --sim <h5> --real <dir> --out reports/<name>/
ringsim converge --job jobs/converge.yaml --dx 2.0,1.5,1.0
```

Kaggle is not a `ringsim` subcommand: `python tools/kaggle_run.py run jobs/<job>.yaml` (INFRA.md, D-038).

## JobSpec

```yaml
# jobs/<name>.yaml
job_id: p4_phone_openclose_001
device: iphone17pro            # devices/<name>.yaml
scene: hand                    # free | plate | hand
dx_mm: 2.0                     # TBD(Q-006) after Phase 3
placement:                     # GEOMETRY.md
  hand_offset_mm: [0, 0, 100]
  hand_rot_deg: [180, 0, 0]
  plate: null                  # {size_mm: [100,100,5], distance_mm: 150} for scene=plate
  rig_boxes: []                # phone jobs: [{center: [x,y,z], size: [w,l,t]}] mm, the clamp from session.json (D-047)
poses:
  source: walk:42              # or grab:<seq_id> / ih26m:<seq_id>
  n_frames: 120
motion:                        # required when scene = hand and n_frames > 1 (D-041)
  snapshot_disp_mm: TBD(Q-019) # max hand-vertex move between snapshots
dr:                            # DATASET.md
  betas_std: 1.0
  ring_rot_deg: TBD(Q-014)     # ring config only
  ring_shift_mm: [-10, 10]     # ring config only; along bone (D-026)
seed: 7
```

## Notes

- All internal units SI (metres, seconds, Hz). Configs use mm; convert at load.
- Every shard stores the full resolved JobSpec + DeviceConfig + build id, so any
  frame is reproducible from the shard alone.
