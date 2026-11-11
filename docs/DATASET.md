# DATASET.md — HDF5 shard schema, labels, domain randomization, reader

## Overview

One sim job writes one shard (D-010) containing raw IRs, the exact parameters that
produced them, and labels (D-009). Profiles are computed at read time through
`ringsim.proc`. Load-time randomization (noise) is applied by the reader.

## Non-goals

- No PyTorch Dataset class (training suite).
- No profile caching to disk in v1. If read-time processing becomes the bottleneck,
  log it in STATUS.md and add a decision before caching.
- No dataset versioning system; shards are immutable and named by job_id.

## Shard schema

```
<job_id>.h5
  attrs:
    schema_version   = 1
    job_yaml         str     full resolved JobSpec
    device_yaml      str     full resolved DeviceConfig
    build_id         str     env RINGSIM_BUILD_ID if set (Kaggle kernels have no git, INFRA.md), else `git rev-parse --short HEAD` + "+dirty" if uncommitted changes
    kwave_version    str
    created_utc      str     ISO 8601
    dx_m, c_sim      float   c_sim = 343.0 for every frame (D-037)
    fs_device        int
    k_decim          int     fs_sim = k_decim * fs_device (constant per shard)
  /pulse             float32 [n_t]        identical for every frame in the shard
  /seq/<seq_id>/
    attrs: source str, fps float, n_frames int, n_snapshots int,
           ring_rot_deg float, ring_shift_mm float, seed int,
           failed bool, failed_reason str (GEOMETRY.md rejection rule; "" if not failed),
           dr_attempts int (sequence DR draws used, 1..6)
    ir             float32 [S, n_t]        one row per snapshot (one static k-Wave run, D-041);
                                           mic pressure, sim rate, gzip level 4, chunk (1, n_t)
    snap_t         float64 [S]             snapshot time, s from sequence start; frame f starts
                                           at f*frame_period_s; sorted
    frame_snap     int32   [T]             index into snap_t of each frame's start snapshot
    snap_pose      float32 [S, 48]         MANO pose at each snapshot
    ref_ir         float32 [n_t]           device-only snapshot (same device body, ports, dx, ring
                                           size and DR as this sequence; no hand, no plate): the
                                           sim free-air reference (D-042)
    betas          float32 [10]
    joints21       float32 [T, 21, 3]      at frame start, device/ring frame, metres
    labels         float32 [T, 20, 3]      at frame start, MediaPipe order, canonical wrist frame, metres (D-025)
    hand_scale_m   float32 scalar          |wrist → little MCP| for Ring-a-Pose size normalization
    wall_s         float32 [S]             per-snapshot solver wall time
    port_covered   bool    [S]             a port voxel was inside a solid before being forced to
                                           air (sensor occluded, e.g. fist; D-048)
    undersampled_frames int32 [U]          frames that needed more than SNAP_MAX_PER_FRAME
                                           snapshots (GEOMETRY.md); kept, but flagged
```

A moving sequence has S ≥ T + 1: every frame start, any mid-frame snapshots, and one end
snapshot at `n_frames*frame_period_s` (no label). Frame f's labels use the pose of
`snap_pose[frame_snap[f]]`.

Scenes without a hand (`free`, `plate`) are static: S = 1, `snap_t = [0.0]`,
`frame_snap = zeros(T)`, and they omit `snap_pose`, `betas`, `joints21`, `labels`,
`hand_scale_m`; the reader then returns `labels=None`. Their seq attrs are
`source = "<scene>"`, `fps = 1/frame_period_s`, `ring_rot_deg = ring_shift_mm = 0`,
`seed` from the job (user decision 2026-10-14). A static hand frame (the convergence study's
base and moved poses) is also S = 1.

No temperature is stored: every frame is simulated at `c_sim`, and temperature is
applied by the reader (D-037). One shard serves any temperature in `TEMP_RANGE_C`.

## Domain randomization v1 (D-013)

| Param | Where applied | Distribution | Notes |
|---|---|---|---|
| `betas` | sim time, per sequence | N(0, `betas_std`²) per dim, clip ±2.5 | real sequences keep their own betas unless `dr.override_betas: true` |
| `ring_rot_deg` | sim time, per sequence | U(job range, Q-014) | ring config only; ignored for phone |
| `ring_shift_mm` | sim time, per sequence | U(−10, 10) | ring config only; along bone. Physical analogue of Ring-a-Pose's ±3-bin vertical-shift augmentation (±10.3 mm, §3.6.4), their most effective augmentation (§8.2.6) (D-026) |
| `temp_c` | read time, per sequence | U(reader range ⊆ `TEMP_RANGE_C`) | applied as `alpha = temperature_alpha(temp_c)` in `rx_from_ir`; exact time-scaling (D-037). Not stored; each read can draw a new temperature |
| sensor noise | read time, per frame | AWGN, SNR U(`snr_db` range) relative to direct-path power in band | job reader arg, not stored |
| device residual | read time, per sequence | TBD(Q-021) | what the free-air reference (D-042) cannot cancel: directivity, real-only internal paths, drift between reference and data. Ranges set from Protocol A phase and the near-field / ring checks, not guessed |

```python
# src/ringsim/dataset/randomize.py
def sample_sequence_dr(job: JobSpec, rng: np.random.Generator) -> dict   # betas, ring_rot_deg, ring_shift_mm (sim time)
def add_noise(stream: np.ndarray, snr_db: float, ref_power: float,
              rng: np.random.Generator) -> np.ndarray
```

## Reader

```python
# src/ringsim/dataset/reader.py
@dataclass
class SeqSample:
    profiles: np.ndarray        # [T, n_bins] signed; referenced (D-042) unless mode="raw"
    diff: np.ndarray            # [T-1, n_bins]
    labels: np.ndarray | None   # [T, 20, 3]
    meta: dict
def load_sequence(path: str, seq_id: str, snr_db: float | None,
                  temp_c: float | None, rng: np.random.Generator | None,
                  mode: str = "ref", dev_override: "DeviceConfig | None" = None) -> SeqSample
    # temp_c None → C_SIM (alpha = 1); otherwise alpha = proc.temperature_alpha(temp_c)
    # dev = DeviceConfig from attrs, or dev_override (D-046): same body and ports required
    #   (raise otherwise); fs_hz, chirp and n_range_bins may differ, e.g. if the ring hardware ends
    #   up at another sample rate. Raise if the override's window needs more IR than the shard has
    #   (C_MAX/C_SIM * n_range_bins/fs_hz beyond the sim window) or its chirp band leaves [16, 28] kHz.
    # tx = proc.tx_frame(dev.chirp, dev.fs_hz)
    # proc.rx_stream(ir, snap_t, pulse, k_decim * fs_device, tx, band, dev.fs_hz, n_frames, alpha)   (D-041)
    # → add_noise (if snr_db) → proc.align_direct_path → frame_stream
    # mode "ref" (default, D-042): ref = proc.free_air_reference(rx_stream of ref_ir as a single
    #   snapshot over n_avg + 1 frames, same alpha, no noise)
    #   → proc.cfmcw_profiles_ref(frames, tx, ref, band, fs, n_bins = dev.n_range_bins)
    # mode "raw": proc.cfmcw_profiles(n_bins = dev.n_range_bins)  (Ring-a-Pose parity, D-029)
    # → differential
    # raises SequenceFailed if the sequence's `failed` attr is set (D-048)
def list_sequences(path: str, include_failed: bool = False) -> list[str]
```

The reader reconstructs `DeviceConfig` from the shard, never from the current repo
YAML, so old shards stay valid after config edits.

## Writer

```python
# src/ringsim/dataset/writer.py
class ShardWriter:
    def __init__(self, path: str, job: JobSpec, dev: DeviceConfig): ...
    def begin_sequence(self, seq_id: str, attrs: dict, schedule: "Schedule", n_t: int): ...
        # schedule from geometry.scene.snapshot_schedule: snap_t, snap_pose, frame_snap, undersampled
    def write_snapshot(self, seq_id: str, s: int, res: SimResult, joints21): ...
        # joints21 used only when s is a frame-start snapshot (fills joints21/labels for that frame)
    def fail_sequence(self, seq_id: str, reason: str): ...   # whole sequence, never single frames (D-048)
    def close(self): ...
```

Writes are snapshot-incremental and flushed every 10 snapshots, so a Kaggle kernel killed
by the time limit leaves a valid partial shard. `n_snapshots_written` attr per sequence
records how far it got; the reader uses only frames whose snapshots through the next frame
start were all written.

## Notes

- Do not store profiles or rx streams. If a debugging need arises, write them to a
  separate `debug/` file, never into a shard.
- Rough size: n_t ≈ 1.3k float32 → ~5 KB/snapshot before compression.
