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
    git_hash         str     `git rev-parse HEAD`, "+dirty" if uncommitted changes
    kwave_version    str
    created_utc      str     ISO 8601
    dx_m, c_ref      float   (c per sequence below; c_ref = 343.0 for bin math)
    fs_device        int
    k_decim          int     fs_sim = k_decim * fs_device (constant per sequence)
  /pulse             float32 [n_t]        identical for all frames of a sequence with same c
  /seq/<seq_id>/
    attrs: source str, fps float, n_frames int, temp_c float, c float,
           ring_rot_deg float, ring_shift_mm float, seed int, k_decim int
    ir             float32 [T, n_t]        mic pressure, sim rate, gzip level 4, chunk (1, n_t)
    pulse          float32 [n_t]           present when this seq's dt differs from /pulse
    mano_pose      float32 [T, 48]
    betas          float32 [10]
    joints21       float32 [T, 21, 3]      device/ring frame, metres
    labels         float32 [T, 20, 3]      MediaPipe order, canonical wrist frame, metres (D-025)
    hand_scale_m   float32 scalar          |wrist → little MCP| for Ring-a-Pose size normalization
    wall_s         float32 [T]             per-frame solver wall time
    failed_frames  int32   [F]             frames that hit the rejection rule (GEOMETRY.md)
```

Why per-sequence `c`: temperature is randomized per sequence (DR), which changes `dt`
and `k_decim`. Within a sequence `c` is constant.

## Domain randomization v1 (D-013)

| Param | Where applied | Distribution | Notes |
|---|---|---|---|
| `betas` | sim time, per sequence | N(0, `betas_std`²) per dim, clip ±2.5 | real sequences keep their own betas unless `dr.override_betas: true` |
| `ring_rot_deg` | sim time, per sequence | U(job range, Q-014) | ring config only; ignored for phone |
| `ring_shift_mm` | sim time, per sequence | U(−10, 10) | ring config only; along bone. Physical analogue of Ring-a-Pose's ±3-bin vertical-shift augmentation (±10.3 mm, §3.6.4), their most effective augmentation (§8.2.6) (D-026) |
| `temp_c` → `c` | sim time, per sequence | U(job range) | `speed_of_sound(temp_c)` |
| sensor noise | read time, per frame | AWGN, SNR U(`snr_db` range) relative to direct-path power in band | job reader arg, not stored |

```python
# src/ringsim/dataset/randomize.py
def sample_sequence_dr(job: JobSpec, rng: np.random.Generator) -> dict   # betas, ring_rot_deg, ring_shift_mm, temp_c
def add_noise(stream: np.ndarray, snr_db: float, ref_power: float,
              rng: np.random.Generator) -> np.ndarray
```

## Reader

```python
# src/ringsim/dataset/reader.py
@dataclass
class SeqSample:
    profiles: np.ndarray        # [T, n_bins] signed
    diff: np.ndarray            # [T-1, n_bins]
    labels: np.ndarray | None   # [T, 20, 3]
    meta: dict
def load_sequence(path: str, seq_id: str, snr_db: float | None,
                  rng: np.random.Generator | None) -> SeqSample
    # dev = DeviceConfig from attrs; tx = proc.tx_frame(dev.chirp, fs)
    # per frame: proc.rx_from_ir → proc.overlap_add_stream
    # → add_noise (if snr_db) → proc.align_direct_path → frame_stream
    # → proc.cfmcw_profiles(n_bins = dev.n_range_bins) → differential
def list_sequences(path: str) -> list[str]
```

The reader reconstructs `DeviceConfig` from the shard, never from the current repo
YAML, so old shards stay valid after config edits.

## Writer

```python
# src/ringsim/dataset/writer.py
class ShardWriter:
    def __init__(self, path: str, job: JobSpec, dev: DeviceConfig): ...
    def begin_sequence(self, seq_id: str, attrs: dict, n_frames: int, n_t: int): ...
    def write_frame(self, seq_id: str, t: int, res: SimResult, pose48, joints21): ...
    def fail_frame(self, seq_id: str, t: int, reason: str): ...
    def close(self): ...
```

Writes are frame-incremental and flushed every 10 frames, so a Kaggle kernel killed
by the time limit leaves a valid partial shard. `n_frames_written` attr per sequence
records how far it got.

## Notes

- Do not store profiles or rx streams. If a debugging need arises, write them to a
  separate `debug/` file, never into a shard.
- Rough size: n_t ≈ 1.3k float32 → ~5 KB/frame before compression.
