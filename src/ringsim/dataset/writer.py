"""HDF5 shard writer (DATASET.md). One shard per sim job (D-010); raw IRs only (D-009)."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from ringsim.devices.config import DeviceConfig
from ringsim.proc.physics import C_SIM

SCHEMA_VERSION = 1
FLUSH_EVERY = 10


def git_hash() -> str:
    """`git rev-parse HEAD` (+dirty); else the wheel's +g<hash> (Kaggle, D-034)."""
    repo = Path(__file__).resolve().parents[3]
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                               text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        try:
            v = version("ringsim")
            return v.split("+", 1)[1] if "+" in v else "unknown"
        except PackageNotFoundError:
            return "unknown"


class ShardWriter:
    def __init__(self, path: str, job, dev: DeviceConfig):
        from ringsim.solver.kwave_runner import kwave_version
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w")
        self.dev = dev
        a = self.f.attrs
        a["schema_version"] = SCHEMA_VERSION
        a["job_yaml"] = job.to_yaml()
        a["device_yaml"] = dev.to_yaml()
        a["git_hash"] = git_hash()
        a["kwave_version"] = kwave_version()
        a["created_utc"] = datetime.now(timezone.utc).isoformat()
        a["dx_m"] = float(job.dx_m)
        a["c_sim"] = C_SIM                                  # D-037
        a["fs_device"] = int(dev.fs_hz)
        self._pending = 0

    def begin_sequence(self, seq_id: str, attrs: dict, n_frames: int, n_t: int) -> None:
        g = self.f.create_group(f"seq/{seq_id}")
        for k in ("source", "fps", "ring_rot_deg", "ring_shift_mm", "seed"):
            g.attrs[k] = attrs[k]
        g.attrs["n_frames"] = int(n_frames)
        g.attrs["n_frames_written"] = 0
        g.create_dataset("ir", (n_frames, n_t), np.float32, chunks=(1, n_t),
                         compression="gzip", compression_opts=4)
        g.create_dataset("wall_s", (n_frames,), np.float32, fillvalue=np.nan)
        g.create_dataset("failed_frames", (0,), np.int32, maxshape=(None,))
        g.create_dataset("failed_reasons", (0,), h5py.string_dtype(), maxshape=(None,))
        for k in ("betas",):
            if attrs.get(k) is not None:
                g.create_dataset(k, data=np.asarray(attrs[k], np.float32))

    def _check_pulse(self, res) -> None:
        if "pulse" not in self.f:
            self.f.create_dataset("pulse", data=res.pulse.astype(np.float32))
            k = res.fs_sim / self.dev.fs_hz
            assert abs(k - round(k)) < 1e-6, k
            self.f.attrs["k_decim"] = int(round(k))
        else:
            assert np.array_equal(self.f["pulse"][()], res.pulse), "pulse differs within shard"
            assert self.f.attrs["k_decim"] * self.dev.fs_hz == round(res.fs_sim)

    def write_frame(self, seq_id: str, t: int, res, pose48=None, joints21=None) -> None:
        """pose48/joints21 are None for scenes without a hand (free, plate): those
        datasets are then omitted from the shard (DATASET.md)."""
        self._check_pulse(res)
        g = self.f[f"seq/{seq_id}"]
        T = g.attrs["n_frames"]
        g["ir"][t] = res.ir
        g["wall_s"][t] = res.wall_s
        for name, val, shape in (("mano_pose", pose48, (48,)), ("joints21", joints21, (21, 3))):
            if val is None:
                continue
            if name not in g:
                g.create_dataset(name, (T, *shape), np.float32, fillvalue=np.nan)
            g[name][t] = val
        g.attrs["n_frames_written"] = max(int(g.attrs["n_frames_written"]), t + 1)
        self._tick()

    def fail_frame(self, seq_id: str, t: int, reason: str) -> None:
        g = self.f[f"seq/{seq_id}"]
        for name, v in (("failed_frames", t), ("failed_reasons", reason)):
            d = g[name]
            d.resize((d.shape[0] + 1,))
            d[-1] = v
        g.attrs["n_frames_written"] = max(int(g.attrs["n_frames_written"]), t + 1)
        self._tick()

    def _tick(self) -> None:
        self._pending += 1
        if self._pending >= FLUSH_EVERY:
            self.f.flush()
            self._pending = 0

    def close(self) -> None:
        self.f.flush()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
