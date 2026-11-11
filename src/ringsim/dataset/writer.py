"""ShardWriter: one HDF5 shard per sim job, exactly per DATASET.md "Shard schema" and "Writer".

Snapshot-incremental; flushed every FLUSH_EVERY snapshots, so a killed Kaggle kernel leaves a valid partial shard
(`n_snapshots_written` says how far each sequence got).

Phase 1 covers static scenes (`free`, `plate`): S = 1, no hand datasets. A sequence whose `source` is not a static
scene needs the label frame (D-025, Phase 3) and raises NotImplementedError until that is specified.

Two calls DATASET.md's list does not name, added because the schema needs them (posted to the room):
  write_ref_ir(seq_id, res)   the device-only snapshot, /seq/<id>/ref_ir (D-042, D-056)
  write_snapshot(..., port_covered=False)   the occlusion flag from the solver/geometry side (D-048)
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = 1
C_SIM = 343.0            # same value as ringsim.proc.physics.C_SIM (D-037); stored, not imported, to keep
                         # the shard format independent of proc
FLUSH_EVERY = 10
STATIC_SOURCES = ("free", "plate")


def _git(args: list[str]) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run(["git", "-C", here, *args], capture_output=True, text=True, check=True).stdout.strip()


def build_id() -> str:
    """env RINGSIM_BUILD_ID if set (Kaggle has no git, INFRA.md), else short HEAD + "+dirty" if uncommitted."""
    env = os.environ.get("RINGSIM_BUILD_ID")
    if env:
        return env
    sha = _git(["rev-parse", "--short", "HEAD"])
    return sha + ("+dirty" if _git(["status", "--porcelain"]) else "")


def _kwave_version() -> str:
    try:
        from importlib.metadata import version
        return version("k-wave-python")
    except Exception:  # not installed: say so, do not guess
        return "not-installed"


class ShardWriter:
    def __init__(self, path: str, job, dev):
        from ringsim.dataset.jobspec import job_yaml
        from ringsim.devices.config import device_yaml
        self.path = path
        self.job = job
        self.dev = dev
        self._f = h5py.File(path, "w")
        a = self._f.attrs
        a["schema_version"] = SCHEMA_VERSION
        a["job_yaml"] = job_yaml(job)
        a["device_yaml"] = device_yaml(dev)
        a["build_id"] = build_id()
        a["kwave_version"] = _kwave_version()
        a["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        a["dx_m"] = float(job.dx_m)
        a["c_sim"] = C_SIM
        a["fs_device"] = int(dev.fs_hz)
        # k_decim is known from the first SimResult (fs_sim / fs_device); set in write_snapshot
        self._pulse: np.ndarray | None = None
        self._seq: dict[str, dict[str, Any]] = {}
        self._since_flush = 0

    # ---- sequence lifecycle -------------------------------------------------------------------------
    def begin_sequence(self, seq_id: str, attrs: dict, schedule, n_t: int) -> None:
        """attrs: source, n_frames, seed (+ optional ring_rot_deg, ring_shift_mm, dr_attempts).
        schedule: geometry.scene.snapshot_schedule result (snap_t, snap_pose, frame_snap, undersampled),
        or None for a static scene (S = 1)."""
        source = attrs["source"]
        T = int(attrs["n_frames"])
        g = self._f.create_group(f"seq/{seq_id}")
        if schedule is None:
            if source not in STATIC_SOURCES:
                raise ValueError(f"source {source!r} needs a schedule; only {STATIC_SOURCES} are static")
            snap_t = np.zeros(1)
            frame_snap = np.zeros(T, dtype=np.int32)
            undersampled = np.zeros(0, dtype=np.int32)
        else:
            if source in STATIC_SOURCES:
                raise ValueError(f"a {source!r} scene is static: pass schedule=None")
            raise NotImplementedError("hand sequences (joints21, labels, snap_pose): Phase 3, D-025 label frame")
        S = len(snap_t)
        g.attrs["source"] = source
        g.attrs["fps"] = float(attrs.get("fps", 1.0 / self.dev.chirp.frame_period_s))
        g.attrs["n_frames"] = T
        g.attrs["n_snapshots"] = S
        g.attrs["ring_rot_deg"] = float(attrs.get("ring_rot_deg", 0.0))
        g.attrs["ring_shift_mm"] = float(attrs.get("ring_shift_mm", 0.0))
        g.attrs["seed"] = int(attrs["seed"])
        g.attrs["failed"] = False
        g.attrs["failed_reason"] = ""
        g.attrs["dr_attempts"] = int(attrs.get("dr_attempts", 1))
        g.attrs["n_snapshots_written"] = 0
        g.create_dataset("ir", shape=(S, n_t), dtype="float32", chunks=(1, n_t), compression="gzip",
                         compression_opts=4)
        g.create_dataset("snap_t", data=np.asarray(snap_t, dtype=np.float64))
        g.create_dataset("frame_snap", data=np.asarray(frame_snap, dtype=np.int32))
        g.create_dataset("wall_s", data=np.zeros(S, dtype=np.float32))
        g.create_dataset("port_covered", data=np.zeros(S, dtype=bool))
        g.create_dataset("undersampled_frames", data=np.asarray(undersampled, dtype=np.int32))
        self._seq[seq_id] = {"S": S, "n_t": n_t, "written": 0}
        self._flush()

    def write_snapshot(self, seq_id: str, s: int, res, joints21=None, port_covered: bool = False) -> None:
        st = self._seq[seq_id]
        g = self._f[f"seq/{seq_id}"]
        if not 0 <= s < st["S"]:
            raise IndexError(f"snapshot {s} outside 0..{st['S'] - 1}")
        ir = np.asarray(res.ir, dtype=np.float32)
        if ir.shape != (st["n_t"],):
            raise ValueError(f"ir shape {ir.shape}, sequence declared n_t = {st['n_t']}")
        self._check_pulse_and_rate(res)
        g["ir"][s] = ir
        g["wall_s"][s] = res.wall_s
        g["port_covered"][s] = bool(port_covered)
        st["written"] = max(st["written"], s + 1)
        g.attrs["n_snapshots_written"] = st["written"]
        self._tick()

    def write_ref_ir(self, seq_id: str, res) -> None:
        st = self._seq[seq_id]
        g = self._f[f"seq/{seq_id}"]
        ir = np.asarray(res.ir, dtype=np.float32)
        if ir.shape != (st["n_t"],):
            raise ValueError(f"ref ir shape {ir.shape}, sequence declared n_t = {st['n_t']}")
        self._check_pulse_and_rate(res)
        if "ref_ir" in g:
            del g["ref_ir"]
        g.create_dataset("ref_ir", data=ir)
        self._tick()

    def fail_sequence(self, seq_id: str, reason: str) -> None:
        g = self._f[f"seq/{seq_id}"] if f"seq/{seq_id}" in self._f else self._f.create_group(f"seq/{seq_id}")
        g.attrs["failed"] = True
        g.attrs["failed_reason"] = str(reason)
        self._flush()

    def close(self) -> None:
        if self._f:
            self._f.flush()
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- internals ----------------------------------------------------------------------------------
    def _check_pulse_and_rate(self, res) -> None:
        pulse = np.asarray(res.pulse, dtype=np.float32)
        k = res.fs_sim / self.dev.fs_hz
        if abs(k - round(k)) > 1e-6 * k:    # fs_sim = 1/dt is a float (lead, 2026-11-11)
            raise ValueError(f"fs_sim {res.fs_sim} is not an integer multiple of fs_device {self.dev.fs_hz}")
        if self._pulse is None:
            self._pulse = pulse
            self._f.create_dataset("pulse", data=pulse)
            self._f.attrs["k_decim"] = int(round(k))
        elif pulse.shape != self._pulse.shape or not np.array_equal(pulse, self._pulse):
            raise ValueError("pulse differs from the shard's /pulse (identical for every frame)")
        elif int(round(k)) != int(self._f.attrs["k_decim"]):
            raise ValueError("k_decim changed within a shard")

    def _tick(self) -> None:
        self._since_flush += 1
        if self._since_flush >= FLUSH_EVERY:
            self._flush()

    def _flush(self) -> None:
        self._f.flush()
        self._since_flush = 0
