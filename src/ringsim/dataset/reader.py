"""Shard reader: IR snapshots -> C-FMCW profiles through ringsim.proc (DATASET.md "Reader").

The DeviceConfig is rebuilt from the shard's `device_yaml` (D-055), never from the repo YAML.
Profiles are never stored; this is the only place they are made from a shard.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

N_AVG = 50                       # free_air_reference default, PROCESSING.md
PULSE_BAND_HZ = (16_000.0, 28_000.0)   # solver pulse band (SOLVER.md); a chirp must sit inside it
C_SIM = 343.0                    # = ringsim.proc.physics.C_SIM; shard attr c_sim is checked against it


class SequenceFailed(Exception):
    """The sequence's `failed` attr is set (GEOMETRY.md rejection rule, D-048)."""


@dataclass
class SeqSample:
    profiles: np.ndarray        # [T, n_bins] signed; referenced (D-042) unless mode="raw"
    diff: np.ndarray            # [T-1, n_bins]
    labels: np.ndarray | None   # [T, 20, 3]
    meta: dict


def list_sequences(path: str, include_failed: bool = False) -> list[str]:
    with h5py.File(path, "r") as f:
        if "seq" not in f:
            return []
        return sorted(k for k, g in f["seq"].items() if include_failed or not bool(g.attrs["failed"]))


def _usable_frames(frame_snap: np.ndarray, n_snapshots: int, written: int) -> int:
    """Frames whose snapshots through the next frame start (or the end snapshot) were all written."""
    T = len(frame_snap)
    n = 0
    for f in range(T):
        last = frame_snap[f + 1] if f + 1 < T else n_snapshots - 1
        if last < written:
            n += 1
        else:
            break
    return n


def _check_override(shard_dev, ov, n_t: int, fs_sim: float) -> None:
    """D-046 / D-058: same body and ports; fs, chirp, n_range_bins may differ."""
    if ov.body != shard_dev.body:
        raise ValueError("dev_override has a different body than the shard's device")
    if ov.ports != shard_dev.ports or ov.active_pair != shard_dev.active_pair:
        raise ValueError("dev_override has different ports or active_pair than the shard's device")
    lo, hi = sorted((ov.chirp.f0_hz, ov.chirp.f1_hz))
    if lo < PULSE_BAND_HZ[0] or hi > PULSE_BAND_HZ[1]:
        raise ValueError(f"dev_override chirp band {lo:.0f}-{hi:.0f} Hz leaves the solver band {PULSE_BAND_HZ}")
    # Window: SOLVER.md time_axis t_end = fixed terms + (C_MAX/C_SIM) * 2 * window / C_SIM. Only the window term
    # differs between the shard's device and the override, and the shard holds exactly the IR its own window
    # needed (to one sample), so any larger window needs IR the shard does not have.
    if ov.range_window_m > shard_dev.range_window_m + 1e-12:
        raise ValueError(
            f"dev_override window {ov.range_window_m:.4f} m exceeds the shard's {shard_dev.range_window_m:.4f} m "
            f"(IR of {n_t / fs_sim * 1e3:.3f} ms is not enough)")


def profiles_from_arrays(irs: np.ndarray, snap_t: np.ndarray, pulse: np.ndarray, fs_sim: float,
                         ref_ir: np.ndarray | None, dev, n_frames: int, alpha: float,
                         mode: str) -> tuple[np.ndarray, int]:
    """The IR -> profile chain of DATASET.md "Reader", on arrays. Shared with validation (the ideal monopole goes
    through exactly this code). Returns (profiles [n_frames, n_bins], direct-path offset)."""
    from ringsim.proc.chirp import tx_frame
    from ringsim.proc.ir import rx_stream
    from ringsim.proc.profile import (align_direct_path, cfmcw_profiles, cfmcw_profiles_ref, frame_stream,
                                      free_air_reference)
    band = (min(dev.chirp.f0_hz, dev.chirp.f1_hz), max(dev.chirp.f0_hz, dev.chirp.f1_hz))
    tx = tx_frame(dev.chirp, dev.fs_hz)
    rx = rx_stream(irs, snap_t, pulse, fs_sim, tx, band, dev.fs_hz, n_frames, alpha)
    offset = align_direct_path(rx, tx, band, dev.fs_hz)
    frames = frame_stream(rx, offset, len(tx))[:n_frames]
    if mode == "ref":
        rx_ref = rx_stream(ref_ir[None, :], np.zeros(1), pulse, fs_sim, tx, band, dev.fs_hz, N_AVG + 1, alpha)
        ref_offset = align_direct_path(rx_ref, tx, band, dev.fs_hz)
        ref = free_air_reference(rx_ref, tx, band, dev.fs_hz, ref_offset, N_AVG)
        return cfmcw_profiles_ref(frames, tx, ref, band, dev.fs_hz, dev.n_range_bins), int(offset)
    return cfmcw_profiles(frames, tx, band, dev.fs_hz, dev.n_range_bins), int(offset)


def load_sequence(path: str, seq_id: str, snr_db: float | None, temp_c: float | None,
                  rng: np.random.Generator | None, mode: str = "ref",
                  dev_override=None) -> SeqSample:
    if mode not in ("ref", "raw"):
        raise ValueError(f"mode must be 'ref' or 'raw', got {mode!r}")
    if snr_db is not None:
        raise NotImplementedError("Q-016: the noise model (SNR range, reference power) is not specified; "
                                  "load_sequence takes snr_db=None until it is")
    from ringsim.devices.config import load_device_yaml
    from ringsim.proc import physics
    with h5py.File(path, "r") as f:
        g = f["seq"][seq_id]
        if bool(g.attrs["failed"]):
            raise SequenceFailed(f"{seq_id}: {g.attrs['failed_reason']}")
        if float(f.attrs["c_sim"]) != physics.C_SIM:
            raise ValueError(f"shard c_sim {f.attrs['c_sim']} != {physics.C_SIM}")
        shard_dev = load_device_yaml(str(f.attrs["device_yaml"]))
        fs_device = int(f.attrs["fs_device"])
        k_decim = int(f.attrs["k_decim"])
        fs_sim = float(k_decim * fs_device)
        pulse = f["pulse"][:]
        S = int(g.attrs["n_snapshots"])
        written = int(g.attrs["n_snapshots_written"])
        frame_snap = g["frame_snap"][:]
        T = _usable_frames(frame_snap, S, written)
        if T == 0:
            raise ValueError(f"{seq_id}: no complete frame in the shard ({written}/{S} snapshots written)")
        irs = g["ir"][:written]
        snap_t = g["snap_t"][:written]
        ref_ir = g["ref_ir"][:] if "ref_ir" in g else None
        labels = g["labels"][:T] if "labels" in g else None
        source = str(g.attrs["source"])
        n_t = irs.shape[1]
        partial = T < int(g.attrs["n_frames"])

    dev = dev_override if dev_override is not None else shard_dev
    if dev_override is not None:
        _check_override(shard_dev, dev_override, n_t, fs_sim)
    if mode == "ref" and ref_ir is None:
        raise ValueError(f"{seq_id}: no ref_ir in the shard (mode='ref' needs it, D-042)")

    alpha = 1.0 if temp_c is None else physics.temperature_alpha(temp_c)
    profiles, offset = profiles_from_arrays(irs, snap_t, pulse, fs_sim, ref_ir if mode == "ref" else None,
                                            dev, T, alpha, mode)

    meta = {"seq_id": seq_id, "source": source, "mode": mode, "alpha": alpha, "temp_c": temp_c,
            "device": dev.name, "fs_hz": dev.fs_hz, "n_frames": T, "partial": partial,
            "offset": int(offset), "build_id": None}
    from ringsim.proc.profile import differential
    return SeqSample(profiles=profiles, diff=differential(profiles), labels=labels, meta=meta)
