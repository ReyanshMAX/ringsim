"""Shard reader: IRs -> profiles through ringsim.proc (DATASET.md, D-009, D-037)."""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from ringsim import proc
from ringsim.devices.config import DeviceConfig, device_from_yaml


@dataclass
class SeqSample:
    profiles: np.ndarray        # [T, n_bins] signed
    diff: np.ndarray            # [T-1, n_bins]
    labels: np.ndarray | None   # [T, 20, 3]; None for scenes without a hand
    meta: dict


def list_sequences(path: str) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(f["seq"].keys())


def shard_device(f: h5py.File) -> DeviceConfig:
    """DeviceConfig from the shard, never from the current repo YAML."""
    return device_from_yaml(f.attrs["device_yaml"])


def rx_stream(f: h5py.File, seq_id: str, alpha: float = 1.0) -> tuple[np.ndarray, DeviceConfig]:
    """Overlap-added received stream at the device rate for the written frames."""
    dev = shard_device(f)
    g = f[f"seq/{seq_id}"]
    n = int(g.attrs["n_frames_written"])
    failed = set(g["failed_frames"][()].tolist())
    tx = proc.tx_frame(dev.chirp, dev.fs_hz)
    pulse = f["pulse"][()]
    k = int(f.attrs["k_decim"])
    rx = []
    for t in range(n):
        if t in failed:
            raise NotImplementedError(f"frame {t} failed; gap handling is not specified")
        rx.append(proc.rx_from_ir(g["ir"][t], pulse, k, tx, dev.band, dev.fs_hz, alpha))
    return proc.overlap_add_stream(rx, dev.frame_len), dev


def load_sequence(path: str, seq_id: str, snr_db: float | None,
                  temp_c: float | None, rng: np.random.Generator | None) -> SeqSample:
    if snr_db is not None:
        # DATASET.md: SNR is relative to direct-path power in band, but the window that
        # power is measured over is not specified (Q-016).
        raise NotImplementedError("snr_db: direct-path reference power undefined (Q-016)")
    alpha = 1.0 if temp_c is None else proc.temperature_alpha(temp_c)
    with h5py.File(path, "r") as f:
        stream, dev = rx_stream(f, seq_id, alpha)
        g = f[f"seq/{seq_id}"]
        labels = g["labels"][()] if "labels" in g else None
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in g.attrs.items()}
    tx = proc.tx_frame(dev.chirp, dev.fs_hz)
    off = proc.align_direct_path(stream, tx, dev.band, dev.fs_hz)
    frames = proc.frame_stream(stream, off, dev.frame_len)
    prof = proc.cfmcw_profiles(frames, tx, dev.band, dev.fs_hz, dev.n_range_bins)
    meta.update(offset=off, temp_c=temp_c, alpha=alpha)
    return SeqSample(profiles=prof, diff=proc.differential(prof), labels=labels, meta=meta)
