"""Phase 1 free-field check: sim profile vs the profile of an ideal monopole.

k-Wave's additive pressure source is a mass source, so the free-field pressure is the
time derivative of the delayed source: H_ref(f) = j*2*pi*f * exp(-2*pi*i*f*tau),
tau = |spk - mic| / C_SIM (measured: phase 96-98 deg vs pure delay across 17-27 kHz).
The reference stream is built through the same proc path (band mask, align, frame,
C-FMCW); one complex gain is fitted on the analytic profiles before the L2 error.
"""
from __future__ import annotations

import h5py
import numpy as np
from scipy import fft, signal

from ringsim import proc
from ringsim.dataset.reader import rx_stream
from ringsim.proc.ir import _band_mask


def _profiles(stream, dev):
    tx = proc.tx_frame(dev.chirp, dev.fs_hz)
    off = proc.align_direct_path(stream, tx, dev.band, dev.fs_hz)
    fr = proc.frame_stream(stream, off, dev.frame_len)
    return proc.cfmcw_profiles(fr, tx, dev.band, dev.fs_hz, dev.n_range_bins), off


def ideal_stream(dev, n_frames: int, n_out: int) -> np.ndarray:
    tx = proc.tx_frame(dev.chirp, dev.fs_hz)
    d = np.linalg.norm(np.subtract(dev.mic.points_mm[0], dev.speaker.points_mm[0])) * 1e-3
    tau = d / proc.C_SIM
    N = fft.next_fast_len(max(len(tx) + n_out, 2**14))
    f = fft.rfftfreq(N, 1 / dev.fs_hz)
    H = 2j * np.pi * f * np.exp(-2j * np.pi * f * tau) * _band_mask(f, dev.band)
    one = fft.irfft(fft.rfft(tx, N) * H, N)[:n_out]
    return proc.overlap_add_stream([one.astype(np.float32)] * n_frames, dev.frame_len)


def freefield_check(path: str, seq_id: str) -> dict:
    with h5py.File(path, "r") as f:
        stream, dev = rx_stream(f, seq_id)
        n = int(f[f"seq/{seq_id}"].attrs["n_frames_written"])
    sim, off_sim = _profiles(stream, dev)
    n_one = len(stream) - (n - 1) * dev.frame_len
    ref, off_ref = _profiles(ideal_stream(dev, n, n_one), dev)
    a, b = signal.hilbert(sim[0]), signal.hilbert(ref[0])
    g = np.vdot(b, a) / np.vdot(b, b)
    err = float(np.linalg.norm(a - g * b) / np.linalg.norm(a))
    env = np.abs(a)
    return {"peak_bin": int(np.argmax(env)), "peak_bin_signed": int(np.argmax(np.abs(sim[0]))),
            "ref_peak_bin": int(np.argmax(np.abs(b))), "rel_l2_vs_ideal": err,
            "gain_phase_deg": float(np.degrees(np.angle(g))),
            "offset_sim": off_sim, "offset_ref": off_ref}
