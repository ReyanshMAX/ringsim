"""Motion within a frame (PROCESSING.md Required tests, D-041).

Ring chirp timing (20-24 kHz, 12 ms, fs 50 kHz) with window="none": the ring window is TBD(Q-002)
and only "none" is specified (D-064); revisit when Q-002 resolves.
Direct path plus two point reflectors at 50 mm and 110 mm moving radially at +v and -0.6 v.
Snapshot IRs = pulse delayed by tau(t_s) (fractional delay by FFT phase). Exact stream =
sum a * tx(t - tau(t)) from the chirp formula, at reception time t. Differential profiles over
frames 4..11 are compared by relative L2.
"""
import numpy as np
import pytest
from scipy import fft

from ringsim.proc import (C_SIM, ChirpSpec, align_direct_path, cfmcw_profiles, differential,
                          frame_stream, rx_stream, tx_frame)

FS, FS_SIM = 50_000, 600_000
SPEC = ChirpSpec(20000, 24000, 0.012, 0.012, "none")
BAND = (20e3, 24e3)
TX = tx_frame(SPEC, FS)
L = len(TX)
P = L / FS
N_FRAMES = 14
N_BINS = 54
N_T = 1200                                   # 2 ms at fs_sim
TAU_D = 1.0e-4                               # direct path delay, s
REFL = [(0.5, 0.050, 1.0), (0.3, 0.110, -0.6)]   # (amplitude, range at t=0 [m], speed / v)


def taus(t, v):
    """Path delays [K, len(t)] at reception time t: direct path, then the reflectors."""
    t = np.atleast_1d(t)
    out = [np.full_like(t, TAU_D)]
    for _, r0, k in REFL:
        out.append(TAU_D + 2 * (r0 + k * v * t) / C_SIM)
    return np.stack(out)


AMPS = np.array([1.0] + [a for a, _, _ in REFL])


def pulse():
    sigma = np.sqrt(2 * np.log(2)) / (2 * np.pi * 6000.0)
    t = np.arange(N_T) / FS_SIM - 4 * sigma
    return (np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * 22000.0 * t)).astype(np.float32)


def snapshot_irs(snap_t, v):
    p = pulse()
    Nf = 1 << 13
    Pf = fft.rfft(p, Nf)
    f = fft.rfftfreq(Nf, 1 / FS_SIM)
    tau = taus(snap_t, v)                                   # [K, S]
    H = np.einsum("k,ksf->sf", AMPS, np.exp(-2j * np.pi * f[None, None, :] * tau[:, :, None]))
    return fft.irfft(Pf[None] * H, Nf, axis=1)[:, :N_T].astype(np.float32), p


def tx_c(t):
    """The periodic transmitted chirp, from scipy.signal.chirp's linear formula."""
    ph = np.mod(t, P)
    k = (SPEC.f1_hz - SPEC.f0_hz) / SPEC.duration_s
    return np.cos(2 * np.pi * (SPEC.f0_hz * ph + 0.5 * k * ph**2))


def exact_stream(n, v):
    t = np.arange(n) / FS
    return (AMPS[:, None] * tx_c(t[None] - taus(t, v))).sum(axis=0)


def diff_profiles(stream, off):
    return differential(cfmcw_profiles(frame_stream(stream, off, L), TX, BAND, FS, N_BINS))


def motion_error(v, snaps_per_frame, interp):
    snap_t = np.arange((N_FRAMES + 2) * snaps_per_frame) * (L / snaps_per_frame) / FS
    irs, p = snapshot_irs(snap_t, v)
    sim = rx_stream(irs, snap_t, p, FS_SIM, TX, BAND, FS, N_FRAMES, interp=interp)
    ex = exact_stream(len(sim), v)
    off = align_direct_path(ex, TX, BAND, FS)
    d_sim, d_ex = diff_profiles(sim, off)[4:12], diff_profiles(ex, off)[4:12]
    return np.linalg.norm(d_sim - d_ex) / np.linalg.norm(d_ex)


def test_frame_start_snapshots_slow_motion():
    err = motion_error(0.1, 1, "cubic")
    print(f"frame-start, v=0.1: {err:.4f}")
    assert err < 0.05


def test_half_frame_snapshots_fast_motion():
    err = motion_error(0.3, 2, "cubic")
    print(f"T/2, v=0.3: {err:.4f}")
    assert err < 0.10


def test_hold_cannot_follow_motion():
    err = motion_error(0.3, 1, "hold")
    print(f"hold, v=0.3: {err:.4f}")
    assert err > 0.5
