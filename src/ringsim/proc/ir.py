"""Simulated impulse responses -> received device stream (PROCESSING.md, D-037, D-041, D-046)."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import fft, signal

TAPER_HZ = 1000.0                # raised-cosine taper width outside each band edge
RESAMPLE_TRANSITION_HZ = 1000.0  # Kaiser FIR transition, centred 500 Hz below fs_device/2
RESAMPLE_ATTEN_DB = 80.0


def _ratio(fs_sim: int, fs_device: int) -> tuple[int, int]:
    """q/p = fs_sim/fs_device as a reduced fraction (D-046)."""
    r = Fraction(int(fs_sim), int(fs_device))
    return r.numerator, r.denominator


@lru_cache(maxsize=16)
def _resample_fir(fs_sim: int, fs_device: int) -> np.ndarray:
    """Kaiser low-pass h for both resample_poly calls, designed at fs_device*q (= fs_sim*p).

    80 dB stop band, 1 kHz transition centred 500 Hz below fs_device/2. Unit DC gain;
    resample_poly multiplies a given window by `up` itself. Odd length, so resample_poly's
    delay compensation is exact.
    """
    q, _ = _ratio(fs_sim, fs_device)
    fs_hi = fs_device * q
    n, beta = signal.kaiserord(RESAMPLE_ATTEN_DB, RESAMPLE_TRANSITION_HZ / (fs_hi / 2))
    h = signal.firwin(n | 1, fs_device / 2 - RESAMPLE_TRANSITION_HZ / 2,
                      window=("kaiser", beta), fs=fs_hi)
    h.setflags(write=False)
    return h


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """1 inside band, 1 kHz raised-cosine taper outside each edge, 0 elsewhere."""
    lo, hi = band
    f = np.asarray(freqs, dtype=float)
    m = np.zeros_like(f)
    m[(f >= lo) & (f <= hi)] = 1.0
    lt = (f >= lo - TAPER_HZ) & (f < lo)
    m[lt] = 0.5 * (1 - np.cos(np.pi * (f[lt] - (lo - TAPER_HZ)) / TAPER_HZ))
    ht = (f > hi) & (f <= hi + TAPER_HZ)
    m[ht] = 0.5 * (1 + np.cos(np.pi * (f[ht] - hi) / TAPER_HZ))
    return m


def temperature_scale(H: np.ndarray, freqs: np.ndarray, alpha: float) -> np.ndarray:
    """H_{alpha c}(f) = alpha * H_c(f / alpha) (D-037; derivation in PROCESSING.md).

    Linear interpolation of real and imaginary parts on the rfft grid, which must be dense
    (rx_from_ir uses N >= 2**16 at fs_sim). 0 where f/alpha > freqs[-1]. alpha = 1 is the identity.
    """
    if alpha == 1.0:
        return H
    fq = freqs / alpha
    out = (np.interp(fq, freqs, H.real, right=0.0)
           + 1j * np.interp(fq, freqs, H.imag, right=0.0))
    return alpha * out


def rx_from_ir(ir: np.ndarray, pulse: np.ndarray, fs_sim: int, tx: np.ndarray,
               band: tuple[float, float], fs_device: int, alpha: float = 1.0,
               eps: float = 1e-3) -> np.ndarray:
    """Response of ONE static snapshot to the tx samples given (any length), at fs_device.

    The source pulse (and its delay) is removed by regularized deconvolution. Output length
    ceil((len(tx_up) + len(ir)) * p / q). The first and last ~20 device samples carry
    resampling edge errors (samples beyond the ends of tx count as 0).
    """
    q, p = _ratio(fs_sim, fs_device)
    h = _resample_fir(fs_sim, fs_device)
    tx_up = signal.resample_poly(np.asarray(tx, dtype=float), up=q, down=p, window=h)
    n_y = len(tx_up) + len(ir)
    N = fft.next_fast_len(max(n_y, 2**16), real=True)
    R = fft.rfft(ir, N)
    P = fft.rfft(pulse, N)
    P2 = np.abs(P) ** 2
    H = R * np.conj(P) / (P2 + eps * P2.max())
    freqs = fft.rfftfreq(N, 1.0 / fs_sim)
    if alpha != 1.0:
        H = temperature_scale(H, freqs, alpha)
    H = H * _band_mask(freqs, band)
    y = fft.irfft(H * fft.rfft(tx_up, N), N)[:n_y]
    return signal.resample_poly(y, up=p, down=q, window=h).astype(np.float32)


def snapshot_weights(snap_t: np.ndarray, t: np.ndarray, interp: str = "cubic") -> np.ndarray:
    """W [S, len(t)]: weight of snapshot s at time t[n]; every column sums to 1 (D-041).

    "cubic": cubic Hermite between neighbouring snapshots, tangent at snapshot i
    (p[i+1] - p[i-1]) / (snap_t[i+1] - snap_t[i-1]), one-sided at the two ends; Catmull-Rom
    when evenly spaced. Outside [snap_t[0], snap_t[-1]] the nearest end snapshot is held.
    "hold": the latest snapshot with snap_t <= t (the first one before snap_t[0]); with
    snapshots at frame starts this is the frame-frozen model. Validation reports only.
    """
    st = np.asarray(snap_t, dtype=float)
    t = np.asarray(t, dtype=float)
    S = len(st)
    if S == 1:
        return np.ones((1, len(t)))
    if np.any(np.diff(st) <= 0):
        raise ValueError("snap_t must be strictly increasing")
    if interp == "hold":
        idx = np.clip(np.searchsorted(st, t, side="right") - 1, 0, S - 1)
        W = np.zeros((S, len(t)))
        W[idx, np.arange(len(t))] = 1.0
        return W
    if interp != "cubic":
        raise ValueError(f"interp must be 'cubic' or 'hold', got {interp!r}")

    # tangents m = M @ p (linear in p)
    M = np.zeros((S, S))
    M[0, 0], M[0, 1] = -1 / (st[1] - st[0]), 1 / (st[1] - st[0])
    M[-1, -2], M[-1, -1] = -1 / (st[-1] - st[-2]), 1 / (st[-1] - st[-2])
    for i in range(1, S - 1):
        d = st[i + 1] - st[i - 1]
        M[i, i - 1], M[i, i + 1] = -1 / d, 1 / d

    i = np.clip(np.searchsorted(st, t, side="right") - 1, 0, S - 2)
    hseg = st[i + 1] - st[i]
    u = np.clip((t - st[i]) / hseg, 0.0, 1.0)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    cols = np.arange(len(t))
    W = (M[i].T * (h10 * hseg) + M[i + 1].T * (h11 * hseg))
    W[i, cols] += h00
    W[i + 1, cols] += h01
    return W


def rx_stream(irs: np.ndarray, snap_t: np.ndarray, pulse: np.ndarray, fs_sim: int,
              tx: np.ndarray, band: tuple[float, float], fs_device: int, n_frames: int,
              alpha: float = 1.0, eps: float = 1e-3, interp: str = "cubic") -> np.ndarray:
    """Continuous received stream for n_frames chirp periods from IR snapshots (D-041).

    irs [S, n_t] at fs_sim; snap_t [S] seconds; frame f's chirp starts at f*len(tx)/fs_device.
    The tx stream runs from frame -1 (one warm-up period) through frame n_frames. Each snapshot
    is convolved over the support of its weights, padded by one period each side (padding
    discarded), and out[n] = sum_s W[s, n] * y_s[n] with W at the reception time n/fs_device.
    Flight time (<= 2 ms) is ignored when evaluating W.
    Returns float32 of length n_frames*len(tx) + ceil(n_t*fs_device/fs_sim).
    """
    irs = np.atleast_2d(np.asarray(irs))
    S, n_t = irs.shape
    L = len(tx)
    q, p = _ratio(fs_sim, fs_device)
    n_out = n_frames * L + -(-n_t * p // q)
    tx_s = np.tile(np.asarray(tx, dtype=float), n_frames + 2)   # index m <-> time (m - L)/fs
    W = snapshot_weights(snap_t, np.arange(n_out) / fs_device, interp)
    out = np.zeros(n_out)
    for s in range(S):
        nz = np.flatnonzero(W[s])
        if nz.size == 0:
            continue
        n_lo, n_hi = nz[0], nz[-1] + 1
        a = max(0, n_lo)                       # tx index of output n_lo minus one period
        b = min(len(tx_s), n_hi + 2 * L)       # tx index of output n_hi plus one period
        y = rx_from_ir(irs[s], pulse, fs_sim, tx_s[a:b], band, fs_device, alpha, eps)
        k0 = n_lo - a + L                      # y index of output n_lo
        out[n_lo:n_hi] += W[s, n_lo:n_hi] * y[k0:k0 + (n_hi - n_lo)]
    return out.astype(np.float32)
