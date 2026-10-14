"""Impulse response -> received stream (PROCESSING.md, D-002, D-037)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import fft, signal

TAPER_HZ = 1000.0               # raised-cosine taper width outside each band edge
RESAMPLE_TRANSITION_HZ = 1000.0 # anti-alias FIR transition, centred 500 Hz below Nyquist
RESAMPLE_ATTEN_DB = 80.0


@lru_cache(maxsize=8)
def _resample_fir(k: int, fs_device: int) -> np.ndarray:
    """Kaiser low-pass for resample_poly at fs_sim = k*fs_device.

    resample_poly's default filter droops 21% at 22 kHz (48 kHz device, k=12, up+down);
    this one is flat (<0.01%) up to fs_device/2 - 1 kHz.
    """
    fs_sim = k * fs_device
    n, beta = signal.kaiserord(RESAMPLE_ATTEN_DB, RESAMPLE_TRANSITION_HZ / (fs_sim / 2))
    return signal.firwin(n | 1, fs_device / 2 - RESAMPLE_TRANSITION_HZ / 2,
                         window=("kaiser", beta), fs=fs_sim)


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    m = np.zeros_like(freqs)
    m[(freqs >= lo) & (freqs <= hi)] = 1.0
    lt = (freqs >= lo - TAPER_HZ) & (freqs < lo)
    m[lt] = 0.5 * (1 - np.cos(np.pi * (freqs[lt] - (lo - TAPER_HZ)) / TAPER_HZ))
    ht = (freqs > hi) & (freqs <= hi + TAPER_HZ)
    m[ht] = 0.5 * (1 + np.cos(np.pi * (freqs[ht] - hi) / TAPER_HZ))
    return m


def temperature_scale(H: np.ndarray, freqs: np.ndarray, alpha: float) -> np.ndarray:
    """H_{alpha c}(f) = alpha * H_c(f / alpha), exact for this model (D-037).

    Linear interpolation of real and imaginary parts on the (dense) rfft grid;
    0 where f/alpha > freqs[-1].
    """
    if alpha == 1.0:
        return H
    fq = freqs / alpha
    out = np.interp(fq, freqs, H.real, right=0.0) + 1j * np.interp(fq, freqs, H.imag, right=0.0)
    return alpha * out


def rx_from_ir(ir: np.ndarray, pulse: np.ndarray, k: int, tx: np.ndarray,
               band: tuple[float, float], fs_device: int, alpha: float = 1.0,
               eps: float = 1e-3) -> np.ndarray:
    """Received stream at fs_device for one tx frame, from a sim IR at fs_sim = k*fs_device.

    The pulse (and its delay) is removed by regularized deconvolution.
    """
    h = _resample_fir(k, fs_device)
    tx_up = signal.resample_poly(tx, up=k, down=1, window=h)
    n_out = len(tx_up) + len(ir)
    N = fft.next_fast_len(max(n_out, 2**16), real=True)
    R = fft.rfft(ir, N)
    P = fft.rfft(pulse, N)
    P2 = np.abs(P) ** 2
    H = R * np.conj(P) / (P2 + eps * P2.max())
    freqs = fft.rfftfreq(N, 1.0 / (k * fs_device))
    if alpha != 1.0:
        H = temperature_scale(H, freqs, alpha)
    H = H * _band_mask(freqs, band)
    y = fft.irfft(H * fft.rfft(tx_up, N), N)[:n_out]
    return signal.resample_poly(y, up=1, down=k, window=h).astype(np.float32)


def overlap_add_stream(frame_rx: list[np.ndarray], frame_len: int) -> np.ndarray:
    """Frame t's response starts at t*frame_len; echoes spill into frame t+1."""
    n = max((t * frame_len + len(r) for t, r in enumerate(frame_rx)), default=0)
    out = np.zeros(n, np.float32)
    for t, r in enumerate(frame_rx):
        out[t * frame_len: t * frame_len + len(r)] += r
    return out
