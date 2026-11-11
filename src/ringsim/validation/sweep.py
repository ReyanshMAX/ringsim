"""Transducer response from sweep captures (VALIDATION.md Protocol A step 1).

Per take: H = R conj(S) / (|S|^2 + EPS max|S|^2), R the capture from its own direct-path delay on, S the sweep.
Each take is aligned on its own delay (integer + parabolic sub-sample, applied as a phase ramp) before the complex
average: every take has its own start latency, so unaligned phases would cancel. No window length is invented: the
capture is used from the delay to its end (common length = shortest take).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.signal import hilbert

from ringsim.validation.real import SAMPLE_RATE, read_wav

EPS = 1e-3                      # same regularisation as proc's deconvolutions (PROCESSING.md)
BAND_HZ = (18000.0, 22000.0)    # Protocol A step 1 phase band


def direct_delay(capture: np.ndarray, sweep: np.ndarray) -> float:
    """Delay of the sweep in the capture, samples (float): earliest local max of the xcorr envelope >= 0.5 max,
    refined by a parabola through the envelope (D-019's rule, applied to the sweep)."""
    n = next_fast_len(len(capture) + len(sweep))
    xc = irfft(rfft(capture, n) * np.conj(rfft(sweep, n)), n)[: len(capture) - len(sweep) + 1]
    env = np.abs(hilbert(xc))
    thr = 0.5 * env.max()
    idx = np.flatnonzero((env[1:-1] >= thr) & (env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:])) + 1
    if len(idx) == 0:
        raise ValueError("no direct-path peak found in the capture")
    k = int(idx[0])
    a, b, c = env[k - 1], env[k], env[k + 1]
    den = a - 2 * b + c
    return k + (0.5 * (a - c) / den if den != 0 else 0.0)


def complex_response(sweep_wav: str, capture_wavs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(freqs, H complex): aligned deconvolution of each take against the sweep, averaged."""
    if not capture_wavs:
        raise ValueError("no captures")
    sweep = read_wav(sweep_wav).astype(np.float64)
    caps, ints, fracs = [], [], []
    for p in capture_wavs:
        x = read_wav(p).astype(np.float64)
        d = direct_delay(x, sweep)
        di = int(np.floor(d + 0.5))
        caps.append(x), ints.append(di), fracs.append(d - di)
    L = min(len(x) - di for x, di in zip(caps, ints))
    if L < len(sweep):
        raise ValueError("a capture is shorter than the sweep after alignment")
    n = next_fast_len(L + len(sweep))
    freqs = rfftfreq(n, 1.0 / SAMPLE_RATE)
    S = rfft(sweep, n)
    denom = np.abs(S) ** 2 + EPS * np.abs(S).max() ** 2
    Hs = []
    for x, di, fr in zip(caps, ints, fracs):
        H = rfft(x[di:di + L], n) * np.conj(S) / denom
        Hs.append(H * np.exp(2j * np.pi * freqs * fr / SAMPLE_RATE))     # remove the sub-sample remainder
    return freqs, np.mean(Hs, axis=0)


def response(sweep_wav: str, capture_wavs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(freqs Hz, magnitude dB) of the transducer chain, 20 log10 |mean_k H_k|."""
    f, H = complex_response(sweep_wav, capture_wavs)
    return f, 20 * np.log10(np.maximum(np.abs(H), 1e-30))


def phase_residual(sweep_wav: str, capture_wavs: list[str], band: tuple[float, float] = BAND_HZ
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(freqs in band, residual phase rad): unwrapped phase of H minus its least-squares straight line over `band`
    (the part a pure delay cannot explain)."""
    f, H = complex_response(sweep_wav, capture_wavs)
    m = (f >= band[0]) & (f <= band[1])
    fb, ph = f[m], np.unwrap(np.angle(H[m]))
    return fb, ph - np.polyval(np.polyfit(fb, ph, 1), fb)
