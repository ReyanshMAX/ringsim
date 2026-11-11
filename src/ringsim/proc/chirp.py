"""Chirp synthesis and tx WAV export (PROCESSING.md)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.io import wavfile


@dataclass(frozen=True)
class ChirpSpec:
    """Chirp keys of a device config (DEVICES.md). devices.config imports this class."""
    f0_hz: float
    f1_hz: float
    duration_s: float
    frame_period_s: float
    window: str                     # only "none" is specified; any other window is Q-002


def _window(name: str, n: int) -> np.ndarray:
    if name == "none":
        return np.ones(n)
    raise NotImplementedError(f"chirp window {name!r}: only 'none' is specified (Q-002)")


def synth_chirp(spec: ChirpSpec, fs: int) -> np.ndarray:
    """Linear chirp f0 -> f1 over duration_s, window applied, float32, peak 1.0."""
    n = int(round(spec.duration_s * fs))
    t = np.arange(n) / fs
    x = signal.chirp(t, spec.f0_hz, spec.duration_s, spec.f1_hz, method="linear")
    x = x * _window(spec.window, n)
    return (x / np.max(np.abs(x))).astype(np.float32)


def tx_frame(spec: ChirpSpec, fs: int) -> np.ndarray:
    """synth_chirp zero-padded to round(frame_period_s * fs) samples."""
    c = synth_chirp(spec, fs)
    n = int(round(spec.frame_period_s * fs))
    if n < len(c):
        raise ValueError(f"frame_period_s gives {n} samples, shorter than the {len(c)}-sample chirp")
    out = np.zeros(n, np.float32)
    out[: len(c)] = c
    return out


def export_tx_wav(spec: ChirpSpec, fs: int, path: str) -> None:
    """One tx_frame as a float32 mono WAV; the capture app loops it seamlessly."""
    wavfile.write(path, fs, tx_frame(spec, fs))


def export_sweep_wav(f0: float, f1: float, dur_s: float, fs: int, path: str) -> None:
    """Log sweep f0 -> f1 for the transducer response (Q-011), Hann fade 10 ms at each end.

    dur_s < 20 ms (the two fades overlap) is refused with ValueError; the spec defines no shorter sweep.
    """
    if dur_s < 0.020:
        raise ValueError(f"dur_s = {dur_s} s is shorter than the two 10 ms fades")
    n = int(round(dur_s * fs))
    t = np.arange(n) / fs
    x = signal.chirp(t, f0, dur_s, f1, method="logarithmic")
    nf = int(round(0.010 * fs))
    fade = np.hanning(2 * nf)
    x[:nf] *= fade[:nf]
    x[-nf:] *= fade[nf:]
    wavfile.write(path, fs, x.astype(np.float32))
