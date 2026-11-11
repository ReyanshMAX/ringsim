"""Shared test fixtures (seat 4 owns this file).

Nothing here needs k-Wave. Markers in use: "slow" only (declared in pyproject.toml).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

C_SIM = 343.0


@dataclass
class FakeSimResult:
    """Same fields as solver.kwave_runner.SimResult (ARCHITECTURE.md), built without k-Wave."""
    ir: np.ndarray       # float32 [n_t]
    pulse: np.ndarray    # float32 [n_t]
    fs_sim: float
    wall_s: float


@pytest.fixture
def shard_path(tmp_path):
    return str(tmp_path / "synthetic.h5")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def make_pulse(n_t: int, fs_sim: float, f_lo: float = 16e3, f_hi: float = 28e3,
               delay_s: float = 2.0e-4) -> np.ndarray:
    """Band-limited test pulse (Hann-windowed Gaussian-free chirp); not the solver's pulse."""
    from scipy.signal import chirp
    n = int(round(4.0e-4 * fs_sim))
    t = np.arange(n) / fs_sim
    p = chirp(t, f_lo, t[-1], f_hi) * np.hanning(n)
    out = np.zeros(n_t, dtype=np.float32)
    d = int(round(delay_s * fs_sim))
    out[d:d + n] = p[: max(0, min(n, n_t - d))]
    return out


@pytest.fixture
def synth_ir():
    """Factory: monopole-like IR at `dist_m`, built by delaying/scaling the pulse (no k-Wave).

    Returns a FakeSimResult. The arrival sits at dist_m/c + pulse delay, 1/r amplitude.
    """
    def _make(dist_m: float = 0.100, fs_sim: float = 576_000.0, n_t: int = 4096,
              c: float = C_SIM, delay_s: float = 2.0e-4) -> FakeSimResult:
        pulse = make_pulse(n_t, fs_sim, delay_s=delay_s)
        ir = np.zeros(n_t, dtype=np.float32)
        shift = dist_m / c * fs_sim
        k = int(np.floor(shift))
        frac = shift - k
        ir[k] += (1 - frac) / dist_m
        ir[k + 1] += frac / dist_m
        # fold the pulse into the IR as k-Wave records it: pulse response at the mic
        resp = np.convolve(pulse, ir)[:n_t].astype(np.float32)
        return FakeSimResult(ir=resp, pulse=pulse, fs_sim=fs_sim, wall_s=0.0)
    return _make
