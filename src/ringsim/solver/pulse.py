"""Band-limited source pulse (docs/SOLVER.md, Pulse)."""
import numpy as np

from ringsim.solver.constants import PULSE_DELAY, PULSE_FC_HZ, SIGMA_T


def make_pulse(fs_sim: float, n_t: int) -> np.ndarray:
    """Gaussian-windowed 22 kHz sine centred at PULSE_DELAY, float32 [n_t]."""
    t = np.arange(n_t) / fs_sim - PULSE_DELAY
    return (np.exp(-t**2 / (2 * SIGMA_T**2)) * np.sin(2 * np.pi * PULSE_FC_HZ * t)).astype(np.float32)
