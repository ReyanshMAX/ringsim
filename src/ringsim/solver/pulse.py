"""Band-limited source pulse (SOLVER.md)."""
import numpy as np

from ringsim.solver.constants import PULSE_DELAY, PULSE_FC_HZ, SIGMA_T


def make_pulse(fs_sim: float, n_t: int) -> np.ndarray:
    t = np.arange(n_t) / fs_sim - PULSE_DELAY
    return (np.exp(-t**2 / (2 * SIGMA_T**2)) * np.sin(2 * np.pi * PULSE_FC_HZ * t)).astype(np.float32)
