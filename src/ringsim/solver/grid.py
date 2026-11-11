"""Time axis and grid sizing (docs/SOLVER.md, Time and grid)."""
import numpy as np

from ringsim.solver.constants import (C_MAX, C_SIM, CFL, PML_SIZE, PPW_F_HZ, PPW_MIN, PULSE_DELAY, PULSE_TAIL,
                                      T_MARGIN_S)


def time_axis(dx: float, range_window_m: float, fs_device: int, d_direct_m: float) -> tuple[float, int, int]:
    """(dt, n_t, k). fs_sim = k * fs_device, so sim -> the job's device rate is plain decimation.

    d_direct_m: distance between the nearest speaker and mic port voxels. t_end is sized for the hottest
    supported air, C_MAX (D-044).
    """
    k = int(np.ceil(C_SIM / (CFL * dx * fs_device)))     # guarantees CFL_actual <= CFL
    dt = 1.0 / (k * fs_device)
    t_end = (PULSE_DELAY + PULSE_TAIL + d_direct_m / C_SIM
             + (C_MAX / C_SIM) * (2 * range_window_m / C_SIM) + T_MARGIN_S)
    return dt, int(np.ceil(t_end / dt)), k


def check_ppw(dx: float, c: float = C_SIM) -> None:
    """Raise if dx gives fewer than PPW_MIN points per wavelength at 28 kHz."""
    ppw = c / PPW_F_HZ / dx
    if ppw < PPW_MIN:
        raise ValueError(f"dx = {dx} m gives {ppw:.2f} points per wavelength at 28 kHz (< {PPW_MIN})")


def _max_prime_factor(n: int) -> int:
    p, m, f = 1, n, 2
    while f * f <= m:
        while m % f == 0:
            p, m = f, m // f
        f += 1
    return max(p, m)


def fast_dims(n: int) -> int:
    """Smallest m >= n such that (m + 2*PML_SIZE) has no prime factor > 7."""
    m = n
    while _max_prime_factor(m + 2 * PML_SIZE) > 7:
        m += 1
    return m
