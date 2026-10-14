"""Time axis and grid sizing (SOLVER.md)."""
import numpy as np

from ringsim.solver.constants import (C_MIN, C_SIM, CFL, F_MAX_HZ, PML_SIZE, PPW_MIN,
                                      PULSE_DELAY, T_MARGIN_S)


def check_ppw(dx: float, c: float = C_SIM) -> None:
    ppw = c / F_MAX_HZ / dx
    if ppw < PPW_MIN:
        raise ValueError(f"dx={dx} m gives {ppw:.2f} points per wavelength at 28 kHz < {PPW_MIN}")


def time_axis(dx: float, range_window_m: float, fs_device: int) -> tuple[float, int, int]:
    """(dt, n_t, k). fs_sim = k*fs_device so sim -> device is exact decimation."""
    k = int(np.ceil(C_SIM / (CFL * dx * fs_device)))     # guarantees CFL_actual <= CFL
    dt = 1.0 / (k * fs_device)
    t_end = PULSE_DELAY + 2 * range_window_m / C_MIN + T_MARGIN_S   # coldest supported air
    return dt, int(np.ceil(t_end / dt)), k


def _max_prime(n: int) -> int:
    p, m, f = 1, n, 2
    while f * f <= m:
        while m % f == 0:
            p, m = f, m // f
        f += 1
    return max(p, m)


def fast_dims(n: int) -> int:
    """Smallest m >= n such that m + 2*PML_SIZE has no prime factor > 7."""
    m = n
    while _max_prime(m + 2 * PML_SIZE) > 7:
        m += 1
    return m
