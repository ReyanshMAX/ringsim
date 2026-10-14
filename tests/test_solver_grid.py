import numpy as np
import pytest

from ringsim.solver.constants import C_SIM, CFL, PML_SIZE
from ringsim.solver.grid import _max_prime, check_ppw, fast_dims, time_axis


def test_time_axis_phone():
    dt, n_t, k = time_axis(0.002, 84 * 343.0 / 96000, 48000)
    assert k == 12 and abs(dt - 1 / 576000) < 1e-15
    assert C_SIM * dt / 0.002 <= CFL
    assert 1200 < n_t < 1350          # SOLVER.md: ~1.27k steps


def test_fast_dims():
    for n in range(5, 200):
        m = fast_dims(n)
        assert m >= n and _max_prime(m + 2 * PML_SIZE) <= 7
        assert all(_max_prime(j + 2 * PML_SIZE) > 7 for j in range(n, m))


def test_ppw():
    check_ppw(0.002)
    with pytest.raises(ValueError):
        check_ppw(0.004)
