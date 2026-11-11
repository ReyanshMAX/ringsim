"""Solver without k-Wave: time axis, grid sizing, pulse, refusals before k-Wave (docs/SOLVER.md)."""
import numpy as np
import pytest

import ringsim.solver.kwave_runner as kr
from ringsim.devices.config import load_device
from ringsim.geometry.scene import free_scene
from ringsim.geometry.voxelize import Medium, SolidNearGridFace, build_medium
from ringsim.proc.physics import C_SIM
from ringsim.solver.constants import DOMAIN_MARGIN_M, PML_SIZE, PULSE_DELAY, SIGMA_T
from ringsim.solver.grid import _max_prime_factor, check_ppw, fast_dims, time_axis
from ringsim.solver.pulse import make_pulse


def test_constants_match_spec():
    assert SIGMA_T == pytest.approx(31.2e-6, rel=2e-3)
    assert PULSE_DELAY == pytest.approx(125e-6, rel=2e-3)


def test_time_axis_test_pair():
    """SOLVER.md worked numbers: test_pair (100 mm apart, dx 2 mm, 48 kHz, 84 bins) -> k = 12, t_end ~ 2.49 ms."""
    dev = load_device("test_pair")
    dt, n_t, k = time_axis(0.002, dev.range_window_m, dev.fs_hz, 0.100)
    assert k == 12
    assert dt == pytest.approx(1 / (12 * 48000))
    assert n_t * dt == pytest.approx(2.49e-3, abs=0.01e-3)
    assert C_SIM * dt / 0.002 <= 0.3                    # CFL_actual <= CFL


def test_time_axis_phone_and_ring_worked_numbers():
    dt, n_t, k = time_axis(0.002, 84 * 343.0 / 96000, 48000, 5.6e-3)
    assert k == 12 and n_t * dt == pytest.approx(2.21e-3, abs=0.01e-3)
    assert n_t == pytest.approx(1270, abs=10)           # "~1.27k steps"
    dt, n_t, k = time_axis(0.002, 54 * 343.0 / 100000, 50000, 5.4e-3)
    assert n_t * dt == pytest.approx(1.52e-3, abs=0.01e-3)


@pytest.mark.parametrize("n", [1, 11, 12, 45, 61, 97, 115, 220])
def test_fast_dims(n):
    m = fast_dims(n)
    assert m >= n and _max_prime_factor(m + 2 * PML_SIZE) <= 7
    assert all(_max_prime_factor(j + 2 * PML_SIZE) > 7 for j in range(n, m))   # smallest such m


def test_ppw_check():
    check_ppw(0.002)                                    # 6.1 points per wavelength
    with pytest.raises(ValueError, match="points per wavelength"):
        check_ppw(0.0035)                               # 3.5


def test_pulse_peaks_at_delay():
    fs = 12 * 48000
    p = make_pulse(fs, 600)
    assert p.dtype == np.float32
    env_peak_t = np.argmax(np.abs(p)) / fs
    assert env_peak_t == pytest.approx(PULSE_DELAY, abs=1 / 22000)


@pytest.fixture
def no_kwave(monkeypatch):
    """Any k-Wave call fails the test: proves the refusal happens before k-Wave."""
    def boom():
        raise AssertionError("k-Wave was called")
    monkeypatch.setattr(kr, "_kwave", boom)


def _free_medium():
    return build_medium(free_scene(load_device("test_pair")), 0.002, C_SIM, DOMAIN_MARGIN_M)


def test_port_collision_refused_before_kwave(no_kwave):
    m = _free_medium()
    m.mic_idx = list(m.src_idx)
    with pytest.raises(kr.PortCollision):
        kr.run_ir(m, 0.3, 48000, gpu=False)


def test_solid_at_face_refused_by_run_ir_before_kwave(no_kwave):
    m = _free_medium()
    m.occ = m.occ.copy()
    m.occ[1, 5, 5] = True                               # within 3 cells of the x = 0 face
    with pytest.raises(SolidNearGridFace):
        kr.run_ir(m, 0.3, 48000, gpu=False)


def test_run_ir_requires_c_sim(no_kwave):
    m = _free_medium()
    m.c = 337.0
    with pytest.raises(ValueError, match="C_SIM"):
        kr.run_ir(m, 0.3, 48000, gpu=False)


def test_simulate_has_no_c_assert_but_still_guards(no_kwave):
    m = _free_medium()
    m.c = 337.0
    with pytest.raises(AssertionError, match="k-Wave was called"):     # got past every refusal
        kr.simulate(m, 1e-6, 10, gpu=False)


def test_medium_is_the_import_shared_with_geometry():
    assert kr.Medium is Medium
