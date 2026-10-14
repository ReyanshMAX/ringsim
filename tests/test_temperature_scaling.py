"""D-037 check on CPU k-Wave: one sim at C_SIM, time-scaled to 337 m/s, vs a direct sim.

Fixture (designed 2026-10-14, PROCESSING.md): 64^3 grid, dx 2 mm, smoothed density
(D-036); a 60x60x4 mm plate 40 mm above the source and two 8x8x60 mm bars, all rigid
(rho 1000). Speaker and mic voxels 6 mm apart near the bottom of the grid.
"""
import numpy as np
import pytest
import trimesh
from scipy import fft

from ringsim.geometry.voxelize import Medium, occupancy, smooth_rho
from ringsim.proc import C_SIM, temperature_scale
from ringsim.solver.grid import time_axis
from ringsim.solver.kwave_runner import _simulate

pytestmark = pytest.mark.slow

DX = 0.002
DIMS = (64, 64, 64)
C_COLD = 337.0
FS_DEVICE = 48000
RANGE_M = 0.30


def _fixture_medium(c: float) -> Medium:
    origin = np.zeros(3)
    # +dx/2 keeps every box face between grid points: mesh.contains is not
    # deterministic for points exactly on a face.
    ctr = np.array(DIMS) * DX / 2 + DX / 2
    boxes = [
        trimesh.creation.box(extents=[0.060, 0.060, 0.004]).apply_translation(ctr + [0, 0, 0.040 - 0.040]),
        trimesh.creation.box(extents=[0.008, 0.008, 0.060]).apply_translation(ctr + [-0.030, 0.020, -0.010]),
        trimesh.creation.box(extents=[0.008, 0.060, 0.008]).apply_translation(ctr + [0.025, -0.005, -0.025]),
    ]
    occ = occupancy(boxes, origin, DIMS, DX)
    src = (32, 30, 12)                 # z = 24 mm; plate centre at z = 64 mm
    mic = (32, 33, 12)
    assert not occ[src] and not occ[mic] and occ.sum() > 0
    return Medium(rho=smooth_rho(occ), c=c, dx=DX, origin_m=origin, src_idx=[src], mic_idx=[mic])


def _H(ir, pulse, fs, N):
    R, P = fft.rfft(ir, N), fft.rfft(pulse, N)
    P2 = np.abs(P) ** 2
    return R * np.conj(P) / (P2 + 1e-3 * P2.max())


@pytest.fixture(scope="module")
def fitted():
    dt, n_t, k = time_axis(DX, RANGE_M, FS_DEVICE)     # equal dt for both sims
    ir_ref, pulse, _ = _simulate(_fixture_medium(C_SIM), dt, n_t, gpu=False)
    ir_cold, pulse2, _ = _simulate(_fixture_medium(C_COLD), dt, n_t, gpu=False)
    assert np.array_equal(pulse, pulse2)
    N = 1 << 16
    freqs = fft.rfftfreq(N, dt)
    band = (freqs >= 17e3) & (freqs <= 27e3)
    alpha = C_COLD / C_SIM
    H_ref, H_cold = _H(ir_ref, pulse, 1 / dt, N), _H(ir_cold, pulse, 1 / dt, N)

    def fit(a, b):                                    # b ~ g * a
        g = np.vdot(a[band], b[band]) / np.vdot(a[band], a[band])
        return g, np.linalg.norm(b[band] - g * a[band]) / np.linalg.norm(b[band])

    g, err = fit(temperature_scale(H_ref, freqs, alpha), H_cold)
    _, err_none = fit(H_ref, H_cold)
    print(f"\nalpha={alpha:.5f} err_scaled={err:.4%} err_unscaled={err_none:.4%} "
          f"|g|={abs(g):.5f} alpha^2={alpha**2:.5f} 1/alpha={1/alpha:.5f}")
    return dict(alpha=alpha, g=g, err=err, err_none=err_none)


def test_time_scaling_matches_direct_sim(fitted):
    assert fitted["err"] < 0.005
    assert fitted["err_none"] > 10 * fitted["err"]


def test_fitted_gain_is_inverse_alpha(fitted):
    # PROCESSING.md (corrected 2026-10-14): direct = g * temperature_scale(H_343),
    # |g| = 1/alpha at equal dt (k-Wave source normalization).
    assert abs(abs(fitted["g"]) * fitted["alpha"] - 1) < 0.01
