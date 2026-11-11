"""D-037 check on CPU k-Wave: a 343 m/s sim time-scaled to 337 m/s vs a direct 337 m/s sim.

Fixture as PROCESSING.md "Required tests": 64^3 grid, dx 2 mm, grid origin at 0, c = grid
centre + 1 mm = 65 mm; plate 60x60x4 mm centred on c, bars 8x8x60 mm at
c + (-30, 20, -10) mm and 8x60x8 mm at c + (25, -5, -25) mm, all rho = 1000 then smoothed (D-036);
speaker voxel (32, 30, 12), mic voxel (32, 33, 12). Both sims use the same dt and n_t, and the
same occupancy (built once), so the comparison does not depend on how face points voxelize.
Open (posted to the lead 2026-11-11): the second bar's faces fall on grid points (x 86/94, y 30/90,
z 36/44 mm), contrary to the spec's "no face on a grid point".
run_ir asserts c == C_SIM, so both go through solver.kwave_runner.simulate (ARCHITECTURE, 22af4bc).
k-Wave runs only through the Kaggle runner (D-052, D-054); slow, deselected by default.
"""
import numpy as np
import pytest
from scipy import fft

from ringsim.proc import C_SIM, temperature_scale

pytestmark = pytest.mark.slow

DX = 0.002
DIMS = (64, 64, 64)
C_COLD = 337.0
SPK = (32, 30, 12)
MIC = (32, 33, 12)
FS_DEVICE = 48_000              # test_pair rate: fs_sim = 12 * 48 kHz = 576 kHz
RANGE_M = 84 * C_SIM / (2 * FS_DEVICE)   # test_pair range window (n_range_bins 84), 0.30 m
BAND_HZ = (17e3, 27e3)


def fixture_occupancy():
    import trimesh

    from ringsim.geometry.voxelize import occupancy
    origin = np.zeros(3)
    ctr = np.array(DIMS) * DX / 2 + 0.001
    boxes = [
        trimesh.creation.box(extents=[0.060, 0.060, 0.004]).apply_translation(ctr),
        trimesh.creation.box(extents=[0.008, 0.008, 0.060]).apply_translation(ctr + [-0.030, 0.020, -0.010]),
        trimesh.creation.box(extents=[0.008, 0.060, 0.008]).apply_translation(ctr + [0.025, -0.005, -0.025]),
    ]
    occ = occupancy(boxes, origin, DIMS, DX)
    assert occ.any() and not occ[SPK] and not occ[MIC]
    return occ


def fixture_medium(c: float, occ: np.ndarray):
    from scipy import ndimage

    from ringsim.geometry.voxelize import RHO_AIR, RHO_SMOOTH_CELLS, RHO_SOLID, Medium
    rho = ndimage.gaussian_filter(np.where(occ, RHO_SOLID, RHO_AIR).astype(np.float32),
                                  RHO_SMOOTH_CELLS).astype(np.float32)
    rho[SPK] = rho[MIC] = RHO_AIR
    return Medium(rho=rho, c=float(c), dx=DX, origin_m=np.zeros(3), src_idx=[SPK], mic_idx=[MIC],
                  occ=occ, port_covered=False)


def transfer(ir, pulse, N):
    R, P = fft.rfft(ir, N), fft.rfft(pulse, N)
    P2 = np.abs(P) ** 2
    return R * np.conj(P) / (P2 + 1e-3 * P2.max())


@pytest.fixture(scope="module")
def fitted():
    from ringsim.solver.grid import time_axis
    from ringsim.solver.kwave_runner import simulate

    d_direct = DX * float(np.linalg.norm(np.subtract(SPK, MIC)))
    dt, n_t, _ = time_axis(DX, RANGE_M, FS_DEVICE, d_direct)        # equal dt and n_t for both
    occ = fixture_occupancy()
    ir_ref, pulse = simulate(fixture_medium(C_SIM, occ), dt, n_t, gpu=False)
    ir_cold, pulse2 = simulate(fixture_medium(C_COLD, occ), dt, n_t, gpu=False)
    assert np.array_equal(pulse, pulse2)
    N = 1 << 16
    freqs = fft.rfftfreq(N, dt)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    alpha = C_COLD / C_SIM
    H_ref, H_cold = transfer(ir_ref, pulse, N), transfer(ir_cold, pulse, N)

    def fit(a, b):                                    # b ~ g * a over the band
        g = np.vdot(a[band], b[band]) / np.vdot(a[band], a[band])
        return g, np.linalg.norm(b[band] - g * a[band]) / np.linalg.norm(b[band])

    g, err = fit(temperature_scale(H_ref, freqs, alpha), H_cold)
    _, err_none = fit(H_ref, H_cold)
    print(f"\nalpha={alpha:.5f} n_t={n_t} fs_sim={1 / dt:.0f} err_scaled={err:.4%} "
          f"err_unscaled={err_none:.4%} |g|={abs(g):.5f} 1/alpha={1 / alpha:.5f}")
    return dict(alpha=alpha, g=g, err=err, err_none=err_none)


def test_scaled_matches_direct(fitted):
    assert fitted["err"] < 0.005


def test_gain_is_one_over_alpha(fitted):
    assert abs(abs(fitted["g"]) * fitted["alpha"] - 1) < 0.01

