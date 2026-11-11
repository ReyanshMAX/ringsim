"""Analytic box voxelization (Q-017 resolved, Q-020 for boxes; USER ANSWER 2026-11-11) and the plate scenes."""
import numpy as np
import pytest

from ringsim.devices.config import load_device
from ringsim.geometry.scene import Placement, Scene, active_ports, device_scene, free_scene, plate_scene
from ringsim.geometry.voxelize import (RHO_AIR, RHO_SOLID, RHO_SMOOTH_CELLS, box_fraction, build_medium,
                                       density_from_fraction)
from ringsim.proc.physics import C_SIM
from ringsim.solver.constants import DOMAIN_MARGIN_M
from scipy import ndimage

DX = 0.002


@pytest.fixture
def phone():
    with pytest.warns(UserWarning):          # TBD(Q-005) ports outside active_pair are dropped
        return load_device("iphone17pro")


def _rho64(boxes, origin, dims, dx=DX):
    """build_medium's steps 3-4 in float64 (before the float32 cast)."""
    return ndimage.gaussian_filter(density_from_fraction(box_fraction(boxes, origin, dims, dx)), RHO_SMOOTH_CELLS)


# ---------- fraction ----------

def test_fraction_whole_and_half_cells():
    origin = np.zeros(3)
    dims = (10, 10, 10)
    # box covering cells 2..5 exactly (faces on cell boundaries x_i +- dx/2)
    f = box_fraction([(np.full(3, 3.5 * DX), np.full(3, 4 * DX))], origin, dims, DX)
    assert f[2:6, 2:6, 2:6].min() == pytest.approx(1.0) and f.sum() == pytest.approx(64.0)
    assert f[1].max() == 0.0 and f[6].max() == 0.0
    # faces exactly on voxel centres: the boundary cells are half full
    f = box_fraction([(np.full(3, 4 * DX), np.full(3, 4 * DX))], origin, dims, DX)
    assert f[2, 4, 4] == pytest.approx(0.5) and f[6, 4, 4] == pytest.approx(0.5) and f[4, 4, 4] == pytest.approx(1)
    assert f.sum() == pytest.approx(64.0)                    # total volume (4 cells)^3, any alignment


def test_fraction_volume_any_offset():
    rng = np.random.default_rng(3)
    for _ in range(20):
        c = 10 * DX + rng.uniform(-DX, DX, 3)
        s = rng.uniform(2 * DX, 7 * DX, 3)
        f = box_fraction([(c, s)], np.zeros(3), (24, 24, 24), DX)
        assert f.sum() * DX ** 3 == pytest.approx(np.prod(s), rel=1e-12)
        assert f.max() <= 1.0 and f.min() >= 0.0


def test_fraction_union_not_sum():
    """Overlapping boxes count once (clamp on the body); flush boxes add exactly (plateau on the body)."""
    origin, dims = np.zeros(3), (20, 20, 20)
    a = (np.full(3, 10 * DX), np.full(3, 6 * DX))
    b = (np.full(3, 10 * DX) + 1.3 * DX, np.full(3, 6 * DX))
    fa, fb, fu = (box_fraction(x, origin, dims, DX) for x in ([a], [b], [a, b]))
    inter = np.prod(6 * DX - np.full(3, 1.3 * DX))
    assert fu.max() <= 1.0 + 1e-12
    assert fu.sum() * DX ** 3 == pytest.approx(fa.sum() * DX ** 3 + fb.sum() * DX ** 3 - inter, rel=1e-12)
    # flush: two halves of one box == the box
    whole = (np.array([10, 10, 10.3]) * DX, np.array([6, 6, 6.0]) * DX)
    lo = (np.array([10, 10, 8.8]) * DX, np.array([6, 6, 3.0]) * DX)
    hi = (np.array([10, 10, 11.8]) * DX, np.array([6, 6, 3.0]) * DX)
    np.testing.assert_allclose(box_fraction([lo, hi], origin, dims, DX),
                               box_fraction([whole], origin, dims, DX), atol=1e-12)


def test_fraction_continuous_in_face_position():
    """Q-020 for boxes: a 0.2 mm face move changes the boundary cell's fraction by 0.1 at dx 2 mm."""
    origin, dims = np.zeros(3), (20, 20, 20)
    s = np.full(3, 6 * DX)
    f0 = box_fraction([(np.full(3, 10.1 * DX), s)], origin, dims, DX)
    f1 = box_fraction([(np.full(3, 10.1 * DX) + np.array([0.0002, 0, 0]), s)], origin, dims, DX)
    d = f1 - f0
    assert d.sum() * DX ** 3 == pytest.approx(0.0, abs=1e-15)          # pure translation keeps volume
    assert np.abs(d[:, 10, 10]).max() == pytest.approx(0.1, rel=1e-9)


# ---------- Q-017 ----------

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_q017_face_on_voxel_centre_vs_nudged(axis):
    """BUILD.md Phase 2: identical voxelization with box faces on and off grid points (USER ANSWER: same rho to
    1e-9). A face exactly on a voxel centre vs nudged +-1e-12 m: max |d rho| / RHO_SOLID <= 1e-9."""
    origin, dims = np.zeros(3), (24, 24, 24)
    c = np.full(3, 12.25 * DX)
    s = np.full(3, 8.5 * DX)
    c[axis] = 12.0 * DX
    s[axis] = 8.0 * DX                              # faces at 8 dx and 16 dx: on voxel centres
    r0 = _rho64([(c, s)], origin, dims)
    for eps in (1e-12, -1e-12):
        c2 = c.copy(); c2[axis] += eps
        assert np.abs(_rho64([(c2, s)], origin, dims) - r0).max() / RHO_SOLID <= 1e-9
        s2 = s.copy(); s2[axis] += 2 * eps          # both faces move outward / inward
        assert np.abs(_rho64([(c, s2)], origin, dims) - r0).max() / RHO_SOLID <= 1e-9


def test_q017_phone_and_plate_on_and_off_grid(phone):
    """Phone body + plateau + plate. d = 80.995 mm, t = 6 mm puts both plate y faces on lattice points (voxel
    centres, D-075 origin snap); d = 81.1 mm puts them off. Each scene vs itself with every face nudged by +-1e-12 m:
    same rho to 1e-9 (float64), and the built float32 medium matches the float64 rho."""
    for d in (80.995, 81.1):
        sc = plate_scene(phone, Placement.from_job({"plate": {"size_mm": [100, 100, 6], "distance_mm": d}}))
        m = build_medium(sc, DX, C_SIM, DOMAIN_MARGIN_M)
        c, s = sc.boxes[-1]
        on = np.abs(((c[1] + s[1] / 2 - m.origin_m[1]) / DX) % 1.0) < 1e-6
        assert bool(on) is (d == 80.995)
        r0 = _rho64(sc.boxes, m.origin_m, m.rho.shape)
        for eps in (1e-12, -1e-12):
            nudged = [(cc + eps, ss) for cc, ss in sc.boxes]     # every face moves by eps
            assert np.abs(_rho64(nudged, m.origin_m, m.rho.shape) - r0).max() / RHO_SOLID <= 1e-9
        r0[_port_mask(m)] = RHO_AIR
        assert np.abs(m.rho.astype(np.float64) - r0).max() <= 1e-6 * RHO_SOLID


def _port_mask(m):
    mask = np.zeros(m.rho.shape, bool)
    for i in m.src_idx + m.mic_idx:
        mask[i] = True
    return mask


# ---------- scenes and media ----------

def test_device_scene_boxes(phone):
    sc = device_scene(phone, Placement.from_job({"rig_boxes": [{"center": [0, 87.5, 0], "size": [80, 25, 20]}]}))
    assert sc.meshes == [] and len(sc.boxes) == 3          # body, plateau, clamp
    np.testing.assert_allclose(sc.boxes[0][1], [0.07185, 0.15001, 0.00875])
    assert sc.joints21 is None and sc.labels is None and sc.hand_scale_m is None


def test_plate_scene_background_is_device_scene(phone):
    pl = Placement.from_job({"plate": None, "rig_boxes": []})
    a, b = plate_scene(phone, pl), device_scene(phone, pl)
    assert len(a.boxes) == len(b.boxes) == 2
    for (c1, s1), (c2, s2) in zip(a.boxes, b.boxes):
        np.testing.assert_array_equal(c1, c2); np.testing.assert_array_equal(s1, s2)


def test_plate_scene_plate_box(phone):
    pl = Placement.from_job({"plate": {"size_mm": [100, 120, 6], "distance_mm": 110}})
    c, s = plate_scene(phone, pl).boxes[-1]
    np.testing.assert_allclose(s * 1e3, [100, 6, 120])     # w -> x, t along the -y normal, h -> z
    np.testing.assert_allclose(c * 1e3, [-17.145, -75.005 - 110 - 3, 0], atol=1e-9)


def test_free_scene_has_no_boxes():
    assert free_scene(load_device("test_pair")).boxes == []


def test_phone_medium(phone):
    m = build_medium(device_scene(phone, Placement()), DX, C_SIM, DOMAIN_MARGIN_M)
    assert m.rho.dtype == np.float32 and m.port_covered is False
    vol = m.occ.sum()
    assert vol > 0
    # solid volume from rho (before smoothing it is exact; smoothing conserves the sum away from the faces)
    solid = (m.rho.astype(np.float64) - RHO_AIR).sum() / (RHO_SOLID - RHO_AIR) * DX ** 3
    expect = 0.07185 * 0.15001 * 0.00875 + 0.07185 * 0.05801 * 0.00255
    assert solid == pytest.approx(expect, rel=0.01)
    for i in m.src_idx + m.mic_idx:
        assert m.rho[i] == np.float32(RHO_AIR)
        assert i[1] * DX + m.origin_m[1] == pytest.approx(-0.075005 - DX, abs=DX / 2 + 1e-12)


def test_port_covered_by_fraction(phone):
    """A rig box below the bottom edge whose top face cuts the speaker port voxel: covered <=> frac >= 0.5."""
    m0 = build_medium(device_scene(phone, Placement()), DX, C_SIM, DOMAIN_MARGIN_M)
    yv = m0.origin_m[1] + m0.src_idx[0][1] * DX                  # port voxel centre (y), 1 dx below the edge
    lo_y = yv - 6 * DX
    for frac, covered in ((0.4, False), (0.5, True), (0.6, True)):
        hi_y = yv - DX / 2 + frac * DX
        box = {"center": [-17.145, (lo_y + hi_y) / 2 * 1e3, 0.0], "size": [20.0, (hi_y - lo_y) * 1e3, 6.0]}
        m = build_medium(device_scene(phone, Placement.from_job({"rig_boxes": [box]})), DX, C_SIM, DOMAIN_MARGIN_M)
        assert m.port_covered is covered, frac


def test_origin_on_device_lattice(phone):
    """D-075: every scene of a device shares one lattice, so the phone and its port voxels do not move with the plate."""
    ref = build_medium(device_scene(phone, Placement()), DX, C_SIM, DOMAIN_MARGIN_M)
    np.testing.assert_allclose(ref.origin_m / DX, np.round(ref.origin_m / DX), atol=1e-9)
    for d in (80.0, 80.2, 80.4, 81.8):
        m = build_medium(plate_scene(phone, Placement.from_job({"plate": {"size_mm": [100, 100, 6], "distance_mm": d}})),
                         DX, C_SIM, DOMAIN_MARGIN_M)
        np.testing.assert_allclose(m.origin_m / DX, np.round(m.origin_m / DX), atol=1e-9)
        off = np.round((ref.origin_m - m.origin_m) / DX).astype(int)
        assert [tuple(np.add(i, off)) for i in ref.src_idx] == m.src_idx
        assert [tuple(np.add(i, off)) for i in ref.mic_idx] == m.mic_idx
        # phone region identical voxel for voxel (the plate is > 3 cells away in y)
        n = [min(a, b - o) for a, b, o in zip(ref.occ.shape, m.occ.shape, off)]
        a = ref.rho[:n[0], 10:n[1], :n[2]]
        b = m.rho[off[0]:off[0] + n[0], off[1] + 10:off[1] + n[1], off[2]:off[2] + n[2]]
        np.testing.assert_array_equal(a, b)


def test_placement_from_job_refusals():
    assert Placement.from_job(None) == Placement()
    with pytest.raises(ValueError, match="unknown placement keys"):
        Placement.from_job({"plate": None, "tilt": 3})
    with pytest.raises(ValueError, match="plate needs"):
        Placement.from_job({"plate": {"size_mm": [100, 100, 5]}})
    with pytest.raises(ValueError, match="rig box needs"):
        Placement.from_job({"rig_boxes": [{"center": [0, 0, 0]}]})
