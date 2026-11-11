"""Phase 2 solid builders: phone BoxBody incl. plateau, plate, rig boxes (docs/GEOMETRY.md Placement, DEVICES.md)."""
import numpy as np
import pytest

from ringsim.devices.config import BoxBody, Port, load_device
from ringsim.geometry.scene import active_ports, body_meshes, box_mesh, plate_mesh, rig_box_meshes


@pytest.fixture
def phone():
    with pytest.warns(UserWarning):          # bottom_mic1 / front_mic are TBD(Q-005), outside active_pair: dropped
        return load_device("iphone17pro")


def test_box_mesh_bounds_and_watertight():
    m = box_mesh((0.01, -0.02, 0.03), (0.004, 0.006, 0.008))
    assert m.is_watertight and m.volume == pytest.approx(0.004 * 0.006 * 0.008)
    np.testing.assert_allclose(m.bounds, [[0.008, -0.023, 0.026], [0.012, -0.017, 0.034]], atol=1e-12)


@pytest.mark.parametrize("size", [(0, 1, 1), (1, -1, 1), (1, 1)])
def test_box_mesh_refuses_bad_size(size):
    with pytest.raises(ValueError):
        box_mesh((0, 0, 0), size)


def test_phone_body_and_plateau(phone):
    body, plateau = body_meshes(phone.body)
    assert body.is_watertight and plateau.is_watertight
    np.testing.assert_allclose(body.bounds * 1e3, [[-35.925, -75.005, -4.375], [35.925, 75.005, 4.375]], atol=1e-9)
    # plateau on the rear face (z < 0), flush with it, reaching the top edge (DEVICES.md iphone17pro.yaml)
    np.testing.assert_allclose(plateau.bounds * 1e3, [[-35.925, 16.995, -6.925], [35.925, 75.005, -4.375]],
                               atol=1e-9)


def test_body_without_plateau():
    b = BoxBody(kind="box", size_mm=(10.0, 20.0, 5.0), plateau_size_mm=None, plateau_center_mm=None)
    (m,) = body_meshes(b)
    np.testing.assert_allclose(m.extents, [0.010, 0.020, 0.005])


def test_body_plateau_size_without_center_refused():
    b = BoxBody(kind="box", size_mm=(10.0, 20.0, 5.0), plateau_size_mm=(1.0, 1.0, 1.0), plateau_center_mm=None)
    with pytest.raises(ValueError, match="plateau_center_mm"):
        body_meshes(b)


def test_active_ports_on_body_surface(phone):
    """The active speaker and mic holes sit on the body box's -y face (DEVICES.md bottom edge)."""
    spk, mic = active_ports(phone)
    body = body_meshes(phone.body)[0]
    for p in (spk, mic):
        y = np.asarray(p.points_mm, float)[:, 1] * 1e-3
        np.testing.assert_allclose(y, body.bounds[0, 1], atol=1e-12)


@pytest.mark.parametrize("d", [80.0, 110.0, 140.0, 170.0, 200.0])
def test_plate_on_speaker_axis(phone, d):
    spk, _ = active_ports(phone)
    t = 5.0
    m = plate_mesh(spk, {"size_mm": [100, 100, t], "distance_mm": d})
    assert m.is_watertight
    c = np.asarray(spk.points_mm, float).mean(0)            # (-17.145, -75.005, 0)
    lo, hi = m.bounds * 1e3
    assert hi[1] == pytest.approx(c[1] - d)                  # near face at d from the port centroid, along -y
    assert lo[1] == pytest.approx(c[1] - d - t)              # thickness away from the port
    np.testing.assert_allclose([(lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2], [c[0], c[2]], atol=1e-9)
    np.testing.assert_allclose([hi[0] - lo[0], hi[2] - lo[2]], [100, 100], atol=1e-9)


def test_plate_refuses_open_cases():
    spk = Port(name="s", kind="speaker", points_mm=((0.0, 0.0, 0.0),), normal=(0.0, 0.6, 0.8))
    with pytest.raises(NotImplementedError, match="S2-6"):
        plate_mesh(spk, {"size_mm": [100, 100, 5], "distance_mm": 80})
    spk = Port(name="s", kind="speaker", points_mm=((0.0, 0.0, 0.0),), normal=(0.0, -1.0, 0.0))
    with pytest.raises(NotImplementedError, match="S2-6"):
        plate_mesh(spk, {"size_mm": [100, 80, 5], "distance_mm": 80})
    with pytest.raises(ValueError):
        plate_mesh(spk, {"size_mm": [100, 100, 0], "distance_mm": 80})


def test_rig_box_meshes_mm_to_m():
    (m,) = rig_box_meshes([{"center": [0, 87.5, 0], "size": [80, 25, 20]}])
    np.testing.assert_allclose(m.bounds, [[-0.040, 0.075, -0.010], [0.040, 0.100, 0.010]], atol=1e-12)
    assert rig_box_meshes([]) == []
