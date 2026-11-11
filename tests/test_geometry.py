"""Free scene + build_medium (docs/GEOMETRY.md Voxelization; D-036, D-048, D-053)."""
import numpy as np
import pytest
import trimesh

from ringsim.devices.config import load_device
from ringsim.geometry.scene import Scene, free_scene
from ringsim.geometry.voxelize import (FACE_GUARD_CELLS, RHO_AIR, RHO_SOLID, SolidNearGridFace, build_medium,
                                       save_medium_png)
from ringsim.proc.physics import C_SIM
from ringsim.solver.constants import DOMAIN_MARGIN_M


@pytest.fixture
def dev():
    return load_device("test_pair")


def test_free_scene(dev):
    s = free_scene(dev)
    assert s.meshes == [] and s.speaker.name == "spk" and s.mic.name == "mic"
    assert s.joints21 is None and s.labels is None and s.hand_scale_m is None


def test_free_medium_test_pair(dev):
    m = build_medium(free_scene(dev), 0.002, C_SIM, DOMAIN_MARGIN_M)
    assert m.rho.shape == (61, 12, 12)                  # 120 x 20 x 20 mm span, fast_dims
    assert m.rho.dtype == np.float32 and np.all(m.rho == np.float32(RHO_AIR))
    assert not m.occ.any() and m.port_covered is False
    assert m.c == C_SIM and m.dx == 0.002
    np.testing.assert_allclose(m.origin_m, [-0.010, -0.010, -0.010])
    assert m.src_idx == [(5, 5, 6)] and m.mic_idx == [(55, 5, 6)]   # port + dx*normal (+z)
    d = 0.002 * np.linalg.norm(np.subtract(m.src_idx[0], m.mic_idx[0]))
    assert d == pytest.approx(0.100)


def _box_scene(dev, center_m, size_m=(0.02, 0.02, 0.02)):
    s = free_scene(dev)
    box = trimesh.creation.box(extents=size_m)
    box.apply_translation(center_m)
    return Scene(meshes=[box], speaker=s.speaker, mic=s.mic, joints21=None, labels=None, hand_scale_m=None)


def test_box_voxelized_and_smoothed(dev):
    m = build_medium(_box_scene(dev, (0.05, 0.0, 0.03)), 0.002, C_SIM, DOMAIN_MARGIN_M)
    assert m.occ.sum() > 0
    assert m.occ.sum() == pytest.approx(10 * 10 * 10, abs=250)       # 20 mm cube at 2 mm, binary
    i = tuple(int(v) for v in np.round(np.argwhere(m.occ).mean(0)))   # centre voxel of the cube
    assert m.rho[i] == pytest.approx(RHO_SOLID, rel=0.05)
    assert 0 < m.rho.min() and m.rho.max() <= RHO_SOLID                # gaussian-smoothed (D-036)
    edge = (m.rho > RHO_AIR * 1.01) & (m.rho < RHO_SOLID * 0.99)
    assert edge.any()
    assert m.port_covered is False


def test_solid_at_face_refused_by_build_medium(dev):
    """BUILD.md: a test scene with a solid at a face is refused by build_medium (D-053)."""
    with pytest.raises(SolidNearGridFace, match="solid_near_grid_face"):
        build_medium(_box_scene(dev, (0.05, 0.0, 0.03)), 0.002, C_SIM, margin_m=0.0)


def test_solid_just_inside_guard_allowed(dev):
    margin = FACE_GUARD_CELLS * 0.002 + 0.0015                         # first solid voxel at index >= 3
    m = build_medium(_box_scene(dev, (0.05, 0.0, 0.03)), 0.002, C_SIM, margin)
    assert m.occ.any()


def test_covered_port_forced_to_air(dev):
    m = build_medium(_box_scene(dev, (0.0, 0.0, 0.0)), 0.002, C_SIM, DOMAIN_MARGIN_M)   # cube over the speaker
    assert m.port_covered is True
    for i in m.src_idx:
        assert m.occ[i] and m.rho[i] == np.float32(RHO_AIR)


def test_medium_png(dev, tmp_path):
    m = build_medium(_box_scene(dev, (0.05, 0.0, 0.03)), 0.002, C_SIM, DOMAIN_MARGIN_M)
    p = tmp_path / "x_medium.png"
    save_medium_png(m, p)
    assert p.stat().st_size > 1000
