import numpy as np
import trimesh

from ringsim.devices.config import load_device
from ringsim.geometry.scene import free_scene
from ringsim.geometry.voxelize import RHO_AIR, RHO_SOLID, build_medium, occupancy, smooth_rho


def test_free_medium():
    m = build_medium(free_scene(load_device("test_pair")), 0.002, 343.0, 0.010)
    assert m.rho.dtype == np.float32 and np.allclose(m.rho, RHO_AIR)
    (s,), (q,) = m.src_idx, m.mic_idx
    assert q[0] - s[0] == 50 and s[1:] == q[1:]        # 100 mm at 2 mm
    assert s[2] == 6                                   # one dx along +z normal from z=0 (origin -10 mm)


def test_occupancy_and_smoothing():
    box = trimesh.creation.box(extents=[0.02, 0.02, 0.02]).apply_translation([0.031, 0.031, 0.031])  # faces between grid points
    occ = occupancy([box], np.zeros(3), (30, 30, 30), 0.002)
    assert occ[15, 15, 15] and not occ[0, 0, 0]
    assert occ.sum() == 1000                            # 10^3 cells
    rho = smooth_rho(occ)
    assert abs(rho[15, 15, 15] - RHO_SOLID) < 1 and abs(rho[0, 0, 0] - RHO_AIR) < 1e-3
    edge = rho[occ ^ np.roll(occ, 1, 0)]
    assert edge.min() > RHO_AIR and edge.max() < RHO_SOLID
