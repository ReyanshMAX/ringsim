"""Scene -> density grid for k-Wave (GEOMETRY.md Voxelization, D-004, D-036)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

RHO_AIR = 1.2
RHO_SOLID = 1000.0              # D-004
RHO_SMOOTH_CELLS = 0.5          # D-036
CONTAINS_BATCH = 1_000_000


@dataclass
class Medium:
    rho: np.ndarray                         # float32 [Nx,Ny,Nz]
    c: float
    dx: float
    origin_m: np.ndarray                    # world coords of voxel (0,0,0)
    src_idx: list[tuple[int, int, int]]     # one per speaker hole (deduplicated)
    mic_idx: list[tuple[int, int, int]]     # one per mic hole (deduplicated); pressures averaged


def _port_points_m(port) -> np.ndarray:
    return np.asarray(port.points_mm, float) * 1e-3


def occupancy(meshes, origin: np.ndarray, dims: tuple[int, int, int], dx: float) -> np.ndarray:
    occ = np.zeros(dims, bool)
    for m in meshes:
        lo = np.maximum(np.floor((m.bounds[0] - origin) / dx).astype(int), 0)
        hi = np.minimum(np.ceil((m.bounds[1] - origin) / dx).astype(int) + 1, dims)
        if np.any(hi <= lo):
            continue
        ii = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        pts = origin + ii * dx
        inside = np.concatenate([m.contains(pts[s: s + CONTAINS_BATCH])
                                 for s in range(0, len(pts), CONTAINS_BATCH)])
        occ[tuple(ii[inside].T)] = True
    return occ


def smooth_rho(occ: np.ndarray) -> np.ndarray:
    rho = np.where(occ, RHO_SOLID, RHO_AIR).astype(np.float32)
    return ndimage.gaussian_filter(rho, RHO_SMOOTH_CELLS).astype(np.float32)


def port_voxels(port, origin: np.ndarray, dx: float, occ: np.ndarray) -> list[tuple[int, int, int]]:
    """round((pos + dx*normal - origin)/dx), deduplicated, checked against UNSMOOTHED occupancy."""
    n = np.asarray(port.normal, float)
    out: list[tuple[int, int, int]] = []
    for p in _port_points_m(port):
        idx = tuple(int(v) for v in np.round((p + dx * n - origin) / dx))
        assert all(0 <= v < s for v, s in zip(idx, occ.shape)), f"{port.name}: voxel {idx} outside grid"
        assert not occ[idx], f"{port.name}: voxel {idx} is inside a solid"
        if idx not in out:
            out.append(idx)
    return out


def build_medium(scene, dx: float, c: float, margin_m: float) -> Medium:
    from ringsim.solver.grid import fast_dims     # grid sizing lives with the solver

    pts = [_port_points_m(scene.speaker), _port_points_m(scene.mic)]
    pts += [m.bounds for m in scene.meshes]       # forearm exclusion: Phase 3 (D-031)
    allp = np.vstack(pts)
    lo = allp.min(0) - margin_m
    hi = allp.max(0) + margin_m
    dims = tuple(fast_dims(int(np.ceil((h - l) / dx)) + 1) for l, h in zip(lo, hi))
    occ = occupancy(scene.meshes, lo, dims, dx)
    rho = smooth_rho(occ)
    return Medium(rho=rho, c=float(c), dx=float(dx), origin_m=lo,
                  src_idx=port_voxels(scene.speaker, lo, dx, occ),
                  mic_idx=port_voxels(scene.mic, lo, dx, occ))
