"""Scene -> density grid for k-Wave (docs/GEOMETRY.md Voxelization; D-004, D-036, D-048, D-053).

Axis-aligned boxes (Scene.boxes: phone body, plateau, rig boxes, plate) are voxelized analytically as the
fraction of each cell inside their union (USER ANSWER 2026-11-11: Q-017 resolved, Q-020 for boxes). Meshes
(hands, Phase 3) are still binary (generalized winding number >= 0.5): allowed for static scenes only until Q-020
covers meshes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ringsim.solver.grid import fast_dims

RHO_AIR = 1.2
RHO_SOLID = 1000.0              # D-004
RHO_SMOOTH_CELLS = 0.5          # D-036
FACE_GUARD_CELLS = 3            # D-053
INSIDE_BATCH = 1_000_000        # query points per winding-number call (GEOMETRY.md Notes)
PORT_COVERED_FRAC = 0.5         # port voxel covered <=> its solid fraction >= this (USER ANSWER 2026-11-11)


def density_from_fraction(frac: np.ndarray) -> np.ndarray:
    """rho = RHO_AIR + frac*(RHO_SOLID - RHO_AIR), float64 (USER ANSWER 2026-11-11: linear in rho)."""
    return RHO_AIR + np.asarray(frac, np.float64) * (RHO_SOLID - RHO_AIR)


class SolidNearGridFace(ValueError):
    """A solid voxel lies within FACE_GUARD_CELLS of a grid face (D-053). Sequence failure reason:
    `solid_near_grid_face`."""
    reason = "solid_near_grid_face"


@dataclass
class Medium:
    rho: np.ndarray                         # float32 [Nx,Ny,Nz], smoothed (D-036), port voxels = RHO_AIR
    c: float
    dx: float
    origin_m: np.ndarray                    # world coords of voxel (0,0,0)
    src_idx: list[tuple[int, int, int]]     # one per speaker hole (deduplicated)
    mic_idx: list[tuple[int, int, int]]     # one per mic hole (deduplicated); recorded pressures are averaged
    occ: np.ndarray                         # bool [Nx,Ny,Nz] frac > 0, unsmoothed (run_ir re-checks the guard on it)
    port_covered: bool                      # frac(port voxel) >= 0.5 before being forced to air (D-048)


def check_face_guard(occ: np.ndarray) -> None:
    """Raise SolidNearGridFace if any solid voxel lies within FACE_GUARD_CELLS of any of the six faces."""
    g = FACE_GUARD_CELLS
    if not occ.any():
        return
    for ax, n in enumerate(occ.shape):
        idx = np.nonzero(occ.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        if idx.min() < g or idx.max() > n - 1 - g:
            raise SolidNearGridFace(f"solid_near_grid_face: solid voxel within {g} cells of a face on axis {ax} "
                                    f"(occupied {idx.min()}..{idx.max()} of {n})")


def occupancy(meshes, origin: np.ndarray, dims: tuple[int, int, int], dx: float) -> np.ndarray:
    """OR over meshes of (fast winding number >= 0.5) at the voxel centres inside each mesh's AABB (D-048)."""
    import igl
    occ = np.zeros(dims, bool)
    for m in meshes:
        lo = np.maximum(np.floor((m.bounds[0] - origin) / dx).astype(int), 0)
        hi = np.minimum(np.ceil((m.bounds[1] - origin) / dx).astype(int) + 1, dims)
        if np.any(hi <= lo):
            continue
        ii = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        q = origin + ii * dx
        V = np.ascontiguousarray(m.vertices, dtype=np.float64)
        F = np.ascontiguousarray(m.faces, dtype=np.int64)
        w = np.concatenate([igl.fast_winding_number(V, F, np.ascontiguousarray(q[s:s + INSIDE_BATCH]))
                            for s in range(0, len(q), INSIDE_BATCH)])
        occ[tuple(ii[w >= 0.5].T)] = True
    return occ


def _axis_overlap(lo: float, hi: float, origin: float, n: int, dx: float) -> np.ndarray:
    """Fraction of each cell [x_i - dx/2, x_i + dx/2] (x_i = origin + i*dx) covered by [lo, hi], float64 [n]."""
    c = origin + np.arange(n) * dx
    return np.clip(np.minimum(c + dx / 2, hi) - np.maximum(c - dx / 2, lo), 0.0, dx) / dx


def box_fraction(boxes, origin: np.ndarray, dims: tuple[int, int, int], dx: float) -> np.ndarray:
    """float64 [Nx,Ny,Nz]: volume of (union of axis-aligned boxes) inside each cell / cell volume.

    Exact union: the boxes are cut on all their edge coordinates into disjoint pieces; a piece is solid if its
    midpoint lies in any box; each solid piece adds its separable per-axis overlap. So a clamp overlapping the body
    or a plateau flush on the body is counted once, and frac is continuous in every face position (no tie at a face
    on a voxel centre, Q-017)."""
    frac = np.zeros(dims, np.float64)
    if not boxes:
        return frac
    lohi = [(np.asarray(c, float) - np.asarray(s, float) / 2, np.asarray(c, float) + np.asarray(s, float) / 2)
            for c, s in boxes]
    cuts = [np.unique(np.concatenate([[lo[a], hi[a]] for lo, hi in lohi])) for a in range(3)]
    for i in range(len(cuts[0]) - 1):
        for j in range(len(cuts[1]) - 1):
            for k in range(len(cuts[2]) - 1):
                plo = np.array([cuts[0][i], cuts[1][j], cuts[2][k]])
                phi = np.array([cuts[0][i + 1], cuts[1][j + 1], cuts[2][k + 1]])
                mid = (plo + phi) / 2
                if not any(np.all(mid > lo) and np.all(mid < hi) for lo, hi in lohi):
                    continue
                f = [_axis_overlap(plo[a], phi[a], origin[a], dims[a], dx) for a in range(3)]
                nz = [np.nonzero(v)[0] for v in f]
                if any(len(z) == 0 for z in nz):
                    continue
                sl = tuple(slice(z[0], z[-1] + 1) for z in nz)
                frac[sl] += np.einsum("i,j,k->ijk", f[0][sl[0]], f[1][sl[1]], f[2][sl[2]])
    return np.minimum(frac, 1.0)


def _port_voxels(port, origin: np.ndarray, dx: float, shape) -> list[tuple[int, int, int]]:
    """round((pos + dx*normal - origin)/dx) per hole; a voxel shared by two holes is used once (DEVICES.md)."""
    n = np.asarray(port.normal, float)
    out: list[tuple[int, int, int]] = []
    for p in np.asarray(port.points_mm, float) * 1e-3:
        idx = tuple(int(v) for v in np.round((p + dx * n - origin) / dx))
        if not all(0 <= v < s for v, s in zip(idx, shape)):
            raise ValueError(f"{port.name}: port voxel {idx} outside the grid {tuple(shape)}")
        if idx not in out:
            out.append(idx)
    return out


def build_medium(scene, dx: float, c: float, margin_m: float) -> Medium:
    """GEOMETRY.md Voxelization steps 1-5. Phase 1 scenes have no forearm, so step 1's forearm exclusion
    (D-031) does not apply yet."""
    # 1. bbox = mesh bounds + port positions, padded by margin_m
    pts = [np.asarray(scene.speaker.points_mm, float) * 1e-3, np.asarray(scene.mic.points_mm, float) * 1e-3]
    pts += [np.asarray(m.bounds, float) for m in scene.meshes]
    boxes = list(getattr(scene, "boxes", []))
    pts += [np.vstack([np.asarray(c, float) - np.asarray(s, float) / 2, np.asarray(c, float) + np.asarray(s, float) / 2])
            for c, s in boxes]
    allp = np.vstack(pts)
    # origin snapped to the device-frame lattice (D-075): every scene of a device shares one lattice, so the body
    # and the port voxels are identical across plate positions and in the device-only reference; the air margin
    # becomes margin_m .. margin_m + dx. The 1e-9 absorbs float error when (min - margin)/dx is an integer.
    origin = np.floor((allp.min(0) - margin_m) / dx + 1e-9) * dx
    hi = allp.max(0) + margin_m
    # 2. grid dims via fast_dims; voxel centres x_i = origin + i*dx
    dims = tuple(fast_dims(int(np.ceil((h - o) / dx - 1e-9)) + 1) for o, h in zip(origin, hi))
    # 3. occupancy: analytic box fractions, OR'd (max) with binary mesh occupancy
    frac = box_fraction(boxes, origin, dims, dx)
    if scene.meshes:
        frac = np.maximum(frac, occupancy(scene.meshes, origin, dims, dx))
    occ = frac > 0
    # 3b. face guard (D-053)
    check_face_guard(occ)
    # 4. density linear in the fraction, then D-036 smoothing
    rho = ndimage.gaussian_filter(density_from_fraction(frac), RHO_SMOOTH_CELLS).astype(np.float32)
    # 5. port voxels: covered <=> frac >= 0.5 before forcing to air (D-048)
    src = _port_voxels(scene.speaker, origin, dx, dims)
    mic = _port_voxels(scene.mic, origin, dx, dims)
    covered = any(bool(frac[i] >= PORT_COVERED_FRAC) for i in src + mic)
    for i in src + mic:
        rho[i] = RHO_AIR
    return Medium(rho=rho, c=float(c), dx=float(dx), origin_m=origin, src_idx=src, mic_idx=mic,
                  occ=occ, port_covered=covered)


def save_medium_png(medium: Medium, path) -> None:
    """3-slice PNG of rho (xy / xz / yz through the first mic voxel), GEOMETRY.md Notes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i, j, k = medium.mic_idx[0]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im = None
    for a, sl, title, (xl, yl) in ((ax[0], medium.rho[:, :, k].T, f"xy @ z={k}", ("x", "y")),
                                   (ax[1], medium.rho[:, j, :].T, f"xz @ y={j}", ("x", "z")),
                                   (ax[2], medium.rho[i, :, :].T, f"yz @ x={i}", ("y", "z"))):
        im = a.imshow(sl, origin="lower", cmap="gray_r", vmin=0, vmax=RHO_SOLID)
        a.set_title(title); a.set_xlabel(f"{xl} (cells)"); a.set_ylabel(f"{yl} (cells)")
    for (si, sj, sk) in medium.src_idx:
        ax[0].plot(si, sj, "r^"); ax[1].plot(si, sk, "r^"); ax[2].plot(sj, sk, "r^")
    for (mi, mj, mk) in medium.mic_idx:
        ax[0].plot(mi, mj, "bo"); ax[1].plot(mi, mk, "bo"); ax[2].plot(mj, mk, "bo")
    fig.colorbar(im, ax=ax, shrink=0.7, label="rho (kg/m^3)")
    fig.suptitle(f"medium {medium.rho.shape}, dx {medium.dx * 1e3:.2f} mm (red: speaker, blue: mic)")
    fig.savefig(path, dpi=100)
    plt.close(fig)
