"""One k-Wave call per snapshot: pulse at the speaker voxels, pressure at the mic voxels (docs/SOLVER.md, Runner).

k-wave-python 0.6.2 (D-035). k-Wave is imported inside `simulate`, so importing this module needs no k-Wave.
"""
from __future__ import annotations

import tempfile
import time
import warnings
from dataclasses import dataclass

import numpy as np

from ringsim.geometry.voxelize import Medium, check_face_guard
from ringsim.solver.constants import C_SIM, PML_SIZE
from ringsim.solver.grid import check_ppw, time_axis
from ringsim.solver.pulse import make_pulse


class PortCollision(Exception):
    """A speaker voxel equals a mic voxel. Ports are never shifted silently."""


@dataclass
class SimResult:
    ir: np.ndarray          # float32 [n_t] pressure at mic (pulse response), mic holes averaged
    pulse: np.ndarray       # float32 [n_t] source signal
    fs_sim: float           # = k * device fs (grid.time_axis)
    wall_s: float


def _kwave():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from kwave.data import Vector
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksensor import kSensor
        from kwave.ksource import kSource
        from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
        from kwave.options.simulation_execution_options import SimulationExecutionOptions
        from kwave.options.simulation_options import SimulationOptions
    return (Vector, kWaveGrid, kWaveMedium, kSensor, kSource, kspaceFirstOrder3D,
            SimulationExecutionOptions, SimulationOptions)


def simulate(medium: Medium, dt: float, n_t: int, gpu: bool) -> tuple[np.ndarray, np.ndarray]:
    """k-Wave call at the medium's own c (no C_SIM check). Returns (ir float32 [n_t], pulse float32 [n_t]).

    Refuses before calling k-Wave: speaker/mic voxel collision (PortCollision), dx below PPW_MIN, a solid
    within FACE_GUARD_CELLS of a grid face on medium.occ (SolidNearGridFace, D-053).
    """
    ir, pulse, _ = _simulate_timed(medium, dt, n_t, gpu)
    return ir, pulse


def _simulate_timed(medium: Medium, dt: float, n_t: int, gpu: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """simulate() plus the wall time of the kspaceFirstOrder3D call alone (SimResult.wall_s scope)."""
    shared = set(map(tuple, medium.src_idx)) & set(map(tuple, medium.mic_idx))
    if shared:
        raise PortCollision(f"speaker and mic share voxel(s) {sorted(shared)}")
    check_ppw(medium.dx, medium.c)
    check_face_guard(medium.occ)

    (Vector, kWaveGrid, kWaveMedium, kSensor, kSource, kspaceFirstOrder3D,
     SimulationExecutionOptions, SimulationOptions) = _kwave()
    # Fresh objects every call: with pml_inside=False kspaceFirstOrder3D modifies them in place (SOLVER.md).
    N = medium.rho.shape
    kgrid = kWaveGrid(Vector(list(N)), Vector([medium.dx] * 3))
    kgrid.setTime(n_t, dt)
    km = kWaveMedium(sound_speed=medium.c, density=medium.rho)      # rho already smoothed (D-036)
    src = kSource()
    src.p_mask = np.zeros(N, bool)
    for i in medium.src_idx:
        src.p_mask[tuple(i)] = True                                  # multi-hole speaker, same pulse
    pulse = make_pulse(1.0 / dt, n_t)
    src.p = np.repeat(pulse[None, :], len(medium.src_idx), axis=0)
    src.p_mode = "additive"
    sen = kSensor(np.zeros(N, bool))
    for i in medium.mic_idx:
        sen.mask[tuple(i)] = True
    sen.record = ["p"]
    with tempfile.TemporaryDirectory(prefix="ringsim_kw_") as tmp:
        opts = SimulationOptions(pml_inside=False, pml_size=PML_SIZE, data_cast="single",
                                 save_to_disk=True, data_path=tmp)
        ex = SimulationExecutionOptions(is_gpu_simulation=gpu)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)   # the one 0.6.2 deprecation (SOLVER.md)
            t0 = time.perf_counter()
            out = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=src, sensor=sen,
                                     simulation_options=opts, execution_options=ex)
            wall = time.perf_counter() - t0
    # out["p"] is [n_t, n_mic_pts] (smoke run 2026-10-14); average the mic holes
    ir = np.asarray(out["p"]).reshape(n_t, len(medium.mic_idx)).mean(1).astype(np.float32)
    return ir, pulse, wall


def medium_time_axis(medium: Medium, range_window_m: float, fs_device: int) -> tuple[float, int, int]:
    """grid.time_axis with d_direct = distance between the nearest speaker and mic voxels (what run_ir uses)."""
    d_direct = medium.dx * min(float(np.linalg.norm(np.subtract(s, m)))
                               for s in medium.src_idx for m in medium.mic_idx)
    return time_axis(medium.dx, range_window_m, fs_device, d_direct)


def run_ir(medium: Medium, range_window_m: float, fs_device: int, gpu: bool) -> SimResult:
    """Impulse response at C_SIM for one static medium (SOLVER.md)."""
    if medium.c != C_SIM:
        raise ValueError(f"run_ir needs medium.c == C_SIM = {C_SIM} (D-037), got {medium.c}")
    dt, n_t, _ = medium_time_axis(medium, range_window_m, fs_device)
    ir, pulse, wall = _simulate_timed(medium, dt, n_t, gpu)
    return SimResult(ir=ir, pulse=pulse, fs_sim=1.0 / dt, wall_s=wall)
