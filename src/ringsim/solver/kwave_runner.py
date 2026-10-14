"""One k-Wave run per frame: pulse at speaker voxels, pressure at mic voxels (SOLVER.md)."""
from __future__ import annotations

import tempfile
import time
import warnings
from dataclasses import dataclass

import numpy as np

from ringsim.geometry.voxelize import Medium
from ringsim.solver.constants import C_SIM, PML_SIZE
from ringsim.solver.grid import check_ppw, time_axis
from ringsim.solver.pulse import make_pulse


class PortCollision(Exception):
    """A speaker voxel coincides with a mic voxel."""


@dataclass
class SimResult:
    ir: np.ndarray          # float32 [n_t] pressure at mic (pulse response)
    pulse: np.ndarray       # float32 [n_t] source signal
    fs_sim: float           # = k * device fs
    wall_s: float


def kwave_version() -> str:
    from importlib.metadata import version
    return version("k-wave-python")


def _simulate(medium: Medium, dt: float, n_t: int, gpu: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Raw k-Wave call at the medium's own c (no C_SIM check). Returns (ir, pulse, wall_s)."""
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

    if set(medium.src_idx) & set(medium.mic_idx):
        raise PortCollision(f"speaker and mic share voxel(s) {set(medium.src_idx) & set(medium.mic_idx)}")
    N = medium.rho.shape
    kgrid = kWaveGrid(Vector(list(N)), Vector([medium.dx] * 3))
    kgrid.setTime(n_t, dt)
    km = kWaveMedium(sound_speed=medium.c, density=medium.rho)    # rho already smoothed (D-036)
    src = kSource()
    src.p_mask = np.zeros(N, bool)
    for i in medium.src_idx:
        src.p_mask[i] = True                                       # multi-hole speaker, same pulse
    pulse = make_pulse(1 / dt, n_t)
    src.p = np.repeat(pulse[None, :], len(medium.src_idx), axis=0)
    src.p_mode = "additive"
    sen = kSensor(np.zeros(N, bool))
    for i in medium.mic_idx:
        sen.mask[i] = True
    sen.record = ["p"]
    with tempfile.TemporaryDirectory(prefix="ringsim_kw_") as tmp:
        opts = SimulationOptions(pml_inside=False, pml_size=PML_SIZE, data_cast="single",
                                 save_to_disk=True, data_path=tmp)
        ex = SimulationExecutionOptions(is_gpu_simulation=gpu, show_sim_log=False)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            out = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=src, sensor=sen,
                                     simulation_options=opts, execution_options=ex)
        wall = time.perf_counter() - t0
    # out["p"] is [n_t, n_mic_pts] (smoke run 2026-10-14); average the mic holes
    p = np.asarray(out["p"]).reshape(n_t, len(medium.mic_idx)).mean(1).astype(np.float32)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("k-Wave output contains NaN/Inf")
    return p, pulse, wall


def run_ir(medium: Medium, range_window_m: float, fs_device: int, gpu: bool) -> SimResult:
    assert medium.c == C_SIM, medium.c                            # D-037
    check_ppw(medium.dx)
    dt, n_t, k = time_axis(medium.dx, range_window_m, fs_device)
    ir, pulse, wall = _simulate(medium, dt, n_t, gpu)
    return SimResult(ir=ir, pulse=pulse, fs_sim=1 / dt, wall_s=wall)
