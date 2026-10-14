# SOLVER.md — k-Wave setup, pulse, IR length, grid sizing, convergence

## Overview

Runs one 3D k-Wave simulation per frame: a band-limited pulse injected at the speaker
voxel, pressure recorded at the mic voxel. Output is the band-limited impulse response
used by `ringsim.proc.rx_from_ir` (D-002).

## Non-goals

- No absorption or nonlinearity (air absorption ≈0.5 dB/m near 22 kHz; paths ≤0.6 m).
- No elastic/solid wave propagation (D-004).
- No multi-source batching inside one k-Wave call in v1. One frame = one call.
- No full-chirp simulation (D-002).

## Constants

```python
# src/ringsim/solver/constants.py
PULSE_FC_HZ   = 22_000.0
PULSE_HALF_BW = 6_000.0          # -6 dB at 16 and 28 kHz
SIGMA_T       = np.sqrt(2*np.log(2)) / (2*np.pi*PULSE_HALF_BW)   # ≈ 31.2e-6 s
PULSE_DELAY   = 4*SIGMA_T        # ≈ 125e-6 s
CFL           = 0.3
PML_SIZE      = 10               # cells, outside the domain (pml_inside=False)
T_MARGIN_S    = 0.3e-3
DOMAIN_MARGIN_M = 0.010          # air padding around solids and ports
PPW_MIN       = 4                # points per wavelength at 28 kHz, asserted

from ringsim.proc.physics import speed_of_sound, C_SIM, C_MIN   # D-037

# Every sim runs at C_SIM = 343.0 m/s. Temperature is applied later, at read time, by
# exact time-scaling of the impulse response (PROCESSING.md `temperature_scale`, D-037).
# t_end uses C_MIN so that the slowest supported air still fits in the IR window.
```

## Pulse

```python
# src/ringsim/solver/pulse.py
def make_pulse(fs_sim: float, n_t: int) -> np.ndarray:
    t = np.arange(n_t) / fs_sim - PULSE_DELAY
    return (np.exp(-t**2 / (2*SIGMA_T**2)) * np.sin(2*np.pi*PULSE_FC_HZ*t)).astype(np.float32)
```

Any device chirp band must lie in [16, 28] kHz (DEVICES.md loader rule); outside it
`|P(f)|` is too small to deconvolve.

## Time and grid

```python
# src/ringsim/solver/grid.py
def time_axis(dx: float, range_window_m: float, fs_device: int) -> tuple[float, int, int]:
    # fs_sim is forced to an integer multiple k of the device rate so that
    # sim → device resampling is exact decimation (resample_poly(up=1, down=k)).
    k = int(np.ceil(C_SIM / (CFL * dx * fs_device)))     # guarantees CFL_actual <= CFL
    dt = 1.0 / (k * fs_device)
    t_end = PULSE_DELAY + 2*range_window_m / C_MIN + T_MARGIN_S   # coldest supported air
    return dt, int(np.ceil(t_end / dt)), k
    # dx=2mm, c=343, fs=48k, range 0.30 m → k=12, dt≈1.74 µs, ~1.27k steps, t_end≈2.2 ms

def fast_dims(n: int) -> int:
    # smallest m >= n such that (m + 2*PML_SIZE) has no prime factor > 7
```

PPW check: `c / 28e3 / dx >= PPW_MIN`, else raise. (dx = 2 mm → 6.1.)

## Runner

k-wave-python 0.6.2 API, as exercised by the 2026-10-14 T4 smoke run (D-035). In 0.6.2
`kspaceFirstOrder3D` emits a FutureWarning (deprecated in favour of
`kwave.kspaceFirstOrder.kspaceFirstOrder`). It works on the pinned version; keep it, and
silence that one warning with `warnings.filterwarnings`. Migrating is a version bump, which
needs a new DECISIONS entry and a smoke re-run. Grid sizes passed to `kWaveGrid` are
wrapped in `kwave.data.Vector`.

```python
# src/ringsim/solver/kwave_runner.py
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

def run_ir(medium: Medium, range_window_m: float, fs_device: int, gpu: bool) -> SimResult:
    assert medium.c == C_SIM                                # D-037
    dt, n_t, k = time_axis(medium.dx, range_window_m, fs_device)
    N = medium.rho.shape
    kgrid = kWaveGrid(Vector(list(N)), Vector([medium.dx]*3))
    kgrid.setTime(n_t, dt)
    km = kWaveMedium(sound_speed=medium.c, density=medium.rho)   # rho already smoothed (D-036)
    src = kSource()
    src.p_mask = np.zeros(N, bool)
    for i in medium.src_idx: src.p_mask[i] = True       # multi-hole speaker, same pulse
    pulse = make_pulse(1/dt, n_t)
    src.p = np.repeat(pulse[None, :], len(medium.src_idx), axis=0)
    src.p_mode = "additive"
    sen = kSensor(np.zeros(N, bool))
    for i in medium.mic_idx: sen.mask[i] = True
    sen.record = ["p"]
    opts = SimulationOptions(pml_inside=False, pml_size=PML_SIZE,
                             data_cast="single", save_to_disk=True)
    ex = SimulationExecutionOptions(is_gpu_simulation=gpu)
    out = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=src, sensor=sen,
                             simulation_options=opts, execution_options=ex)
    # out["p"] is [n_t, n_mic_pts] (measured, smoke run 2026-10-14); average the mic holes
    return SimResult(ir=np.asarray(out["p"]).reshape(n_t, len(medium.mic_idx)).mean(1).astype(np.float32),
                     pulse=pulse, fs_sim=1/dt, wall_s=...)
```

Speaker/mic coincidence: if any `src_idx` equals any `mic_idx` (possible on a ring at coarse dx),
raise `PortCollision`; never silently shift a port.

## Convergence study (resolves Q-006)

```
ringsim converge --job jobs/converge.yaml --dx 2.0,1.5,1.0 --ref-dx 0.75
```

The model learns from motion, so convergence is judged on **differential** profiles,
not static ones (D-033). A static noise floor is far smaller than the differential
signal and would force an unaffordable grid.

1. Base poses (phone config, bottom pair, fixed placement, forearm attached):
   - 5 random-walk poses (seed 0)
   - 5 adducted-finger poses where neighbouring fingers are 1–3 mm apart: flat hand
     with fingers together, fist, and Ring-a-Pose's "Close", "ASL0", "Thumb Up" shapes
     (hand-set MANO params in `jobs/converge_poses.yaml`, checked in)
2. For each base pose b, three small motions m ∈ {index, middle, ring}: flex that finger's
   MCP by +5°. Differential `D(b,m,dx) = P(b+m, dx) − P(b, dx)` (signed C-FMCW profiles).
3. Discretization error: `e(dx) = median_{b,m} ||D(b,m,dx) − D(b,m,1.0)|| / ||D(b,m,1.0)||`.
4. Separation: `s = median_{b, m≠m'} ||D(b,m,1.0) − D(b,m',1.0)|| / ||D(b,m,1.0)||`, i.e. how
   different two different finger motions look. Discretization error must stay well
   below the difference between motions the model has to tell apart.
5. Choose the largest dx with `e(dx) < 0.25·s`, evaluated separately on the random and
   the adducted subsets; both must pass. Record in DECISIONS.md, delete Q-006.
6. Reference check: on 2 adducted poses, compute `e(1.0)` against dx = 0.75 mm. If it is
   not below `0.25·s`, 1.0 mm is not a valid reference: stop and log it in STATUS.md as
   blocked; do not pick a dx.
7. Report static-profile error and the Phase 2 real noise floor `n` alongside, for
   information only. Report T4 wall time per frame per dx → feeds Q-007.

## Notes

- Expected T4 cost is unmeasured. Phase 1 must log `wall_s` per frame; do not
  quote throughput numbers anywhere until measured.
- Sub-cell port positions round to the nearest voxel; this error is part of what the
  convergence study measures.
- Use float32 everywhere; IR amplitudes are relative (linear system, arbitrary source gain).
- Measured on T4 (smoke run 2026-10-14, smoothed plate, CFL 0.3): 64³ grid × 1200 steps =
  2.7 s; 124³ × 1300 steps = 13.0 s. Each call carries ~2.5 s of fixed overhead (HDF5 file
  write + binary launch), so small grids are overhead-bound.
