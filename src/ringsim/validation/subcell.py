"""Q-020 sub-cell motion check on plate shards (BUILD.md Phase 2; D-072, D-076, D-082).

Per shard (one static plate snapshot + its device-only `ref_ir`):
- echo = ir[0] - ref_ir (the plate's echo; the phone body's own echoes cancel because both scenes share one lattice,
  D-075); direct = ref_ir.
- estimator: cross-correlate with the shard's pulse, take |hilbert| of the correlation (the envelope), the largest
  envelope sample inside the gate, parabolic interpolation through it and its two neighbours -> lag in seconds.
- gates: echo 2(d - 3dx)/c .. 2(d + 3dx)/c (D-076). Direct: (min speaker-mic voxel distance - 3dx)/c .. (max + 3dx)/c,
  the same ±3-cell allowance about the range of direct paths (D-076 names no direct gate; seat 2's choice, lower edge
  widened here because the Hilbert transform now covers negative lags).
- expected lags (D-082): closed-form ideal signals through the SAME estimator. Direct = sum_i m(t - r_i/c)/r_i, echo =
  sum_i m(t - R_i/c)/R_i, with m the ideal monopole response of the pulse (D-057, `freefield.monopole_ir`), r_i the
  speaker-voxel -> mic-voxel distances, R_i the image-source paths speaker voxel -> near-face plane -> mic voxel.
  Mic holes: the IR is the mean over mic voxels (SOLVER.md), so the ideal signals are averaged over them too.
- offset_mm = c/2 * [(meas_echo - meas_direct) - (ideal_echo - ideal_direct)]; offset_bins = offset_mm / bin, bin =
  c / (2 fs_device). Removing the direct path takes out the voxel offset of the ports and the source's carrier phase,
  so the offset is the effective-surface shift of the voxelized plate (D-036 smoothing, D-067 fractions).

Report over the 10 steps: delay = meas_echo - sys_lag (sys_lag = meas_direct - ideal_direct); monotonic increase
with d; slope of delay vs d within 10 % of 2/c; max |offset_bins| <= 0.25. D-036 re-measure: echo/direct envelope
peak ratio of the 100 mm plate and of the 300 mm rigid reference, the image-source ratio (ideal echo / ideal direct
peak) for the rigid reference, and rigid / image source.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import yaml
from scipy.signal import correlate, hilbert

from ringsim.validation.freefield import monopole_ir

RESIDUAL_LAG_SAMPLES = 1e-3         # tests/test_validation_subcell.py bound; measured 1e-4 sim samples (docstring)
GATE_CELLS = 3                     # D-076
SLOPE_TOL = 0.10                   # BUILD.md Phase 2: slope within 10 % of 2/c
OFFSET_TOL_BINS = 0.25             # BUILD.md Phase 2: within 0.25 bin of the rigid delay


def envelope_lag(sig: np.ndarray, pulse: np.ndarray, fs: float, gate_s: tuple[float, float]) -> tuple[float, float]:
    """(lag_s, peak) of |hilbert(xcorr(sig, pulse))| inside gate_s, parabolic sub-sample peak (D-076).

    The Hilbert transform is taken over the FULL cross-correlation, negative lags included: cutting it at lag 0
    distorts the envelope near 0, where the 2-8 mm direct paths sit (6-23 us; seat 2's draft read a 5 us delay as
    15.6 us). The gate is applied after. Gate lags may be negative.

    Residual bias of the three-point parabola on this envelope: measured on ideal monopole signals at fs_sim = c/(CFL*dx)
    = 571.7 kHz (dx 2 mm), delays 20, 200 and 466 us, 21 sub-sample phases each: |lag - tau| <= 1e-4 sim samples
    (0.0002 us). It is not zero in general, so `test_envelope_lag_residual` bounds it at RESIDUAL_LAG_SAMPLES; the Q-020
    criterion (0.25 bin of 48 kHz = ~3 sim samples) is far above it. Seat 2's draft measured 0.2 sample: that was the
    lag >= 0 cut of the Hilbert transform, not the parabola. Real k-Wave IRs add whatever the solver adds (noise,
    dispersion); that part is what the report measures.
    """
    sig = np.asarray(sig, np.float64)
    pulse = np.asarray(pulse, np.float64)
    xc = correlate(sig, pulse, mode="full", method="fft")           # lags -(len(pulse)-1) .. len(sig)-1
    pad = len(xc)                                                   # zero-padded: scipy's hilbert is circular (as in
    env = np.abs(hilbert(np.concatenate([np.zeros(pad), xc, np.zeros(pad)])))[pad:-pad]   # proc.align_direct_path)
    off = len(pulse) - 1                                            # array index of lag 0
    lo = int(np.ceil(gate_s[0] * fs)) + off
    hi = int(np.floor(gate_s[1] * fs)) + off
    lo, hi = max(lo, 1), min(hi, len(env) - 2)
    if hi < lo:
        raise ValueError(f"gate {gate_s} s holds no lag at fs {fs}")
    k = lo + int(np.argmax(env[lo:hi + 1]))
    if k == lo or k == hi:
        raise ValueError(f"envelope peak at the gate edge (lag {(k - off) / fs:.3e} s, gate {gate_s}): "
                         f"no peak inside the gate")
    a, b, c = env[k - 1], env[k], env[k + 1]
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if den != 0 else 0.0
    return (k - off + delta) / fs, float(b - 0.25 * (a - c) * delta)


def _geometry(f: h5py.File):
    """Rebuild the shard's medium from its device_yaml + job_yaml: port voxels and the plate's near-face plane."""
    from ringsim.devices.config import load_device_yaml
    from ringsim.geometry.scene import Placement, active_ports, plate_scene
    from ringsim.geometry.voxelize import build_medium
    from ringsim.solver.constants import DOMAIN_MARGIN_M

    dev = load_device_yaml(str(f.attrs["device_yaml"]))
    job = yaml.safe_load(str(f.attrs["job_yaml"]))
    pl = Placement.from_job(job["placement"])
    if pl.plate is None:
        raise ValueError("subcell check needs a plate shard (placement.plate is null)")
    dx = float(f.attrs["dx_m"])
    m = build_medium(plate_scene(dev, pl), dx, float(f.attrs["c_sim"]), DOMAIN_MARGIN_M)
    spk, _ = active_ports(dev)
    n = np.asarray(spk.normal, float)
    near = (np.asarray(spk.points_mm, float).mean(0) * 1e-3) + n * pl.plate["distance_mm"] * 1e-3   # a point on it
    src = [m.origin_m + np.asarray(i) * dx for i in m.src_idx]
    mic = [m.origin_m + np.asarray(i) * dx for i in m.mic_idx]
    return dev, pl, dx, src, mic, near, n


def _ideal(pulse, fs, dists_per_mic: list[list[float]], c: float) -> np.ndarray:
    """Mean over mic voxels of sum_i monopole(t - D_i/c) / D_i."""
    out = np.zeros(len(pulse))
    for ds in dists_per_mic:
        for d in ds:
            out += monopole_ir(pulse, fs, d / c).astype(np.float64) / d
    return out / len(dists_per_mic)


def echo_delay(path: str, seq: str = "plate") -> dict:
    """Q-020 numbers for one plate shard (module docstring)."""
    with h5py.File(path, "r") as f:
        g = f["seq"][seq]
        if bool(g.attrs["failed"]):
            raise ValueError(f"{path}: sequence {seq} failed: {g.attrs['failed_reason']}")
        ir = np.asarray(g["ir"][0], np.float64)
        ref = np.asarray(g["ref_ir"], np.float64)
        pulse = np.asarray(f["pulse"], np.float64)
        fs = float(f.attrs["k_decim"]) * float(f.attrs["fs_device"])
        c = float(f.attrs["c_sim"])
        fs_dev = float(f.attrs["fs_device"])
        dev, pl, dx, src, mic, near, n = _geometry(f)
    d = pl.plate["distance_mm"] * 1e-3
    r = [[float(np.linalg.norm(s - mm)) for s in src] for mm in mic]
    # image of each speaker voxel across the near-face plane (normal n through `near`)
    img = [s - 2 * np.dot(s - near, n) * n for s in src]
    R = [[float(np.linalg.norm(s - mm)) for s in img] for mm in mic]
    gate_e = (2 * (d - GATE_CELLS * dx) / c, 2 * (d + GATE_CELLS * dx) / c)
    gate_d = ((min(min(x) for x in r) - GATE_CELLS * dx) / c, (max(max(x) for x in r) + GATE_CELLS * dx) / c)
    me, ae = envelope_lag(ir - ref, pulse, fs, gate_e)
    md, ad = envelope_lag(ref, pulse, fs, gate_d)
    ie, aie = envelope_lag(_ideal(pulse, fs, R, c), pulse, fs, gate_e)
    idd, aid = envelope_lag(_ideal(pulse, fs, r, c), pulse, fs, gate_d)
    off_mm = c / 2 * ((me - md) - (ie - idd)) * 1e3
    bin_mm = c / (2 * fs_dev) * 1e3
    return {"path": path, "d_mm": pl.plate["distance_mm"], "plate_size_mm": pl.plate["size_mm"],
            "lag_echo_s": me, "lag_direct_s": md, "ideal_echo_s": ie, "ideal_direct_s": idd,
            "sys_lag_s": md - idd, "rigid_s": ie, "delay_s": me - (md - idd),
            "offset_mm": off_mm, "offset_bins": off_mm / bin_mm,
            "echo_direct_ratio": ae / ad, "image_source_ratio": aie / aid}


def subcell_report(step_shards: list[str], rigid_shard: str) -> dict:
    """BUILD.md Q-020 criterion over the plate steps + the D-036 re-measure against the rigid reference."""
    rows = sorted((echo_delay(p) for p in step_shards), key=lambda x: x["d_mm"])
    if len(rows) < 2:
        raise ValueError("need at least 2 plate steps")
    c = 343.0
    with h5py.File(rows[0]["path"], "r") as f:
        c = float(f.attrs["c_sim"])
    dm = np.array([x["d_mm"] for x in rows]) * 1e-3
    dl = np.array([x["delay_s"] for x in rows])
    slope = float(np.polyfit(dm, dl, 1)[0])
    rel = abs(slope / (2 / c) - 1)
    mono = bool(np.all(np.diff(dl) > 0))
    max_off = float(np.max(np.abs([x["offset_bins"] for x in rows])))
    rig = echo_delay(rigid_shard)
    d036 = {"ratio_100mm": rows[0]["echo_direct_ratio"], "ratio_rigid": rig["echo_direct_ratio"],
            "image_source": rig["image_source_ratio"],
            "rigid_vs_image": rig["echo_direct_ratio"] / rig["image_source_ratio"],
            "100mm_vs_rigid": rows[0]["echo_direct_ratio"] / rig["echo_direct_ratio"],
            "rigid_offset_mm": rig["offset_mm"]}
    return {"rows": rows, "slope_s_per_m": slope, "expected_s_per_m": 2 / c, "slope_rel_err": rel,
            "monotonic": mono, "max_offset_bins": max_off,
            "pass": bool(mono and rel <= SLOPE_TOL and max_off <= OFFSET_TOL_BINS), "d036": d036}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ringsim.validation.subcell", description=__doc__.splitlines()[0])
    p.add_argument("shards", nargs="+", help="plate-step shards (seq 'plate')")
    p.add_argument("--rigid", required=True, help="rigid-reference plate shard")
    a = p.parse_args(argv)
    print(json.dumps(subcell_report(a.shards, a.rigid), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
