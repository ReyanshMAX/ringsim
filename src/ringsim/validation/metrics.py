"""Protocol B (flat plate) metrics, VALIDATION.md. Inputs are profiles already computed through ringsim.proc and
already background-subtracted; frame skipping (frame 0 of real takes) is the caller's job.

Nothing here has a pass/fail default the spec does not give: peak-bin error <= 1 and range-linearity slope within 5 %
are the two spec thresholds; everything else is reported.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ringsim.proc import magnitude, speed_of_sound

MODES = ("ref", "raw")
PEAK_MIN_BIN = 3                # "beyond bin 2" (VALIDATION.md metric 1)
PEAK_TOL_BINS = 1               # metric 1 pass
SLOPE_TOL = 0.05                # metric 2 pass
SHAPE_HALF_WIDTH = 10           # metric 5: bins either side of the plate peak
SHAPE_MAX_DELTA = 1.0           # metric 5: |delta| <= 1 bin


def _mean_profile(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).mean(axis=0)


def peak_bin(profiles: np.ndarray) -> tuple[int, float]:
    """(bin, |magnitude|) of the largest magnitude of the mean profile beyond bin 2."""
    mag = magnitude(_mean_profile(profiles)[None, :])[0]
    k = PEAK_MIN_BIN + int(np.argmax(mag[PEAK_MIN_BIN:]))
    return k, float(mag[k])


def _shift(x: np.ndarray, delta: float) -> np.ndarray:
    """x delayed by `delta` bins (cubic spline, edge value held)."""
    n = np.arange(len(x))
    return CubicSpline(n, x)(np.clip(n - delta, 0, len(x) - 1))


def signed_shape(sim: np.ndarray, real: np.ndarray, center_bin: int) -> dict:
    """Fit one sub-bin delay (|delta| <= 1) to the sim mean profile, then Pearson r with the real one over
    center_bin +- SHAPE_HALF_WIDTH. Returns {delta_bins, r}."""
    s, r = _mean_profile(sim), _mean_profile(real)
    lo, hi = max(0, center_bin - SHAPE_HALF_WIDTH), min(len(r), center_bin + SHAPE_HALF_WIDTH + 1)

    def corr(delta):
        a, b = _shift(s, delta)[lo:hi], r[lo:hi]
        a, b = a - a.mean(), b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / den) if den > 0 else 0.0

    grid = np.linspace(-SHAPE_MAX_DELTA, SHAPE_MAX_DELTA, 201)
    delta = float(grid[int(np.argmax([corr(d) for d in grid]))])
    return {"delta_bins": delta, "r": corr(delta)}


def _loglog_slope(d: list[float], a: list[float]) -> float:
    return float(np.polyfit(np.log(d), np.log(a), 1)[0])


def plate_metrics(sim: dict, real: dict, fs_hz: float, temp_c: float) -> dict:
    """sim, real: {"ref"|"raw": {distance_mm: [frames, n_bins]}}, background-subtracted. Returns
      peak_bin   {mode: {"sim"|"real": {d: bin}}}, peak_bin_error {mode: {d: bins}}, peak_bin_pass {mode: bool}
      linearity  {mode: {slope_bins_per_m, expected_bins_per_m, rel_err, pass}}   (real data, metric 2)
      amp_trend  {mode: {"sim", "real": log-log slope of peak magnitude vs distance}}  (report only)
      signed_shape {mode: {d: {delta_bins, r}}}   (centred on the real peak; report only)
    Metric 1 is gated on "ref" (D-074); "raw" is reported alongside with its own flag."""
    out: dict = {"peak_bin": {}, "peak_bin_error": {}, "peak_bin_pass": {}, "linearity": {}, "amp_trend": {},
                 "signed_shape": {}}
    expected = 2.0 * fs_hz / speed_of_sound(temp_c)
    for m in MODES:
        if set(sim[m]) != set(real[m]):
            raise ValueError(f"{m}: sim distances {sorted(sim[m])} != real distances {sorted(real[m])}")
        ds = sorted(real[m])
        ps = {d: peak_bin(sim[m][d]) for d in ds}
        pr = {d: peak_bin(real[m][d]) for d in ds}
        err = {d: ps[d][0] - pr[d][0] for d in ds}
        out["peak_bin"][m] = {"sim": {d: ps[d][0] for d in ds}, "real": {d: pr[d][0] for d in ds}}
        out["peak_bin_error"][m] = err
        out["peak_bin_pass"][m] = all(abs(e) <= PEAK_TOL_BINS for e in err.values())
        if len(ds) >= 2:
            slope = float(np.polyfit(np.array(ds) / 1000.0, [pr[d][0] for d in ds], 1)[0])
            rel = slope / expected - 1.0
            out["linearity"][m] = {"slope_bins_per_m": slope, "expected_bins_per_m": expected, "rel_err": rel,
                                   "pass": abs(rel) <= SLOPE_TOL}
            out["amp_trend"][m] = {"sim": _loglog_slope(ds, [ps[d][1] for d in ds]),
                                   "real": _loglog_slope(ds, [pr[d][1] for d in ds])}
        out["signed_shape"][m] = {d: signed_shape(sim[m][d], real[m][d], pr[d][0]) for d in ds}
    return out


def noise_floor(profiles: np.ndarray, *, skip_frames: int) -> float:
    """Static noise floor n (metric 4): median over frames of ||P_k - mean P|| / ||mean P||, from a no-plate take.
    The caller passes the take's referenced profiles NOT background-subtracted (subtracting makes mean P ~ 0 and the
    ratio meaningless), computed with the session's reference; skip_frames drops frame 0 (and any others) first."""
    p = np.asarray(profiles, dtype=np.float64)[skip_frames:]
    if len(p) < 2:
        raise ValueError("need at least 2 frames after skipping")
    mean = p.mean(axis=0)
    return float(np.median(np.linalg.norm(p - mean, axis=1) / np.linalg.norm(mean)))


def free_air_drift(start: np.ndarray, end: np.ndarray) -> dict:
    """Drift between the session's first and last free-air recording: ||mean(end) - mean(start)|| / ||mean(start)||
    of their profiles. The caller aligns each recording on its own offset, uses the SAME reference (the start one) for
    both, and passes only frames after the reference's own n_avg frames (index >= n_avg + 1), so the number measures
    drift and not how well the reference fits the frames that built it."""
    a = np.asarray(start, dtype=np.float64).mean(axis=0)
    b = np.asarray(end, dtype=np.float64).mean(axis=0)
    return {"rel_l2": float(np.linalg.norm(b - a) / np.linalg.norm(a))}
