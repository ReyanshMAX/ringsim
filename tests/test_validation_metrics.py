"""plate_metrics, noise_floor, free_air_drift on synthetic band-limited profiles."""
import numpy as np
import pytest

from ringsim.proc import speed_of_sound
from ringsim.validation.metrics import free_air_drift, noise_floor, peak_bin, plate_metrics, signed_shape

FS = 48000.0
N_BINS = 84
DIST = (80, 110, 140, 170, 200)


def _profile(bin_pos, amp=1.0, n_bins=N_BINS, frames=4, noise=0.0, seed=0):
    """Band-limited signed echo: Gaussian-windowed carrier centred at `bin_pos` (bins), 1/r-like amp already in amp."""
    n = np.arange(n_bins)
    p = amp * np.exp(-0.5 * ((n - bin_pos) / 2.0) ** 2) * np.cos(2 * np.pi * 0.25 * (n - bin_pos))
    out = np.tile(p, (frames, 1))
    return out + noise * np.random.default_rng(seed).standard_normal(out.shape)


def _set(slope, offset=0.0, amp_exp=-1.0, shift=0.0):
    return {d: _profile(slope * d / 1000.0 + offset + shift, amp=(d / 100.0) ** amp_exp) for d in DIST}


def test_perfect_match():
    temp = 20.0
    slope = 2 * FS / speed_of_sound(temp)
    sim, real = _set(slope), _set(slope)
    m = plate_metrics({"ref": sim, "raw": sim}, {"ref": real, "raw": real}, FS, temp)
    for mode in ("ref", "raw"):
        assert m["peak_bin_pass"][mode] and all(e == 0 for e in m["peak_bin_error"][mode].values())
        assert m["linearity"][mode]["pass"] and abs(m["linearity"][mode]["rel_err"]) < 0.03
        assert abs(m["amp_trend"][mode]["real"] + 1.0) < 0.15 and m["amp_trend"][mode]["sim"] == pytest.approx(
            m["amp_trend"][mode]["real"])
        assert all(v["r"] > 0.999 and abs(v["delta_bins"]) < 1e-9 for v in m["signed_shape"][mode].values())


def test_peak_error_detected_and_slope_fail():
    temp = 20.0
    slope = 2 * FS / speed_of_sound(temp)
    sim = _set(slope)
    real = _set(slope * 1.2)                       # 20 % too steep
    m = plate_metrics({"ref": sim, "raw": sim}, {"ref": real, "raw": real}, FS, temp)
    assert not m["linearity"]["ref"]["pass"] and m["linearity"]["ref"]["rel_err"] > 0.1
    assert not m["peak_bin_pass"]["ref"]
    with pytest.raises(ValueError, match="distances"):
        plate_metrics({"ref": sim, "raw": sim}, {"ref": {80: real[80]}, "raw": {80: real[80]}}, FS, temp)


def test_peak_ignores_bins_0_to_2():
    p = _profile(30.0)
    p[:, 1] += 1.5                                 # direct-path residue at bin 1, larger than the echo peak after the hilbert magnitude
    assert peak_bin(p)[0] in (29, 30, 31)


def test_signed_shape_absorbs_subbin_delay_and_sees_sign_flip():
    real = _profile(40.0)
    sim = _profile(40.6)                          # 0.6 bin late
    s = signed_shape(sim, real, 40)
    assert s["r"] > 0.99 and abs(s["delta_bins"] + 0.6) < 0.05
    assert signed_shape(-real, real, 40)["r"] < -0.9 or signed_shape(-real, real, 40)["r"] < 0.5


def test_noise_floor_and_drift():
    base = _profile(30.0, frames=200, noise=0.02, seed=1)
    n = noise_floor(base, skip_frames=1)
    assert 0.02 < n < 0.2
    assert noise_floor(_profile(30.0, frames=5), skip_frames=1) == 0.0
    with pytest.raises(ValueError):
        noise_floor(base[:2], skip_frames=1)
    a, b = _profile(30.0, frames=10), _profile(30.0, amp=1.1, frames=10)
    assert free_air_drift(a, b)["rel_l2"] == pytest.approx(0.1)
