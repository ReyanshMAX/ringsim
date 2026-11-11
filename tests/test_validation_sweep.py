"""sweep.response / phase_residual on synthetic captures with known delay, gain and phase."""
import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.proc import export_sweep_wav
from ringsim.validation.sweep import direct_delay, phase_residual, response

FS = 48000


@pytest.fixture
def sweep_path(tmp_path):
    p = str(tmp_path / "sweep.wav")
    export_sweep_wav(16000.0, 24000.0, 0.5, FS, p)
    return p


def _capture(tmp_path, name, sweep, delay, gain=0.5, extra_phase=None, noise=0.0, seed=0, n=FS * 2):
    """sweep delayed by `delay` samples (fractional ok, via phase ramp), scaled; optional extra phase(f)."""
    x = np.zeros(n)
    x[:len(sweep)] = sweep
    f = np.fft.rfftfreq(n, 1 / FS)
    H = gain * np.exp(-2j * np.pi * f * delay / FS)
    if extra_phase is not None:
        H = H * np.exp(1j * extra_phase(f))
    y = np.fft.irfft(np.fft.rfft(x) * H, n) + noise * np.random.default_rng(seed).standard_normal(n)
    p = str(tmp_path / f"{name}.wav")
    wavfile.write(p, FS, y.astype(np.float32))
    return p


def _sweep(sweep_path):
    return wavfile.read(sweep_path)[1].astype(np.float64)


def test_delay_subsample(tmp_path, sweep_path):
    sw = _sweep(sweep_path)
    for d in (100.0, 137.3, 200.5):
        p = _capture(tmp_path, "c", sw, d)
        assert abs(direct_delay(wavfile.read(p)[1].astype(np.float64), sw) - d) < 0.1


def test_response_flat_gain_with_different_delays(tmp_path, sweep_path):
    sw = _sweep(sweep_path)
    caps = [_capture(tmp_path, f"t{i}", sw, d, noise=1e-4, seed=i) for i, d in enumerate((120.0, 311.4, 87.7))]
    f, mag = response(sweep_path, caps)
    band = (f >= 18000) & (f <= 22000)
    assert np.all(np.abs(mag[band] - 20 * np.log10(0.5)) < 0.5)


def test_phase_residual_zero_for_pure_delay_and_sees_curvature(tmp_path, sweep_path):
    sw = _sweep(sweep_path)
    caps = [_capture(tmp_path, f"a{i}", sw, d) for i, d in enumerate((120.0, 311.4, 87.7))]
    f, r = phase_residual(sweep_path, caps)
    assert f[0] >= 18000 and f[-1] <= 22000 and np.max(np.abs(r)) < 0.1
    quad = lambda ff: 0.8 * ((ff - 20000.0) / 2000.0) ** 2          # 0.8 rad of curvature across the band
    caps2 = [_capture(tmp_path, f"b{i}", sw, d, extra_phase=quad) for i, d in enumerate((120.0, 311.4))]
    _, r2 = phase_residual(sweep_path, caps2)
    assert 0.2 < np.max(np.abs(r2)) < 1.0


def test_no_captures(sweep_path):
    with pytest.raises(ValueError):
        response(sweep_path, [])
