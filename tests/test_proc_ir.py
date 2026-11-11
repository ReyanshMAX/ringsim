"""ringsim.proc.ir (PROCESSING.md Required tests: rx_from_ir, linearity, rx_stream S=1,
snapshot_weights, temperature_scale)."""
import numpy as np
import pytest
from scipy import fft, signal

from ringsim.proc import (ChirpSpec, rx_from_ir, rx_stream, snapshot_weights, synth_chirp,
                          temperature_scale, tx_frame)

FS_SIM = 600_000
N_T = 2400


def gauss_pulse(n_t=N_T, fs=FS_SIM, fc=22e3, half_bw=6e3):
    """Gaussian-modulated pulse shaped like the solver's (SOLVER.md constants), test-local."""
    sigma = np.sqrt(2 * np.log(2)) / (2 * np.pi * half_bw)
    t = np.arange(n_t) / fs - 4 * sigma
    return (np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * fc * t)).astype(np.float32)


def delayed(x, m):
    out = np.zeros_like(x)
    out[m:] = x[:len(x) - m]
    return out


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


RATES = [  # fs_device, band (chirp of 600 samples)
    (50_000, (20e3, 24e3)),
    (48_000, (18e3, 22e3)),
    (44_100, (17e3, 21e3)),
]


@pytest.mark.parametrize("fs_dev,band", RATES)
def test_rx_from_ir_delay_and_amplitude(fs_dev, band):
    spec = ChirpSpec(band[0], band[1], 600 / fs_dev, 600 / fs_dev, "none")
    # zeros around the chirp keep its sharp edges away from the array ends, where resample_poly
    # truncates (the documented ~20-sample edge error that rx_stream discards as padding)
    z = np.zeros(100, np.float32)
    tx = np.concatenate([z, synth_chirp(spec, fs_dev), z])
    pulse = gauss_pulse()
    m = 120
    y = rx_from_ir(delayed(pulse, m), pulse, FS_SIM, tx, band, fs_dev)
    expect = m * fs_dev / FS_SIM
    xc = signal.correlate(y, tx, mode="full", method="fft")
    lag = np.argmax(np.abs(signal.hilbert(xc))) - (len(tx) - 1)
    assert abs(lag - expect) <= 1, (lag, expect)
    M = 1 << 14
    Y, X = fft.rfft(y, M), fft.rfft(tx, M)
    f = fft.rfftfreq(M, 1 / fs_dev)
    sel = (f >= band[0]) & (f <= band[1])
    ratio = np.abs(Y[sel]) / np.abs(X[sel])
    assert np.max(np.abs(ratio - 1)) < 0.03, np.max(np.abs(ratio - 1))


def test_non_integer_rate_refused():
    tx = synth_chirp(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), 48_000)
    pulse = gauss_pulse()
    with pytest.raises(ValueError, match="fs_sim"):
        rx_from_ir(pulse, pulse, 799_999.9999999999, tx, (18e3, 22e3), 48_000)


def test_rx_from_ir_linearity():
    fs_dev, band = 48_000, (18e3, 22e3)
    tx = tx_frame(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), fs_dev)
    pulse = gauss_pulse()
    rng = np.random.default_rng(1)
    ir1 = delayed(pulse, 100) + 0.3 * delayed(pulse, 700)
    ir2 = (rng.standard_normal(N_T) * 0.01).astype(np.float32) + delayed(pulse, 400)
    a, b = 0.7, -1.3
    lhs = rx_from_ir(a * ir1 + b * ir2, pulse, FS_SIM, tx, band, fs_dev)
    rhs = a * rx_from_ir(ir1, pulse, FS_SIM, tx, band, fs_dev) + b * rx_from_ir(ir2, pulse, FS_SIM, tx, band, fs_dev)
    assert rel(lhs, rhs) < 1e-5


def test_rx_stream_linearity():
    fs_dev, band = 50_000, (20e3, 24e3)
    tx = tx_frame(ChirpSpec(20e3, 24e3, 0.012, 0.012, "none"), fs_dev)
    pulse = gauss_pulse(1200)
    rng = np.random.default_rng(2)
    S, n_frames = 5, 4
    snap_t = np.arange(S) * 0.012
    irs1 = np.stack([delayed(pulse, 100 + 3 * s) for s in range(S)])
    irs2 = (rng.standard_normal((S, 1200)) * 0.05).astype(np.float32)
    a, b = 1.5, -0.4
    args = (snap_t, pulse, FS_SIM, tx, band, fs_dev, n_frames)
    lhs = rx_stream(a * irs1 + b * irs2, *args)
    rhs = a * rx_stream(irs1, *args) + b * rx_stream(irs2, *args)
    assert rel(lhs, rhs) < 1e-5


def test_rx_stream_static_equals_rx_from_ir():
    fs_dev, band = 48_000, (18e3, 22e3)
    tx = tx_frame(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), fs_dev)
    L, n_frames = len(tx), 6
    pulse = gauss_pulse(1500)
    ir = delayed(pulse, 150) + 0.4 * delayed(pulse, 900)
    out = rx_stream(ir[None], np.array([0.0]), pulse, FS_SIM, tx, band, fs_dev, n_frames)
    assert len(out) == n_frames * L + int(np.ceil(1500 * fs_dev / FS_SIM))
    warm = 3                                     # longer warm-up than rx_stream's one period
    ref = rx_from_ir(ir, pulse, FS_SIM, np.tile(tx, n_frames + warm + 1), band, fs_dev)
    assert rel(out[:n_frames * L], ref[warm * L: (warm + n_frames) * L]) < 1e-5


def test_snapshot_weights_columns_sum_to_one():
    rng = np.random.default_rng(3)
    snap_t = np.cumsum(rng.uniform(0.002, 0.01, 9))
    t = np.linspace(snap_t[0] - 0.01, snap_t[-1] + 0.01, 3001)
    for interp in ("cubic", "hold"):
        W = snapshot_weights(snap_t, t, interp)
        assert W.shape == (9, len(t))
        assert np.max(np.abs(W.sum(axis=0) - 1)) < 1e-12
    assert np.array_equal(snapshot_weights(np.array([0.3]), t), np.ones((1, len(t))))


def test_snapshot_weights_catmull_rom_when_even():
    h = 0.006
    snap_t = np.arange(8) * h
    t = np.linspace(h, 6 * h, 1001, endpoint=False)          # interior intervals 1..5
    W = snapshot_weights(snap_t, t, "cubic")
    i = np.floor(t / h + 1e-12).astype(int)
    u = t / h - i
    basis = 0.5 * np.stack([-u**3 + 2 * u**2 - u, 3 * u**3 - 5 * u**2 + 2,
                            -3 * u**3 + 4 * u**2 + u, u**3 - u**2])
    expect = np.zeros_like(W)
    for k, off in enumerate((-1, 0, 1, 2)):
        expect[i + off, np.arange(len(t))] = basis[k]
    assert np.max(np.abs(W - expect)) < 1e-12


def test_snapshot_weights_hold_switches_at_frame_starts():
    P, fs = 0.012, 50_000
    L = int(round(P * fs))
    snap_t = np.arange(6) * L / fs           # frame starts, computed as rx_stream's device times are
    t = np.arange(5 * L) / fs
    W = snapshot_weights(snap_t, t, "hold")
    assert set(np.unique(W)) <= {0.0, 1.0}
    frame = np.arange(len(t)) // L
    assert np.array_equal(np.argmax(W, axis=0), frame)


def test_temperature_scale_pure_delay():
    N = 1 << 16
    freqs = fft.rfftfreq(N, 1 / FS_SIM)
    tau = 0.25e-3
    H = np.exp(-2j * np.pi * freqs * tau)
    sel = (freqs >= 17e3) & (freqs <= 27e3)
    for alpha in (0.983, 1.026):
        out = temperature_scale(H, freqs, alpha)
        expect = alpha * np.exp(-2j * np.pi * freqs * tau / alpha)
        assert np.max(np.abs(out[sel] - expect[sel])) / alpha < 1e-4
    assert temperature_scale(H, freqs, 1.0) is H
