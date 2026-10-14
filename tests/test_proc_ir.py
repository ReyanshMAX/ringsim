import numpy as np
from scipy import fft, signal

from conftest import BAND, FS, TEST_CHIRP
from ringsim.proc import overlap_add_stream, rx_from_ir, temperature_scale, tx_frame

K = 12                                  # fs_sim = 576 kHz (SOLVER.md time_axis, dx 2 mm)
FS_SIM = K * FS


def _pulse(n_t):
    # Same form as solver.pulse.make_pulse; proc must not import solver.
    sigma = np.sqrt(2 * np.log(2)) / (2 * np.pi * 6000.0)
    t = np.arange(n_t) / FS_SIM - 4 * sigma
    return (np.exp(-t**2 / (2 * sigma**2)) * np.sin(2 * np.pi * 22000.0 * t)).astype(np.float32)


def _ref_delayed(tx, d):
    """tx (band-limited the same way) delayed by integer d device samples."""
    out = np.zeros(len(tx) + 400)
    out[d: d + len(tx)] = tx
    return out


def test_rx_from_ir_delay_and_amplitude():
    n_t = 1300
    pulse = _pulse(n_t)
    m = 144                                             # 12 device samples
    ir = np.zeros(n_t, np.float32)
    ir[m:] = pulse[: n_t - m]
    tx = tx_frame(TEST_CHIRP, FS)
    rx = rx_from_ir(ir, pulse, K, tx, BAND, FS)
    assert len(rx) == int(np.ceil((len(tx) * K + n_t) / K))
    xc = signal.correlate(rx, tx, mode="full")
    lag = np.argmax(np.abs(signal.hilbert(xc))) - (len(tx) - 1)
    assert abs(lag - m / K) <= 1
    # amplitude inside band: |RX(f)| / |TX(f) delayed| within 3%
    N = 1 << 14
    R = fft.rfft(rx, N); T = fft.rfft(tx, N)
    f = fft.rfftfreq(N, 1 / FS)
    inb = (f >= BAND[0] + 200) & (f <= BAND[1] - 200)
    ratio = np.abs(R[inb]) / np.abs(T[inb])
    assert np.all(np.abs(ratio - 1) < 0.03), (ratio.min(), ratio.max())


def test_rx_from_ir_linearity(rng):
    n_t = 1300
    pulse = _pulse(n_t)
    ir1 = rng.standard_normal(n_t).astype(np.float32)
    ir2 = rng.standard_normal(n_t).astype(np.float32)
    tx = tx_frame(TEST_CHIRP, FS)
    a, b = 0.7, -1.3
    lhs = rx_from_ir((a * ir1 + b * ir2).astype(np.float32), pulse, K, tx, BAND, FS)
    rhs = a * rx_from_ir(ir1, pulse, K, tx, BAND, FS) + b * rx_from_ir(ir2, pulse, K, tx, BAND, FS)
    assert np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) < 1e-5


def test_temperature_scale_pure_delay():
    N = 1 << 16
    freqs = fft.rfftfreq(N, 1 / FS_SIM)                 # df ~ 8.8 Hz, as in rx_from_ir
    tau = 0.100 / 343.0                                 # Phase 1 direct path
    H = np.exp(-2j * np.pi * freqs * tau)
    inb = (freqs >= 17e3) & (freqs <= 27e3)
    for alpha in (0.983, 1.026):
        out = temperature_scale(H, freqs, alpha)
        ref = alpha * np.exp(-2j * np.pi * freqs * tau / alpha)
        err = np.abs(out[inb] - ref[inb]) / np.abs(ref[inb])
        assert err.max() < 1e-4, err.max()
    np.testing.assert_array_equal(temperature_scale(H, freqs, 1.0), H)


def test_overlap_add_stream():
    a = np.ones(5, np.float32); b = 2 * np.ones(5, np.float32)
    s = overlap_add_stream([a, b], 3)
    np.testing.assert_array_equal(s, [1, 1, 1, 3, 3, 2, 2, 2])
