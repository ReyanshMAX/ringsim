"""ringsim.proc.profile (PROCESSING.md Required tests: synthetic bins 0 and 20)."""
import numpy as np

from ringsim.proc import (ChirpSpec, align_direct_path, cfmcw_profiles, differential,
                          frame_stream, magnitude, tx_frame)

FS = 48_000
BAND = (18e3, 22e3)
TX = tx_frame(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), FS)
N_BINS = 84


def test_synthetic_bins_0_and_20():
    L, d, n_per = len(TX), 37, 8
    base = np.tile(TX, n_per).astype(float)
    stream = np.zeros(d + 20 + len(base))
    stream[d:d + len(base)] += base
    stream[d + 20:d + 20 + len(base)] += 0.3 * base
    off = align_direct_path(stream, TX, BAND, FS)
    prof = cfmcw_profiles(frame_stream(stream, off, L), TX, BAND, FS, N_BINS)
    assert prof.dtype == np.float32 and prof.shape[1] == N_BINS
    for p in prof[2:-1]:                                   # steady-state frames, signed peaks
        assert np.argmax(p) <= 1                           # measured 2026-11-11: bin 1
        assert abs(10 + np.argmax(p[10:]) - 20) <= 1       # measured: bin 19


def test_frame_stream_drops_tail():
    x = np.arange(1000.0)
    f = frame_stream(x, 7, 100)
    assert f.shape == (9, 100) and f[0, 0] == 7 and f[-1, -1] == 906


def test_magnitude_and_differential():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((5, N_BINS)).astype(np.float32)
    assert magnitude(p).shape == p.shape and np.all(magnitude(p) >= 0)
    d = differential(p)
    assert d.shape == (4, N_BINS) and np.allclose(d[1], p[2] - p[1])
