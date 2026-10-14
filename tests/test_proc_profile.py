import numpy as np
from scipy import signal

from conftest import BAND, FS, TEST_CHIRP
from ringsim.proc import (align_direct_path, cfmcw_profiles, differential, frame_stream,
                          magnitude, tx_frame)

N_BINS = 84


def _stream(paths, T=8):
    tx = tx_frame(TEST_CHIRP, FS)
    tile = np.tile(tx, T + 2)
    s = np.zeros(T * len(tx) + 400, np.float32)
    for d, a in paths:
        s[d: d + T * len(tx)] += a * tile[: T * len(tx)]
    return s, tx


def _peak_bins(p, rel=0.2):
    e = magnitude(p)
    pk, _ = signal.find_peaks(np.concatenate([[0], e, [0]]))
    pk = pk - 1
    return pk[e[pk] >= rel * e.max()], e


def test_two_path_peaks_at_0_and_20():
    d = 37
    s, tx = _stream([(d, 1.0), (d + 20, 0.3)])
    off = align_direct_path(s, tx, BAND, FS)
    fr = frame_stream(s, off, len(tx))
    P = cfmcw_profiles(fr, tx, BAND, FS, N_BINS)
    assert P.dtype == np.float32 and P.shape == (fr.shape[0], N_BINS)
    p = P[3]
    assert abs(int(np.argmax(np.abs(p)))) <= 1
    pk, e = _peak_bins(p, rel=0.3)
    assert any(abs(b) <= 1 for b in pk) and any(abs(b - 20) <= 1 for b in pk), pk


def test_frame_stream_shape():
    x = np.arange(1050)
    f = frame_stream(x, 10, 100)
    assert f.shape == (10, 100) and f[0, 0] == 10


def test_differential():
    p = np.arange(12, dtype=np.float32).reshape(3, 4) ** 2
    d = differential(p)
    assert d.shape == (2, 4)
    np.testing.assert_array_equal(d[0], p[1] - p[0])
