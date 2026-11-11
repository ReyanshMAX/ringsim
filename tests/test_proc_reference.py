"""Reference-normalized profiles (PROCESSING.md Required tests, D-042).

Stream = direct path + 4 echoes, a steady periodic stream built one period at a time. Real-side
copies pass through 50 random device responses (magnitude tilt up to -8 dB across the band; random
constant, linear and quadratic phase); each copy's reference comes from free_air_reference on its
own direct-only stream.
"""
import numpy as np
from scipy import fft

from ringsim.proc import (ChirpSpec, align_direct_path, cfmcw_profiles_ref, frame_stream,
                          free_air_reference, tx_frame)
from ringsim.proc.ir import _band_mask

FS = 48_000
BAND = (18e3, 22e3)
TX = tx_frame(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), FS)
L = len(TX)
N_BINS = 84
N_PERIODS = 56
F = fft.rfftfreq(L, 1 / FS)
TXF = fft.rfft(TX, L)
D0 = 3.3                                                    # direct path, samples
ECHO_D = D0 + np.array([7.2, 18.5, 33.1, 60.7])
ECHO_A = np.array([0.4, -0.25, 0.15, 0.1])


def delay(d):
    return np.exp(-2j * np.pi * F * d / FS)


H_DIRECT = delay(D0)
H_SCENE = H_DIRECT + sum(a * delay(d) for a, d in zip(ECHO_A, ECHO_D))


def stream(H):
    return np.tile(fft.irfft(TXF * H, L), N_PERIODS)


def device_response(rng):
    x = (F - BAND[0]) / (BAND[1] - BAND[0])                 # 0..1 across the band
    tilt_db = -rng.uniform(0, 8) * x
    c0 = rng.uniform(-np.pi, np.pi)
    c1 = rng.uniform(-6, 6)                                 # linear phase = delay in samples
    c2 = rng.uniform(-np.pi, np.pi)                         # quadratic phase across the band
    ph = c0 - 2 * np.pi * F * c1 / FS + c2 * (2 * x - 1) ** 2
    return 10 ** (tilt_db / 20) * np.exp(1j * ph)


def referenced(H_dev):
    data, direct = stream(H_SCENE * H_dev), stream(H_DIRECT * H_dev)
    off = align_direct_path(direct, TX, BAND, FS)
    ref = free_air_reference(direct, TX, BAND, FS, off)
    return cfmcw_profiles_ref(frame_stream(data, off, L), TX, ref, BAND, FS, N_BINS)[1:51]


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_device_response_cancels():
    rng = np.random.default_rng(20261110)
    base = referenced(np.ones_like(F, dtype=complex))
    errs = [rel(referenced(device_response(rng)), base) for _ in range(50)]
    print(f"max rel L2 over 50 device responses: {max(errs):.2e}")
    assert max(errs) < 1e-2


def test_self_reference_is_chirp_autocorrelation():
    s = stream(H_DIRECT)
    off = align_direct_path(s, TX, BAND, FS)
    ref = free_air_reference(s, TX, BAND, FS, off)
    q = cfmcw_profiles_ref(frame_stream(s, off, L), TX, ref, BAND, FS, N_BINS)
    expect = fft.irfft(np.abs(TXF) ** 2 * _band_mask(F, BAND), L)[:N_BINS]
    errs = [rel(row, expect) for row in q]
    print(f"self-reference max rel L2: {max(errs):.2e}")
    assert max(errs) < 1e-2 and np.argmax(q[1]) == 0


def test_too_few_frames_refused():
    import pytest
    with pytest.raises(ValueError):
        free_air_reference(np.tile(TX, 10), TX, BAND, FS, 0)
