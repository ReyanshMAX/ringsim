"""align_direct_path (PROCESSING.md, D-086): absolute offset for late first chirps, refusal otherwise.

Looped tx_frame after `start` samples of silence, 1e-3 noise. The offset is start + the causal band-pass delay
(13 samples for the phone chirp, measured 2026-11-11), not (start mod frame_len)."""
import numpy as np
import pytest

from ringsim.proc import ChirpSpec, align_direct_path, tx_frame

FS = 48_000
BAND = (18e3, 22e3)
TX = tx_frame(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), FS)
L = len(TX)
BP_DELAY = 13
WINDOW_SAMPLES = int(round(0.25 * FS))


def stream(start, n_frames=40, seed=0, amp=1.0, noise=1e-3):
    rng = np.random.default_rng(seed)
    x = noise * rng.standard_normal(start + n_frames * L + 100)
    x[start:start + n_frames * L] += amp * np.tile(TX, n_frames)
    return x.astype(np.float32)


@pytest.mark.parametrize("start", [0, 200, 500, 587, 600, 650, 777, 1000, 1500, 2400, 6000, WINDOW_SAMPLES - 20])
def test_absolute_offset(start):
    assert align_direct_path(stream(start), TX, BAND, FS) - start == BP_DELAY


def test_start_past_the_window_raises():
    for start in (WINDOW_SAMPLES, WINDOW_SAMPLES + 100, WINDOW_SAMPLES + 3 * L, 24_000):
        with pytest.raises(ValueError):
            align_direct_path(stream(start), TX, BAND, FS)


def test_window_is_a_parameter():
    x = stream(15_000)
    with pytest.raises(ValueError):
        align_direct_path(x, TX, BAND, FS)
    assert align_direct_path(x, TX, BAND, FS, search_window_s=0.4) - 15_000 == BP_DELAY


def test_noise_only_raises():
    rng = np.random.default_rng(1)
    for sigma in (1e-3, 1e-1, 1.0):
        with pytest.raises(ValueError, match="no chirp"):
            align_direct_path(sigma * rng.standard_normal(40 * L), TX, BAND, FS)


def test_short_stream_raises():
    with pytest.raises(ValueError, match="shorter"):
        align_direct_path(np.zeros(L - 1, np.float32), TX, BAND, FS)


def test_weak_chirp_still_found():
    """0 dB per-sample SNR (chirp amplitude = noise sigma) is above the refusal level."""
    assert align_direct_path(stream(777, amp=1e-2, noise=1e-2), TX, BAND, FS) - 777 == BP_DELAY


def test_short_periodic_stream_is_used_as_far_as_it_goes():
    """8 periods = 4800 samples < W + len(tx): the window is clamped, not refused."""
    x = np.tile(TX, 8).astype(np.float32)
    assert align_direct_path(x, TX, BAND, FS) in range(BP_DELAY - 1, BP_DELAY + 2)


def test_clamped_window_ending_on_a_period_peak_does_not_raise():
    """Length 5L + 13: the correlation's last lag (13 + 3L) is a period peak. The window is cut short by the stream
    end, and a period peak at the last lag must neither raise nor be mistaken for lag 0 (circular Hilbert leak)."""
    x = np.tile(TX, 8).astype(np.float32)[:5 * L + BP_DELAY]
    assert align_direct_path(x, TX, BAND, FS) == BP_DELAY
