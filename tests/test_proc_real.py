"""Real-capture call path through ringsim.proc (PROCESSING.md Notes, CAPTURE_APP.md, VALIDATION.md Protocol B).

A synthetic "capture" stands in for the iOS app's output: float32 mono 48 kHz WAV, the looped tx frame
played from some sample after the recording started (so the stream begins with silence and the first
period lacks the previous period's echo tail), a device response, noise. Every recording (free-air
start, free-air end, each plate take) gets its OWN random start offset and is aligned on its own.

Call order under test, one recording at a time:
  align_direct_path -> frame_stream -> free_air_reference (free-air takes) -> cfmcw_profiles_ref
Real data takes skip frame 0 (user answer 2026-11-11, PROCESSING.md Notes); the checks below still run with
both choices, first_frame in (0, 1), so the bins are shown not to depend on the skip.
"""
import numpy as np
import pytest
from scipy import fft, signal
from scipy.io import wavfile

from ringsim.proc import (C_SIM, ChirpSpec, align_direct_path, cfmcw_profiles, cfmcw_profiles_ref,
                          frame_stream, free_air_reference, magnitude, tx_frame)

FS = 48_000
BAND = (18e3, 22e3)
TX = tx_frame(ChirpSpec(18e3, 22e3, 0.0125, 0.0125, "none"), FS)
L = len(TX)
N_BINS = 84
F = fft.rfftfreq(L, 1 / FS)
TXF = fft.rfft(TX, L)
D0 = 3.3                                              # speaker -> mic direct path, samples
DIST_MM = np.array([80.0, 110.0, 140.0, 170.0, 200.0])
FIRST_FRAMES = (0, 1)


def delay(d):
    return np.exp(-2j * np.pi * F * d / FS)


def device_response():
    """Fixed device response: -5 dB tilt across the band, 2.4-sample delay, constant phase."""
    x = (F - BAND[0]) / (BAND[1] - BAND[0])
    return 10 ** (-5 * x / 20) * np.exp(-2j * np.pi * F * 2.4 / FS + 0.7j)


def scene(dist_mm=None, amp=0.3):
    """Direct path, plus one plate echo whose excess round-trip path is 2*dist (VALIDATION.md)."""
    H = delay(D0)
    if dist_mm is not None:
        d_ex = 2 * dist_mm * 1e-3 / C_SIM * FS            # excess delay in samples (= bins)
        H = H + amp * delay(D0 + d_ex)
    return H * device_response()


def capture(tmp_path, name, H, start, seconds, rng):
    """Write and read back a float32 48 kHz WAV: tx loop starting `start` samples into the recording."""
    n = int(seconds * FS)
    loop = np.tile(TX, n // L + 2).astype(float)
    imp = fft.irfft(H, L)                                 # band-limited impulse response, 600 samples
    y = signal.fftconvolve(loop, imp)[:len(loop)]
    y = np.concatenate([np.zeros(start), y])[:n]
    y = y + 5e-3 * rng.standard_normal(n)
    path = tmp_path / f"{name}.wav"
    wavfile.write(path, FS, y.astype(np.float32))
    rate, x = wavfile.read(path)
    assert rate == FS and x.dtype == np.float32
    return x


def free_ref(x):
    off = align_direct_path(x, TX, BAND, FS)
    return off, free_air_reference(x, TX, BAND, FS, off)


def referenced(x, ref, first_frame):
    off = align_direct_path(x, TX, BAND, FS)
    return off, cfmcw_profiles_ref(frame_stream(x, off, L), TX, ref, BAND, FS, N_BINS)[first_frame:]


def peak_bin(profiles, bg):
    """Protocol B metric 1: bin of the largest magnitude beyond bin 2 in the background-subtracted
    mean profile (bg = mean profile of the no-plate take, same ref)."""
    m = magnitude(profiles.mean(axis=0) - bg)
    return 3 + int(np.argmax(m[3:]))


def background(x, ref, first_frame):
    return referenced(x, ref, first_frame)[1].mean(axis=0)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cap")
    rng = np.random.default_rng(20261111)
    H_free = scene(None)
    free_a = capture(tmp, "free_air_0", H_free, 123, 5.0, rng)
    free_b = capture(tmp, "free_air_1", H_free, 41, 5.0, rng)
    takes = {d: capture(tmp, f"plate_{int(d)}mm", scene(d), int(rng.integers(0, 300)), 3.0, rng)
             for d in DIST_MM}
    bg = capture(tmp, "noplate", scene(None), 77, 3.0, rng)
    return free_a, free_b, takes, bg


@pytest.mark.parametrize("first_frame", FIRST_FRAMES)
def test_peak_bins_and_linearity(session, first_frame):
    free_a, _, takes, bg_x = session
    _, ref = free_ref(free_a)
    bg = background(bg_x, ref, first_frame)
    bins = []
    for d, x in takes.items():
        _, prof = referenced(x, ref, first_frame)
        b = peak_bin(prof, bg)
        expect = 2 * d * 1e-3 / C_SIM * FS
        print(f"first_frame={first_frame} d={d:.0f} mm expect bin {expect:.2f}, measured {b}")
        assert abs(b - expect) <= 1
        bins.append(b)
    slope = np.polyfit(DIST_MM * 1e-3, bins, 1)[0]
    ideal = 2 * FS / C_SIM
    print(f"first_frame={first_frame} slope {slope:.1f} bins/m vs 2*fs/c {ideal:.1f}")
    assert abs(slope / ideal - 1) < 0.05


def test_references_from_start_and_end_agree(session):
    """Two free-air recordings with different start offsets give the same REF (drift check input)."""
    free_a, free_b = session[:2]
    (oa, ra), (ob, rb) = free_ref(free_a), free_ref(free_b)
    err = np.linalg.norm(ra - rb) / np.linalg.norm(ra)
    print(f"free-air offsets {oa}, {ob}; REF rel L2 {err:.2e}")
    assert oa != ob and err < 1e-2


def test_offset_tracks_start_sample():
    """Per-recording alignment: the offset moves one-for-one with the start sample."""
    rng = np.random.default_rng(3)
    base = None
    for s in (0, 17, 150, 299):
        x = np.concatenate([np.zeros(s), np.tile(TX, 60)])
        x = (x + 1e-3 * rng.standard_normal(len(x))).astype(np.float32)
        off = align_direct_path(x, TX, BAND, FS)
        base = off - s if base is None else base
        assert abs((off - s) - base) <= 1


@pytest.mark.parametrize("first_frame", FIRST_FRAMES)
def test_separate_offsets_matter(session, first_frame):
    """A data take referenced with a REF aligned at the wrong offset moves every profile by that
    many lags (why each recording is aligned on its own; Q-026 is the sim-reader analogue)."""
    free_a, _, takes, bg_x = session
    x = takes[110.0]
    off, ref = free_ref(free_a)
    bg = background(bg_x, ref, first_frame)
    own = peak_bin(referenced(x, ref, first_frame)[1], bg)
    off_data = align_direct_path(x, TX, BAND, FS)
    shifted = cfmcw_profiles_ref(frame_stream(x, off_data + 5, L), TX, ref, BAND, FS, N_BINS)[first_frame:]
    print(f"first_frame={first_frame} own-offset peak {own}, +5 sample offset peak {peak_bin(shifted, bg)}")
    assert abs(own - peak_bin(shifted, bg)) >= 4


def test_frame0_vs_steady_state(session):
    """Report-only numbers for the open frame-0 question; asserts only that frame 0 still has the
    plate peak within 1 bin (the data stream here starts with silence, no echo tail in period 0)."""
    free_a, _, takes, bg_x = session
    _, ref = free_ref(free_a)
    x = takes[140.0]
    bg = background(bg_x, ref, 1)                         # steady-state background (frames 1..)
    _, prof = referenced(x, ref, 0)
    steady = prof[5:].mean(axis=0) - bg
    rel0 = np.linalg.norm(prof[0] - bg - steady) / np.linalg.norm(steady)
    exp = 2 * 140.0 * 1e-3 / C_SIM * FS
    b0 = 3 + int(np.argmax(magnitude(prof[0] - bg)[3:]))
    print(f"frame 0: peak bin {b0} (expect {exp:.2f}), rel L2 to steady-state mean {rel0:.3f}")
    assert abs(b0 - exp) <= 1


def test_raw_profiles_share_the_call_path(session):
    """cfmcw_profiles (unreferenced) on the same frames, background-subtracted: plate echo within 1 bin."""
    takes, bg_x = session[2:]
    x = takes[140.0]

    def raw_mean(y):
        off = align_direct_path(y, TX, BAND, FS)
        return cfmcw_profiles(frame_stream(y, off, L), TX, BAND, FS, N_BINS)[5:].mean(axis=0)

    m = magnitude(raw_mean(x) - raw_mean(bg_x))
    exp = 2 * 140.0 * 1e-3 / C_SIM * FS
    b = 3 + int(np.argmax(m[3:]))
    print(f"raw, background-subtracted: plate bin {b}, expect {exp:.2f}")
    assert abs(b - exp) <= 1
