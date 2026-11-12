"""C-FMCW echo profiles, reference-normalized profiles and differential profiles (PROCESSING.md).

The raw profile follows the C-FMCW echo profiles described in Ring-a-Pose §3.4 and ActSonic
(Cornell SciFi Lab): cross-correlation of tx with the 5th-order Butterworth band-passed rx,
direct path at lag 0, differential = current - previous frame. Details their paper leaves open
are Q-002. The reference normalization (D-042) is this project's addition.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, signal

from ringsim.proc.physics import C_SIM
from ringsim.proc.ir import _band_mask

BUTTER_ORDER = 5
REF_GATE_M = 0.70          # free_air_reference gate: excess path < 0.70 m
REF_TUKEY_ALPHA = 0.5
REF_DECONV_EPS = 1e-3
ALIGN_MIN_PEAK_TO_MEDIAN = 8.0   # D-086: noise-only streams give 3.3-4.2, a chirp at 0 dB per-sample SNR 14-15


def _bandpass_sos(band: tuple[float, float], fs: int) -> np.ndarray:
    """Butterworth order 5, edges = chirp band (Ring-a-Pose §3.4.2)."""
    return signal.butter(BUTTER_ORDER, list(band), btype="bandpass", fs=fs, output="sos")


def align_direct_path(stream: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                      fs: int, search_window_s: float = 0.25) -> int:
    """ABSOLUTE index into `stream` of the first chirp's direct path (D-019, D-086).

    The direct path may lie up to `search_window_s` into the stream (the player starts a variable time after the
    tap, and iOS output latency adds to it). Band-pass (same filter as cfmcw_profiles, causal) the first
    W + 2*len(tx) samples, W = round(search_window_s*fs), and cross-correlate with tx in mode "valid": every lag
    0 .. W + len(tx) is a FULL overlap with tx, so a partial overlap cannot win. The extra period beyond W lets a
    chirp that starts just past the window show its true peak, which then outranks the ramp of sidelobes before it.
    Return the EARLIEST local maximum of the envelope |hilbert(xcorr)| that is >= 0.5 * its max.

    A stream shorter than W + 2*len(tx) is used as far as it goes (a sim stream, a short reference stream).

    Raises ValueError, never misaligns: the stream is shorter than tx; max / median of the envelope is below
    ALIGN_MIN_PEAK_TO_MEDIAN (no chirp found); or the earliest qualifying peak lies past lag W (the first chirp's
    direct path is beyond the window). The window bounds the DIRECT-PATH index (chirp start + the band-pass delay,
    13 samples), not the chirp start.
    """
    tx = np.asarray(tx)
    if len(stream) < len(tx):
        raise ValueError(f"stream of {len(stream)} samples is shorter than tx ({len(tx)})")
    n_win = int(round(search_window_s * fs))
    seg = signal.sosfilt(_bandpass_sos(band, fs), np.asarray(stream[: n_win + 2 * len(tx)], dtype=float))
    xc = signal.correlate(seg, tx, mode="valid", method="fft")
    # zero-padded Hilbert transform: it is FFT-based (circular), and a period peak at the last lag leaks into lag 0
    # (measured: envelope at lag 0 = 56 % of the maximum, which picked lag 0 as "the earliest peak")
    pad = len(tx)
    env = np.abs(signal.hilbert(np.concatenate([np.zeros(pad), xc, np.zeros(pad)])))[pad:-pad]
    if env.max() < ALIGN_MIN_PEAK_TO_MEDIAN * np.median(env):
        raise ValueError(f"no chirp found in the first {search_window_s} s: envelope max / median = "
                         f"{env.max() / max(np.median(env), 1e-300):.1f} < {ALIGN_MIN_PEAK_TO_MEDIAN}")
    padded = np.concatenate([[-np.inf], env, [-np.inf]])
    pk, _ = signal.find_peaks(padded)
    pk = pk - 1
    first = int(pk[env[pk] >= 0.5 * env.max()][0])
    if first > n_win:
        raise ValueError(f"the first chirp's direct path (sample {first}) lies after the {search_window_s} s search "
                         f"window ({n_win} samples)")
    return first


def frame_stream(stream: np.ndarray, offset: int, frame_len: int) -> np.ndarray:
    """[T, frame_len], T = floor((len(stream) - offset) / frame_len); the tail is dropped."""
    T = (len(stream) - offset) // frame_len
    return np.asarray(stream[offset: offset + T * frame_len]).reshape(T, frame_len)


def cfmcw_profiles(frames: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                   fs: int, n_bins: int) -> np.ndarray:
    """Signed C-FMCW profiles [T, n_bins] float32; lag 0 = direct path after alignment.

    Correlation-based; Ring-a-Pose parity unverified (Q-002).
    Per frame: causal Butterworth band-pass (sosfilt; filtfilt vs sosfilt is Q-002), then
    corr[j] = sum_n frame[n + j] * tx[n], j = 0..n_bins-1 (linear: samples past the frame end
    count as 0). One bin = c/(2*fs) one-way.
    """
    fr = signal.sosfilt(_bandpass_sos(band, fs), np.asarray(frames, dtype=float), axis=1)
    L = fr.shape[1]
    N = fft.next_fast_len(L + len(tx), real=True)
    c = fft.irfft(fft.rfft(fr, N, axis=1) * np.conj(fft.rfft(tx, N)), N, axis=1)
    return c[:, :n_bins].astype(np.float32)


def free_air_reference(stream: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                       fs: int, offset: int, n_avg: int = 50) -> np.ndarray:
    """Device-only response REF (complex [L//2+1]) for cfmcw_profiles_ref (D-042).

    stream: a capture (real) or rx_stream (sim) with no target in range; offset from
    align_direct_path. Mean of frames 1..n_avg, regularized deconvolution by tx with the band
    mask, gate |lag| <= floor(0.70*fs/C_SIM) with a Tukey(0.5) taper, REF = rfft(gated) * TX.
    """
    L = len(tx)
    frames = frame_stream(stream, offset, L)
    if len(frames) < n_avg + 1:
        raise ValueError(f"free_air_reference needs {n_avg + 1} frames after offset, got {len(frames)}")
    g = frames[1:n_avg + 1].astype(float).mean(axis=0)
    TX = fft.rfft(tx, L)
    TX2 = np.abs(TX) ** 2
    mask = _band_mask(fft.rfftfreq(L, 1.0 / fs), band)
    e = fft.irfft(fft.rfft(g, L) * np.conj(TX) / (TX2 + REF_DECONV_EPS * TX2.max()) * mask, L)
    G = int(np.floor(REF_GATE_M * fs / C_SIM))
    w = signal.windows.tukey(2 * G + 1, REF_TUKEY_ALPHA)
    gate = np.zeros(L)
    lags = np.arange(-G, G + 1)
    gate[lags % L] = w
    return fft.rfft(e * gate, L) * TX


def cfmcw_profiles_ref(frames: np.ndarray, tx: np.ndarray, ref: np.ndarray,
                       band: tuple[float, float], fs: int, n_bins: int,
                       eps: float = 1e-3) -> np.ndarray:
    """Reference-normalized signed profiles [T, n_bins] float32 (D-042); the model input.

    q = irfft(FR * |TX|^2 * conj(ref) / (|ref|^2 + eps * max_band|ref|^2) * mask, L)[:n_bins],
    FR = rfft(frame, L) (circular). The device response cancels and the direct path sits at
    lag 0. No Butterworth: the band mask limits the band.
    """
    L = len(tx)
    freqs = fft.rfftfreq(L, 1.0 / fs)
    mask = _band_mask(freqs, band)
    TX2 = np.abs(fft.rfft(tx, L)) ** 2
    R2 = np.abs(ref) ** 2
    inband = (freqs >= band[0]) & (freqs <= band[1])
    filt = TX2 * np.conj(ref) / (R2 + eps * R2[inband].max()) * mask
    FR = fft.rfft(np.asarray(frames, dtype=float), L, axis=1)
    return fft.irfft(FR * filt, L, axis=1)[:, :n_bins].astype(np.float32)


def magnitude(profiles: np.ndarray) -> np.ndarray:
    """|hilbert(profile)| along bins. Display and the Protocol B peak-bin metric; never model input."""
    return np.abs(signal.hilbert(profiles, axis=-1)).astype(np.float32)


def differential(profiles: np.ndarray) -> np.ndarray:
    """profiles[1:] - profiles[:-1], [T-1, n_bins]."""
    return profiles[1:] - profiles[:-1]
