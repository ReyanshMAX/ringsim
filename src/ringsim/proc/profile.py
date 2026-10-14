"""C-FMCW echo profiles and differential profiles (PROCESSING.md).

Follows Ring-a-Pose / ActSonic (Yu et al., IMWUT 2024, §3.4): cross-correlation of tx
with the 5th-order Butterworth band-passed rx, direct path at lag 0.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, signal

BUTTER_ORDER = 5


def bandpass_sos(band: tuple[float, float], fs: int) -> np.ndarray:
    """Butterworth order 5, edges = chirp band (Ring-a-Pose §3.4.2)."""
    return signal.butter(BUTTER_ORDER, list(band), btype="bandpass", fs=fs, output="sos")


def align_direct_path(stream: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                      fs: int, search_frames: int = 2) -> int:
    """Index of the direct path (D-019).

    Band-pass the stream, cross-correlate its first search_frames*len(tx) samples with
    tx, and return the EARLIEST local maximum of the correlation envelope
    |hilbert(xcorr)| that is >= 0.5 * its max. The envelope avoids picking a carrier
    ripple on the main lobe's leading edge.
    """
    seg = signal.sosfilt(bandpass_sos(band, fs), stream)[: search_frames * len(tx)]
    xc = signal.correlate(seg, tx, mode="valid", method="fft")
    env = np.abs(signal.hilbert(xc))
    pk, _ = signal.find_peaks(np.concatenate([[0.0], env, [0.0]]))
    pk = pk - 1
    return int(pk[env[pk] >= 0.5 * env.max()][0])


def frame_stream(stream: np.ndarray, offset: int, frame_len: int) -> np.ndarray:
    """[T, frame_len], T = floor((len(stream)-offset)/frame_len); tail dropped."""
    T = (len(stream) - offset) // frame_len
    return np.asarray(stream[offset: offset + T * frame_len]).reshape(T, frame_len)


def cfmcw_profiles(frames: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                   fs: int, n_bins: int) -> np.ndarray:
    """Signed correlation profiles [T, n_bins], lag 0 = direct path after alignment.

    Correlation-based; Ring-a-Pose parity unverified (Q-002).
    Per frame: causal Butterworth band-pass (sosfilt), then
    corr[j] = sum_n frame[n + j] * tx[n], j = 0..n_bins-1 (linear: samples past the
    frame end count as 0).
    """
    sos = bandpass_sos(band, fs)
    fr = signal.sosfilt(sos, frames, axis=1)
    L = frames.shape[1]
    N = fft.next_fast_len(L + len(tx), real=True)
    c = fft.irfft(fft.rfft(fr, N, axis=1) * np.conj(fft.rfft(tx, N)), N, axis=1)
    return c[:, :n_bins].astype(np.float32)


def magnitude(profiles: np.ndarray) -> np.ndarray:
    """|hilbert(profile)| along bins. Display only."""
    return np.abs(signal.hilbert(profiles, axis=-1)).astype(np.float32)


def differential(profiles: np.ndarray) -> np.ndarray:
    """profiles[1:] - profiles[:-1], [T-1, n_bins]."""
    return profiles[1:] - profiles[:-1]
