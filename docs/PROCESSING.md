# PROCESSING.md — shared signal processing (`ringsim.proc`)

## Overview

The only processing code in the project (D-008). Converts simulated IRs into the
received audio stream a real device would record, and converts any received stream
(sim or real) into C-FMCW echo profiles and differential profiles. Hardware code
imports this package directly.

## Non-goals

- No imports from `ringsim.geometry`, `solver`, `dataset`, `validation` (enforced by
  `tests/test_proc_isolation.py`, which greps imports).
- No ML preprocessing (normalization, augmentation beyond noise). That belongs to the
  training suite.
- No device-specific branches; everything is parameterized by `ChirpSpec`, `fs`, and
  `range_window_m`. `ringsim.proc` defines its own minimal `ChirpSpec` dataclass;
  `devices.config` imports it from here, not the reverse.

## Signatures

```python
# src/ringsim/proc/chirp.py
def synth_chirp(spec: ChirpSpec, fs: int) -> np.ndarray
    # linear chirp f0→f1 over duration_s, scipy.signal.chirp(method="linear"),
    # window applied, float32, peak 1.0
def tx_frame(spec: ChirpSpec, fs: int) -> np.ndarray
    # synth_chirp zero-padded to round(frame_period_s*fs) samples
def export_tx_wav(spec: ChirpSpec, fs: int, path: str) -> None
    # one tx_frame, float32 mono WAV; the capture app loops it seamlessly
def export_sweep_wav(f0: float, f1: float, dur_s: float, fs: int, path: str) -> None
    # log sweep for transducer response (Q-011), Hann fade 10 ms each end

# src/ringsim/proc/ir.py
def rx_from_ir(ir: np.ndarray, pulse: np.ndarray, k: int, tx: np.ndarray,
               band: tuple[float, float], fs_device: int, eps: float = 1e-3) -> np.ndarray
    # 1. tx_up = resample_poly(tx, up=k, down=1)            (fs_sim = k*fs_device)
    # 2. N = next_fast_len(len(tx_up) + len(ir))
    # 3. H = R*conj(P) / (|P|^2 + eps*max|P|^2)               R=rfft(ir,N), P=rfft(pulse,N)
    # 4. H *= raised-cosine mask: 1 inside band, 1 kHz cosine taper each side, 0 outside
    # 5. y = irfft(H * rfft(tx_up, N), N)
    # 6. return resample_poly(y, up=1, down=k).astype(float32)
    # Output length: ceil((len(tx_up)+len(ir))/k). Pulse delay is removed by the deconvolution.
def overlap_add_stream(frame_rx: list[np.ndarray], frame_len: int) -> np.ndarray
    # frame t's response starts at t*frame_len; echoes spill into frame t+1

# src/ringsim/proc/profile.py
def align_direct_path(stream: np.ndarray, tx: np.ndarray, search_frames: int = 2) -> int
    # bandpass (see below), xcorr first search_frames*len(tx) samples with tx,
    # return index of the EARLIEST peak >= 0.5 * max |xcorr|  (direct path, D-019)
def frame_stream(stream: np.ndarray, offset: int, frame_len: int) -> np.ndarray
    # [T, frame_len], T = floor((len(stream)-offset)/frame_len); drop the tail
def cfmcw_profiles(frames: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                   fs: int, n_bins: int) -> np.ndarray
    # n_bins comes from DeviceConfig.n_range_bins (ring: 54, phone: 84).
    # per frame: bandpass = Butterworth order 5, edges = chirp band (Ring-a-Pose §3.4.2),
    #   applied with sosfilt (causal) — filtfilt vs sosfilt parity is Q-002;
    # corr[j] = sum_n frame[n + j] * chirp[n], j = 0..n_bins-1, via rfft;
    # returns SIGNED float32 [T, n_bins]. Lag 0 = direct path after alignment.
    # One bin = c/(2*fs) one-way: 3.43 mm @ 50 kHz, 3.57 mm @ 48 kHz.
def magnitude(profiles: np.ndarray) -> np.ndarray      # |hilbert(profile)| along bins, display only
def differential(profiles: np.ndarray) -> np.ndarray   # profiles[1:] - profiles[:-1], [T-1, n_bins]
```

Range interpretation: bin j corresponds to excess round-trip path `j*c/fs` beyond the
direct speaker→mic path, not absolute distance. Same definition in sim and real.

## Ring-a-Pose parity

Ring-a-Pose / ActSonic define C-FMCW in their own way (chirp mixing and filtering).
From the paper (§3.4): cross-correlation of tx with the 5th-order Butterworth
band-passed rx, direct path = strongest peak t0, keep 54 bins starting at t0, original
profile = stacked frames, differential = current − previous frame. The implementation
above follows that. Still unverified without their code (Q-002): chirp window, causal vs
zero-phase filtering, whether frames are taken as signed correlation or magnitude, and
whether per-frame alignment re-finds t0 every frame or once per session. Per-channel
input normalization (§3.6.2) belongs to the training suite. Until resolved, `cfmcw_profiles` docstring
must say "correlation-based; Ring-a-Pose parity unverified (Q-002)".

## Required tests (`tests/test_proc_*.py`)

- Synthetic: stream = tx delayed by d samples + 0.3·tx delayed by d+20 (both n_bins > 20) → after
  alignment, signed profile peaks at bins 0 and 20 (±1).
- `rx_from_ir` with an analytic IR (pulse delayed by m sim samples) returns tx delayed
  by m/k device samples (±1 sample), amplitude within 3% inside band.
- Linearity: `rx_from_ir(a*ir1 + b*ir2) == a*rx1 + b*rx2` to 1e-5 relative.
- `export_tx_wav` → read back → identical to `tx_frame` to float32 precision.
- Isolation test: no forbidden imports in `src/ringsim/proc/`.

## Notes

- `frame_len = round(frame_period_s * fs)` must be an integer number of samples;
  `load_device` asserts `abs(frame_period_s*fs - round(...)) < 1e-9`.
- Real iOS captures are float32 at 48 kHz; if `AVAudioSession` reports a different
  hardware rate the capture is rejected, never resampled (CAPTURE_APP.md).
