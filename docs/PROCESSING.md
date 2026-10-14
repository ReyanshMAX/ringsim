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
# src/ringsim/proc/physics.py  (solver and dataset import these from here, D-037)
C_SIM = 343.0                                   # every k-Wave sim runs at this speed
TEMP_RANGE_C = (10.0, 35.0)                     # supported read-time temperatures
def speed_of_sound(temp_c: float) -> float:
    return 331.3 * np.sqrt(1 + temp_c / 273.15)
C_MIN = speed_of_sound(TEMP_RANGE_C[0])         # ≈ 337.3 m/s, sets the sim time window
def temperature_alpha(temp_c: float) -> float:  # α = c(T) / C_SIM
    assert TEMP_RANGE_C[0] <= temp_c <= TEMP_RANGE_C[1]
    return speed_of_sound(temp_c) / C_SIM

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
               band: tuple[float, float], fs_device: int, alpha: float = 1.0,
               eps: float = 1e-3) -> np.ndarray
    # 1. tx_up = resample_poly(tx, up=k, down=1, window=h)  (fs_sim = k*fs_device)
    #    h = Kaiser FIR, 80 dB, 1 kHz transition centred 500 Hz below fs_device/2.
    #    resample_poly's default filter droops 21% at 22 kHz over the up+down round
    #    trip (48 kHz, k=12), failing the 3% amplitude test; h is flat to <0.01%
    #    up to fs_device/2 - 1 kHz. Same h in step 6.
    # 2. N = next_fast_len(max(len(tx_up) + len(ir), 2**16))
    # 3. H = R*conj(P) / (|P|^2 + eps*max|P|^2)               R=rfft(ir,N), P=rfft(pulse,N)
    # 3b. if alpha != 1: H = temperature_scale(H, freqs, alpha)   (D-037)
    # 4. H *= raised-cosine mask: 1 inside band, 1 kHz cosine taper each side, 0 outside
    # 5. y = irfft(H * rfft(tx_up, N), N)
    # 6. return resample_poly(y[:len(tx_up)+len(ir)], up=1, down=k, window=h).astype(float32)
    # Output length: ceil((len(tx_up)+len(ir))/k). Pulse delay is removed by the deconvolution.
def temperature_scale(H: np.ndarray, freqs: np.ndarray, alpha: float) -> np.ndarray
    # Exact for this model (derivation below): H_{αc}(f) = α · H_c(f/α).
    # Evaluate H at f/α by linear interpolation of real and imaginary parts on the rfft
    # grid. The grid must be dense: rx_from_ir zero-pads to N ≥ 2**16 at fs_sim, so
    # df ≤ fs_sim / 65536 (≈ 9 Hz at 576 kHz). Returns α · H(f/α); 0 where f/α > freqs[-1].
def overlap_add_stream(frame_rx: list[np.ndarray], frame_len: int) -> np.ndarray
    # frame t's response starts at t*frame_len; echoes spill into frame t+1

# src/ringsim/proc/profile.py
def align_direct_path(stream: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                      fs: int, search_frames: int = 2) -> int
    # bandpass (same filter as cfmcw_profiles, see below), xcorr first
    # search_frames*len(tx) samples with tx, env = |hilbert(xcorr)|,
    # return index of the EARLIEST local max of env >= 0.5 * max env  (direct path, D-019)
    # Envelope, not |xcorr|: local maxima of |xcorr| ripple at the carrier, and the
    # earliest one >= 0.5 max sits ~5 samples early on the main lobe's leading edge
    # (user decision 2026-10-14). band, fs added because the bandpass needs them.
def frame_stream(stream: np.ndarray, offset: int, frame_len: int) -> np.ndarray
    # [T, frame_len], T = floor((len(stream)-offset)/frame_len); drop the tail
def cfmcw_profiles(frames: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                   fs: int, n_bins: int) -> np.ndarray
    # n_bins comes from DeviceConfig.n_range_bins (ring: 54, phone: 84).
    # per frame: bandpass = Butterworth order 5, edges = chirp band (Ring-a-Pose §3.4.2),
    #   applied with sosfilt (causal) — filtfilt vs sosfilt parity is Q-002;
    # corr[j] = sum_n frame[n + j] * tx[n], j = 0..n_bins-1, via zero-padded rfft
    #   (linear: frame samples past the frame end count as 0);
    # returns SIGNED float32 [T, n_bins]. Lag 0 = direct path after alignment.
    # One bin = c/(2*fs) one-way: 3.43 mm @ 50 kHz, 3.57 mm @ 48 kHz.
def magnitude(profiles: np.ndarray) -> np.ndarray      # |hilbert(profile)| along bins, display only
def differential(profiles: np.ndarray) -> np.ndarray   # profiles[1:] - profiles[:-1], [T-1, n_bins]
```

## Temperature by time-scaling (D-037)

All sims run at `C_SIM`. With one sound speed c everywhere (D-004) the model is

  ρ₀(x) ∂ₜu = −∇p,   ∂ₜp = −ρ₀(x) c² ∇·u + c² S(x,t)

For c' = αc, the pair p'(x,t) = p(x,αt), u'(x,t) = α⁻¹u(x,αt) satisfies both equations
when S'(x,t) = α⁻¹S(x,αt). For an impulse source S = qδ(x−x_s)δ(t) this gives

  h_{αc}(x_r, t) = α² h_c(x_r, αt)   ⇔   H_{αc}(f) = α H_c(f/α)

It holds exactly when: c is uniform (D-004); ρ₀ is fixed (a uniform scaling of ρ₀
leaves p unchanged anyway); there is no absorption or dispersion (SOLVER.md non-goal; if
absorption is ever added this breaks); geometry does not move; H is sampled inside the
pulse band at f/α (TEMP_RANGE_C gives α ∈ [0.983, 1.026], so a 20–24 kHz chirp needs
19.5–24.4 kHz, inside the 16–28 kHz pulse band); and the IR window covers the slowest
air (t_end uses C_MIN).

Verified 2026-10-14 (k-wave-python 0.6.2, CPU, 64³, plate + two bars, smoothed ρ):
scaled-from-343 vs direct sims at 349 and 337 m/s differ by 0.10% and 0.03% over
17–27 kHz after a single complex gain; without scaling the error is 5.4–5.6%.
Re-measured 2026-10-14 by `tests/test_temperature_scaling.py` (fixture below): 0.035%
scaled vs 5.67% unscaled at 337 m/s, identical over 3 runs. The fitted gain g in `direct = g·temperature_scale(H_343)`
is |g| = 1/α at equal dt (1.0179 measured, 1/α = 1.0178), i.e. the direct sim equals
H_343(f/α) with unit gain; the earlier "α²" statement was wrong (k-Wave's source
normalization absorbs one α).
That gain is a per-sequence constant and is irrelevant downstream (noise SNR is relative
to the direct path; training normalizes per channel).

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
- `temperature_scale`: for an analytic H = exp(−2πifτ) (pure delay τ), output equals
  α·exp(−2πifτ/α) to 1e-4 relative inside 17–27 kHz; alpha = 1 is the identity.
- `tests/test_temperature_scaling.py` (marked `slow`, needs k-Wave on CPU): the
  2026-10-14 check above — 64³ grid, smoothed plate + two bars, sims at 343 and 337 m/s
  with equal dt; scaled-from-343 must match the 337 sim to < 0.5% over 17–27 kHz after
  fitting one complex scalar, and |scalar| must equal 1/α to 1%.
  Fixture (dx 2 mm, grid origin at 0, c = grid centre + 1 mm so no face lies on a grid
  point): plate 60×60×4 mm centred on c,
  bars 8×8×60 mm at c + (−30, 20, −10) mm and 8×60×8 mm at c + (25, −5, −25) mm,
  all ρ = 1000 then smoothed (D-036); speaker voxel (32, 30, 12), mic voxel (32, 33, 12).
- Isolation test: no forbidden imports in `src/ringsim/proc/`.

## Notes

- `frame_len = round(frame_period_s * fs)` must be an integer number of samples;
  `load_device` asserts `abs(frame_period_s*fs - round(...)) < 1e-9`.
- Real iOS captures are float32 at 48 kHz; if `AVAudioSession` reports a different
  hardware rate the capture is rejected, never resampled (CAPTURE_APP.md).
