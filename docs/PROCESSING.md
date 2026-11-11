# PROCESSING.md — shared signal processing (`ringsim.proc`)

## Overview

The only processing code in the project (D-008). Converts simulated IR snapshots into the
received audio stream a real device would record (D-041), and converts any received stream
(sim or real) into C-FMCW echo profiles and differential profiles. Hardware code
imports this package directly.

## Non-goals

- No imports from `ringsim.geometry`, `solver`, `dataset`, `validation`, and no third-party
  imports beyond numpy and scipy (enforced by `tests/test_proc_isolation.py`, see Required tests;
  packaging in INFRA.md, D-045).
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
C_MIN = speed_of_sound(TEMP_RANGE_C[0])         # ≈ 337.3 m/s
C_MAX = speed_of_sound(TEMP_RANGE_C[1])         # ≈ 351.9 m/s, sets the sim time window (D-044)
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
    # log sweep for transducer response (Q-011), Hann fade 10 ms each end;
    # dur_s < 20 ms (fades overlap) raises ValueError

# src/ringsim/proc/ir.py
def rx_from_ir(ir: np.ndarray, pulse: np.ndarray, fs_sim: int, tx: np.ndarray,
               band: tuple[float, float], fs_device: int, alpha: float = 1.0,
               eps: float = 1e-3) -> np.ndarray
    # Response of ONE static snapshot to the tx samples given. tx may be any length; rx_stream
    # passes a stretch of the periodic tx stream (several periods), not a single period.
    # fs_device need not divide fs_sim (D-046): q/p = fs_sim/fs_device as a reduced fraction
    # (12/1 for a 50 kHz shard read at 50 kHz, 25/2 at 48 kHz, 2000/147 at 44.1 kHz).
    # Both rates must be integers: a float such as 1/dt = 799999.9999999999 (dx 1.5 mm, 50 kHz) raises
    # ValueError rather than being truncated; the caller rounds after checking (DATASET.md reader).
    # 1. tx_up = resample_poly(tx, up=q, down=p, window=h)
    #    h = Kaiser FIR, 80 dB, 1 kHz transition centred 500 Hz below fs_device/2, designed at
    #    fs_device*q (= fs_sim*p); cached per (fs_sim, fs_device). Scratch 2026-11-10 from 600 kHz:
    #    round-trip error ≤ 4e-4 at 44.1/48/50 kHz; 443k taps at 44.1 kHz, < 0.2 s.
    #    resample_poly's default filter droops 21% at 22 kHz over the up+down round
    #    trip (48 kHz, k=12), failing the 3% amplitude test; h is flat to <0.01%
    #    up to fs_device/2 - 1 kHz. Same h in step 6.
    # 2. N = next_fast_len(max(len(tx_up) + len(ir), 2**16))
    # 3. H = R*conj(P) / (|P|^2 + eps*max|P|^2)               R=rfft(ir,N), P=rfft(pulse,N)
    # 3b. if alpha != 1: H = temperature_scale(H, freqs, alpha)   (D-037)
    # 4. H *= raised-cosine mask: 1 inside band, 1 kHz cosine taper each side, 0 outside
    # 5. y = irfft(H * rfft(tx_up, N), N)
    # 6. return resample_poly(y[:len(tx_up)+len(ir)], up=p, down=q, window=h).astype(float32)
    # Output length: ceil((len(tx_up)+len(ir))*p/q). Pulse delay is removed by the deconvolution.
    # Edge effects: resample_poly treats samples beyond the ends of tx as 0, so the first and last
    # ~20 device samples carry errors up to ~0.5 of peak. rx_stream keeps them in padding it discards.
def temperature_scale(H: np.ndarray, freqs: np.ndarray, alpha: float) -> np.ndarray
    # Exact for this model (derivation below): H_{αc}(f) = α · H_c(f/α).
    # Evaluate H at f/α by linear interpolation of real and imaginary parts on the rfft
    # grid. The grid must be dense: rx_from_ir zero-pads to N ≥ 2**16 at fs_sim, so
    # df ≤ fs_sim / 65536 (≈ 9 Hz at 576 kHz). Returns α · H(f/α); 0 where f/α > freqs[-1].
def snapshot_weights(snap_t: np.ndarray, t: np.ndarray, interp: str = "cubic") -> np.ndarray
    # W [S, len(t)]: weight of snapshot s at time t[n]; every column sums to 1 (D-041).
    # "cubic": cubic Hermite between neighbouring snapshots, tangent at snapshot i
    #   = (p[i+1] - p[i-1]) / (snap_t[i+1] - snap_t[i-1]) (one-sided at the two ends). Equals
    #   Catmull-Rom when snapshots are evenly spaced; the tangents are linear in the p's, so W is a
    #   fixed weight matrix. Outside [snap_t[0], snap_t[-1]] the nearest end snapshot is held.
    # "hold": zero-order hold of the latest snapshot with snap_t <= t (the first snapshot held
    #   before snap_t[0]) (D-065). The old frame-frozen D-005 model = the caller passes only the
    #   frame-start snapshots. Frame starts as device times (f*len(tx)/fs_device), so a sample
    #   at a frame start takes that frame's snapshot. For comparison in validation reports only;
    #   never for datasets.
    # S == 1 (static scene) → W = 1 everywhere.
def rx_stream(irs: np.ndarray, snap_t: np.ndarray, pulse: np.ndarray, fs_sim: int, tx: np.ndarray,
              band: tuple[float, float], fs_device: int, n_frames: int, alpha: float = 1.0,
              eps: float = 1e-3, interp: str = "cubic") -> np.ndarray
    # irs [S, n_t] at fs_sim; snap_t [S] seconds; frame f's chirp starts at f*len(tx)/fs_device.
    # 1. tx_stream = tx tiled from frame -1 through frame n_frames: one warm-up period before t = 0
    #    so frame 0 already holds the previous period's echo tail, as in a real continuous capture.
    # 2. per snapshot s: y_s = rx_from_ir(irs[s], pulse, fs_sim, tx_stream over the support of W[s]
    #    padded by one period each side, band, fs_device, alpha, eps); the padding absorbs the
    #    resampling edge errors and is discarded.
    # 3. out[n] = sum_s W[s, n] * y_s[n], W evaluated at device sample times (reception time).
    #    An echo takes the geometry at its reception time; the <= 2 ms flight time is ignored
    #    (<= 1.2 mm of motion at 0.6 m/s).
    # Returns out for n in [0, n_frames*len(tx) + ceil(n_t*fs_device/fs_sim)), float32.
    # Cost: each snapshot is convolved over its support (~2-4 periods), ~3x the old per-frame cost.

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
def free_air_reference(stream: np.ndarray, tx: np.ndarray, band: tuple[float, float],
                       fs: int, offset: int, n_avg: int = 50) -> np.ndarray
    # Device-only response, used to normalize profiles (D-042). stream = a capture (real) or an
    # rx_stream (sim) with no target in range; offset from align_direct_path.
    # 1. g = mean of frames 1..n_avg (frame 0 skipped) from frame_stream(stream, offset, len(tx))
    # 2. impulse estimate e = irfft(rfft(g) * conj(TX) / (|TX|^2 + 1e-3 * max|TX|^2) * mask, L),
    #    TX = rfft(tx), L = len(tx), mask = the rx_from_ir raised-cosine band mask
    # 3. gate: keep circular lags |j| <= G, G = floor(0.70 * fs / C_SIM) (102 samples ring, 97 phone:
    #    excess path < 0.70 m, i.e. reflectors closer than ~35 cm), Tukey taper (alpha 0.5), zero
    #    elsewhere. Removes room echoes; free-air captures keep everything >= 40 cm away (D-018).
    #    The device's own body (phone far end, ring band) stays in the reference, in sim and real.
    #    A narrow gate (~2 main lobes, 25 samples) truncates the band-limited impulse and costs 8-17 %
    #    error (scratch check 2026-11-10); 100 samples is within 0.6 % of no gate.
    # 4. return REF = rfft(gated e, L) * TX   (complex [L//2+1]; = device response × direct path × TX)
def cfmcw_profiles_ref(frames: np.ndarray, tx: np.ndarray, ref: np.ndarray,
                       band: tuple[float, float], fs: int, n_bins: int,
                       eps: float = 1e-3) -> np.ndarray
    # Reference-normalized C-FMCW profiles (D-042); the model input in this project.
    # per frame: FR = rfft(frame, L) (circular: frames of a steady periodic stream),
    #   q = irfft(FR * |TX|^2 * conj(ref) / (|ref|^2 + eps * max_band|ref|^2) * mask, L)[:n_bins]
    # ≈ irfft(H_scene / H_direct * |TX|^2): the device response (phase and magnitude) cancels,
    # the direct path sits exactly at lag 0. Returns SIGNED float32 [T, n_bins]. No Butterworth:
    # the band mask does the band limiting. Same `ref` for every frame of a sequence / session.
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
19.5–24.4 kHz, inside the 16–28 kHz pulse band); and the IR window covers the hottest
air: H(f/α) needs the 343 m/s IR up to α times the read window, and α > 1 when it is
warmer than 20 °C (t_end uses C_MAX, D-044).

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

Temperature and motion are independent: `alpha` scales each snapshot's IR; snapshot times
(the pose timeline) do not change.

## Motion within a frame (D-041)

A chirp that sweeps B = 4 kHz over T = 12 ms is a slow chirp (B/T ≈ 333 kHz/s). A reflector moving
at radial speed v shifts the echo frequency by f_D = 2·v·f/c, and the cross-correlation turns a
frequency shift into a lag shift of f_D·T/B. At 22 kHz that is about 2 bins per 0.1 m/s, which a
frame-frozen sim (one pose for the whole chirp) cannot produce. k-Wave cannot move geometry during a
run, so each run is a static snapshot; `rx_stream` interpolates the snapshot responses along the
chirp. This is valid because the received signal is linear in the IR.

Accuracy is set by how far the geometry moves between snapshots: the IR at a fixed delay oscillates
in time at f_D, and the snapshots have to sample that oscillation. Scratch check 2026-11-10 (analytic
moving point reflectors, this chain): differential error ≤ 7 % when no reflector moves more than
1.8 mm between snapshots; 2 % at 1.2 mm. The threshold the generator uses is `snapshot_disp_mm` (Q-019).

## Reference-normalized profiles (D-042)

Every echo, the direct path included, passes through the same speaker and mic, so the received
spectrum is H_dev(f) · H_scene(f) · TX(f), with H_dev unknown (transducer magnitude and phase,
amplifier, and in the sim k-Wave's mass-source derivative). Signed profiles depend on H_dev's
phase: a 90° carrier phase change takes the sim-vs-real correlation of a signed profile to about 0
(audit scratch check 2026-11-10), while envelopes barely change. A free-air recording of the same
device measures H_dev · H_direct; dividing by it leaves H_scene / H_direct, the same quantity in sim
and real.

Conditions: the reference is taken with nothing inside the range window (real: device ≥ 40 cm from
any surface; sim: device-only snapshot); H_dev must not change between reference and data
(same session); H_direct has no near-zeros in band (the direct path dominates the gated reference).
What it does not remove: parts of H_dev that differ by direction (transducer directivity), and
paths that exist only in the real device (e.g. sound through the ring's plastic body); those stay
as residual DR (Q-021).

For a comparison against the Ring-a-Pose baseline, the training suite should run the baseline on
both raw and referenced input, so the input change is not confused with the sim-to-real method.

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
  by m·fs_device/fs_sim device samples (±1 sample), amplitude within 3% inside band, for
  fs_sim = 600 kHz and fs_device ∈ {50, 48, 44.1} kHz (D-046). tx = the 600-sample chirp with
  100 zeros on each side: a chirp that starts and ends at the array ends puts the documented
  resampling edge error into the spectrum (in-band ratio 0.91-1.04 measured 2026-11-11; padded:
  0.997-1.000).
- Linearity: `rx_from_ir(a*ir1 + b*ir2) == a*rx1 + b*rx2` to 1e-5 relative; same for `rx_stream`
  over the IR stack.
- `rx_stream` with S = 1 equals `rx_from_ir` of that IR against the tiled tx stream (steady state)
  to 1e-5 relative over frames 0..n_frames-1.
- `snapshot_weights`: columns sum to 1 to 1e-12; even spacing reproduces Catmull-Rom weights;
  `interp="hold"` gives 0/1 weights that switch at frame starts.
- `tests/test_proc_motion.py` (ring ChirpSpec, fs 50 kHz): direct path plus two point reflectors at
  50 mm and 110 mm moving radially at +v and −0.6·v. Snapshot IRs = pulse delayed by τ(t_s) at
  fs_sim (fractional delay by FFT phase). Exact stream = Σ a·tx(t − τ(t)) evaluated from the chirp
  formula. Compare differential profiles over frames 4..11 (relative L2):
  - frame-start snapshots only, v = 0.1 m/s: < 0.05 (scratch 2026-11-10: 0.02)
  - snapshots every T/2, v = 0.3 m/s: < 0.10 (scratch: 0.07)
  - `interp="hold"`, v = 0.3 m/s: > 0.5 (shows the test can see the effect; scratch: 1.28)
- `export_tx_wav` → read back → identical to `tx_frame` to float32 precision.
- `temperature_scale`: for an analytic H = exp(−2πifτ) (pure delay τ), output equals
  α·exp(−2πifτ/α) to 1e-4 relative inside 17–27 kHz; alpha = 1 is the identity.
- `tests/test_temperature_scaling.py` (marked `slow`, needs k-Wave on CPU): the
  2026-10-14 check above — 64³ grid, smoothed plate + two bars, sims at 343 and 337 m/s
  with equal dt; scaled-from-343 must match the 337 sim to < 0.5% over 17–27 kHz after
  fitting one complex scalar, and |scalar| must equal 1/α to 1%.
  Fixture (dx 2 mm, grid origin at 0, c = grid centre N·dx/2 = 64 mm, + 1 mm = 65 mm per
  axis, so no face lies on a grid point): plate 60×60×4 mm centred on c,
  bars 8×8×60 mm at c + (−30, 20, −10) mm and 8×60×8 mm at c + (24, −6, −26) mm
  (shifted 1 mm per axis 2026-11-11, user: at (25, −5, −25) all its faces fell on grid points),
  all ρ = 1000 then smoothed (D-036); speaker voxel (32, 30, 12), mic voxel (32, 33, 12).
- `tests/test_proc_reference.py`: stream = direct path + 4 echoes; real-side copies pass through 50
  random device responses (magnitude tilt up to −8 dB across the band, random constant, quadratic
  and linear phase), with matching references built by `free_air_reference` on direct-only streams.
  `cfmcw_profiles_ref` of every copy matches the unfiltered one to 1e-2 relative L2 (scratch
  2026-11-10: max 6e-3). A reference applied to its own stream gives `irfft(|TX|^2 * mask)[:n_bins]`
  (the chirp's own autocorrelation, peak at bin 0) to 1e-2 relative L2.
- Isolation test (D-045): (a) no import in `src/ringsim/proc/` other than the standard library, numpy,
  scipy and `ringsim.proc` itself; (b) in a fresh subprocess, `import ringsim.proc` leaves none of
  kwave, torch, smplx, trimesh, h5py, yaml, matplotlib, manifold3d in `sys.modules`.

## Notes

- `frame_len = round(frame_period_s * fs)` must be an integer number of samples;
  `load_device` asserts `abs(frame_period_s*fs - round(...)) < 1e-9`.
- Real iOS captures are float32 at 48 kHz; if `AVAudioSession` reports a different
  hardware rate the capture is rejected, never resampled (CAPTURE_APP.md).
- Real captures through proc, one recording at a time (VALIDATION.md capture layout; user answer 2026-11-11):
  1. `tx = tx_frame(spec, 48000)` from the device config's chirp (never from the WAV's `tx_file`), `frame_len = len(tx)`.
  2. Every recording, free-air and data, is aligned on its own: `offset = align_direct_path(stream, tx, band, 48000)`.
     Offsets differ between takes (the player starts a variable time after the tap). Using another recording's offset
     moves every profile by the offset difference (synthetic check, `tests/test_proc_real.py`: 5 samples moved the peak
     from bin 31 to 6).
  3. `ref = free_air_reference(free_air_stream, tx, band, 48000, offset_free)`: needs `n_avg + 1` frames after that
     recording's offset (ValueError otherwise); a 5 s free-air file has about 400. One reference at session start and one
     at the end; the two are compared for drift.
  4. `frames = frame_stream(stream, offset_data, len(tx))`, then `cfmcw_profiles_ref(frames, tx, ref, band, 48000, n_bins)`
     (and `cfmcw_profiles` for the raw comparison). Real data takes skip frame 0, as `free_air_reference` does: frame 0
     has no echo tail from a previous period. In the synthetic check frame 0 keeps the plate peak bin but differs from
     the steady-state profile by 0.44 relative L2.
  5. `align_direct_path` takes the earliest envelope peak >= 0.5 of the maximum. Synthetic check: stable (constant
     offset minus start sample) while the target echo is up to 0.6x the direct path, 1 sample later at 1.0-1.5x, and it
     locks onto the echo at 3x. A target echo stronger than the direct path is not expected from a phone at >= 8 cm.
