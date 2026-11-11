"""Phase 1 free-field check (BUILD.md Phase 1, D-057, D-060).

A free scene with a point speaker and point mic has an ideal monopole transfer function
    H(f) = j*2*pi*f * exp(-2*pi*i*f*tau),  tau = d / C_SIM,  d = speaker-mic distance (0.100 m for test_pair)
(k-Wave's mass source makes the direct path a time derivative: the j*2*pi*f factor, ~90 degrees of phase).
The reference IR is `irfft(rfft(pulse) * H)`; it goes through exactly the reader's IR -> profile chain
(`dataset.reader.profiles_from_arrays`), so sim and reference differ only in what k-Wave actually produced.

Gate (D-057): mode="raw" profiles. mode="ref" is computed and reported, never gating: a free scene referenced to itself
gives the chirp autocorrelation, so that comparison passes trivially.

Comparison: profiles are real; "one complex gain" is applied to their analytic signals (scipy.signal.hilbert along
bins): g = <B, A> / <B, B>, rel L2 = ||A - g B|| / ||A||. This reading of BUILD.md is posted to the lead.
"""
from __future__ import annotations

import h5py
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import hilbert

PEAK_BIN_TOL = 1          # BUILD.md: envelope peak within 1 bin of the reference's
REL_L2_MAX = 0.10         # BUILD.md: rel L2 < 10 % after one complex gain
ARRIVAL_TOL_SAMPLES = 2   # BUILD.md: within 2 sim samples
IR_REL_L2_MAX = 1e-3      # BUILD.md: CPU vs T4 IR


def monopole_ir(pulse: np.ndarray, fs_sim: float, tau_s: float) -> np.ndarray:
    """IR an ideal monopole at delay tau would record for `pulse`: irfft(rfft(pulse) * H), length len(pulse)."""
    n_t = len(pulse)
    N = next_fast_len(2 * n_t)
    f = np.fft.rfftfreq(N, 1.0 / fs_sim)
    H = 1j * 2 * np.pi * f * np.exp(-2j * np.pi * f * tau_s)
    return irfft(rfft(pulse.astype(np.float64), N) * H, N)[:n_t].astype(np.float32)


def _complex_gain_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, complex]:
    """rel L2 of analytic(a) against g * analytic(b), g the least-squares complex gain."""
    A, B = hilbert(a), hilbert(b)
    g = np.vdot(B, A) / np.vdot(B, B)
    return float(np.linalg.norm(A - g * B) / np.linalg.norm(A)), complex(g)


def _peak_bin(profile: np.ndarray) -> int:
    return int(np.argmax(np.abs(hilbert(profile))))


def shard_has_ref_ir(path: str) -> bool:
    """BUILD.md: every sequence (failed ones excluded) has ref_ir."""
    with h5py.File(path, "r") as f:
        seqs = [g for g in f["seq"].values() if not bool(g.attrs["failed"])]
        return len(seqs) > 0 and all("ref_ir" in g for g in seqs)


def freefield_check(path: str, seq_id: str, dev_override=None) -> dict:
    from ringsim.dataset.reader import load_sequence, profiles_from_arrays
    from ringsim.devices.config import load_device_yaml
    from ringsim.proc import physics
    from ringsim.solver.constants import PULSE_DELAY

    with h5py.File(path, "r") as f:
        g = f["seq"][seq_id]
        shard_dev = load_device_yaml(str(f.attrs["device_yaml"]))
        fs_sim = float(int(f.attrs["k_decim"]) * int(f.attrs["fs_device"]))
        pulse = f["pulse"][:]
        ir0 = g["ir"][0]
        ref_ir = g["ref_ir"][:] if "ref_ir" in g else None
        if g["ir"].shape[0] != 1:
            raise ValueError("freefield_check is for static (S = 1) free scenes")

    dev = dev_override if dev_override is not None else shard_dev
    pts = {p.name: np.asarray(p.points_mm, dtype=float) for p in shard_dev.ports}
    spk, mic = shard_dev.active_pair
    d = float(np.linalg.norm(pts[spk].mean(axis=0) - pts[mic].mean(axis=0))) / 1000.0
    tau = d / physics.C_SIM

    out: dict = {"seq_id": seq_id, "device": dev.name, "fs_hz": dev.fs_hz, "distance_m": d}
    mono = monopole_ir(pulse, fs_sim, tau)
    for mode in ("raw", "ref"):
        s = load_sequence(path, seq_id, None, None, None, mode=mode, dev_override=dev_override)
        ref_p, _ = profiles_from_arrays(mono[None, :], np.zeros(1), pulse, fs_sim, mono if mode == "ref" else None,
                                        dev, s.profiles.shape[0], 1.0, mode)
        rel, gain = _complex_gain_fit(s.profiles[0], ref_p[0])
        out[mode] = {"peak_bin_sim": _peak_bin(s.profiles[0]), "peak_bin_ref": _peak_bin(ref_p[0]),
                     "rel_l2": rel, "gain_abs": abs(gain), "gain_phase_rad": float(np.angle(gain))}
        out[mode]["peak_bin_diff"] = abs(out[mode]["peak_bin_sim"] - out[mode]["peak_bin_ref"])

    # arrival: envelope peak of the recorded IR vs (d/c + PULSE_DELAY), in sim samples
    env = np.abs(hilbert(ir0.astype(np.float64)))
    arrival = int(np.argmax(env))
    expected = (tau + PULSE_DELAY) * fs_sim
    out["arrival_err_samples"] = float(arrival - expected)
    out["has_ref_ir"] = ref_ir is not None
    raw = out["raw"]
    out["pass_peak"] = raw["peak_bin_diff"] <= PEAK_BIN_TOL
    out["pass_l2"] = raw["rel_l2"] < REL_L2_MAX
    out["pass_arrival"] = abs(out["arrival_err_samples"]) <= ARRIVAL_TOL_SAMPLES
    out["pass"] = bool(out["pass_peak"] and out["pass_l2"] and out["pass_arrival"] and out["has_ref_ir"])
    return out


def compare_ir(path_a: str, path_b: str, seq_id: str) -> float:
    """Relative L2 of the two shards' IRs, ||a - b|| / ||a||; the gate is IR_REL_L2_MAX (BUILD.md, CPU vs T4)."""
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        a = fa["seq"][seq_id]["ir"][:].astype(np.float64)
        b = fb["seq"][seq_id]["ir"][:].astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"IR shapes differ: {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))
