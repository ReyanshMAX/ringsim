"""Protocol B glue (VALIDATION.md, D-070, D-078, D-080): sim directory + session directory -> `plate_metrics` inputs
through `ringsim.proc`, then the report numbers. `ringsim validate plate` calls `plate_report`.

Sim side: every *.h5 in `sim_dir`; sequence "plate" (distance from the job's placement.plate.distance_mm) and one
sequence "background". Read with the session's temp_c; the sim background is subtracted from every sim plate profile.
Real side: per recording, `align_direct_path` on its own offset, frame 0 dropped; REF from free_air_<lowest n>;
the mean over all background takes' frames is subtracted from every plate take (referenced and raw alike).
Take stems (D-080): plate_<mm>mm_t<k>, background_t<k>, free_air_<n>. Any other stem raises.
"""
from __future__ import annotations

import glob
import os
import re

import h5py
import numpy as np
import yaml

from ringsim.dataset.reader import N_AVG, list_sequences, load_sequence
from ringsim.devices.config import load_device_yaml
from ringsim.proc import (align_direct_path, cfmcw_profiles, cfmcw_profiles_ref, frame_stream, free_air_reference,
                          tx_frame)
from ringsim.validation.metrics import free_air_drift, noise_floor, plate_metrics
from ringsim.validation.real import SAMPLE_RATE, load_capture, load_session

PLATE_RE = re.compile(r"^plate_(\d+)mm_t(\d+)$")
BACKGROUND_RE = re.compile(r"^background_t(\d+)$")
FREE_AIR_RE = re.compile(r"^free_air_(\d+)$")


def _sim_profiles(sim_dir: str, temp_c: float) -> tuple[dict, dict, object]:
    """({mode: {distance_mm: [T, n_bins]}} background-subtracted, {mode: background mean [n_bins]}, device)."""
    paths = sorted(glob.glob(os.path.join(sim_dir, "*.h5")))
    if not paths:
        raise FileNotFoundError(f"no .h5 shards in {sim_dir}")
    plates: dict[str, dict[int, np.ndarray]] = {"ref": {}, "raw": {}}
    bg: dict[str, np.ndarray] = {}
    dev = None
    for p in paths:
        seqs = list_sequences(p, include_failed=True)
        with h5py.File(p, "r") as f:
            if dev is None:
                dev = load_device_yaml(str(f.attrs["device_yaml"]))
            job = yaml.safe_load(str(f.attrs["job_yaml"]))
        for sid in seqs:
            if sid == "background":
                for m in plates:
                    bg[m] = load_sequence(p, sid, None, temp_c, None, mode=m).profiles.astype(np.float64).mean(axis=0)
            elif sid == "plate":
                d = int(round(float(job["placement"]["plate"]["distance_mm"])))
                for m in plates:
                    plates[m][d] = load_sequence(p, sid, None, temp_c, None, mode=m).profiles.astype(np.float64)
            else:
                raise ValueError(f"{p}: unexpected sequence {sid!r} (only 'plate' and 'background')")
    if not bg:
        raise ValueError(f"no 'background' sequence in {sim_dir}")
    if not plates["ref"]:
        raise ValueError(f"no 'plate' sequence in {sim_dir}")
    sim = {m: {d: v - bg[m] for d, v in plates[m].items()} for m in plates}
    return sim, bg, dev


def _classify(session: dict) -> tuple[dict[int, list[str]], list[str], list[str]]:
    plates: dict[int, list[str]] = {}
    backgrounds: list[str] = []
    free: list[tuple[int, str]] = []
    for p in session["takes"] + session["free_air"]:
        stem = os.path.splitext(os.path.basename(p))[0]
        if m := PLATE_RE.match(stem):
            plates.setdefault(int(m.group(1)), []).append(p)
        elif BACKGROUND_RE.match(stem):
            backgrounds.append(p)
        elif m := FREE_AIR_RE.match(stem):
            free.append((int(m.group(1)), p))
        else:
            raise ValueError(f"unrecognised take {stem!r}: expected plate_<mm>mm_t<k>, background_t<k>, free_air_<n>")
    free.sort()                                        # integer n, not the string (free_air_10 after free_air_2)
    return plates, backgrounds, [p for _, p in free]


def _frames(path: str, tx: np.ndarray, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(frames [T, len(tx)] on this recording's own offset, the stream)."""
    stream, _ = load_capture(path)
    off = align_direct_path(stream, tx, band, SAMPLE_RATE)
    return frame_stream(stream, off, len(tx)), stream


def plate_inputs(sim_dir: str, session_dir: str) -> tuple[dict, dict, float, int, dict]:
    """(sim, real, temp_c, fs_hz, extras) = the plate_metrics arguments; extras = noise-floor and drift inputs."""
    session = load_session(session_dir)
    if "temp_c" not in session["session"]:
        raise ValueError("session.json has no temp_c")
    temp_c = float(session["session"]["temp_c"])
    sim, _, dev = _sim_profiles(sim_dir, temp_c)
    fs = dev.fs_hz
    if fs != SAMPLE_RATE:
        raise ValueError(f"device fs {fs} Hz; real captures are {SAMPLE_RATE} Hz only")
    band = (min(dev.chirp.f0_hz, dev.chirp.f1_hz), max(dev.chirp.f0_hz, dev.chirp.f1_hz))
    tx = tx_frame(dev.chirp, fs)
    nb = dev.n_range_bins

    plates, backgrounds, free = _classify(session)
    if not plates or not backgrounds:
        raise ValueError("session needs plate_<mm>mm_t<k> takes and background_t<k> takes")
    if len(free) < 2:
        raise ValueError("session needs a free_air start and end recording (free_air_<n>, n lowest and highest)")

    # REF from the start free-air recording, on its own offset (D-042)
    fa_start_frames, fa_start_stream = _frames(free[0], tx, band)
    off0 = align_direct_path(fa_start_stream, tx, band, fs)
    ref = free_air_reference(fa_start_stream, tx, band, fs, off0, N_AVG)

    def profiles(frames: np.ndarray, mode: str) -> np.ndarray:
        f = frames[1:]                                  # frame 0 skipped (D-070)
        if mode == "ref":
            return cfmcw_profiles_ref(f, tx, ref, band, fs, nb).astype(np.float64)
        return cfmcw_profiles(f, tx, band, fs, nb).astype(np.float64)

    bg_frames = [_frames(p, tx, band)[0] for p in backgrounds]
    bg_prof = {m: [profiles(f, m) for f in bg_frames] for m in ("ref", "raw")}
    bg_mean = {m: np.concatenate(bg_prof[m]).mean(axis=0) for m in bg_prof}

    real = {"ref": {}, "raw": {}}
    n_takes = {}
    for d, paths in sorted(plates.items()):
        fr = [_frames(p, tx, band)[0] for p in paths]
        n_takes[d] = len(paths)
        for m in real:
            real[m][d] = np.concatenate([profiles(f, m) - bg_mean[m] for f in fr])

    # drift: start and end free-air, both with the START ref, frames index >= n_avg + 1 only (D-079)
    fa_end_frames, _ = _frames(free[-1], tx, band)
    first = N_AVG + 1
    if len(fa_start_frames) <= first or len(fa_end_frames) <= first:
        raise ValueError(f"free-air recordings need more than {first} frames for the drift number")
    drift_start = cfmcw_profiles_ref(fa_start_frames[first:], tx, ref, band, fs, nb)
    drift_end = cfmcw_profiles_ref(fa_end_frames[first:], tx, ref, band, fs, nb)

    extras = {"noise_profiles": np.concatenate(bg_prof["ref"]),      # referenced, NOT background-subtracted, frame 0 gone
              "drift_start": drift_start, "drift_end": drift_end, "n_takes": n_takes, "device": dev.name}
    return sim, real, temp_c, fs, extras


def plate_report(sim_dir: str, session_dir: str) -> dict:
    """plate_metrics(...) plus temp_c, fs_hz, distances_mm, noise_floor, free_air_drift, n_takes."""
    sim, real, temp_c, fs, ex = plate_inputs(sim_dir, session_dir)
    out = plate_metrics(sim, real, fs, temp_c)
    out.update({"temp_c": temp_c, "fs_hz": fs, "distances_mm": sorted(real["ref"]), "n_takes": ex["n_takes"],
                "noise_floor": noise_floor(ex["noise_profiles"], skip_frames=0),
                "free_air_drift": free_air_drift(ex["drift_start"], ex["drift_end"])})
    return out
