"""Load a real capture (CAPTURE_APP.md): the .wav the app saved plus its <take>.json metadata.

`stream` is the recording exactly as saved: mono float32 at 48 kHz, unaligned and unresampled. Alignment to the tx
frame is `ringsim.proc`'s job, done per recording. A file in any other format or rate is refused, never converted
(PROCESSING.md Notes, CAPTURE_APP.md).

Only the .wav form is implemented. The directory form (a whole session) waits on a user answer.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import wavfile

SCHEMA_VERSION = 1
SAMPLE_RATE = 48000
META_KEYS = ("schema_version", "take", "started_utc", "duration_s", "tx_file", "tx_channel", "tx_sha256",
             "route", "mic_source", "sample_rate", "io_buffer_s", "input_latency_s", "output_latency_s",
             "data_source_name", "polar_pattern", "ios_version", "device_model", "label")


def read_wav(path: str) -> np.ndarray:
    """Mono float32 48 kHz samples of a WAV, exactly as stored; any other format or rate is refused."""
    fs, data = wavfile.read(path)
    if data.dtype != np.float32:
        raise ValueError(f"{path}: WAV must be float32, got {data.dtype}")
    if data.ndim != 1:
        raise ValueError(f"{path}: WAV must be mono, got shape {data.shape}")
    if fs != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {fs} Hz, only {SAMPLE_RATE} Hz is accepted (never resampled)")
    return data


def load_session(path: str) -> dict:
    """{session: session.json dict, free_air: sorted free_air_*.wav paths, takes: sorted other .wav paths}
    for a session directory (VALIDATION.md "Capture file layout")."""
    jpath = os.path.join(path, "session.json")
    if not os.path.isfile(jpath):
        raise FileNotFoundError(f"session.json missing: {jpath}")
    with open(jpath, encoding="utf-8") as f:
        session = json.load(f)
    wavs = sorted(f for f in os.listdir(path) if f.lower().endswith(".wav"))
    free = [os.path.join(path, f) for f in wavs if f.startswith("free_air_")]
    takes = [os.path.join(path, f) for f in wavs if not f.startswith("free_air_")]
    return {"session": session, "free_air": free, "takes": takes}


def load_capture(path: str) -> tuple[np.ndarray, dict]:
    """(stream float32 [N], meta dict) for `<take>.wav` with `<take>.json` beside it."""
    if os.path.isdir(path):
        raise NotImplementedError("load_capture on a session directory is not specified yet (held for a user answer)")
    if not path.lower().endswith(".wav"):
        raise ValueError(f"expected a .wav file, got {path!r}")
    data = read_wav(path)
    fs = SAMPLE_RATE
    jpath = os.path.splitext(path)[0] + ".json"
    if not os.path.isfile(jpath):
        raise FileNotFoundError(f"metadata file missing: {jpath}")
    with open(jpath, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{jpath}: schema_version {meta.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{jpath}: missing keys {missing}")
    if meta["sample_rate"] != fs:
        raise ValueError(f"{jpath}: sample_rate {meta['sample_rate']} disagrees with the WAV's {fs}")
    spath = os.path.join(os.path.dirname(os.path.abspath(path)), "session.json")
    session = None
    if os.path.isfile(spath):
        with open(spath, encoding="utf-8") as f:
            session = json.load(f)
    meta["session"] = session                      # session.json beside the take, None if absent (D-074 ruling)
    return data, meta
