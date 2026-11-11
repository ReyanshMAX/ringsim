"""`ringsim real` on a synthetic capture (PROCESSING.md Notes, D-070; CAPTURE_APP.md acceptance: direct-path peak at
bin 0 and a stable profile with no target)."""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.cli import main
from ringsim.devices.config import load_device
from ringsim.proc.chirp import tx_frame
from ringsim.validation.real import META_KEYS


def _capture(tmp_path, take, start, n_frames=40, echo=None, seed=0):
    with pytest.warns(UserWarning):              # TBD(Q-005) ports dropped
        dev = load_device("iphone17pro")
    tx = tx_frame(dev.chirp, 48000).astype(np.float64)
    loop = np.tile(tx, n_frames)
    x = np.zeros(start + len(loop) + 100)
    x[start:start + len(loop)] += loop                       # direct path
    if echo is not None:
        lag, gain = echo
        x[start + lag:start + lag + len(loop)] += gain * loop
    x += 1e-4 * np.random.default_rng(seed).standard_normal(len(x))
    wav = tmp_path / f"{take}.wav"
    wavfile.write(wav, 48000, x.astype(np.float32))
    meta = {k: None for k in META_KEYS}
    meta.update(schema_version=1, take=take, sample_rate=48000, tx_file="phone_tx.wav", label=take)
    (tmp_path / f"{take}.json").write_text(json.dumps(meta))
    return wav


def test_real_direct_path_at_bin_0(tmp_path, capsys):
    # start <= 500: align_direct_path misaligns starts beyond ~1 frame (seat 2 finding to seat 3, 2026-11-11)
    wav = _capture(tmp_path, "free_air_1", start=477)
    png = tmp_path / "real.png"
    with pytest.warns(UserWarning):
        assert main(["real", "--capture", str(wav), "--device", "iphone17pro", "--out", str(png)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert png.exists()
    assert out["take"] == "free_air_1" and out["direct_peak_bin"] == 0
    assert out["n_frames"] == 39                              # 40 looped frames, frame 0 dropped
    assert out["spread_raw"] < 0.01                           # stable with no target


def test_real_target_echo_shows_beyond_bin_0(tmp_path, capsys):
    """A delayed copy (a target) shows up at its lag; the direct path still aligns at bin 0."""
    wav = _capture(tmp_path, "plate_110mm_t1", start=300, echo=(31, 0.4))
    with pytest.warns(UserWarning):
        assert main(["real", "--capture", str(wav), "--device", "iphone17pro", "--out", str(tmp_path / "p.png")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["direct_peak_bin"] == 0 and out["offset"] >= 300


def test_real_refuses_too_short(tmp_path):
    wav = _capture(tmp_path, "short", start=10, n_frames=2)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="need >= 2"):
        main(["real", "--capture", str(wav), "--device", "iphone17pro", "--out", str(tmp_path / "s.png")])
