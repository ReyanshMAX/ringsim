"""Protocol A analysis on synthetic sweep captures (gain per condition, no hardware)."""
import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.proc import export_sweep_wav
from ringsim.validation.protocol_a import protocol_a, write_protocol_a_report

FS = 48000
N = FS * 2


@pytest.fixture
def sweep_path(tmp_path):
    p = str(tmp_path / "sweep.wav")
    export_sweep_wav(16000.0, 24000.0, 0.5, FS, p)
    return p


class Rig:
    def __init__(self, tmp_path, sweep_path):
        self.dir = tmp_path / "ses"
        self.dir.mkdir()
        self.sw = wavfile.read(sweep_path)[1].astype(np.float64)
        self.rng = np.random.default_rng(0)

    def take(self, stem, gain, delay=None, noise=1e-5):
        delay = float(self.rng.integers(50, 800)) if delay is None else delay
        x = np.zeros(N)
        x[:len(self.sw)] = self.sw
        f = np.fft.rfftfreq(N, 1 / FS)
        y = np.fft.irfft(np.fft.rfft(x) * gain * np.exp(-2j * np.pi * f * delay / FS), N)
        y = y + noise * self.rng.standard_normal(N)
        wavfile.write(str(self.dir / f"{stem}.wav"), FS, y.astype(np.float32))

    def step0(self, ch, ear, grille, k=2):
        for t in range(1, k + 1):
            self.take(f"protA0_{ch}_open_t{t}", 0.5)
            self.take(f"protA0_{ch}_ear_t{t}", 0.5 * ear)
            self.take(f"protA0_{ch}_grille_t{t}", 0.5 * grille)


def _full(rig):
    rig.step0("left", ear=0.9, grille=0.02)           # grille -34 dB, ear -0.9 dB: bottom speaker
    rig.step0("right", ear=0.02, grille=0.9)          # earpiece side
    for t in (1, 2, 3):
        rig.take(f"protA1_sweep_t{t}", 0.5)
    rig.take("protA2_mic2_t1", 0.5 * 0.05)            # -26 dB
    rig.take("protA2_mic1grp_t1", 0.5 * 0.9)
    rig.take("protA3_noplay_t1", 0.0, noise=1e-3)


def test_full_session(tmp_path, sweep_path):
    rig = Rig(tmp_path, sweep_path)
    _full(rig)
    r = protocol_a(str(rig.dir), sweep_path)
    assert r["tx_channel"] == "left" and r["step0_verdict"] == "ok"
    assert r["step0"]["left"]["drop_grille_db"] > 25 and abs(r["step0"]["left"]["drop_ear_db"]) < 3
    assert r["step0"]["right"]["drop_ear_db"] > 20 and not r["step0"]["right"]["bottom_only"]
    assert r["step1"]["phase_residual_max_rad"] < 0.1
    assert r["step2"]["bottom_mic"] == "mic2" and r["step2"]["drop_db"]["mic2"] > 10
    assert r["step3"]["pass"] and r["step3"]["margin_db"] >= 20
    rep = tmp_path / "report.md"
    write_protocol_a_report(r, str(rep))
    txt = rep.read_text(encoding="utf-8")
    assert "txChannel = `left`" in txt and "bottom mic: **mic2**" in txt and "pass" in txt


def test_blocked_and_ambiguous(tmp_path, sweep_path):
    rig = Rig(tmp_path, sweep_path)
    rig.step0("left", ear=0.02, grille=0.9)            # earpiece-only on both channels
    rig.step0("right", ear=0.02, grille=0.9)
    r = protocol_a(str(rig.dir), sweep_path)
    assert r["tx_channel"] is None and r["step0_verdict"].startswith("blocked")
    rig2 = Rig(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, sweep_path)
    rig2.step0("left", ear=0.9, grille=0.02)
    rig2.step0("right", ear=0.9, grille=0.02)
    assert protocol_a(str(rig2.dir), sweep_path)["step0_verdict"].startswith("ambiguous")


def test_floor_fails_when_band_is_weak(tmp_path, sweep_path):
    rig = Rig(tmp_path, sweep_path)
    for t in (1, 2, 3):
        rig.take(f"protA1_sweep_t{t}", 0.001, noise=1e-3)
    rig.take("protA3_noplay_t1", 0.0, noise=1e-3)
    r = protocol_a(str(rig.dir), sweep_path)
    assert not r["step3"]["pass"]
    assert r["step0"] == {"status": "no takes"} and r["step2"] == {"status": "no takes"}


def test_unknown_stem_raises(tmp_path, sweep_path):
    rig = Rig(tmp_path, sweep_path)
    rig.take("protA9_what_t1", 0.5)
    with pytest.raises(ValueError, match="unrecognised"):
        protocol_a(str(rig.dir), sweep_path)


def test_mirrored_speaker(tmp_path, sweep_path):
    rig = Rig(tmp_path, sweep_path)
    for ch in ("left", "right"):
        rig.step0(ch, ear=0.02, grille=0.9)              # no channel qualifies with the left grille taped
    for t in (1, 2):
        rig.take(f"protA0_left_rgrille_t{t}", 0.5 * 0.9)  # left channel: ear taped silences (earpiece), rgrille not
        rig.take(f"protA0_right_rgrille_t{t}", 0.5 * 0.02)
    # right channel: ear drop 34 dB would disqualify; make right's ear take loud so only rgrille silences it
    for t in (1, 2):
        rig.take(f"protA0_right_ear_t{t}", 0.5 * 0.9)
    r = protocol_a(str(rig.dir), sweep_path)
    assert r["step0_verdict"].startswith("mirrored") and r["tx_channel"] == "right"
    assert r["step0"]["right"]["mirrored"] and not r["step0"]["left"]["mirrored"]


def test_noise_floor_needs_no_sweep_alignment(tmp_path, sweep_path):
    from ringsim.validation.sweep import noise_floor_response
    wavfile.write(str(tmp_path / "n.wav"), FS, (1e-3 * np.random.default_rng(1).standard_normal(N)).astype(np.float32))
    f, mag = noise_floor_response(sweep_path, [str(tmp_path / "n.wav")])
    band = (f >= 18000) & (f <= 22000)
    assert np.all(np.isfinite(mag)) and np.mean(mag[band]) < -40
    with pytest.raises(ValueError):
        noise_floor_response(sweep_path, [])
