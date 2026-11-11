"""ringsim.proc.physics + chirp (PROCESSING.md Required tests: export_tx_wav round trip)."""
import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.proc import (C_MAX, C_MIN, C_SIM, TEMP_RANGE_C, ChirpSpec, export_sweep_wav,
                          export_tx_wav, speed_of_sound, synth_chirp, temperature_alpha, tx_frame)

RING = ChirpSpec(20000, 24000, 0.012, 0.012, "none")     # ring.yaml timing; ring window is TBD(Q-002)
PHONE = ChirpSpec(18000, 22000, 0.0125, 0.0125, "none")  # iphone17pro.yaml / test_pair.yaml


def test_physics_constants():
    assert C_SIM == 343.0 and TEMP_RANGE_C == (10.0, 35.0)
    assert C_MIN == pytest.approx(337.3, abs=0.05) and C_MAX == pytest.approx(351.9, abs=0.05)
    assert temperature_alpha(20.0) == pytest.approx(speed_of_sound(20.0) / C_SIM)
    with pytest.raises(AssertionError):
        temperature_alpha(40.0)


def test_chirp_shape_and_peak():
    c = synth_chirp(PHONE, 48000)
    assert c.dtype == np.float32 and len(c) == 600 and np.max(np.abs(c)) == pytest.approx(1.0)
    f = tx_frame(RING, 50000)
    assert len(f) == 600 and f.dtype == np.float32


def test_unknown_window_refused():
    with pytest.raises(NotImplementedError, match="Q-002"):
        synth_chirp(ChirpSpec(20000, 24000, 0.012, 0.012, "hann"), 50000)


def test_tx_frame_zero_padded():
    spec = ChirpSpec(18000, 22000, 0.010, 0.0125, "none")
    f = tx_frame(spec, 48000)
    assert len(f) == 600 and np.all(f[480:] == 0) and np.array_equal(f[:480], synth_chirp(spec, 48000))


def test_export_tx_wav_round_trip(tmp_path):
    p = tmp_path / "tx.wav"
    export_tx_wav(PHONE, 48000, str(p))
    fs, x = wavfile.read(p)
    assert fs == 48000 and x.dtype == np.float32
    assert np.array_equal(x, tx_frame(PHONE, 48000))


def test_export_sweep_wav(tmp_path):
    p = tmp_path / "sweep.wav"
    export_sweep_wav(100.0, 24000.0, 1.0, 48000, str(p))
    fs, x = wavfile.read(p)
    assert fs == 48000 and x.dtype == np.float32 and len(x) == 48000
    assert abs(x[0]) < 1e-6 and abs(x[-1]) < 1e-3
