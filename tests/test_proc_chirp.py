import numpy as np
from scipy.io import wavfile

from conftest import FS, TEST_CHIRP
from ringsim.proc import export_sweep_wav, export_tx_wav, synth_chirp, tx_frame


def test_synth_chirp_shape_peak():
    c = synth_chirp(TEST_CHIRP, FS)
    assert c.dtype == np.float32 and len(c) == 600
    assert np.isclose(np.abs(c).max(), 1.0)


def test_tx_frame_padding():
    from dataclasses import replace
    spec = replace(TEST_CHIRP, frame_period_s=0.02)
    f = tx_frame(spec, FS)
    assert len(f) == 960 and np.all(f[600:] == 0)


def test_export_tx_wav_roundtrip(tmp_path):
    p = tmp_path / "tx.wav"
    export_tx_wav(TEST_CHIRP, FS, str(p))
    fs, x = wavfile.read(p)
    assert fs == FS and x.dtype == np.float32
    np.testing.assert_array_equal(x, tx_frame(TEST_CHIRP, FS))


def test_export_sweep_wav(tmp_path):
    p = tmp_path / "sweep.wav"
    export_sweep_wav(16000, 24000, 1.0, FS, str(p))
    fs, x = wavfile.read(p)
    assert fs == FS and len(x) == FS and abs(x[0]) < 1e-6 and abs(x[-1]) < 1e-3
