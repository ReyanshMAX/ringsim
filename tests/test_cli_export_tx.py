"""`ringsim export-tx` (D-077): one call writes the device tx frame and the Protocol A sweep."""
import hashlib

import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.cli import main
from ringsim.devices.config import load_device
from ringsim.proc.chirp import export_sweep_wav, export_tx_wav, tx_frame


def test_export_tx_writes_both_files(tmp_path):
    with pytest.warns(UserWarning):          # TBD(Q-005) ports outside active_pair are dropped
        assert main(["export-tx", "--device", "iphone17pro", "--out", str(tmp_path / "tx")]) == 0
    files = sorted(p.name for p in (tmp_path / "tx").iterdir())
    assert files == ["phone_tx.wav", "sweep_16k_24k_2s.wav"]

    with pytest.warns(UserWarning):
        dev = load_device("iphone17pro")
    fs, x = wavfile.read(tmp_path / "tx" / "phone_tx.wav")
    assert fs == 48000 and x.dtype == np.float32 and x.ndim == 1 and x.shape == (600,)
    np.testing.assert_array_equal(x, tx_frame(dev.chirp, 48000))
    fs, s = wavfile.read(tmp_path / "tx" / "sweep_16k_24k_2s.wav")
    assert fs == 48000 and s.dtype == np.float32 and s.ndim == 1 and s.shape == (96000,)

    # byte-identical to proc's writers called directly (CAPTURE_APP.md: the app plays exactly the sim's signal)
    export_tx_wav(dev.chirp, 48000, str(tmp_path / "a.wav"))
    export_sweep_wav(16000, 24000, 2.0, 48000, str(tmp_path / "b.wav"))
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()   # noqa: E731
    assert h(tmp_path / "a.wav") == h(tmp_path / "tx" / "phone_tx.wav")
    assert h(tmp_path / "b.wav") == h(tmp_path / "tx" / "sweep_16k_24k_2s.wav")


def test_export_tx_refuses_unnamed_device(tmp_path):
    with pytest.raises(ValueError, match="no tx file name"):
        main(["export-tx", "--device", "test_pair", "--out", str(tmp_path)])
    assert not any(tmp_path.iterdir())
