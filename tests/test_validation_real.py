"""load_capture on synthetic WAV + JSON (no hardware)."""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.validation.real import META_KEYS, load_capture


def _meta(**kw):
    m = dict.fromkeys(META_KEYS)
    m.update(schema_version=1, sample_rate=48000, take="t")
    m.update(kw)
    return m


def _write(tmp_path, data, fs=48000, meta=None, name="t"):
    wavfile.write(str(tmp_path / f"{name}.wav"), fs, data)
    if meta is not None:
        (tmp_path / f"{name}.json").write_text(json.dumps(meta))
    return str(tmp_path / f"{name}.wav")


def test_roundtrip_exact(tmp_path):
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    s, m = load_capture(_write(tmp_path, x, meta=_meta(label="hi")))
    assert s.dtype == np.float32 and np.array_equal(s, x) and m["label"] == "hi"


@pytest.mark.parametrize("data, fs, match", [
    (np.zeros(10, np.float64), 48000, "float32"),
    (np.zeros(10, np.int16), 48000, "float32"),
    (np.zeros((10, 2), np.float32), 48000, "mono"),
    (np.zeros(10, np.float32), 44100, "44100"),
])
def test_bad_format_refused(tmp_path, data, fs, match):
    with pytest.raises(ValueError, match=match):
        load_capture(_write(tmp_path, data, fs=fs, meta=_meta()))


def test_metadata_checks(tmp_path):
    x = np.zeros(10, np.float32)
    with pytest.raises(FileNotFoundError):
        load_capture(_write(tmp_path, x, meta=None))
    with pytest.raises(ValueError, match="schema_version"):
        load_capture(_write(tmp_path, x, meta=_meta(schema_version=2), name="a"))
    bad = _meta(); del bad["route"]
    with pytest.raises(ValueError, match="route"):
        load_capture(_write(tmp_path, x, meta=bad, name="b"))
    with pytest.raises(ValueError, match="disagrees"):
        load_capture(_write(tmp_path, x, meta=_meta(sample_rate=44100), name="c"))


def test_directory_form_not_built(tmp_path):
    with pytest.raises(NotImplementedError):
        load_capture(str(tmp_path))
