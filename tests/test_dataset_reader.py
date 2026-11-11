"""Reader on synthetic shards (no k-Wave)."""
import dataclasses

import numpy as np
import pytest

from ringsim.dataset.jobspec import job_from_dict
from ringsim.dataset.reader import SequenceFailed, _usable_frames, list_sequences, load_sequence
from ringsim.dataset.writer import ShardWriter
from ringsim.devices.config import load_device

RAW = {"job_id": "t", "device": "test_pair", "scene": "free", "dx_mm": 2.0, "placement": None,
       "poses": {"source": "free", "n_frames": 3}, "dr": None, "seed": 5}


@pytest.fixture
def shard(shard_path, synth_ir):
    dev = load_device("test_pair")
    w = ShardWriter(shard_path, job_from_dict(dict(RAW)), dev)
    res = synth_ir(n_t=2048)
    w.begin_sequence("free", {"source": "free", "n_frames": 3, "seed": 5}, None, 2048)
    w.write_snapshot("free", 0, res)
    w.write_ref_ir("free", res)
    w.begin_sequence("bad", {"source": "free", "n_frames": 3, "seed": 5}, None, 2048)
    w.fail_sequence("bad", "hand collides with phone")
    w.close()
    return shard_path


def test_list_sequences_hides_failed(shard):
    assert list_sequences(shard) == ["free"]
    assert list_sequences(shard, include_failed=True) == ["bad", "free"]


def test_failed_sequence_refused(shard):
    with pytest.raises(SequenceFailed, match="collides"):
        load_sequence(shard, "bad", None, None, None)


def test_snr_not_implemented_names_q016(shard):
    with pytest.raises(NotImplementedError, match="Q-016"):
        load_sequence(shard, "free", 20.0, None, np.random.default_rng(0))


def test_bad_mode(shard):
    with pytest.raises(ValueError):
        load_sequence(shard, "free", None, None, None, mode="both")


def test_usable_frames_partial_rule():
    fs = np.array([0, 1, 2], dtype=np.int32)       # 3 frames, S = 4 (end snapshot = 3)
    assert _usable_frames(fs, 4, 4) == 3
    assert _usable_frames(fs, 4, 3) == 2           # frame 2 needs the end snapshot
    assert _usable_frames(fs, 4, 2) == 1
    assert _usable_frames(fs, 4, 0) == 0
    assert _usable_frames(np.zeros(3, dtype=np.int32), 1, 1) == 3   # static


def test_override_raises(shard):
    dev = load_device("test_pair")
    other_ports = tuple(dataclasses.replace(p, points_mm=((0.0, 0.0, 50.0),)) if p.name == "mic" else p
                        for p in dev.ports)
    with pytest.raises(ValueError, match="ports"):
        load_sequence(shard, "free", None, None, None, mode="raw",
                      dev_override=dataclasses.replace(dev, ports=other_ports))
    with pytest.raises(ValueError, match="solver band"):
        load_sequence(shard, "free", None, None, None, mode="raw",
                      dev_override=dataclasses.replace(dev, chirp=dataclasses.replace(dev.chirp, f1_hz=30000.0)))
    with pytest.raises(ValueError, match="window"):
        load_sequence(shard, "free", None, None, None, mode="raw",
                      dev_override=dataclasses.replace(dev, n_range_bins=dev.n_range_bins + 1))


def test_missing_ref_ir_refused_in_ref_mode(shard_path, synth_ir):
    w = ShardWriter(shard_path, job_from_dict(dict(RAW)), load_device("test_pair"))
    w.begin_sequence("free", {"source": "free", "n_frames": 3, "seed": 5}, None, 2048)
    w.write_snapshot("free", 0, synth_ir(n_t=2048))
    w.close()
    with pytest.raises(ValueError, match="ref_ir"):
        load_sequence(shard_path, "free", None, None, None, mode="ref")


def test_unwritten_sequence_has_no_frames(shard_path):
    w = ShardWriter(shard_path, job_from_dict(dict(RAW)), load_device("test_pair"))
    w.begin_sequence("free", {"source": "free", "n_frames": 3, "seed": 5}, None, 2048)
    w.close()
    with pytest.raises(KeyError):   # no /pulse yet: nothing was ever written
        load_sequence(shard_path, "free", None, None, None, mode="raw")


# ---- the proc chain ------------------------------------------------------------------------------------
def test_round_trip_both_modes(shard):
    for mode in ("ref", "raw"):
        s = load_sequence(shard, "free", None, None, None, mode=mode)
        assert s.profiles.shape == (3, 84) and s.diff.shape == (2, 84) and s.labels is None
        assert s.profiles.dtype == np.float32 and np.isfinite(s.profiles).all()
        assert s.meta["mode"] == mode and s.meta["alpha"] == 1.0 and not s.meta["partial"]
        assert np.allclose(s.profiles[0], s.profiles[1], atol=1e-4 * np.abs(s.profiles).max())  # static scene
    raw = load_sequence(shard, "free", None, None, None, mode="raw")
    ref = load_sequence(shard, "free", None, None, None, mode="ref")
    assert not np.allclose(raw.profiles, ref.profiles)


def test_temperature_changes_profiles(shard):
    a = load_sequence(shard, "free", None, None, None, mode="raw")
    b = load_sequence(shard, "free", None, 30.0, None, mode="raw")
    assert b.meta["alpha"] > 1.0 and not np.allclose(a.profiles, b.profiles)


def test_override_44k_shape(shard):
    from ringsim.proc import ChirpSpec
    dev = load_device("test_pair")
    ov = dataclasses.replace(dev, fs_hz=44100, n_range_bins=77,
                             chirp=ChirpSpec(f0_hz=17000.0, f1_hz=21000.0, duration_s=600 / 44100,
                                             frame_period_s=600 / 44100, window="none"))
    s = load_sequence(shard, "free", None, None, None, mode="raw", dev_override=ov)
    assert s.profiles.shape == (3, 77) and s.meta["fs_hz"] == 44100


def test_one_frame_shard_both_modes(shard_path, synth_ir):
    """p1_free is a single frame: proc must cope with a short stream; diff is empty."""
    w = ShardWriter(shard_path, job_from_dict(dict(RAW, poses={"source": "free", "n_frames": 1})),
                    load_device("test_pair"))
    res = synth_ir(n_t=2048)
    w.begin_sequence("free", {"source": "free", "n_frames": 1, "seed": 5}, None, 2048)
    w.write_snapshot("free", 0, res)
    w.write_ref_ir("free", res)
    w.close()
    for mode in ("ref", "raw"):
        s = load_sequence(shard_path, "free", None, None, None, mode=mode)
        assert s.profiles.shape == (1, 84) and s.diff.shape == (0, 84) and np.isfinite(s.profiles).all()


def test_reference_stream_has_n_avg_plus_one_frames(shard, monkeypatch):
    """DATASET.md: the reference is built over n_avg + 1 frames, whatever the sequence's n_frames."""
    import ringsim.proc.ir as ir_mod
    seen = []
    orig = ir_mod.rx_stream

    def spy(irs, snap_t, pulse, fs_sim, tx, band, fs, n_frames, *a, **k):
        seen.append((irs.shape[0], n_frames))
        return orig(irs, snap_t, pulse, fs_sim, tx, band, fs, n_frames, *a, **k)

    monkeypatch.setattr(ir_mod, "rx_stream", spy)
    load_sequence(shard, "free", None, None, None, mode="ref")
    assert seen[0] == (1, 3) and seen[1][0] == 1 and seen[1][1] == 51
