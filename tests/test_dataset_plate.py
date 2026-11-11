"""Static `plate` sequences through ShardWriter / reader on synthetic shards (no k-Wave).

The seq_id is arbitrary here on purpose: the plate seq_id convention is not settled (lead ruling pending)."""
import h5py
import numpy as np
import pytest

from ringsim.dataset.jobspec import job_from_dict
from ringsim.dataset.reader import list_sequences, load_sequence
from ringsim.dataset.writer import ShardWriter
from ringsim.devices.config import load_device

RAW = {"job_id": "t", "device": "test_pair", "scene": "free", "dx_mm": 2.0, "placement": None,
       "poses": {"source": "plate", "n_frames": 60}, "dr": None, "seed": 3}
ATTRS = {"source": "plate", "n_frames": 60, "seed": 3}


@pytest.fixture
def plate_shard(shard_path, synth_ir):
    w = ShardWriter(shard_path, job_from_dict(dict(RAW)), load_device("test_pair"))
    ref = synth_ir(dist_m=0.05, n_t=2048)
    w.begin_sequence("p", ATTRS, None, 2048)
    w.write_snapshot("p", 0, synth_ir(dist_m=0.14, n_t=2048))
    w.write_ref_ir("p", ref)
    w.close()
    return shard_path


def test_plate_layout(plate_shard):
    with h5py.File(plate_shard, "r") as f:
        g = f["seq/p"]
        assert g.attrs["source"] == "plate" and g.attrs["n_snapshots"] == 1
        assert g["ir"].shape[0] == 1 and np.all(g["frame_snap"][:] == 0) and len(g["frame_snap"]) == 60
        for k in ("snap_pose", "betas", "joints21", "labels", "hand_scale_m"):
            assert k not in g
        assert g.attrs["ring_rot_deg"] == 0 and g.attrs["ring_shift_mm"] == 0


def test_plate_reads_without_labels(plate_shard):
    assert list_sequences(plate_shard) == ["p"]
    s = load_sequence(plate_shard, "p", None, None, None)
    assert s.labels is None and s.profiles.shape[0] == 60 and s.diff.shape[0] == 59
    assert np.all(np.isfinite(s.profiles))
    # static scene: every frame sees the same IR, so the raw profiles are identical frame to frame
    raw = load_sequence(plate_shard, "p", None, None, None, mode="raw").profiles
    assert np.allclose(raw, raw[0], atol=1e-6 * np.abs(raw).max())


def test_plate_needs_no_schedule(shard_path):
    w = ShardWriter(shard_path, job_from_dict(dict(RAW)), load_device("test_pair"))
    with pytest.raises(NotImplementedError):
        w.begin_sequence("h", {"source": "hand", "n_frames": 2, "seed": 0}, object(), 2048)
    with pytest.raises(ValueError, match="static"):
        w.begin_sequence("p", ATTRS, object(), 2048)
    w.close()
