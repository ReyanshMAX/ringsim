"""ShardWriter + JobSpec on synthetic data (no k-Wave), real test_pair DeviceConfig."""
import sys

import h5py
import numpy as np
import pytest
import yaml

from ringsim.dataset.jobspec import ConfigIncomplete, job_from_dict, job_yaml, load_job
from ringsim.dataset.writer import ShardWriter


def _Dev():
    from ringsim.devices.config import load_device
    return load_device("test_pair")


RAW = {"job_id": "t", "device": "test_pair", "scene": "free", "dx_mm": 2.0, "placement": None,
       "poses": {"source": "free", "n_frames": 3}, "dr": None, "seed": 5}


def _job():
    return job_from_dict(dict(RAW))


def test_job_si_and_yaml_roundtrip():
    j = _job()
    assert j.dx_m == pytest.approx(0.002) and j.n_frames == 3
    assert yaml.safe_load(job_yaml(j))["seed"] == 5


def test_job_tbd_lists_all():
    raw = dict(RAW, motion={"snapshot_disp_mm": "TBD(Q-019)"}, dr={"ring_rot_deg": "TBD(Q-014)"})
    with pytest.raises(ConfigIncomplete) as e:
        job_from_dict(raw)
    assert "Q-019" in str(e.value) and "Q-014" in str(e.value)


def test_job_unknown_key_and_scene_refused():
    with pytest.raises(ValueError):
        job_from_dict(dict(RAW, bogus=1))
    with pytest.raises(ValueError):
        job_from_dict(dict(RAW, scene="wall"))


def test_load_job_file(tmp_path):
    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump(RAW))
    assert load_job(str(p)).job_id == "t"


def _write(path, synth_ir, n_t=2048, with_ref=True):
    w = ShardWriter(path, _job(), _Dev())
    w.begin_sequence("free", {"source": "free", "n_frames": 3, "seed": 5}, None, n_t)
    res = synth_ir(n_t=n_t)
    w.write_snapshot("free", 0, res)
    if with_ref:
        w.write_ref_ir("free", res)
    w.close()
    return res


def test_static_free_schema(shard_path, synth_ir, monkeypatch):
    monkeypatch.setenv("RINGSIM_BUILD_ID", "abc123")
    res = _write(shard_path, synth_ir)
    with h5py.File(shard_path, "r") as f:
        a = f.attrs
        assert a["schema_version"] == 1 and a["build_id"] == "abc123"
        assert a["c_sim"] == 343.0 and a["fs_device"] == 48000 and a["k_decim"] == 12
        assert a["dx_m"] == pytest.approx(0.002)
        for k in ("job_yaml", "device_yaml", "kwave_version", "created_utc"):
            assert k in a
        assert f["pulse"].shape == (2048,)
        g = f["seq/free"]
        assert g.attrs["source"] == "free" and g.attrs["n_snapshots"] == 1
        assert g.attrs["fps"] == pytest.approx(80.0)
        assert g.attrs["ring_rot_deg"] == 0 and g.attrs["ring_shift_mm"] == 0 and g.attrs["seed"] == 5
        assert not g.attrs["failed"] and g.attrs["n_snapshots_written"] == 1
        assert g["ir"].shape == (1, 2048) and g["ir"].chunks == (1, 2048)
        assert g["snap_t"][:].tolist() == [0.0] and g["frame_snap"][:].tolist() == [0, 0, 0]
        assert np.array_equal(g["ir"][0], res.ir) and np.array_equal(g["ref_ir"][:], res.ir)
        for k in ("snap_pose", "betas", "joints21", "labels", "hand_scale_m"):
            assert k not in g
        assert g["wall_s"].shape == (1,) and g["port_covered"].dtype == bool
        assert g["undersampled_frames"].shape == (0,)


def test_build_id_dirty_suffix_from_git(monkeypatch):
    from ringsim.dataset import writer
    monkeypatch.delenv("RINGSIM_BUILD_ID", raising=False)
    calls = iter(["deadbee", " M x.py"])
    monkeypatch.setattr(writer, "_git", lambda args: next(calls))
    assert writer.build_id() == "deadbee+dirty"


def test_pulse_mismatch_refused(shard_path, synth_ir):
    w = ShardWriter(shard_path, _job(), _Dev())
    w.begin_sequence("free", {"source": "free", "n_frames": 1, "seed": 0}, None, 2048)
    res = synth_ir(n_t=2048)
    w.write_snapshot("free", 0, res)
    res2 = synth_ir(n_t=2048)
    res2.pulse = res2.pulse * 2
    with pytest.raises(ValueError):
        w.write_ref_ir("free", res2)
    w.close()


def test_fail_sequence_and_hand_refused(shard_path, synth_ir):
    w = ShardWriter(shard_path, _job(), _Dev())
    w.begin_sequence("a", {"source": "free", "n_frames": 1, "seed": 0}, None, 2048)
    w.fail_sequence("a", "port inside solid")
    with pytest.raises(NotImplementedError):
        w.begin_sequence("h", {"source": "walk:1", "n_frames": 2, "seed": 0}, object(), 2048)
    w.close()
    with h5py.File(shard_path, "r") as f:
        assert f["seq/a"].attrs["failed"] and f["seq/a"].attrs["failed_reason"] == "port inside solid"


_CHILD = """
import os, sys, numpy as np
sys.path.insert(0, {tests!r})
from test_dataset_writer import _job, _Dev
from conftest import FakeSimResult, make_pulse
from ringsim.dataset.writer import ShardWriter
w = ShardWriter({path!r}, _job(), _Dev())
w.begin_sequence("free", {{"source": "free", "n_frames": 1, "seed": 0}}, None, 2048)
pulse = make_pulse(2048, 576000.0)
res = FakeSimResult(ir=np.full(2048, 0.5, np.float32), pulse=pulse, fs_sim=576000.0, wall_s=1.0)
for _ in range(10):
    w.write_snapshot("free", 0, res)
os._exit(1)   # killed: close() never runs
"""


def test_flush_every_10_survives_kill(shard_path):
    import os
    import subprocess
    code = _CHILD.format(tests=os.path.dirname(__file__), path=shard_path)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    with h5py.File(shard_path, "r") as f:
        g = f["seq/free"]
        assert g.attrs["n_snapshots_written"] == 1
        assert float(g["ir"][0, 0]) == 0.5
