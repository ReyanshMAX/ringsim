"""Phase 1 free-field pipeline end to end on CPU k-Wave (~5 s)."""
import h5py
import numpy as np
import pytest
from scipy import signal

from ringsim.cli import main
from ringsim.dataset.reader import list_sequences, load_sequence
from ringsim.solver.constants import PULSE_DELAY

SHARD_ATTRS = {"schema_version", "job_yaml", "device_yaml", "git_hash", "kwave_version",
               "created_utc", "dx_m", "c_sim", "fs_device", "k_decim"}
SEQ_ATTRS = {"source", "fps", "n_frames", "ring_rot_deg", "ring_shift_mm", "seed",
             "n_frames_written"}


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    assert main(["sim", "--job", "jobs/p1_free.yaml", "--out-dir", str(out)]) == 0
    return out / "p1_free.h5"


def test_shard_attrs(shard):
    with h5py.File(shard, "r") as f:
        assert SHARD_ATTRS <= set(f.attrs)
        assert f.attrs["c_sim"] == 343.0 and f.attrs["k_decim"] == 12
        g = f["seq/free"]
        assert SEQ_ATTRS <= set(g.attrs)
        assert g["ir"].shape == (1, f["pulse"].shape[0]) and g["ir"].compression == "gzip"
        assert np.isfinite(g["wall_s"][0]) and g["failed_frames"].shape == (0,)
    assert (shard.parent / "p1_free_medium.png").exists()


def test_ir_arrival(shard):
    with h5py.File(shard, "r") as f:
        ir = f["seq/free/ir"][0]
        fs = f.attrs["k_decim"] * f.attrs["fs_device"]
    t_peak = np.argmax(np.abs(signal.hilbert(ir))) / fs
    assert abs(t_peak - (0.100 / 343.0 + PULSE_DELAY)) <= 2 / fs


def test_reader(shard):
    assert list_sequences(str(shard)) == ["free"]
    s = load_sequence(str(shard), "free", snr_db=None, temp_c=None, rng=None)
    assert s.profiles.shape == (1, 84) and s.diff.shape == (0, 84) and s.labels is None
    s20 = load_sequence(str(shard), "free", snr_db=None, temp_c=20.0, rng=None)
    assert s20.meta["alpha"] > 1


def test_freefield_matches_ideal_monopole(shard):
    from ringsim.validation.freefield import freefield_check
    r = freefield_check(str(shard), "free")
    assert r["rel_l2_vs_ideal"] < 0.10
    assert abs(r["peak_bin"] - r["ref_peak_bin"]) <= 1
