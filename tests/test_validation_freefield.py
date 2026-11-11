"""Free-field check on a synthetic ideal-monopole shard (no k-Wave): the check must pass on a perfect monopole, fail on
corrupted ones, and the 44.1 kHz override must pass the same gate (D-046, D-058)."""
import dataclasses

import h5py
import numpy as np
import pytest

from ringsim.dataset.jobspec import job_from_dict
from ringsim.dataset.writer import ShardWriter
from ringsim.devices.config import load_device
from ringsim.proc import ChirpSpec, C_SIM
from ringsim.solver.constants import PULSE_DELAY, PULSE_FC_HZ, SIGMA_T
from ringsim.validation.freefield import (compare_ir, freefield_check, monopole_ir, shard_has_ref_ir)

FS_SIM = 576_000.0
N_T = 1500
RAW = {"job_id": "t", "device": "test_pair", "scene": "free", "dx_mm": 2.0, "placement": None,
       "poses": {"source": "free", "n_frames": 3}, "dr": None, "seed": 0}


class _Res:
    def __init__(self, ir, pulse):
        self.ir, self.pulse, self.fs_sim, self.wall_s = ir, pulse, FS_SIM, 0.0


def _pulse():
    t = np.arange(N_T) / FS_SIM - PULSE_DELAY       # SOLVER.md make_pulse
    return (np.exp(-t**2 / (2 * SIGMA_T**2)) * np.sin(2 * np.pi * PULSE_FC_HZ * t)).astype(np.float32)


def _write(path, ir, pulse, with_ref=True, seq="free"):
    w = ShardWriter(path, job_from_dict(dict(RAW)), load_device("test_pair"))
    w.begin_sequence(seq, {"source": "free", "n_frames": 3, "seed": 0}, None, N_T)
    w.write_snapshot(seq, 0, _Res(ir, pulse))
    if with_ref:
        w.write_ref_ir(seq, _Res(ir, pulse))
    w.close()


@pytest.fixture
def mono_shard(shard_path):
    pulse = _pulse()
    _write(shard_path, monopole_ir(pulse, FS_SIM, 0.100 / C_SIM), pulse)
    return shard_path


def test_ideal_monopole_passes(mono_shard):
    r = freefield_check(mono_shard, "free")
    assert r["pass"], r
    assert r["raw"]["peak_bin_diff"] <= 1 and r["raw"]["rel_l2"] < 0.10
    assert abs(r["arrival_err_samples"]) <= 2 and r["has_ref_ir"]
    assert r["distance_m"] == pytest.approx(0.100)
    assert set(r["raw"]) >= {"peak_bin_sim", "peak_bin_ref", "rel_l2", "gain_phase_rad"}
    assert "ref" in r and "rel_l2" in r["ref"]      # reported, not gating (D-057)


def test_44k_override_passes_same_gate(mono_shard):
    dev = load_device("test_pair")
    ov = dataclasses.replace(dev, fs_hz=44100, n_range_bins=77,
                             chirp=ChirpSpec(17000.0, 21000.0, 600 / 44100, 600 / 44100, "none"))
    r = freefield_check(mono_shard, "free", dev_override=ov)
    assert r["pass"], r
    assert r["fs_hz"] == 44100


def test_wrong_arrival_fails(shard_path):
    pulse = _pulse()
    ir = np.roll(monopole_ir(pulse, FS_SIM, 0.100 / C_SIM), 20)      # 20 sim samples late
    _write(shard_path, ir, pulse)
    r = freefield_check(shard_path, "free")
    assert not r["pass_arrival"] and not r["pass"]


def test_wrong_distance_fails_profile(shard_path):
    pulse = _pulse()
    _write(shard_path, monopole_ir(pulse, FS_SIM, 0.250 / C_SIM), pulse)   # echo at 250 mm, not 100 mm
    r = freefield_check(shard_path, "free")
    assert not r["pass"]


def test_missing_ref_ir_fails(shard_path):
    pulse = _pulse()
    _write(shard_path, monopole_ir(pulse, FS_SIM, 0.100 / C_SIM), pulse, with_ref=False)
    assert not shard_has_ref_ir(shard_path)
    with pytest.raises(ValueError, match="ref_ir"):
        freefield_check(shard_path, "free")      # mode="ref" read refuses a shard without ref_ir


def test_shard_has_ref_ir(mono_shard):
    assert shard_has_ref_ir(mono_shard)


def test_compare_ir(mono_shard, tmp_path):
    pulse = _pulse()
    ir = monopole_ir(pulse, FS_SIM, 0.100 / C_SIM)
    b = str(tmp_path / "b.h5")
    _write(b, (ir * (1 + 1e-4)).astype(np.float32), pulse)
    assert compare_ir(mono_shard, mono_shard, "free") == 0.0
    assert compare_ir(mono_shard, b, "free") == pytest.approx(1e-4, rel=0.1)
    c = str(tmp_path / "c.h5")
    w = ShardWriter(c, job_from_dict(dict(RAW)), load_device("test_pair"))
    w.begin_sequence("free", {"source": "free", "n_frames": 3, "seed": 0}, None, 100)
    w.close()
    with pytest.raises(ValueError, match="shapes"):
        compare_ir(mono_shard, c, "free")
