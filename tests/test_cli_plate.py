"""`ringsim sim` for scene plate (Protocol B sims, D-047, D-071, D-078) without k-Wave (fake run_ir)."""
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import ringsim.solver.kwave_runner as kr
from ringsim.cli import main
from ringsim.dataset.jobspec import ConfigIncomplete, load_job
from ringsim.proc.physics import C_SIM
from ringsim.solver.pulse import make_pulse
from ringsim.validation.freefield import monopole_ir

REPO = Path(__file__).resolve().parents[1]
P2_JOBS = ["p2_plate_80", "p2_plate_110", "p2_plate_140", "p2_plate_170", "p2_plate_200", "p2_plate_background"]


@pytest.fixture
def fake_kwave(monkeypatch):
    calls = []

    def fake_run_ir(medium, range_window_m, fs_device, gpu):
        dt, n_t, _ = kr.medium_time_axis(medium, range_window_m, fs_device)
        d = medium.dx * np.linalg.norm(np.subtract(medium.src_idx[0], medium.mic_idx[0]))
        pulse = make_pulse(1 / dt, n_t)
        calls.append(medium)
        return kr.SimResult(ir=monopole_ir(pulse, 1 / dt, d / C_SIM), pulse=pulse, fs_sim=1 / dt, wall_s=0.01)

    monkeypatch.setattr(kr, "run_ir", fake_run_ir)
    return calls


def _job(tmp_path, plate, rig_boxes=(), job_id="pl"):
    raw = {"job_id": job_id, "device": "iphone17pro", "scene": "plate", "dx_mm": 2.0,
           "placement": {"plate": plate, "rig_boxes": list(rig_boxes)},
           "poses": {"source": "plate", "n_frames": 1}, "dr": None, "seed": 0}
    p = tmp_path / f"{job_id}.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("name", P2_JOBS)
def test_p2_jobs_raise_until_q027(name):
    """D-071 / Q-027: plate thickness and the clamp box are TBD until the user measures them."""
    with pytest.raises(ConfigIncomplete, match="Q-027"):
        load_job(str(REPO / "jobs" / f"{name}.yaml"))


@pytest.mark.parametrize("name", P2_JOBS)
def test_p2_jobs_match_protocol_b(name):
    raw = yaml.safe_load((REPO / "jobs" / f"{name}.yaml").read_text())
    assert raw["job_id"] == name and raw["device"] == "iphone17pro" and raw["scene"] == "plate"
    assert raw["dx_mm"] == 2.0 and raw["poses"] == {"source": "plate", "n_frames": 1}
    plate = raw["placement"]["plate"]
    if name.endswith("background"):
        assert plate is None
    else:
        assert plate["distance_mm"] == int(name.rsplit("_", 1)[1])      # VALIDATION.md: 80..200 mm
        assert plate["size_mm"][:2] == [100, 100]


def test_plate_sim_writes_plate_seq_and_device_only_ref(tmp_path, fake_kwave):
    clamp = {"center": [0.0, 62.5, 0.0], "size": [80.0, 30.0, 20.0]}
    job = _job(tmp_path, {"size_mm": [100, 100, 6], "distance_mm": 80}, [clamp])
    out = tmp_path / "out"
    assert main(["sim", "--job", str(job), "--out-dir", str(out)]) == 0
    assert len(fake_kwave) == 2
    sim_m, ref_m = fake_kwave
    # ref = body + plateau + clamp only: no solid beyond the bottom edge region where the plate is
    assert sim_m.occ.sum() > ref_m.occ.sum()
    off = np.round((ref_m.origin_m - sim_m.origin_m) / sim_m.dx).astype(int)
    assert [tuple(np.add(i, off)) for i in ref_m.src_idx] == sim_m.src_idx     # one lattice (D-075)
    with h5py.File(out / "pl.h5", "r") as f:
        assert list(f["seq"]) == ["plate"]
        g = f["seq"]["plate"]
        assert g.attrs["source"] == "plate" and not bool(g.attrs["failed"])
        assert g["ir"].shape == (1, g["ref_ir"].shape[0])
        assert bool(g["port_covered"][0]) is False
        job_attr = yaml.safe_load(f.attrs["job_yaml"])
        assert job_attr["placement"]["plate"]["distance_mm"] == 80                # D-078 reads it
    assert (out / "pl_medium.png").exists()


def test_background_sim_seq_id(tmp_path, fake_kwave):
    job = _job(tmp_path, None, job_id="bg")
    assert main(["sim", "--job", str(job), "--out-dir", str(tmp_path / "o")]) == 0
    with h5py.File(tmp_path / "o" / "bg.h5", "r") as f:
        assert list(f["seq"]) == ["background"]
    sim_m, ref_m = fake_kwave
    np.testing.assert_array_equal(sim_m.rho, ref_m.rho)          # background scene == device-only scene


def test_plate_poses_source_must_be_plate(tmp_path, fake_kwave):
    job = _job(tmp_path, None)
    job.write_text(job.read_text().replace("source: plate", "source: free"))
    with pytest.raises(ValueError, match="poses.source"):
        main(["sim", "--job", str(job), "--out-dir", str(tmp_path / "o")])
    assert fake_kwave == []
