"""`ringsim sim` / `profile` wiring without k-Wave: run_ir is replaced by an ideal monopole (docs/ARCHITECTURE.md CLI)."""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import ringsim.solver.kwave_runner as kr
from ringsim.cli import main
from ringsim.proc.physics import C_SIM
from ringsim.solver.pulse import make_pulse
from ringsim.validation.freefield import monopole_ir

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_kwave(monkeypatch):
    """run_ir -> ideal monopole at the medium's speaker-mic voxel distance; records each call."""
    calls = []

    def fake_run_ir(medium, range_window_m, fs_device, gpu):
        dt, n_t, _ = kr.medium_time_axis(medium, range_window_m, fs_device)
        d = medium.dx * np.linalg.norm(np.subtract(medium.src_idx[0], medium.mic_idx[0]))
        pulse = make_pulse(1 / dt, n_t)
        calls.append((medium, gpu))
        return kr.SimResult(ir=monopole_ir(pulse, 1 / dt, d / C_SIM), pulse=pulse, fs_sim=1 / dt, wall_s=0.01)

    monkeypatch.setattr(kr, "run_ir", fake_run_ir)
    return calls


@pytest.fixture
def job(tmp_path):
    src = REPO / "jobs" / "p1_free.yaml"
    dst = tmp_path / "p1_free.yaml"
    shutil.copy(src, dst)
    return dst


def test_sim_writes_shard_png_and_ref_ir(job, tmp_path, fake_kwave, capsys):
    out = tmp_path / "out"
    assert main(["sim", "--job", str(job), "--out-dir", str(out)]) == 0
    assert (out / "p1_free.h5").exists() and (out / "p1_free_medium.png").exists()
    assert len(fake_kwave) == 2                                 # sim snapshot + its own ref_ir call (D-056)
    assert fake_kwave[0][0] is not fake_kwave[1][0]
    with h5py.File(out / "p1_free.h5", "r") as f:
        g = f["seq"]["free"]                                    # D-060 seq_id
        assert g["ir"].shape[0] == 1 and "ref_ir" in g
        assert bool(g["port_covered"][0]) is False
        assert not bool(g.attrs["failed"])


def test_profile_prints_passing_freefield_json(job, tmp_path, fake_kwave, capsys):
    out = tmp_path / "out"
    main(["sim", "--job", str(job), "--out-dir", str(out)])
    capsys.readouterr()
    png = tmp_path / "prof.png"
    assert main(["profile", "--shard", str(out / "p1_free.h5"), "--seq", "free", "--out", str(png)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert png.stat().st_size > 1000
    assert res["pass"] is True and "raw" in res and "ref" in res


def test_cpu_refuses_grids_over_96_cubed(tmp_path, fake_kwave, monkeypatch):
    import ringsim.cli as cli
    monkeypatch.setattr(cli, "CPU_MAX_CELLS", 61 * 12 * 12 - 1)  # the p1_free grid becomes "too large"
    job = tmp_path / "p1_free.yaml"
    shutil.copy(REPO / "jobs" / "p1_free.yaml", job)
    with pytest.raises(cli.CpuGridTooLarge, match="kaggle_run"):
        main(["sim", "--job", str(job), "--out-dir", str(tmp_path / "o")])
    assert fake_kwave == []
    assert main(["sim", "--job", str(job), "--gpu", "--out-dir", str(tmp_path / "o")]) == 0   # GPU: no limit
    assert fake_kwave[0][1] is True


@pytest.mark.parametrize("bad, reason", [("nan_sim", "nonfinite_ir"), ("nan_ref", "nonfinite_ir"),
                                         ("raise", "run_error: RuntimeError: boom")])
def test_run_failure_fails_the_sequence(job, tmp_path, fake_kwave, monkeypatch, bad, reason):
    good = kr.run_ir

    def flaky(medium, *a):
        r = good(medium, *a)
        if bad == "raise":
            raise RuntimeError("boom")
        if (bad == "nan_sim" and len(fake_kwave) == 1) or (bad == "nan_ref" and len(fake_kwave) == 2):
            r.ir[5] = np.nan
        return r

    monkeypatch.setattr(kr, "run_ir", flaky)
    out = tmp_path / "out"
    assert main(["sim", "--job", str(job), "--out-dir", str(out)]) == 1
    with h5py.File(out / "p1_free.h5", "r") as f:
        g = f["seq"]["free"]
        assert bool(g.attrs["failed"]) and g.attrs["failed_reason"] == reason


def test_cpu_limit_is_96_cubed():
    import ringsim.cli as cli
    assert cli.CPU_MAX_CELLS == 96 ** 3


@pytest.mark.parametrize("scene, phase", [("plate", "Phase 2"), ("hand", "Phase 3")])
def test_later_scenes_name_their_phase(tmp_path, scene, phase):
    job = tmp_path / "x.yaml"
    job.write_text((REPO / "jobs" / "p1_free.yaml").read_text().replace("job_id: p1_free", "job_id: x")
                   .replace("scene: free", f"scene: {scene}"))
    with pytest.raises(NotImplementedError, match=phase):
        main(["sim", "--job", str(job), "--out-dir", str(tmp_path / "o")])
