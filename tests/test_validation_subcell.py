"""Q-020 analysis (validation/subcell.py, D-076, D-082) on shards from `ringsim sim` with a fake k-Wave whose IRs are
ideal monopoles: the direct path from every speaker voxel to the mic voxel, plus (plate scenes) the image-source echo
off a plane at the plate's near face moved by a known `shift`, all delayed by a known extra system lag."""
from pathlib import Path

import numpy as np
import pytest
import yaml

import ringsim.solver.kwave_runner as kr
from ringsim.cli import main
from ringsim.proc.physics import C_SIM
from ringsim.solver.pulse import make_pulse
from ringsim.validation.freefield import monopole_ir
from ringsim.validation.subcell import RESIDUAL_LAG_SAMPLES, echo_delay, envelope_lag, subcell_report

Y_EDGE = -0.075005           # iphone17pro bottom face (the speaker normal is -y)
SYS_LAG = 7.3e-6             # a lag the solver adds to every path (e.g. the mass source's phase); must cancel


@pytest.fixture
def fake(monkeypatch):
    state = {"d_mm": None, "shift_m": 0.0}

    def fake_run_ir(medium, range_window_m, fs_device, gpu):
        dt, n_t, _ = kr.medium_time_axis(medium, range_window_m, fs_device)
        fs = 1 / dt
        pulse = make_pulse(fs, n_t)
        src = [medium.origin_m + np.asarray(i) * medium.dx for i in medium.src_idx]
        mic = [medium.origin_m + np.asarray(i) * medium.dx for i in medium.mic_idx]
        ys = medium.origin_m[1] + np.nonzero(medium.occ.any(axis=(0, 2)))[0] * medium.dx
        has_plate = ys.min() < Y_EDGE - 0.01
        ir = np.zeros(n_t)
        for m in mic:
            for s in src:
                r = np.linalg.norm(s - m)
                ir += monopole_ir(pulse, fs, r / C_SIM + SYS_LAG) / r
                if has_plate:
                    y_face = Y_EDGE - state["d_mm"] * 1e-3 + state["shift_m"]
                    img = s.copy(); img[1] = 2 * y_face - s[1]
                    R = np.linalg.norm(img - m)
                    ir += monopole_ir(pulse, fs, R / C_SIM + SYS_LAG) / R
        ir /= len(mic)
        return kr.SimResult(ir=ir.astype(np.float32), pulse=pulse, fs_sim=fs, wall_s=0.01)

    monkeypatch.setattr(kr, "run_ir", fake_run_ir)
    return state


def _shard(tmp_path, fake, d_mm, size=(100, 100, 10), shift_m=0.0, job_id=None):
    job_id = job_id or f"q_{d_mm:.1f}".replace(".", "p")
    raw = {"job_id": job_id, "device": "iphone17pro", "scene": "plate", "dx_mm": 2.0,
           "placement": {"plate": {"size_mm": list(size), "distance_mm": d_mm}, "rig_boxes": []},
           "poses": {"source": "plate", "n_frames": 1}, "dr": None, "seed": 0}
    p = tmp_path / f"{job_id}.yaml"
    p.write_text(yaml.safe_dump(raw))
    fake["d_mm"], fake["shift_m"] = d_mm, shift_m
    with pytest.warns(UserWarning):                 # TBD(Q-005) ports dropped
        assert main(["sim", "--job", str(p), "--gpu", "--out-dir", str(tmp_path / "out")]) == 0
    return str(tmp_path / "out" / f"{job_id}.h5")


def test_envelope_lag_subsample():
    fs = 576_000.0
    pulse = make_pulse(fs, 1500)
    for tau in (200e-6, 200.37e-6, 401.9e-6):
        x = monopole_ir(pulse, fs, tau)
        lag, _ = envelope_lag(x, pulse, fs, (tau - 30e-6, tau + 30e-6))
        lag0, _ = envelope_lag(monopole_ir(pulse, fs, 100e-6), pulse, fs, (70e-6, 130e-6))
        assert (lag - lag0) == pytest.approx(tau - 100e-6, abs=0.02 / fs)     # sub-sample


FS_SIM = C_SIM / (0.3 * 2e-3)                   # c / (CFL * dx) at dx 2 mm, ~571.7 kHz


@pytest.mark.parametrize("base_s", [5e-6, 20e-6, 200e-6, 466e-6])
def test_envelope_lag_residual(base_s):
    """Parabolic-peak bias vs sub-sample phase on ideal monopole signals. The 5 us and 20 us delays sit against lag 0
    (the speaker holes are 2-8 mm from the mic voxel): seat 2's draft cut the xcorr at lag 0 before the Hilbert
    transform and read 5 us as 15.6 us."""
    pulse = make_pulse(FS_SIM, 1500)
    for fr in np.linspace(0, 1, 11):
        tau = base_s + fr / FS_SIM
        lag, _ = envelope_lag(monopole_ir(pulse, FS_SIM, tau), pulse, FS_SIM, (tau - 35e-6, tau + 35e-6))
        assert abs(lag - tau) * FS_SIM < RESIDUAL_LAG_SAMPLES


def test_envelope_lag_refuses_edge_peak():
    fs = 576_000.0
    pulse = make_pulse(fs, 1500)
    with pytest.raises(ValueError, match="gate edge"):
        envelope_lag(monopole_ir(pulse, fs, 300e-6), pulse, fs, (100e-6, 200e-6))


@pytest.mark.parametrize("shift_mm", [0.0, 0.6, -0.9])
def test_offset_recovers_surface_shift(tmp_path, fake, shift_mm):
    """A reflecting plane moved by `shift` toward the phone reads as offset = -shift (shorter path); the solver's
    system lag and the port-voxel geometry cancel."""
    r = echo_delay(_shard(tmp_path, fake, 80.0, shift_m=shift_mm * 1e-3))
    assert r["offset_mm"] == pytest.approx(-shift_mm, abs=0.02)
    assert r["sys_lag_s"] == pytest.approx(SYS_LAG, abs=0.05e-6)
    assert r["offset_bins"] == pytest.approx(r["offset_mm"] / (C_SIM / (2 * 48000) * 1e3))
    # the 1/R amplitude of the shifted plane changes the ratio by ~shift/R (about 1 % at 0.9 mm); exact at shift 0
    assert r["echo_direct_ratio"] == pytest.approx(r["image_source_ratio"], rel=1e-3 if shift_mm == 0 else 0.02)


def test_report_steps_and_d036(tmp_path, fake):
    steps = [_shard(tmp_path, fake, round(80.0 + 0.2 * i, 1)) for i in range(4)]
    rigid = _shard(tmp_path, fake, 80.0, size=(300, 300, 10), job_id="rigid")
    rep = subcell_report(steps, rigid)
    assert rep["monotonic"] and rep["slope_rel_err"] < 0.01 and rep["max_offset_bins"] < 0.01 and rep["pass"]
    assert [x["d_mm"] for x in rep["rows"]] == [80.0, 80.2, 80.4, 80.6]
    assert rep["d036"]["rigid_vs_image"] == pytest.approx(1.0, rel=1e-3)
    assert rep["expected_s_per_m"] == pytest.approx(2 / C_SIM)


def test_report_flags_non_monotonic(tmp_path, fake):
    """A plane that does not follow the plate (quantized occupancy) fails the criterion."""
    steps = [_shard(tmp_path, fake, 80.0 + 0.2 * i, shift_m=0.2e-3 * i) for i in range(3)]
    rigid = _shard(tmp_path, fake, 80.0, size=(300, 300, 10), job_id="rigid")
    rep = subcell_report(steps, rigid)
    assert rep["slope_rel_err"] > 0.9 and not rep["pass"]       # the delay does not move with d


def test_cli_entry(tmp_path, fake, capsys):
    from ringsim.validation.subcell import main as sub_main
    steps = [_shard(tmp_path, fake, 80.0 + 0.2 * i) for i in range(2)]
    rigid = _shard(tmp_path, fake, 80.0, size=(300, 300, 10), job_id="rigid")
    capsys.readouterr()                             # drop the `ringsim sim` output of _shard
    assert sub_main(steps + ["--rigid", rigid]) == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["pass"] is True and len(out["rows"]) == 2
    assert Path(out["rows"][0]["path"]).exists()
