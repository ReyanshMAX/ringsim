"""validation/plate.py on synthetic shards and synthetic captures (no k-Wave, no hardware).

Sim and 'real' are built from the same synthetic IRs (direct path + a plate echo at path 2d + 0.05 m), the real ones
through rx_stream with a different start offset per recording. temp_c is chosen so speed_of_sound = C_SIM, so the
two sides should agree; the test checks the glue (naming, offsets, frame 0, background, drift), not physics.
Note: with ECHO_GAIN = 1 the near plates (80, 110 mm) move align_direct_path by a sample relative to the background
recording, leaving a direct-path residue after subtraction at bin 3 (seen in both sim and real); reported to the lead."""
import json
import os

import numpy as np
import pytest
from scipy.io import wavfile

from ringsim.dataset.jobspec import job_from_dict
from ringsim.dataset.writer import ShardWriter
from ringsim.devices.config import load_device
from ringsim.proc import rx_stream, tx_frame
from ringsim.validation.metrics import peak_bin
from ringsim.validation.plate import _classify, plate_inputs, plate_report
from ringsim.validation.real import META_KEYS, load_session

FS_SIM = 576_000.0
N_T = 4096
N_FRAMES = 80
DISTS = (80, 110, 140, 170, 200)
TEMP_C = 19.63                       # speed_of_sound(19.63) = 343.0 to 4 digits
ECHO_GAIN = 0.3                      # weak enough that the plate echo does not move the direct-path alignment (see note)


class _Res:
    def __init__(self, ir, pulse):
        self.ir, self.pulse, self.fs_sim, self.wall_s = ir, pulse, FS_SIM, 0.0


def _irs(synth_ir, d_mm=None):
    direct = synth_ir(dist_m=0.100, n_t=N_T)
    if d_mm is None:
        return direct.ir, direct
    echo = synth_ir(dist_m=2 * d_mm / 1000.0 + 0.05, n_t=N_T)
    return direct.ir + ECHO_GAIN * echo.ir, direct


def _shard(path, synth_ir, seq, d_mm):
    raw = {"job_id": f"p_{seq}_{d_mm}", "device": "test_pair", "scene": "plate", "dx_mm": 2.0,
           "placement": {"plate": None if d_mm is None else {"size_mm": [100, 100, 5], "distance_mm": d_mm}},
           "poses": {"source": "plate", "n_frames": N_FRAMES}, "dr": None, "seed": 1}
    ir, direct = _irs(synth_ir, d_mm)
    w = ShardWriter(str(path), job_from_dict(raw), load_device("test_pair"))
    w.begin_sequence(seq, {"source": "plate", "n_frames": N_FRAMES, "seed": 1}, None, N_T)
    w.write_snapshot(seq, 0, _Res(ir, direct.pulse))
    w.write_ref_ir(seq, _Res(direct.ir, direct.pulse))
    w.close()


def _wav(path, synth_ir, d_mm, offset, dev):
    ir, direct = _irs(synth_ir, d_mm)
    tx = tx_frame(dev.chirp, dev.fs_hz)
    band = (dev.chirp.f0_hz, dev.chirp.f1_hz)
    s = rx_stream(ir[None, :], np.zeros(1), direct.pulse, round(FS_SIM), tx, band, dev.fs_hz, N_FRAMES, 1.0)
    x = np.concatenate([np.zeros(offset), s * 1e3]).astype(np.float32)
    wavfile.write(str(path) + ".wav", 48000, x)
    meta = dict.fromkeys(META_KEYS)
    meta.update(schema_version=1, sample_rate=48000, take=path.name)
    (path.parent / f"{path.name}.json").write_text(json.dumps(meta))


@pytest.fixture
def rig(tmp_path, synth_ir):
    sim = tmp_path / "sim"
    ses = tmp_path / "session"
    sim.mkdir(), ses.mkdir()
    for d in DISTS:
        _shard(sim / f"plate_{d}.h5", synth_ir, "plate", d)
    _shard(sim / "background.h5", synth_ir, "background", None)
    (ses / "session.json").write_text(json.dumps({"temp_c": TEMP_C}))
    dev = load_device("test_pair")
    rng = np.random.default_rng(0)
    for d in DISTS:
        for k in (1, 2):
            _wav(ses / f"plate_{d}mm_t{k}", synth_ir, d, int(rng.integers(20, 500)), dev)
    for k in (1, 2):
        _wav(ses / f"background_t{k}", synth_ir, None, int(rng.integers(20, 500)), dev)
    for n in (1, 2, 10):
        _wav(ses / f"free_air_{n}", synth_ir, None, int(rng.integers(20, 500)), dev)
    return str(sim), str(ses)


def test_report_on_matching_sim_and_real(rig):
    r = plate_report(*rig)
    assert r["distances_mm"] == list(DISTS) and set(r["n_takes"].values()) == {2}
    assert r["peak_bin_pass"]["ref"] and r["peak_bin_pass"]["raw"]
    assert r["linearity"]["ref"]["pass"], r["linearity"]["ref"]
    assert all(v["r"] > 0.9 for v in r["signed_shape"]["ref"].values())
    assert r["noise_floor"] < 0.05 and r["free_air_drift"]["rel_l2"] < 0.05, (r["noise_floor"], r["free_air_drift"])


def test_inputs_shapes_frame0_dropped(rig):
    sim, real, temp_c, fs, ex = plate_inputs(*rig)
    assert fs == 48000 and temp_c == TEMP_C
    assert set(sim["ref"]) == set(real["ref"]) == set(DISTS)
    # two takes per distance, each loses frame 0 and a few tail frames to the offset
    assert 2 * (N_FRAMES - 3) <= real["ref"][80].shape[0] <= 2 * (N_FRAMES - 1)
    assert sim["ref"][80].shape[0] == N_FRAMES
    assert abs(peak_bin(real["ref"][80])[0] - peak_bin(sim["ref"][80])[0]) <= 1


def test_classify_sorts_integers_and_rejects_unknown(tmp_path):
    for n in ["free_air_10", "free_air_2", "free_air_1", "plate_80mm_t1", "background_t1"]:
        (tmp_path / f"{n}.wav").write_bytes(b"")
    (tmp_path / "session.json").write_text("{}")
    plates, bgs, free = _classify(load_session(str(tmp_path)))
    assert [os.path.basename(p)[len("free_air_"):-4] for p in free] == ["1", "2", "10"]
    assert list(plates) == [80] and len(bgs) == 1
    (tmp_path / "plate_110mm_t2_2.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="unrecognised"):
        _classify(load_session(str(tmp_path)))


def test_missing_pieces_raise(rig, tmp_path):
    sim, ses = rig
    for n in ("free_air_2", "free_air_10"):
        os.remove(os.path.join(ses, n + ".wav"))
        os.remove(os.path.join(ses, n + ".json"))
    with pytest.raises(ValueError, match="free_air"):
        plate_report(sim, ses)
    with pytest.raises(FileNotFoundError):
        plate_report(str(tmp_path / "empty"), ses)


def test_truncated_take_raises_naming_it(rig):
    sim, ses = rig
    p = os.path.join(ses, "plate_80mm_t1.wav")
    x = wavfile.read(p)[1]
    wavfile.write(p, 48000, x[:900])                   # 1 frame
    with pytest.raises(ValueError, match="plate_80mm_t1"):
        plate_report(sim, ses)


def test_report_labels_noise_floor_pooled(rig):
    assert plate_report(*rig)["noise_floor_basis"] == "pooled over takes"
