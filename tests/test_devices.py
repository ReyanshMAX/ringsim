"""DeviceConfig loader rules (docs/DEVICES.md, D-055)."""
import warnings
from pathlib import Path

import pytest
import yaml

import ringsim.devices
from ringsim.devices.config import (ConfigIncomplete, DroppedPortWarning, NoBody, device_yaml, load_device,
                                    load_device_yaml)

_DIR = Path(ringsim.devices.__file__).parent


def _raw(name):
    return yaml.safe_load((_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _load(raw):
    return load_device_yaml(yaml.safe_dump(raw))


def test_test_pair_loads():
    d = load_device("test_pair")
    assert d.fs_hz == 48000 and d.active_pair == ("spk", "mic") and isinstance(d.body, NoBody)
    assert [p.points_mm for p in d.ports] == [((0.0, 0.0, 0.0),), ((100.0, 0.0, 0.0),)]
    assert d.range_window_m == pytest.approx(84 * 343.0 / (2 * 48000), rel=1e-12)   # 0.300125 m


def test_iphone17pro_drops_q005_ports_with_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = load_device("iphone17pro")
    msgs = [str(x.message) for x in w if issubclass(x.category, DroppedPortWarning)]
    assert len(msgs) == 2
    assert any("bottom_mic1" in m and "Q-005" in m for m in msgs)
    assert any("front_mic" in m and "Q-005" in m for m in msgs)
    assert [p.name for p in d.ports] == ["bottom_spk", "bottom_mic2", "earpiece"]
    assert d.active_pair == ("bottom_spk", "bottom_mic2")
    assert len(d.ports[0].points_mm) == 4


def test_ring_raises_listing_q001_and_q002():
    with pytest.raises(ConfigIncomplete) as e:
        load_device("ring")
    msg = str(e.value)
    assert "Q-001" in msg and "Q-002" in msg
    for field in ("chirp.window", "body.sensor_section_arc_deg", "body.gap_center_deg", "body.phalanx",
                  "body.along_bone_frac", "ports.spk.points_mm", "ports.mic.points_mm"):
        assert field in msg


def test_tbd_in_active_port_raises():
    raw = _raw("test_pair")
    raw["ports"][1]["points_mm"] = "TBD(Q-099)"
    with pytest.raises(ConfigIncomplete, match=r"Q-099: ports\.mic\.points_mm"):
        _load(raw)


@pytest.mark.parametrize("f0, f1", [(15000, 22000), (18000, 28500)])
def test_chirp_band_outside_pulse_band_raises(f0, f1):
    raw = _raw("test_pair")
    raw["chirp"].update(f0_hz=f0, f1_hz=f1)
    with pytest.raises(ValueError, match="pulse band"):
        _load(raw)


def test_chirp_band_edges_allowed():
    raw = _raw("test_pair")
    raw["chirp"].update(f0_hz=16000, f1_hz=28000)
    _load(raw)


def test_frame_period_not_integer_samples_raises():
    raw = _raw("test_pair")
    raw["chirp"].update(duration_s=0.01251, frame_period_s=0.01251)    # 600.48 samples
    with pytest.raises(ValueError, match="integer"):
        _load(raw)


@pytest.mark.parametrize("pair", [["spk", "nope"], ["spk"], ["mic", "mic"]])
def test_active_pair_must_be_one_speaker_one_mic_in_ports(pair):
    raw = _raw("test_pair")
    raw["active_pair"] = pair
    with pytest.raises(ValueError, match="active_pair"):
        _load(raw)


def test_box_port_off_surface_raises():
    raw = _raw("iphone17pro")
    raw["ports"][0]["points_mm"][0] = [-20.525, -74.0, 0]    # 1.005 mm inside the bottom face
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DroppedPortWarning)
        with pytest.raises(ValueError, match="box surface"):
            _load(raw)


def test_box_port_within_tolerance_allowed():
    raw = _raw("iphone17pro")
    raw["ports"][0]["points_mm"][0] = [-20.525, -74.6, 0]    # 0.405 mm inside
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DroppedPortWarning)
        _load(raw)


@pytest.mark.parametrize("name", ["test_pair", "iphone17pro"])
def test_device_yaml_round_trip(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DroppedPortWarning)
        d = load_device(name)
        text = device_yaml(d)
    assert "TBD" not in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # dropped ports are absent: reloading warns about nothing
        assert load_device_yaml(text) == d


def test_package_yaml_matches_devices_md():
    """The three YAMLs are verbatim copies of DEVICES.md's blocks (when the spec doc is available)."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "DEVICES.md"
    if not doc.exists():
        pytest.skip("docs/ not shipped (Kaggle run dir)")
    text = doc.read_text(encoding="utf-8")
    for name in ("test_pair", "iphone17pro", "ring"):
        block = text.split(f"## {name}.yaml", 1)[1].split("```yaml\n", 1)[1].split("```", 1)[0]
        assert (_DIR / f"{name}.yaml").read_text(encoding="utf-8") == block, name
