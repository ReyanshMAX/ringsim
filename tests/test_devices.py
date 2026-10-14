import warnings

import pytest

from ringsim.devices.config import (ConfigIncomplete, ConfigTBDWarning, device_from_yaml,
                                    load_device, parse_device)


def test_test_pair():
    d = load_device("test_pair")
    assert d.speaker.points_mm == ((0.0, 0.0, 0.0),) and d.mic.points_mm == ((100.0, 0.0, 0.0),)
    assert d.frame_len == 600
    assert abs(d.range_window_m - 84 * 343.0 / 96000) < 1e-12


def test_phone_drops_tbd_ports_with_q005_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = load_device("iphone17pro")
    msgs = [str(x.message) for x in w if issubclass(x.category, ConfigTBDWarning)]
    assert len(msgs) == 2
    assert any("bottom_mic1" in m and "Q-005" in m for m in msgs)
    assert any("front_mic" in m and "Q-005" in m for m in msgs)
    assert {p.name for p in d.ports} == {"bottom_spk", "bottom_mic2", "earpiece"}
    assert d.active_pair == ("bottom_spk", "bottom_mic2")


def test_ring_raises_naming_q001():
    with pytest.raises(ConfigIncomplete, match="Q-001"):
        load_device("ring")


def test_tbd_in_active_port_raises():
    raw = {"name": "x", "fs_hz": 48000, "n_range_bins": 10, "frame": "device",
           "chirp": {"f0_hz": 18000, "f1_hz": 22000, "duration_s": 0.0125,
                     "frame_period_s": 0.0125, "window": "none"},
           "body": {"kind": "none"}, "active_pair": ["s", "m"],
           "ports": [{"name": "s", "kind": "speaker", "points_mm": [[0, 0, 0]], "normal": [0, 0, 1]},
                     {"name": "m", "kind": "mic", "points_mm": "TBD(Q-099)", "normal": [0, 0, 1]}]}
    with pytest.raises(ConfigIncomplete, match="Q-099: ports.m.points_mm"):
        parse_device(raw)


def test_chirp_band_outside_pulse_band():
    raw = yaml_raw()
    raw["chirp"]["f1_hz"] = 30000
    with pytest.raises(AssertionError, match="pulse band"):
        parse_device(raw)


def test_box_port_off_surface():
    import yaml
    raw = yaml.safe_load(open(_path("iphone17pro")))
    raw["ports"][0]["points_mm"][0] = [-20.525, -70.0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AssertionError, match="not on box surface"):
            parse_device(raw)


def test_yaml_roundtrip():
    d = load_device("test_pair")
    assert device_from_yaml(d.to_yaml()) == d


def _path(name):
    from pathlib import Path
    import ringsim.devices
    return Path(ringsim.devices.__file__).parent / f"{name}.yaml"


def yaml_raw():
    import yaml
    return yaml.safe_load(open(_path("test_pair")))
