"""DeviceConfig schema and loader (DEVICES.md, D-006)."""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from ringsim.proc.chirp import ChirpSpec
from ringsim.proc.physics import C_SIM

PULSE_BAND_HZ = (16_000.0, 28_000.0)    # SOLVER.md pulse band
PORT_SURFACE_TOL_MM = 0.5
_TBD = re.compile(r"^TBD\((Q-\d{3})\)$")
_DIR = Path(__file__).parent


class ConfigIncomplete(Exception):
    """A required config value is TBD(Q-xxx)."""


class ConfigTBDWarning(UserWarning):
    """A non-active port with a TBD value was dropped."""


@dataclass(frozen=True)
class Port:
    name: str
    kind: Literal["speaker", "mic"]
    points_mm: tuple[tuple[float, float, float], ...]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class BoxBody:
    kind: Literal["box"]
    size_mm: tuple[float, float, float]
    plateau_size_mm: tuple[float, float, float] | None
    plateau_center_mm: tuple[float, float, float] | None


@dataclass(frozen=True)
class RingBody:
    kind: Literal["ring"]
    clearance_mm: float
    wall_mm: float
    sensor_section_mm: float
    sensor_section_arc_deg: float
    width_mm: float
    arc_deg: float
    gap_center_deg: float
    finger: Literal["thumb", "index", "middle", "ring", "little"]
    phalanx: Literal["proximal", "middle"]
    along_bone_frac: float


@dataclass(frozen=True)
class NoBody:
    kind: Literal["none"]


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    fs_hz: int
    chirp: ChirpSpec
    n_range_bins: int
    ports: tuple[Port, ...]
    active_pair: tuple[str, str]
    body: BoxBody | RingBody | NoBody
    frame: Literal["device", "ring"]

    @property
    def range_window_m(self) -> float:
        return self.n_range_bins * C_SIM / (2 * self.fs_hz)

    @property
    def frame_len(self) -> int:
        return int(round(self.chirp.frame_period_s * self.fs_hz))

    @property
    def band(self) -> tuple[float, float]:
        return (float(self.chirp.f0_hz), float(self.chirp.f1_hz))

    def port(self, name: str) -> Port:
        return next(p for p in self.ports if p.name == name)

    @property
    def speaker(self) -> Port:
        return self.port(self.active_pair[0])

    @property
    def mic(self) -> Port:
        return self.port(self.active_pair[1])

    def to_yaml(self) -> str:
        """Fully resolved config (TBD ports already dropped), as stored in shards."""
        return yaml.safe_dump(asdict(self), sort_keys=False)


def _find_tbd(obj: Any, path: str) -> list[tuple[str, str]]:
    if isinstance(obj, str):
        m = _TBD.match(obj.strip())
        return [(m.group(1), path)] if m else []
    if isinstance(obj, dict):
        return [t for k, v in obj.items() for t in _find_tbd(v, f"{path}.{k}" if path else k)]
    if isinstance(obj, list):
        return [t for i, v in enumerate(obj) for t in _find_tbd(v, f"{path}[{i}]")]
    return []


def _tup(x) -> tuple:
    return tuple(_tup(v) if isinstance(v, list | tuple) else float(v) for v in x)


def _on_box_surface(p: np.ndarray, center: np.ndarray, size: np.ndarray) -> bool:
    d = np.abs(p - center) - size / 2           # >0 outside along that axis
    outside = np.linalg.norm(np.maximum(d, 0))
    if outside > 0:
        return outside <= PORT_SURFACE_TOL_MM
    return -d.max() <= PORT_SURFACE_TOL_MM      # inside: distance to nearest face


def parse_device(raw: dict) -> DeviceConfig:
    """Validate a raw config dict (DEVICES.md loader rules)."""
    raw = dict(raw)
    active = list(raw.get("active_pair", []))
    ports_raw = []
    missing: list[tuple[str, str]] = []
    for p in raw.get("ports", []):
        tbd = _find_tbd(p, f"ports.{p.get('name')}")
        if tbd and p.get("name") not in active:
            qids = sorted({q for q, _ in tbd})
            warnings.warn(f"{raw.get('name')}: dropping port {p.get('name')!r}, "
                          f"TBD {', '.join(qids)}", ConfigTBDWarning, stacklevel=3)
            continue
        missing += tbd
        ports_raw.append(p)
    missing += _find_tbd({k: v for k, v in raw.items() if k != "ports"}, "")
    if missing:
        raise ConfigIncomplete("; ".join(f"{q}: {f}" for q, f in missing))

    ports = tuple(Port(name=p["name"], kind=p["kind"], points_mm=_tup(p["points_mm"]),
                       normal=_tup(p["normal"])) for p in ports_raw)
    for p in ports:
        assert p.kind in ("speaker", "mic"), p
        assert len(p.points_mm) >= 1 and all(len(q) == 3 for q in p.points_mm), p
        assert abs(np.linalg.norm(p.normal) - 1) < 1e-6, f"{p.name}: normal not unit"
    names = {p.name: p for p in ports}
    assert len(active) == 2 and all(n in names for n in active), active
    assert names[active[0]].kind == "speaker" and names[active[1]].kind == "mic", \
        "active_pair must be (speaker, mic)"

    b = raw["body"]
    kind = b["kind"]
    if kind == "none":
        body: BoxBody | RingBody | NoBody = NoBody(kind="none")
    elif kind == "box":
        body = BoxBody(kind="box", size_mm=_tup(b["size_mm"]),
                       plateau_size_mm=_tup(b["plateau_size_mm"]) if b.get("plateau_size_mm") else None,
                       plateau_center_mm=_tup(b["plateau_center_mm"]) if b.get("plateau_center_mm") else None)
        boxes = [(np.zeros(3), np.array(body.size_mm))]
        if body.plateau_size_mm:
            boxes.append((np.array(body.plateau_center_mm), np.array(body.plateau_size_mm)))
        for p in ports:
            for q in p.points_mm:
                assert any(_on_box_surface(np.array(q), c, s) for c, s in boxes), \
                    f"{p.name}: point {q} not on box surface within {PORT_SURFACE_TOL_MM} mm"
    elif kind == "ring":
        body = RingBody(kind="ring", **{k: v for k, v in b.items() if k != "kind"})
    else:
        raise ValueError(f"unknown body kind {kind!r}")

    chirp = ChirpSpec(**raw["chirp"])
    assert PULSE_BAND_HZ[0] <= chirp.f0_hz < chirp.f1_hz <= PULSE_BAND_HZ[1], \
        f"chirp band outside pulse band {PULSE_BAND_HZ}"
    fs = int(raw["fs_hz"])
    assert abs(chirp.frame_period_s * fs - round(chirp.frame_period_s * fs)) < 1e-9, \
        "frame_period_s * fs_hz is not an integer"
    assert raw["frame"] in ("device", "ring")
    return DeviceConfig(name=raw["name"], fs_hz=fs, chirp=chirp,
                        n_range_bins=int(raw["n_range_bins"]), ports=ports,
                        active_pair=(active[0], active[1]), body=body, frame=raw["frame"])


def device_from_yaml(text: str) -> DeviceConfig:
    return parse_device(yaml.safe_load(text))


def load_device(name: str) -> DeviceConfig:
    """Load devices/<name>.yaml; raises ConfigIncomplete listing every TBD(Q-xxx)."""
    return device_from_yaml((_DIR / f"{name}.yaml").read_text())
