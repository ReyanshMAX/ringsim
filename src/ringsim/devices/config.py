"""DeviceConfig schema and loader (docs/DEVICES.md, D-006).

Every device is data: `src/ringsim/devices/<name>.yaml`, validated by `load_device`.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from ringsim.proc.chirp import ChirpSpec

PULSE_BAND_HZ = (16_000.0, 28_000.0)    # SOLVER.md pulse band; chirp must lie inside
PORT_SURFACE_TOL_MM = 0.5               # box port points lie on the box surface within this
_TBD = re.compile(r"^TBD\((Q-\d{3})\)$")
_DIR = Path(__file__).parent


class ConfigIncomplete(Exception):
    """A required config value is TBD(Q-xxx). The message lists every one: "Q-xxx: <field>; ..."."""


class DroppedPortWarning(UserWarning):
    """A port outside active_pair holds a TBD(Q-xxx) value and was dropped."""


@dataclass(frozen=True)
class Port:
    name: str
    kind: Literal["speaker", "mic"]
    points_mm: tuple[tuple[float, float, float], ...]   # 1+ holes; all driven/summed in phase
    normal: tuple[float, float, float]                  # outward unit vector


@dataclass(frozen=True)
class BoxBody:
    kind: Literal["box"]
    size_mm: tuple[float, float, float]                 # x width, y length, z thickness
    plateau_size_mm: tuple[float, float, float] | None  # camera plateau on rear face
    plateau_center_mm: tuple[float, float, float] | None


@dataclass(frozen=True)
class RingBody:
    kind: Literal["ring"]
    # inner diameter is NOT configured: derived per sequence from the MANO finger (D-024, D-049)
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
        return self.n_range_bins * 343.0 / (2 * self.fs_hz)   # DEVICES.md loader rule


def _find_tbd(obj: Any, path: str) -> list[tuple[str, str]]:
    """(Q-ID, field path) for every TBD(Q-xxx) string inside obj, in file order."""
    if isinstance(obj, str):
        m = _TBD.match(obj.strip())
        return [(m.group(1), path)] if m else []
    if isinstance(obj, dict):
        return [t for k, v in obj.items() for t in _find_tbd(v, f"{path}.{k}" if path else str(k))]
    if isinstance(obj, list):
        return [t for i, v in enumerate(obj) for t in _find_tbd(v, f"{path}[{i}]")]
    return []


def _vec(x) -> tuple:
    return tuple(_vec(v) if isinstance(v, (list, tuple)) else float(v) for v in x)


def _on_box_surface(p: np.ndarray, center: np.ndarray, size: np.ndarray) -> bool:
    d = np.abs(p - center) - size / 2           # > 0 outside along that axis
    outside = float(np.linalg.norm(np.maximum(d, 0.0)))
    if outside > 0:
        return outside <= PORT_SURFACE_TOL_MM
    return -float(d.max()) <= PORT_SURFACE_TOL_MM   # inside: distance to the nearest face


def _parse(raw: dict) -> DeviceConfig:
    if not isinstance(raw, dict):
        raise ValueError("device config must be a YAML mapping")
    name = raw.get("name")
    active = list(raw.get("active_pair") or [])

    # TBD rule: one ConfigIncomplete listing every TBD, except ports outside active_pair (dropped + warning).
    kept, missing = [], []
    for p in raw.get("ports") or []:
        tbd = _find_tbd(p, f"ports.{p.get('name')}")
        if tbd and p.get("name") not in active:
            qids = ", ".join(sorted({q for q, _ in tbd}))
            warnings.warn(f"{name}: dropping port {p.get('name')!r} (TBD {qids})", DroppedPortWarning, stacklevel=3)
            continue
        missing += tbd
        kept.append(p)
    missing += _find_tbd({k: v for k, v in raw.items() if k != "ports"}, "")
    if missing:
        raise ConfigIncomplete("; ".join(f"{q}: {f}" for q, f in missing))

    ports = tuple(Port(name=p["name"], kind=p["kind"], points_mm=_vec(p["points_mm"]), normal=_vec(p["normal"]))
                  for p in kept)
    for p in ports:
        if p.kind not in ("speaker", "mic"):
            raise ValueError(f"{name}: port {p.name!r} kind {p.kind!r}")
        if not p.points_mm or any(len(q) != 3 for q in p.points_mm) or len(p.normal) != 3:
            raise ValueError(f"{name}: port {p.name!r} needs 1+ 3D points and a 3D normal")
        if abs(np.linalg.norm(p.normal) - 1.0) > 1e-6:
            raise ValueError(f"{name}: port {p.name!r} normal is not a unit vector")

    by_name = {p.name: p for p in ports}
    if len(active) != 2 or any(n not in by_name for n in active):
        raise ValueError(f"{name}: active_pair {active} must name two ports in `ports`")
    kinds = sorted(by_name[n].kind for n in active)
    if kinds != ["mic", "speaker"]:
        raise ValueError(f"{name}: active_pair {active} must name one speaker and one mic")

    b = dict(raw["body"])
    kind = b.pop("kind")
    if kind == "none":
        body: BoxBody | RingBody | NoBody = NoBody(kind="none")
    elif kind == "box":
        body = BoxBody(kind="box", size_mm=_vec(b["size_mm"]),
                       plateau_size_mm=_vec(b["plateau_size_mm"]) if b.get("plateau_size_mm") is not None else None,
                       plateau_center_mm=_vec(b["plateau_center_mm"]) if b.get("plateau_center_mm") is not None else None)
        boxes = [(np.zeros(3), np.array(body.size_mm))]
        if body.plateau_size_mm is not None:
            boxes.append((np.array(body.plateau_center_mm), np.array(body.plateau_size_mm)))
        for p in ports:
            for q in p.points_mm:
                if not any(_on_box_surface(np.array(q), c, s) for c, s in boxes):
                    raise ValueError(f"{name}: port {p.name!r} point {q} is not on the box surface "
                                     f"within {PORT_SURFACE_TOL_MM} mm")
    elif kind == "ring":
        floats = {k: float(v) for k, v in b.items() if k not in ("finger", "phalanx")}
        body = RingBody(kind="ring", finger=b["finger"], phalanx=b["phalanx"], **floats)
    else:
        raise ValueError(f"{name}: unknown body kind {kind!r}")

    chirp = ChirpSpec(**raw["chirp"])
    lo, hi = sorted((float(chirp.f0_hz), float(chirp.f1_hz)))
    if not (PULSE_BAND_HZ[0] <= lo and hi <= PULSE_BAND_HZ[1]):
        raise ValueError(f"{name}: chirp band {lo:.0f}-{hi:.0f} Hz outside the pulse band {PULSE_BAND_HZ}")
    fs = int(raw["fs_hz"])
    n = chirp.frame_period_s * fs
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{name}: frame_period_s * fs_hz = {n!r} is not an integer number of samples")
    if raw["frame"] not in ("device", "ring"):
        raise ValueError(f"{name}: frame {raw['frame']!r}")

    return DeviceConfig(name=name, fs_hz=fs, chirp=chirp, n_range_bins=int(raw["n_range_bins"]), ports=ports,
                        active_pair=(active[0], active[1]), body=body, frame=raw["frame"])


def device_yaml(dev: DeviceConfig) -> str:
    """The fully resolved DeviceConfig as YAML (shard attr `device_yaml`); TBD-dropped ports are absent.
    `load_device_yaml(device_yaml(dev)) == dev`."""
    return yaml.safe_dump(_plain(asdict(dev)), sort_keys=False)


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def load_device_yaml(text: str) -> DeviceConfig:
    """DeviceConfig from YAML text, every loader rule applied (D-055; the reader calls it on a shard's device_yaml)."""
    return _parse(yaml.safe_load(text))


def load_device(name: str) -> DeviceConfig:
    """Load `devices/<name>.yaml`; raises ConfigIncomplete listing every TBD(Q-xxx) (DEVICES.md)."""
    return load_device_yaml((_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
