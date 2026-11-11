"""JobSpec: the job YAML of ARCHITECTURE.md "JobSpec" (D-059).

Only `dx_mm` is converted to SI here (`dx_m`). The nested blocks (`placement`, `poses`, `motion`, `dr`) are
kept exactly as written in the YAML, in the YAML's own units, until the module that consumes each of them has a
spec for its typed form (GEOMETRY.md / DATASET.md); converting them here would be inventing units rules.
Any value written `TBD(Q-xxx)` anywhere in the file raises `ConfigIncomplete`, listing every one.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

SCENES = ("free", "plate", "hand")
_TBD = re.compile(r"TBD\((Q-\d+)\)")


class ConfigIncomplete(ValueError):
    """A job value is still TBD(Q-xxx). Same meaning as devices' ConfigIncomplete for device files."""


@dataclass
class JobSpec:
    job_id: str
    device: str
    scene: str                       # free | plate | hand
    dx_m: float                      # from dx_mm
    placement: dict | None
    poses: dict                      # {source, n_frames}
    seed: int
    motion: dict | None = None
    dr: dict | None = None
    extra: dict = field(default_factory=dict)   # keys no spec names are refused, so this stays empty

    @property
    def n_frames(self) -> int:
        return int(self.poses["n_frames"])


_KNOWN = {"job_id", "device", "scene", "dx_mm", "placement", "poses", "motion", "dr", "seed"}


def _find_tbd(node: Any, path: str, out: list[str]) -> None:
    if isinstance(node, str):
        m = _TBD.search(node)
        if m:
            out.append(f"{m.group(1)}: {path}")
    elif isinstance(node, dict):
        for k, v in node.items():
            _find_tbd(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _find_tbd(v, f"{path}[{i}]", out)


def job_from_dict(raw: dict) -> JobSpec:
    tbd: list[str] = []
    _find_tbd(raw, "", tbd)
    if tbd:
        raise ConfigIncomplete("; ".join(tbd))
    unknown = set(raw) - _KNOWN
    if unknown:
        raise ValueError(f"unknown job keys (no spec names them): {sorted(unknown)}")
    missing = {"job_id", "device", "scene", "dx_mm", "poses", "seed"} - set(raw)
    if missing:
        raise ValueError(f"missing job keys: {sorted(missing)}")
    if raw["scene"] not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {raw['scene']!r}")
    poses = raw["poses"]
    if not isinstance(poses, dict) or "source" not in poses or "n_frames" not in poses:
        raise ValueError("poses needs source and n_frames")
    dx_m = float(raw["dx_mm"]) / 1000.0
    if dx_m <= 0:
        raise ValueError("dx_mm must be > 0")
    return JobSpec(job_id=str(raw["job_id"]), device=str(raw["device"]), scene=raw["scene"], dx_m=dx_m,
                   placement=raw.get("placement"), poses=poses, seed=int(raw["seed"]),
                   motion=raw.get("motion"), dr=raw.get("dr"))


def load_job(path: str) -> JobSpec:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: not a mapping")
    return job_from_dict(raw)


def job_yaml(job: JobSpec) -> str:
    """The resolved JobSpec as stored in a shard's `job_yaml` attr."""
    d = asdict(job)
    d.pop("extra", None)
    return yaml.safe_dump(d, sort_keys=True)
