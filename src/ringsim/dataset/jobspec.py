"""JobSpec: one YAML job = one sim run = one shard (ARCHITECTURE.md JobSpec)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCENES = ("free", "plate", "hand")


@dataclass
class JobSpec:
    job_id: str
    device: str
    scene: str
    dx_mm: float
    poses: dict
    seed: int
    placement: dict | None = None
    dr: dict | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def dx_m(self) -> float:
        return self.dx_mm * 1e-3

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.raw, sort_keys=False)


def parse_job(raw: dict) -> JobSpec:
    assert raw["scene"] in SCENES, raw["scene"]
    for k in ("job_id", "device", "scene", "dx_mm", "poses", "seed"):
        assert k in raw, f"job missing {k!r}"
        if isinstance(raw[k], str) and raw[k].startswith("TBD("):
            raise ValueError(f"job field {k} is {raw[k]}")
    return JobSpec(job_id=raw["job_id"], device=raw["device"], scene=raw["scene"],
                   dx_mm=float(raw["dx_mm"]), poses=dict(raw["poses"]), seed=int(raw["seed"]),
                   placement=raw.get("placement"), dr=raw.get("dr"), raw=raw)


def load_job(path: str | Path) -> JobSpec:
    return parse_job(yaml.safe_load(Path(path).read_text()))
