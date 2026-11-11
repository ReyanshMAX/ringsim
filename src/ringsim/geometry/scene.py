"""Scene assembly (docs/GEOMETRY.md). Phase 1: the free scene only; hand scenes (build_scene) are Phase 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ringsim.devices.config import DeviceConfig, Port


@dataclass
class Scene:
    meshes: list[trimesh.Trimesh]       # all solids, device frame, metres
    speaker: Port
    mic: Port
    joints21: np.ndarray | None         # [21,3] scene frame, metres; None without a hand (free, plate)
    labels: np.ndarray | None           # [20,3] canonical wrist frame, metres (D-025); None without a hand
    hand_scale_m: float | None          # |wrist -> little MCP|; None without a hand


def active_ports(dev: DeviceConfig) -> tuple[Port, Port]:
    """(speaker, mic) named by dev.active_pair."""
    by_name = {p.name: p for p in dev.ports}
    a, b = (by_name[n] for n in dev.active_pair)
    return (a, b) if a.kind == "speaker" else (b, a)


def free_scene(dev: DeviceConfig) -> Scene:
    """`scene: free`: no solids, the device's active port pair only (lead ruling 2026-11-11)."""
    spk, mic = active_ports(dev)
    return Scene(meshes=[], speaker=spk, mic=mic, joints21=None, labels=None, hand_scale_m=None)
