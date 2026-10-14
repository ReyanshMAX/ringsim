"""Scene assembly (GEOMETRY.md). Phase 1: free scene only; hand scenes are Phase 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ringsim.devices.config import DeviceConfig, Port


@dataclass
class Placement:
    hand_offset_mm: tuple[float, float, float] | None = None
    hand_rot_deg: tuple[float, float, float] | None = None
    plate: dict | None = None
    ring_rot_deg: float = 0.0
    ring_shift_mm: float = 0.0


@dataclass
class Scene:
    meshes: list[trimesh.Trimesh]       # all solids, device frame, metres
    speaker: Port
    mic: Port
    joints21: np.ndarray | None         # [21,3]; None for scenes without a hand (free, plate)
    labels: np.ndarray | None           # [20,3] canonical wrist frame (D-025); None without a hand
    hand_scale_m: float | None


def free_scene(dev: DeviceConfig) -> Scene:
    """scene: free — no solids, just the active port pair (BUILD.md Phase 1)."""
    assert dev.body.kind == "none", "scene 'free' is for bodiless devices (test_pair)"
    return Scene(meshes=[], speaker=dev.speaker, mic=dev.mic,
                 joints21=None, labels=None, hand_scale_m=None)
