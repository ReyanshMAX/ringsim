"""Scene assembly (docs/GEOMETRY.md). Phase 1: the free scene. Phase 2: solid builders for the phone body
(BoxBody incl. plateau), the plate and the rig boxes (D-047). Hand scenes (build_scene) are Phase 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ringsim.devices.config import BoxBody, DeviceConfig, Port


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


def box_mesh(center_m, size_m) -> trimesh.Trimesh:
    """Axis-aligned box solid, device frame, metres."""
    size = np.asarray(size_m, float)
    if size.shape != (3,) or np.any(size <= 0):
        raise ValueError(f"box size must be 3 positive lengths, got {size_m}")
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.asarray(center_m, float))
    return m


def body_meshes(body: BoxBody) -> list[trimesh.Trimesh]:
    """Phone BoxBody as solids (DEVICES.md): the body box centred on the device origin, plus the camera plateau
    box when the config has one. Separate meshes; occupancy ORs them (GEOMETRY.md Voxelization step 3)."""
    if body.kind != "box":
        raise ValueError(f"body_meshes needs a box body, got {body.kind!r}")
    out = [box_mesh((0.0, 0.0, 0.0), np.asarray(body.size_mm, float) * 1e-3)]
    if body.plateau_size_mm is not None:
        if body.plateau_center_mm is None:
            raise ValueError("plateau_size_mm given without plateau_center_mm")
        out.append(box_mesh(np.asarray(body.plateau_center_mm, float) * 1e-3,
                            np.asarray(body.plateau_size_mm, float) * 1e-3))
    return out


def rig_box_meshes(rig_boxes) -> list[trimesh.Trimesh]:
    """Placement.rig_boxes ({center, size} in device-frame mm, D-047) as solids in metres."""
    return [box_mesh(np.asarray(b["center"], float) * 1e-3, np.asarray(b["size"], float) * 1e-3)
            for b in rig_boxes]


def plate_mesh(speaker: Port, plate: dict) -> trimesh.Trimesh:
    """Placement.plate {size_mm: [w, h, t], distance_mm: d} as a solid (GEOMETRY.md Placement).

    Centred on the speaker port's centroid normal axis; the NEAR face is perpendicular to the normal at d from the
    port centroid (lead ruling 2026-11-11, from VALIDATION.md "distance measured port-to-plate"). Thickness t runs
    along the normal, away from the port.

    Only an axis-aligned normal and a square face (w == h) are built: which in-plane axis w and h map to is open
    (seat 2 question S2-6), and for a square plate on an axis-aligned normal the answer does not change the solid.
    """
    w, h, t = (float(v) for v in plate["size_mm"])
    d = float(plate["distance_mm"])
    if min(w, h, t) <= 0 or d <= 0:
        raise ValueError(f"plate size and distance must be > 0, got size_mm {plate['size_mm']}, distance_mm {d}")
    n = np.asarray(speaker.normal, float)
    n = n / np.linalg.norm(n)
    ax = int(np.argmax(np.abs(n)))
    if not np.isclose(abs(n[ax]), 1.0, atol=1e-9):
        raise NotImplementedError(f"plate on a non-axis-aligned normal {speaker.normal}: in-plane axes unspecified (S2-6)")
    if w != h:
        raise NotImplementedError(f"non-square plate {w} x {h} mm: which in-plane axis w and h map to is open (S2-6)")
    size = np.full(3, w)
    size[ax] = t
    centroid = np.asarray(speaker.points_mm, float).mean(0)
    center = centroid + n * (d + t / 2)
    return box_mesh(center * 1e-3, size * 1e-3)
