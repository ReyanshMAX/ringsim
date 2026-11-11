"""Scene assembly (docs/GEOMETRY.md). Phase 1: the free scene. Phase 2: solid builders for the phone body
(BoxBody incl. plateau), the plate and the rig boxes (D-047). Hand scenes (build_scene) are Phase 3."""
from __future__ import annotations

from dataclasses import dataclass, field

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
    boxes: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # axis-aligned solids (center_m, size_m), device frame, voxelized analytically (Q-017, USER ANSWER 2026-11-11)


@dataclass
class Placement:
    """GEOMETRY.md Placement. Lengths stay in mm, angles in degrees, as the JobSpec writes them."""
    hand_offset_mm: tuple[float, float, float] | None = None
    hand_rot_deg: tuple[float, float, float] | None = None
    plate: dict | None = None                       # {size_mm: [w, h, t], distance_mm: d}
    ring_rot_deg: float = 0.0
    ring_shift_mm: float = 0.0
    rig_boxes: tuple[dict, ...] = ()                # {center, size} in device-frame mm (D-047)

    @classmethod
    def from_job(cls, d: dict | None) -> "Placement":
        """JobSpec `placement` block -> Placement. Unknown keys and malformed plate / rig boxes raise."""
        if d is None:
            return cls()
        if not isinstance(d, dict):
            raise ValueError(f"placement must be a mapping, got {type(d).__name__}")
        known = {"hand_offset_mm", "hand_rot_deg", "plate", "ring_rot_deg", "ring_shift_mm", "rig_boxes"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown placement keys (no spec names them): {sorted(unknown)}")
        vec = lambda v: None if v is None else tuple(float(x) for x in v)   # noqa: E731
        plate = d.get("plate")
        if plate is not None:
            if set(plate) != {"size_mm", "distance_mm"} or len(plate["size_mm"]) != 3:
                raise ValueError(f"plate needs exactly size_mm [w, h, t] and distance_mm, got {plate}")
            plate = {"size_mm": [float(v) for v in plate["size_mm"]], "distance_mm": float(plate["distance_mm"])}
        rig = []
        for b in d.get("rig_boxes") or ():
            if set(b) != {"center", "size"} or len(b["center"]) != 3 or len(b["size"]) != 3:
                raise ValueError(f"rig box needs exactly center [x, y, z] and size [w, l, t] in mm, got {b}")
            rig.append({"center": [float(v) for v in b["center"]], "size": [float(v) for v in b["size"]]})
        return cls(hand_offset_mm=vec(d.get("hand_offset_mm")), hand_rot_deg=vec(d.get("hand_rot_deg")),
                   plate=plate, ring_rot_deg=float(d.get("ring_rot_deg", 0.0)),
                   ring_shift_mm=float(d.get("ring_shift_mm", 0.0)), rig_boxes=tuple(rig))


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

    In-plane axes (D-071): w along device x, h along device z, t along the normal. That rule
    names a normal along y (the phone's bottom edge, D-023); any other normal raises.
    """
    w, h, t = (float(v) for v in plate["size_mm"])
    d = float(plate["distance_mm"])
    if min(w, h, t) <= 0 or d <= 0:
        raise ValueError(f"plate size and distance must be > 0, got size_mm {plate['size_mm']}, distance_mm {d}")
    n = np.asarray(speaker.normal, float)
    n = n / np.linalg.norm(n)
    if not np.isclose(abs(n[1]), 1.0, atol=1e-9):
        raise NotImplementedError(f"plate on normal {speaker.normal}: the w->x, h->z rule names a normal along y only")
    size = np.array([w, t, h])
    centroid = np.asarray(speaker.points_mm, float).mean(0)
    center = centroid + n * (d + t / 2)
    return box_mesh(center * 1e-3, size * 1e-3)


def _as_box(m: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """(center_m, size_m) of an axis-aligned box mesh from the builders above."""
    lo, hi = np.asarray(m.bounds, float)
    return (lo + hi) / 2, hi - lo


def _device_boxes(dev: DeviceConfig, placement: Placement) -> list[tuple[np.ndarray, np.ndarray]]:
    if dev.body.kind == "box":
        solids = body_meshes(dev.body)
    elif dev.body.kind == "none":
        solids = []
    else:
        raise NotImplementedError(f"device body {dev.body.kind!r} in a box scene: ring scenes are Phase 5")
    return [_as_box(m) for m in solids + rig_box_meshes(placement.rig_boxes)]


def device_scene(dev: DeviceConfig, placement: Placement) -> Scene:
    """Device-only scene: body (incl. plateau) + rig boxes, no plate, no hand. The ref_ir scene (D-042, D-056)."""
    spk, mic = active_ports(dev)
    return Scene(meshes=[], speaker=spk, mic=mic, joints21=None, labels=None, hand_scale_m=None,
                 boxes=_device_boxes(dev, placement))


def plate_scene(dev: DeviceConfig, placement: Placement) -> Scene:
    """`scene: plate`: device_scene + the plate. placement.plate None = the Protocol B background (D-047)."""
    s = device_scene(dev, placement)
    if placement.plate is not None:
        s.boxes.append(_as_box(plate_mesh(s.speaker, placement.plate)))
    return s
