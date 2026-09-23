"""The asset types the scenario builders in `humanoidtoolbench.assets.tools` return.

`BenchmarkAsset` describes a scanned mesh with its benchmark role, declared
mass and normalised length; `PrimitiveToolAsset` describes an object built from
primitive `geoms` (the balls, ice blocks and ring markers), which the MuJoCo
engine's object builder draws directly.
"""

from __future__ import annotations

from typing import Any

from humanoidtoolbench.core.asset import Asset
from humanoidtoolbench.core.object import SemanticAnnotated


class BenchmarkAsset(Asset, SemanticAnnotated):
    """A mesh asset with its benchmark role and normalised length attached."""

    def __init__(self, entry: dict[str, Any]) -> None:
        super().__init__(
            uid=entry["uid"],
            collision_meshes_mujoco=list(entry["collision_meshes_mujoco"]),
        )
        self.label = entry["label"]
        self.name = entry["name"]
        self.description = f"{entry['name']} ({entry['group_label']})"
        self.group = entry["group"]
        self.source = entry["source"]
        self.license = entry["license"]
        self.mass = float(entry["mass"])
        self.friction = [
            float(value) for value in entry.get("friction") or [0.9, 0.05, 0.005]
        ]
        self.mesh_scale = list(entry["mesh_scale"])
        self.visual_mesh = entry["visual_mesh"]
        self.texture = entry["texture"]
        self.up_axis = entry["up_axis"]
        self.extent = list(entry["extent"])
        self.height = float(entry["height"])
        self.footprint = float(entry["footprint"])
        self.n_convex_parts = int(entry["n_convex_parts"])
        self.role = entry["role"]
        self.kind = entry["kind"]
        self.length = float(entry["length"])
        self.grip_width = float(entry.get("grip_width") or 0.0)
        self.colour = entry.get("colour")
        self.holds = entry.get("holds")
        self.interior = entry.get("interior")

    def __repr__(self) -> str:
        return (
            f"BenchmarkAsset(uid={self.uid!r}, role={self.role!r}, "
            f"mass={self.mass:.3f}kg)"
        )


class PrimitiveToolAsset(BenchmarkAsset):
    """An object made of primitive geoms rather than meshes.

    Carries the same placement fields as a mesh asset (`height`, `footprint`,
    `extent`) so a task never has to care which it got. The
    engine branches on `geoms` being present.
    """

    def __init__(self, entry: dict[str, Any]) -> None:
        # A primitive has no collider files; keep the field so every consumer
        # that iterates `collision_meshes_mujoco` sees an empty list, not None.
        super().__init__({**entry, "collision_meshes_mujoco": []})
        self.geoms = [dict(g) for g in entry["geoms"]]
        # A static asset gets no free joint, which is what a marker needs: it is
        # drawn where it was put and never simulated.
        self.static = bool(entry.get("static", False))
        self.visual_mesh = None
        self.texture = None

    @property
    def rgba(self) -> list[float]:
        return list(self.geoms[0]["rgba"])


__all__ = ["BenchmarkAsset", "PrimitiveToolAsset"]
