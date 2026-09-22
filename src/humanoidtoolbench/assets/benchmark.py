"""Benchmark assets for the six roles used by Push, Sweep, and Mop.

The focused catalog is built by `scripts/dataset/build_benchmark_assets.py` into
`humanoidtoolbench/resources/benchmark_assets/catalog.json`, ten per role, every mesh
normalised to one length per role.

    from humanoidtoolbench.assets.benchmark import BenchmarkCatalog

    cat = BenchmarkCatalog()
    stick = cat.sample_role("stick", seed=0)
    block = cat.load("block_red_50")

Two kinds of entry come out of the same catalogue: `BenchmarkAsset` for a mesh,
and `PrimitiveToolAsset` for a box, sphere, or capsule built from `geoms`. The
  MuJoCo engine's object builder draws directly. Objaverse has no tabletop
  wooden block and exactly one dustpan, so those roles are procedural, and a
  primitive is also the only way to vary colour independently of shape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import transforms3d as t3d

from humanoidtoolbench.core.asset import Asset
from humanoidtoolbench.core.object import SemanticAnnotated

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources"
DEFAULT_CATALOG = Path(
    os.environ.get(
        "HUMANOIDTOOLBENCH_BENCHMARK_CATALOG",
        RESOURCE_ROOT / "benchmark_assets" / "catalog.json",
    )
)

ROLES = ("stick", "banana", "vase", "dustpan", "ball", "block")

_Y_UP_FIX = t3d.euler.euler2quat(np.pi / 2, 0.0, 0.0)
_X_UP_FIX = t3d.euler.euler2quat(0.0, -np.pi / 2, 0.0)


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
        self._rest_offset = float(entry["rest_offset"])
        self.n_convex_parts = int(entry["n_convex_parts"])
        self.role = entry["role"]
        self.kind = entry["kind"]
        self.length = float(entry["length"])
        self.grip_width = float(entry.get("grip_width") or 0.0)
        self.colour = entry.get("colour")
        self.holds = entry.get("holds")
        self.interior = entry.get("interior")

    @property
    def upright_quat(self) -> list[float]:
        if self.up_axis == "y":
            return [float(value) for value in _Y_UP_FIX]
        if self.up_axis == "x":
            return [float(value) for value in _X_UP_FIX]
        return [1.0, 0.0, 0.0, 0.0]

    def pose_on(
        self,
        x: float,
        y: float,
        surface_z: float,
        yaw: float = 0.0,
        clearance: float = 0.0,
    ) -> tuple[list[float], list[float]]:
        quaternion = t3d.quaternions.qmult(
            t3d.euler.euler2quat(0.0, 0.0, yaw), self.upright_quat
        )
        position = [float(x), float(y), surface_z + self._rest_offset + clearance]
        return position, [float(value) for value in quaternion]

    def __repr__(self) -> str:
        return (
            f"BenchmarkAsset(uid={self.uid!r}, role={self.role!r}, "
            f"mass={self.mass:.3f}kg)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "benchmark",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "group": self.group,
            "source": self.source,
            "mass": self.mass,
            "extent": self.extent,
            "license": self.license,
            "role": self.role,
            "kind": self.kind,
            "length": self.length,
            "colour": self.colour,
        }


class PrimitiveToolAsset(BenchmarkAsset):
    """An object made of primitive geoms rather than meshes.

    Carries the same placement interface as a mesh asset (`pose_on`,
    `rest_offset`, `footprint`) so a task never has to care which it got. The
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


def _resolve(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Turn the catalogue's relative mesh paths into paths on this machine.

    The catalogue ships with paths relative to itself so the folder can be
    committed, copied or unpacked anywhere; absolute paths are left alone, so an
    older catalogue built in place still loads.
    """

    def fix(rel):
        return (
            None if rel is None else str(rel if Path(rel).is_absolute() else root / rel)
        )

    out = dict(entry)
    out["collision_meshes_mujoco"] = [fix(m) for m in entry["collision_meshes_mujoco"]]
    out["visual_mesh"] = fix(entry.get("visual_mesh"))
    out["texture"] = fix(entry.get("texture"))
    return out


def _make(entry: dict[str, Any]) -> BenchmarkAsset:
    return (
        PrimitiveToolAsset(entry)
        if entry["kind"] == "primitive"
        else BenchmarkAsset(entry)
    )


class BenchmarkCatalog:
    """The packaged benchmark asset catalog, queried by role."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_CATALOG)
        if not self.path.exists():
            raise FileNotFoundError(
                f"No benchmark catalogue at {self.path}. "
                "Re-download the release to restore packaged benchmark assets."
            )
        root = self.path.parent
        self._entries = {
            e["uid"]: _resolve(e, root) for e in json.loads(self.path.read_text())
        }

    def load(self, asset_id: str) -> BenchmarkAsset:
        if asset_id not in self._entries:
            raise KeyError(f"unknown benchmark asset: {asset_id!r}")
        return _make(self._entries[asset_id])

    def sample(self, exclude: list[str] | None = None) -> BenchmarkAsset:
        pool = [u for u in self._entries if u not in set(exclude or ())]
        return self.load(str(np.random.choice(pool)))

    # -- querying ----------------------------------------------------------

    @property
    def roles(self) -> list[str]:
        return sorted({e["role"] for e in self._entries.values()})

    def uids_for(
        self,
        role: str,
        kind: str | None = None,
        exclude: Sequence[str] | None = None,
        max_footprint: float | None = None,
    ) -> list[str]:
        banned = set(exclude or ())
        return sorted(
            u
            for u, e in self._entries.items()
            if e["role"] == role
            and u not in banned
            and (kind is None or e["kind"] == kind)
            and (max_footprint is None or e["footprint"] <= max_footprint)
        )

    def sample_role(
        self,
        role: str,
        seed: int | None = None,
        kind: str | None = None,
        exclude: Sequence[str] | None = None,
        max_footprint: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> BenchmarkAsset:
        pool = self.uids_for(
            role, kind=kind, exclude=exclude, max_footprint=max_footprint
        )
        if not pool and max_footprint is not None:  # nothing slim enough left
            pool = self.uids_for(role, kind=kind, exclude=exclude)
        if not pool:
            raise ValueError(f"no benchmark asset for role {role!r}")
        rng = rng or np.random.default_rng(seed)
        return self.load(pool[int(rng.integers(len(pool)))])


__all__ = [
    "BenchmarkAsset",
    "PrimitiveToolAsset",
    "BenchmarkCatalog",
    "ROLES",
]
