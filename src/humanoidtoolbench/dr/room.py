"""The room the tabletop tasks stand in, as a randomiser.

Geometry is fixed and shared: the 5 x 5 x 3 m room measured in
`humanoid-tool-use-benchmark` (toolbench/scene/scene.py), with 0.05 m walls, a
0.08 m baseboard and a floor slab. Every task that asks for a room gets the
same one. What varies per episode is only how it *looks*: the albedo of each
surface, and the material bound to it.

Why a separate randomiser rather than `TabletopSceneDR`: this one owns the
walls, floor, and table finish, while the scene randomizer owns table geometry.

Materials come from the packaged vMaterials index. `humanoidtoolbench.assets.textures`
turns each entry into an image MuJoCo can map, since MuJoCo cannot read MDL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from humanoidtoolbench.assets.primitive import Box
from humanoidtoolbench.assets.textures import material_texture
from humanoidtoolbench.core.randomizer import Randomizer, RandomizerCfg
from humanoidtoolbench.dr.scene import TabletopSceneDR, TabletopSceneDRCfg
from humanoidtoolbench.dr.types import Box as Range
from humanoidtoolbench.utils import resolve_res_path


class RoomDR(Randomizer):
    """Produces the room's primitives, already dressed."""

    def __init__(self, cfg: RoomDRCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self._materials = np.load(
            resolve_res_path("vMaterials_2/material_split.npy"), allow_pickle=True
        ).item()

    # -- helpers -----------------------------------------------------------

    def _pick_material(
        self,
        split: str,
        families: tuple[str, ...],
        listing: str = "ground",
        exclude: tuple[str, ...] = (),
    ) -> dict:
        """One vMaterials entry from the allowed families.

        The index carries two listings per split. `table` is a subset of
        `ground` with the materials that belong on a wall or underfoot taken
        out, so the bench draws from it: on the `ground` listing a fifth of the
        bench's train pool was mosaic tiling and mossy brick facade, 160 of its
        752 entries, and a workbench came out tiled.

        `exclude` names `.mdl` stems to drop after the family filter, for the
        few a family lets through that the surface should not have.
        """
        if split not in self._materials:
            # Falling back to train would hand an eval the materials it was
            # trained on and say nothing, which is the one failure here that
            # looks like a result. `val` was folded into `test` and is the
            # name most likely to arrive stale.
            raise ValueError(
                f"unknown material split {split!r}; the index has "
                f"{sorted(self._materials)}"
            )
        entries = self._materials[split]
        pool = [
            m
            for m in entries[listing]
            if self._family_of(m) in families and self._stem_of(m) not in exclude
        ]
        if not pool:
            pool = self._materials["train"][listing]
        return pool[int(np.random.randint(len(pool)))]

    @staticmethod
    def _family_of(material: dict) -> str:
        """The directory the `.mdl` sits in, which is vMaterials' own family."""
        parts = str(material.get("path", "")).split("/")
        return parts[-2] if len(parts) >= 2 else ""

    @staticmethod
    def _stem_of(material: dict) -> str:
        """The `.mdl` file's name without its suffix."""
        name = str(material.get("path", "")).split("/")[-1]
        return name[:-4] if name.endswith(".mdl") else name

    def _surface(
        self,
        split: str,
        families,
        albedo_range,
        texrepeat,
        specular,
        shininess,
        listing: str = "ground",
        exclude: tuple[str, ...] = (),
    ) -> dict:
        """A `set_material` dict carrying both the image and the albedo."""
        if self.cfg.room_mode == "fixed":
            albedo = 0.5 * (albedo_range[0] + albedo_range[1])
            material = self._materials["train"][listing][0]
        else:
            albedo = float(np.random.uniform(*albedo_range))
            material = self._pick_material(split, families, listing, exclude)

        texture, _ = material_texture(material)
        # The texture already carries the material's colour and MuJoCo
        # multiplies rgba into it, so rgba has to be a plain dimmer. Putting
        # the tint here as well squared it: a beech bench came out averaging
        # (0.14, 0.06, 0.02) where the material's own colour is
        # (0.32, 0.21, 0.11), twice as dark and far deeper.
        rgba = [albedo, albedo, albedo, 1.0]
        return {
            "texture": texture,
            "texrepeat": list(texrepeat),
            "rgba": rgba,
            "specular": specular,
            "shininess": shininess,
            # Recorded so an episode can say which material it drew.
            "vmaterial": material,
        }

    @staticmethod
    def _slab(size, position, material) -> Box:
        box = Box(
            size=list(size), position=list(position), quaternion=[1.0, 0.0, 0.0, 0.0]
        )
        box.set_material(material)
        return box

    # -- Randomizer --------------------------------------------------------

    def surfaces(self, split: str = "train") -> dict[str, dict]:
        """One draw per episode, shared by the room and the tables.

        `apply` asks twice, once through `__call__` for the room and once for
        the tables; the second call gets copies of the first draw.
        """
        if self._inner_state is not None:
            return {name: dict(surface) for name, surface in self._inner_state.items()}
        cfg = self.cfg
        drawn = {
            "wall": self._surface(
                split, cfg.wall_families, cfg.wall_albedo, cfg.texrepeat_wall, 0.05, 0.1
            ),
            "floor": self._surface(
                split,
                cfg.floor_families,
                cfg.floor_albedo,
                cfg.texrepeat_floor,
                0.18,
                0.3,
                exclude=cfg.floor_exclude,
            ),
            "baseboard": self._surface(
                split,
                cfg.wall_families,
                cfg.baseboard_albedo,
                cfg.texrepeat_wall,
                0.08,
                0.15,
            ),
            # The bench takes the index's own `table` listing; the room's
            # surfaces take `ground`. The legs go with the top they hold up.
            "table": self._surface(
                split,
                cfg.table_families,
                cfg.table_albedo,
                cfg.texrepeat_table,
                0.25,
                0.45,
                listing="table",
            ),
            "leg": self._surface(
                split,
                cfg.leg_families,
                cfg.leg_albedo,
                cfg.texrepeat_table,
                0.45,
                0.55,
                listing="table",
            ),
        }
        return super()._transient(drawn)

    def apply(self, layout, split: str = "train") -> dict[str, Box]:
        """Put the room in the layout and dress whatever tables it already has.

        The tables come from the scene randomiser and arrive carrying a
        vMaterials entry, which is MDL and so invisible to MuJoCo. They are
        re-dressed here with the image form of a material drawn from the same
        index, so the bench is randomised rather than a fixed wood.

        Called after the layout is otherwise built: the room is decoration, and
        appending to the layout afterwards cannot move what the spatial
        randomiser placed.
        """
        room = self(split)
        for name, primitive in room.items():
            layout.add_primitive(name, primitive)

        surfaces = self.surfaces(split)
        for key in ("table", "table2", "tool_table"):
            table = layout.actors.get(key)
            if table is None or not hasattr(table, "set_material"):
                continue
            table.set_material(dict(surfaces["table"]))
            for name, part in self._table_parts(key, table, surfaces).items():
                layout.add_primitive(name, part)
                room[name] = part
        return room

    def _table_parts(self, prefix: str, table, surfaces: dict) -> dict[str, Box]:
        """Apron rails and four legs under a table slab.

        Read off the slab that is already in the layout rather than off the
        config, so the legs follow wherever the scene randomiser put the table
        and whatever size it gave it.
        """
        cfg = self.cfg
        cx, cy, cz = table.pose.position
        tx, ty, tz = table.size
        slab_bottom = cz - 0.5 * tz
        if slab_bottom <= 0.0:
            return {}  # a slab sitting on the floor has nothing to stand on

        leg = cfg.leg_side
        lx = 0.5 * tx - cfg.leg_inset - 0.5 * leg
        ly = 0.5 * ty - cfg.leg_inset - 0.5 * leg
        if lx <= 0 or ly <= 0:
            return {}

        parts: dict[str, Box] = {}
        ah, at = cfg.apron_height, cfg.apron_thickness
        apron_z = slab_bottom - 0.5 * ah
        # Each rail spans the gap *between* two legs (2*l - leg), not across
        # their outer faces (2*l + leg). Running through the legs left the two
        # boxes interpenetrating at every corner, which renders as z-fighting on
        # the leg faces and reads as a modelling mistake on camera.
        for name, (size, pos) in {
            f"{prefix}_apron_y_pos": ((2 * lx - leg, at, ah), (cx, cy + ly, apron_z)),
            f"{prefix}_apron_y_neg": ((2 * lx - leg, at, ah), (cx, cy - ly, apron_z)),
            f"{prefix}_apron_x_pos": ((at, 2 * ly - leg, ah), (cx + lx, cy, apron_z)),
            f"{prefix}_apron_x_neg": ((at, 2 * ly - leg, ah), (cx - lx, cy, apron_z)),
        }.items():
            parts[name] = self._slab(size, pos, surfaces["table"])

        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            parts[f"{prefix}_leg_{i}"] = self._slab(
                (leg, leg, slab_bottom),
                (cx + sx * lx, cy + sy * ly, 0.5 * slab_bottom),
                surfaces["leg"],
            )
        return parts

    def __call__(self, split: str = "train", **kwargs) -> dict[str, Box]:
        """Named primitives to hand to `layout.add_primitive`."""
        del kwargs
        cfg = self.cfg
        surfaces = self.surfaces(split)

        w, d = cfg.inner
        t, h = cfg.thickness, cfg.height
        bh, bd, ft, lift = (
            cfg.baseboard_height,
            cfg.baseboard_depth,
            cfg.floor_thickness,
            cfg.floor_lift,
        )
        cx, cy = cfg.centre
        half_w, half_d = 0.5 * w, 0.5 * d

        room: dict[str, Box] = {
            "floor": self._slab(
                (w, d, ft), (cx, cy, lift - 0.5 * ft), surfaces["floor"]
            ),
        }
        for name, (pos, size) in {
            "wall_x_pos": ([cx + half_w + 0.5 * t, cy, 0.5 * h], [t, d + 2 * t, h]),
            "wall_x_neg": ([cx - half_w - 0.5 * t, cy, 0.5 * h], [t, d + 2 * t, h]),
            "wall_y_pos": ([cx, cy + half_d + 0.5 * t, 0.5 * h], [w + 2 * t, t, h]),
            "wall_y_neg": ([cx, cy - half_d - 0.5 * t, 0.5 * h], [w + 2 * t, t, h]),
        }.items():
            room[name] = self._slab(size, pos, surfaces["wall"])
        for name, (pos, size) in {
            "baseboard_x_pos": (
                [cx + half_w - 0.5 * bd, cy, lift + 0.5 * bh],
                [bd, d, bh],
            ),
            "baseboard_x_neg": (
                [cx - half_w + 0.5 * bd, cy, lift + 0.5 * bh],
                [bd, d, bh],
            ),
            "baseboard_y_pos": (
                [cx, cy + half_d - 0.5 * bd, lift + 0.5 * bh],
                [w, bd, bh],
            ),
            "baseboard_y_neg": (
                [cx, cy - half_d + 0.5 * bd, lift + 0.5 * bh],
                [w, bd, bh],
            ),
        }.items():
            room[name] = self._slab(size, pos, surfaces["baseboard"])
        return room


@dataclass
class RoomDRCfg(RandomizerCfg):
    """How the room looks. Its size is not a knob: every task shares one room."""

    # "fixed" keeps the default albedos and one material per surface;
    # "random" redraws both each episode.
    room_mode: str = "random"

    inner: tuple[float, float] = (5.0, 5.0)
    height: float = 3.0
    thickness: float = 0.05
    baseboard_height: float = 0.08
    baseboard_depth: float = 0.02
    floor_thickness: float = 0.02
    # The engine puts an infinite ground plane at z = 0; the floor slab is
    # lifted clear of it rather than left coplanar and z-fighting.
    floor_lift: float = 0.002
    centre: tuple[float, float] = (0.0, 0.0)

    # Albedo ranges, sampled per episode in "random" mode.
    wall_albedo: tuple[float, float] = (0.72, 0.94)
    floor_albedo: tuple[float, float] = (0.55, 0.85)
    baseboard_albedo: tuple[float, float] = (0.35, 0.65)

    # Which vMaterials families each surface may draw from. Restricted on
    # purpose: a floor of brushed metal or a wall of carpet reads as a bug.
    wall_families: tuple[str, ...] = ("Plaster", "Paint", "Paper")
    # The room is indoors. `Ground` is the outdoor family and every one of its
    # entries is outdoor: gravel, mulch, fallen leaves, cobblestone, paving.
    floor_families: tuple[str, ...] = ("Concrete", "Stone", "Ceramic", "Wood")
    # A workbench is wood or a hard wipe-clean surface, never carpet or glass,
    # and never tiled: glazed ceramic is a floor and a splashback, not a bench.
    table_families: tuple[str, ...] = ("Wood", "Stone", "Metal", "Plastic")

    # What a family lets through that the surface should still not have.
    # Matched on the `.mdl` stem, which is the material's identity: one file
    # holds one material at several finishes, and the whole file goes or stays.
    # Two kinds go: the outdoor stragglers left in the floor's families once
    # `Ground` is gone, which are weathered and mossy concrete and tree bark;
    # and the ones that are not flooring at all, aged concrete wall and OSB
    # sheathing board. `Concrete_Wall_Even` stays because plain even concrete
    # reads as a floor slab as much as a wall.
    floor_exclude: tuple[str, ...] = (
        "Concrete_Wall_Aged",
        "Concrete_Wall_Aged_Scratched",
        "OSB_Wood",
        "OSB_Wood_Splattered",
        "Spongy_Concrete_Weathered",
        "Spongy_Concrete_Weathered_Mossy",
        "Stone_Pores_Weathered",
        "Wood_Bark",
    )

    table_albedo: tuple[float, float] = (0.55, 0.95)
    leg_albedo: tuple[float, float] = (0.35, 0.75)
    leg_families: tuple[str, ...] = ("Metal", "Wood")

    # A slab on legs, not a plinth. Sizes from the benchmark's table.
    leg_side: float = 0.055
    leg_inset: float = 0.055  # slab edge to the leg's outer face
    apron_height: float = 0.070
    apron_thickness: float = 0.020

    texrepeat_wall: tuple[float, float] = (1.0, 1.0)
    texrepeat_floor: tuple[float, float] = (0.6, 0.6)
    texrepeat_table: tuple[float, float] = (1.1, 1.1)

    randmizer_class: "Randomizer" = RoomDR


class ToolbenchSceneDR(TabletopSceneDR):
    """Build the benchmark's two table slabs without a background scene asset."""


@dataclass
class ToolbenchSceneDRCfg(TabletopSceneDRCfg):
    """Two tables at the benchmark's measured size, as defaults.

    From humanoid-tool-use-benchmark (toolbench/scene/scene.py): 0.901 x 0.802 m
    tops at z = 0.758, lowered here to 0.70 for easier reach, butted together so
    the pair reads as one bench. The slab is 0.035 m: `table_height` fixes the
    top surface, so its thickness is free, and a thin one is what lets the legs
    below it show.

    One top, centred on the robot, which spawns at about x = 0, y = 0.75 facing
    -y. The placement code works in halves of it: the half at the robot's right
    hand carries the scenario's own objects and the half at its left carries the
    tool row, each within reach of a turn. They were two slabs once, which left
    a seam down the middle and a leg at each side of it.
    """

    # One top, not two butted together. Two slabs left a seam down the middle
    # and eight legs under it; this is the same working surface as a single
    # geom, so the top is unbroken and the legs stand at its four outer
    # corners. The placement code still works in halves, and takes them off
    # this one slab.
    table_size: Range | None = field(
        default_factory=lambda: Range(
            low=[1.802, 0.802, 0.035], high=[1.802, 0.802, 0.035]
        )
    )
    # Centred on the robot, which spawns at about x = 0 facing -y.
    table_position: Range | None = field(
        default_factory=lambda: Range(low=[0.0, 0.0], high=[0.0, 0.0])
    )
    table_height: Range | None = field(
        default_factory=lambda: Range(low=0.70, high=0.70)
    )

    randmizer_class: Any = ToolbenchSceneDR


__all__ = ["RoomDR", "RoomDRCfg", "ToolbenchSceneDR", "ToolbenchSceneDRCfg"]
