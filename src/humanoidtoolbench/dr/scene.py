"""Procedural tabletop scene randomization for MuJoCo tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import transforms3d as t3d

from humanoidtoolbench.assets.primitive import Box as PrimitiveBox
from humanoidtoolbench.core.randomizer import Randomizer, RandomizerCfg
from humanoidtoolbench.core.scene import TabletopScene
from humanoidtoolbench.dr.types import Box


class SceneDR(Randomizer):
    """Base class for scene randomizers."""


class TabletopSceneDR(SceneDR):
    """Build the table slab at the middle of its configuration ranges."""

    def __init__(self, cfg: "TabletopSceneDRCfg") -> None:
        super().__init__(cfg)
        self.cfg = cfg

    def _draw(self, value: Box | None, default: Any) -> Any:
        return default if value is None else value.middle()

    def _make_table(
        self,
        size_range: Box | None,
        position_range: Box | None,
        height_range: Box | None,
        rotation_range: Box | None,
    ) -> PrimitiveBox:
        size = list(self._draw(size_range, [1.0, 0.8, 0.05]))
        xy = list(self._draw(position_range, [0.0, 0.0]))
        height = float(self._draw(height_range, size[2]))
        rotation_z = float(self._draw(rotation_range, 0.0))
        position = [float(xy[0]), float(xy[1]), height - 0.5 * float(size[2])]
        quaternion = t3d.euler.euler2quat(0.0, 0.0, rotation_z).tolist()
        return PrimitiveBox(size=size, position=position, quaternion=quaternion)

    def __call__(self, split: str = "train", **kwargs: Any) -> TabletopScene:
        del split, kwargs
        scene = TabletopScene()
        scene.set_table(
            self._make_table(
                self.cfg.table_size,
                self.cfg.table_position,
                self.cfg.table_height,
                self.cfg.rotation_z,
            )
        )
        return scene


@dataclass
class TabletopSceneDRCfg(RandomizerCfg):
    table_size: Box | None = None
    table_position: Box | None = None
    table_height: Box | None = None
    rotation_z: Box | None = None
    randmizer_class: Type[Randomizer] = TabletopSceneDR


__all__ = ["SceneDR", "TabletopSceneDR", "TabletopSceneDRCfg"]
