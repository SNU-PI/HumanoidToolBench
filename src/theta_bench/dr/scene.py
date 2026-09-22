"""Procedural tabletop scene randomization for MuJoCo tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import transforms3d as t3d

from theta_bench.assets.primitive import Box as PrimitiveBox
from theta_bench.core.randomizer import Randomizer, RandomizerCfg
from theta_bench.core.scene import TabletopScene
from theta_bench.dr.types import Box


class SceneDR(Randomizer):
    """Base class for scene randomizers."""


class TabletopSceneDR(SceneDR):
    """Build one or two table slabs directly from configuration ranges."""

    def __init__(self, cfg: "TabletopSceneDRCfg") -> None:
        super().__init__(cfg)
        self.cfg = cfg

    def _draw(self, value: Box | None, default: Any) -> Any:
        if value is None:
            return default
        return value.middle() if self.cfg.scene_mode == "fixed" else value.sample()

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

    @staticmethod
    def _table_from_state(state: dict[str, Any]) -> PrimitiveBox:
        pose = state["pose"]
        table = PrimitiveBox(
            size=state["size"],
            position=pose["position"],
            quaternion=pose["quaternion"],
        )
        if "material" in state:
            table.set_material(state["material"])
        return table

    def state_dict(self) -> dict[str, Any]:
        return self._inner_state.to_dict() if self._inner_state is not None else {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        scene = TabletopScene(
            uid=state_dict.get("uid", "toolbench"),
            name=state_dict.get("name", "toolbench"),
            data_dir=state_dict.get("data_dir", ""),
        )
        scene.set_table(self._table_from_state(state_dict["table"]))
        if state_dict.get("table2") is not None:
            scene.set_table2(self._table_from_state(state_dict["table2"]))
        if state_dict.get("tool_table") is not None:
            scene.set_tool_table(self._table_from_state(state_dict["tool_table"]))
        self._inner_state = scene

    def __call__(self, split: str = "train", **kwargs: Any) -> TabletopScene:
        del split, kwargs
        if self._inner_state is not None:
            return self._inner_state

        scene = TabletopScene()
        scene.set_table(
            self._make_table(
                self.cfg.table_size,
                self.cfg.table_position,
                self.cfg.table_height,
                self.cfg.rotation_z,
            )
        )
        if self.cfg.enable_table2:
            scene.set_table2(
                self._make_table(
                    self.cfg.table2_size,
                    self.cfg.table2_position,
                    self.cfg.table2_height,
                    self.cfg.table2_rotation_z,
                )
            )
        if self.cfg.enable_tool_table:
            scene.set_tool_table(
                self._make_table(
                    self.cfg.tool_table_size,
                    self.cfg.tool_table_position,
                    self.cfg.tool_table_height,
                    self.cfg.tool_table_rotation_z,
                )
            )
        return self._transient(scene)


@dataclass
class TabletopSceneDRCfg(RandomizerCfg):
    scene_mode: str = "fixed"
    table_size: Box | None = None
    table_position: Box | None = None
    table_height: Box | None = None
    rotation_z: Box | None = None
    enable_table2: bool = False
    table2_size: Box | None = None
    table2_position: Box | None = None
    table2_height: Box | None = None
    table2_rotation_z: Box | None = None
    enable_tool_table: bool = False
    tool_table_size: Box | None = None
    tool_table_position: Box | None = None
    tool_table_height: Box | None = None
    tool_table_rotation_z: Box | None = None
    randmizer_class: Type[Randomizer] = TabletopSceneDR


__all__ = ["SceneDR", "TabletopSceneDR", "TabletopSceneDRCfg"]
