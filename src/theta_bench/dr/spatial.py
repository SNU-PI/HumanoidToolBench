"""Robot spawn randomization for the tool-use scenes."""

from __future__ import annotations

from dataclasses import dataclass

from theta_bench.core.actor import RobotActor
from theta_bench.core.layout import Layout
from theta_bench.core.randomizer import Randomizer, RandomizerCfg
from theta_bench.dr.types import Box


class SpatialDR(Randomizer):
    """Place the robot at a sampled or fixed pose.

    Push, sweep, and mop objects are placed by their task-specific randomizers,
    so the shared spatial randomizer only owns the robot spawn.
    """

    def __init__(self, cfg: "SpatialDRCfg") -> None:
        super().__init__(cfg)
        self.cfg = cfg

    def __call__(
        self,
        split: str,
        layout: Layout,
        table_height: float = 0.0,
    ) -> None:
        del split, table_height
        actor = layout.actors.get("robot")
        if not isinstance(actor, RobotActor):
            return

        uid = str(actor.robot.uid)
        if self._inner_state is not None and uid in self._inner_state:
            state = self._inner_state[uid]
            actor.pose.position = list(state["position"])
            actor.pose.quaternion = list(state["quaternion"])
            return

        position = self._sample(self.cfg.robot_region, [0.0, 0.0, 0.0])
        quaternion = self._sample(
            self.cfg.robot_orientation_region,
            [1.0, 0.0, 0.0, 0.0],
        )
        actor.pose.position = position
        actor.pose.quaternion = quaternion
        self._inner_state = {
            uid: {"position": list(position), "quaternion": list(quaternion)}
        }

    def _sample(self, region: Box | None, default: list[float]) -> list[float]:
        if region is None:
            return list(default)
        value = region.middle() if self.cfg.spatial_mode == "fixed" else region.sample()
        if not isinstance(value, list):
            raise TypeError("robot pose regions must contain vectors")
        return value


@dataclass
class SpatialDRCfg(RandomizerCfg):
    robot_region: Box | None = None
    robot_orientation_region: Box | None = None
    spatial_mode: str = "random"
    randmizer_class: type[Randomizer] = SpatialDR


__all__ = ["SpatialDR", "SpatialDRCfg"]
