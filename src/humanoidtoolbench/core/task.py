"""Base contract shared by the active tool-use tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from gymnasium import spaces

if TYPE_CHECKING:
    from humanoidtoolbench.core.randomizer import RandomizerCfg
    from humanoidtoolbench.dr.manager import DRManager

from humanoidtoolbench.core.layout import Layout
from humanoidtoolbench.core.robot import Robot
from humanoidtoolbench.dr.scene import TabletopSceneDR
from humanoidtoolbench.robots.protocols import HeadCamMountable
from humanoidtoolbench.sensors.config import CameraCfg, SensorCfg


class Task(ABC):
    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "split": "train",  # train, test
    }

    uid: str
    label: str
    description: str

    robot_cfg: dict[str, Any]

    sensor_cfgs: dict[str, SensorCfg]

    dr_cfgs: dict[str, RandomizerCfg]

    def __init__(
        self,
        dr: DRManager,
        split: str | None = None,
        physics_dt: float | None = None,
    ) -> None:
        self.metadata = dict(type(self).metadata)
        self.metadata.update(
            {
                k: v
                for k, v in {"split": split, "physics_dt": physics_dt}.items()
                if v is not None
            }
        )
        self.dr = dr
        self.split = split

    @property
    def render_hz(self) -> int:
        return self.metadata.get("render_hz", 30)

    @property
    def robot(self) -> Robot:
        return self._robot

    @robot.setter
    def robot(self, value: Robot) -> None:
        self._robot = value

    @property
    @abstractmethod
    def layout(self) -> Layout: ...

    @property
    @abstractmethod
    def instruction(self) -> str:
        """A natural language instruction describing the task."""
        # return self.description

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space contributed by configured sensors."""
        obs: dict[str, spaces.Space] = {}
        for key, sensor_cfg in self.sensor_cfgs.items():
            sense_obs = sensor_cfg.observation_space
            if isinstance(sense_obs, spaces.Dict):
                for sub_key, sub_space in sense_obs.spaces.items():
                    obs[f"{key}_{sub_key}"] = sub_space
            else:
                obs[key] = sense_obs
        return spaces.Dict(obs)

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space: ...

    def task_info(self, info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Report the outcome of the current step.

        A task that wants its outcome reported returns it here::

            {"success": bool, "failure": bool, "stage": int,
             "metrics": {"<name>": float, ...}}

        Every key is optional. Whatever is returned is merged into the gym
        ``info`` dict, and the evaluator writes ``metrics`` verbatim to the
        per-step metrics and the episode summary; nothing is inferred from the
        reward.
        """

        del info, kwargs
        return {}

    def reset(self, seed: int | None = None) -> None:
        """Build a new randomized layout from the seed."""
        split = self.metadata.get("split", "train")

        self.dr.reset(seed=seed)

        self._layout = Layout()
        self._layout.add_robot(self.robot)

        scene_dr = self.dr.get_randomizer("scene")
        if isinstance(scene_dr, TabletopSceneDR):
            scene = scene_dr(split)
            self._layout.add_primitive("table", scene.table)
            table_height = scene.table.pose.position[2] + 0.5 * scene.table.size[2]
        else:
            table_height = 0.0

        # Robot spawn randomization. Task-specific randomizers place all tools
        # and targets after this shared setup has completed.
        spatial_dr = self.dr.get_randomizer("spatial")
        if spatial_dr is not None:
            spatial_dr(split, self._layout, table_height=table_height)

        camera_dr = self.dr.get_randomizer("camera")
        if camera_dr is not None:
            for cam_id, cam_info in self.sensor_cfgs.items():
                if isinstance(cam_info, CameraCfg):
                    cam_cfg = camera_dr(split, cam_info)
                    if (
                        cam_cfg.quaternion is None
                        and cam_cfg.mount == "eye_in_head"
                        and isinstance(self.robot, HeadCamMountable)
                    ):
                        cam_cfg.pose["quaternion"] = self.robot.head_camera_orientation

                    self._layout.add_camera(cam_id, cam_cfg)

    def check_success(self, *args, **kwargs) -> bool:
        """Check if the task is successfully completed."""
        raise NotImplementedError

    def check_failure(self, *args: Any, **kwargs: Any) -> bool:
        """Return whether an explicit task failure has occurred."""
        del args, kwargs
        return False

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """Compute the reward for the current state of the task."""
        return 0.0
