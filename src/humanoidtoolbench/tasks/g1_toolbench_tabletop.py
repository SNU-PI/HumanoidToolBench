"""Shared MuJoCo setup for the G1 tool-use tasks."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

from humanoidtoolbench.core.actor import Actor
from humanoidtoolbench.core.layout import Layout
from humanoidtoolbench.core.task import Task
from humanoidtoolbench.dr.camera import CameraDRCfg
from humanoidtoolbench.dr.manager import ToolbenchDRManager
from humanoidtoolbench.dr.mujoco_lighting import MujocoLightingDR, MujocoLightingDRCfg
from humanoidtoolbench.dr.room import RoomDRCfg, ToolbenchSceneDRCfg
from humanoidtoolbench.dr.spatial import SpatialDRCfg
from humanoidtoolbench.dr.types import Box
from humanoidtoolbench.robots.protocols import Controllable
from humanoidtoolbench.robots.registry import RobotRegistry
from humanoidtoolbench.sensors import CameraCfg, SensorCfg, StereoCameraCfg

if TYPE_CHECKING:
    from humanoidtoolbench.core.randomizer import RandomizerCfg


# Where a wrist camera sits, relative to `{side}_hand_camera_base_link`, copied
# from Unitree's own camera_configs.py in unitree_sim_isaaclab. Written for the
# left hand; the engine adds the link's own offset and mirrors y for the right.
# The quaternion is never read: the heading comes from Unitree's own figure in
# the engine, which is the whole point of copying the mount.
WRIST_CAM_POSE = {
    "position": [-0.04012, -0.07441, 0.15711],
    "quaternion": [1.0, 0.0, 0.0, 0.0],
}


# Every stream is 640 x 360, so the three views differ in where they are and
# in nothing else a policy has to normalise for. The lenses stay apart, because
# on the real robot they are two different cameras.
FRAME = {"width": 640, "height": 360}

# Unitree's own wrist lens, from the dex3 entries in `camera_configs.py` in
# unitree_sim_isaaclab: a 12 mm focal length behind a 20 mm horizontal aperture,
# which is 79.61 degrees across. They render it 640 x 480; the lens is the same
# lens at 640 x 360 and only the vertical field follows the frame, 64.0 degrees
# down to 50.2. Their near plane of 0.1 m is kept: the nearest body of the
# robot's own hand sits 0.170 m from this camera, so it clips nothing, and the
# frames at 0.1 and at 0.02 come out pixel for pixel identical.
#
# Their far plane does not survive the trip. Isaac is handed 1.0e5 m, which a
# reversed-Z RTX renderer is happy with and MuJoCo's depth buffer is not; the
# room is 5 m across, so 5 is what it gets.
WRIST_CAM_FOCAL_MM = 12.0
WRIST_CAM_APERTURE_MM = 20.0
WRIST_CAM_FOV = 2.0 * np.arctan(WRIST_CAM_APERTURE_MM / (2.0 * WRIST_CAM_FOCAL_MM))
WRIST_CAM_NEAR = 0.1
SCENE_FAR = 5


class G1ToolbenchTabletop(Task):
    """Common robot, room, sensors, and randomization for tool-use tasks.

    Concrete tasks own their objects, instructions, rewards, and success rules.
    """

    uid = "g1_toolbench_tabletop"
    label = "G1 Toolbench"
    description = "Shared scene for G1 tool-use tasks."

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 1.0,
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 3000,
        "success_criteria": 0.2,
    }

    robot_cfg: dict[str, Any] = {"uid": "g1_sonic"}

    mujoco_lights: list[dict[str, Any]] = MujocoLightingDRCfg().build().fixed()

    # The head pair is what the policy sees; the two wrist cameras are what
    # tells a grasp from a near miss, which the head cannot at this range.
    sensor_cfgs: dict[str, SensorCfg] = {
        "head_stereo": StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            focal_length=1.93,
            fov=np.deg2rad(110),
            near=0.2,
            far=SCENE_FAR,
            baseline=0.05,
            pose={"position": [0.0, 0.0, 0.0]},
            **FRAME,
        ),
        "wrist_right": CameraCfg(
            uid="Unitree_dex3_wrist",
            mount="eye_in_hand",
            focal_length=WRIST_CAM_FOCAL_MM,
            fov=WRIST_CAM_FOV,
            near=WRIST_CAM_NEAR,
            far=SCENE_FAR,
            pose=dict(WRIST_CAM_POSE),
            **FRAME,
        ),
        "wrist_left": CameraCfg(
            uid="Unitree_dex3_wrist",
            mount="eye_in_hand",
            focal_length=WRIST_CAM_FOCAL_MM,
            fov=WRIST_CAM_FOV,
            near=WRIST_CAM_NEAR,
            far=SCENE_FAR,
            pose=dict(WRIST_CAM_POSE),
            **FRAME,
        ),
    }

    dr_cfgs: dict[str, RandomizerCfg] = {
        "spatial": SpatialDRCfg(
            spatial_mode="random",
            robot_region=Box(low=[-0.1, 0.65, 0.0], high=[0.1, 0.85, 0.0]),
            robot_orientation_region=Box(
                low=[0.7071, 0.0, 0.0, -0.7071],
                high=[0.7071, 0.0, 0.0, -0.7071],
            ),
        ),
        "camera": CameraDRCfg(cam_id="head_stereo"),
        "scene": ToolbenchSceneDRCfg(),
        "lighting": MujocoLightingDRCfg(),
        "room": RoomDRCfg(room_mode="random", centre=(0.4, 0.0)),
    }

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._instruction: str | None = None
        self._tools: list[Any] = []
        self._target: Actor | None = None
        self._layout: Layout | None = None
        self._init_target_height: float | None = None
        self.reward = 0.0

        self.robot_cfg = {**self.robot_cfg, "uid": robot_uid}
        self.sensor_cfgs = deepcopy(self.sensor_cfgs)
        self.dr_cfgs = deepcopy(self.dr_cfgs)
        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)

        super().__init__(
            dr=ToolbenchDRManager(level=dr_level, **self.dr_cfgs),
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            *args,
            **kwargs,
        )

    @property
    def layout(self) -> Layout:
        if self._layout is None:
            raise RuntimeError("call reset() first")
        return self._layout

    @property
    def instruction(self) -> str:
        if self._instruction is None:
            raise RuntimeError("call reset() first")
        return self._instruction

    @property
    def target(self) -> Actor:
        if self._target is None:
            raise RuntimeError("call reset() first")
        return self._target

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    @property
    def action_space(self) -> spaces.Space:
        if not isinstance(self.robot, Controllable):
            raise TypeError("task robot must implement Controllable")
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        obs: dict[str, spaces.Space] = {
            "joint_qpos": spaces.Box(
                -np.pi,
                np.pi,
                shape=(self.robot.wholebody_dof,),  # type: ignore[attr-defined]
                dtype=np.float32,
            )
        }
        default_obs = super().observation_space
        if isinstance(default_obs, spaces.Dict):
            obs.update(default_obs.spaces)
        return spaces.Dict(obs)

    def apply_scene_dr(self, split: str) -> None:
        """Apply the per-episode room appearance and MuJoCo lighting rig."""
        room_dr = self.dr.get_randomizer("room")
        if room_dr is not None:
            room_dr.apply(self.layout, split)

        light_dr = self.dr.get_randomizer("lighting")
        if isinstance(light_dr, MujocoLightingDR):
            self.mujoco_lights = light_dr.apply(self.layout, split)
