"""Runtime entities used to assemble a Toolbench MuJoCo scene."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import transforms3d as t3d

from humanoidtoolbench.core.types import Pose

if TYPE_CHECKING:
    from humanoidtoolbench.core.asset import Asset
    from humanoidtoolbench.core.robot import Robot
    from humanoidtoolbench.sensors.config import CameraCfg


class Entity:
    """Something placed in the scene layout."""


class Actor(Entity):
    pose: Pose


class ObjectActor(Actor):
    def __init__(self, asset: Asset, uid: str | None = None) -> None:
        self.uid = asset.uid if uid is None else uid
        self.asset = asset
        self.pose = Pose()
        self.material: dict[str, Any] = {}

    def set_material(self, material: dict[str, Any]) -> None:
        self.material = material


class RobotActor(Actor):
    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.pose = Pose()


class CameraEntity(Entity):
    def __init__(self, cam_id: str, cam_cfg: CameraCfg) -> None:
        self.cam_id = cam_id
        self.cam_cfg = cam_cfg

        if "distance" in cam_cfg.pose:
            distance = cam_cfg.pose["distance"]
            polar = cam_cfg.pose["polar"]
            azimuth = cam_cfg.pose["azimuth"]
            position_array = np.array(
                [
                    distance * np.cos(azimuth) * np.sin(polar),
                    distance * np.sin(azimuth) * np.sin(polar),
                    distance * np.cos(polar),
                ]
            )
            direction = -position_array
            pitch = -np.arctan2(direction[2], np.linalg.norm(direction[:2]))
            yaw = np.arctan2(direction[1], direction[0])
            position = position_array.tolist()
            quaternion = t3d.euler.euler2quat(0.0, pitch, yaw).tolist()
        elif "position" in cam_cfg.pose and "quaternion" in cam_cfg.pose:
            position = cam_cfg.pose["position"]
            quaternion = cam_cfg.pose["quaternion"]
        elif "position" in cam_cfg.pose and "eulers" in cam_cfg.pose:
            position = cam_cfg.pose["position"]
            quaternion = t3d.euler.euler2quat(*cam_cfg.pose["eulers"]).tolist()
        else:
            raise ValueError("invalid camera pose specification")

        self.pose = Pose(position=position, quaternion=quaternion)
        self.fx = cam_cfg.fx
        self.fy = cam_cfg.fy
        self.cx = cam_cfg.cx
        self.cy = cam_cfg.cy
        self.mount = cam_cfg.mount
        self.resolution = cam_cfg.resolution
        self.focal_length = cam_cfg.focal_length
