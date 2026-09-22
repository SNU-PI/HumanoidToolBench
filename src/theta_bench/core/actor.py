"""Runtime entities used to assemble a Toolbench MuJoCo scene."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import transforms3d as t3d

from theta_bench.core.types import Pose

if TYPE_CHECKING:
    from theta_bench.core.asset import Asset
    from theta_bench.core.robot import Robot
    from theta_bench.sensors.config import CameraCfg


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {
            key: _to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


class Entity:
    def to_dict(self) -> dict[str, Any]:
        return {
            key: _to_plain(value)
            for key, value in vars(self).items()
            if not key.startswith("_")
        }


class Actor(Entity):
    pose: Pose


class ObjectActor(Actor):
    def __init__(self, asset: Asset, uid: str | None = None) -> None:
        self.uid = asset.uid if uid is None else uid
        self.asset = asset
        self.pose = Pose()
        self.material: dict[str, Any] = {}
        self.isaac_shaders: dict[str, float] = {}

    def set_material(self, material: dict[str, Any]) -> None:
        self.material = material

    def set_isaac_shaders(self, shaders: dict[str, float]) -> None:
        """PBR constants for the Isaac renderer, kept apart from `material`."""
        self.isaac_shaders = shaders


class RobotActor(Actor):
    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.pose = Pose()
        self.shaders: dict[str, float] = {}

    def set_shaders(self, shaders: dict[str, float]) -> None:
        """PBR constants the Isaac renderer pushes into the robot's shaders."""
        self.shaders = shaders


class Light(Entity):
    """A light in the Isaac renderer's terms, not MuJoCo's.

    MuJoCo's light model has no notion of an emitter's size or its colour
    temperature, so a rig authored for RTX cannot be expressed as `mujoco_lights`
    and is kept separately. The MuJoCo engine ignores these entirely.
    """

    def __init__(self, uid: str, type: str) -> None:
        self.uid = uid
        self.type = type
        self.pose = Pose()
        self.light_radius: float = 0.0
        self.light_length: float = 0.0
        self.light_intensity: float = 0.0
        self.light_color_temperature: float = 6500.0
        self.center_light_position: list[float] = [0.0, 0.0, 0.0]
        self.center_light_orientation: list[float] = [1.0, 0.0, 0.0, 0.0]


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
