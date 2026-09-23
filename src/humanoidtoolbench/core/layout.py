"""A lightweight description of one randomized Toolbench scene."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import transforms3d as t3d

from humanoidtoolbench.core.actor import Actor, CameraEntity, ObjectActor, RobotActor

if TYPE_CHECKING:
    from humanoidtoolbench.core.asset import Asset
    from humanoidtoolbench.core.robot import Robot
    from humanoidtoolbench.core.scene import Scene
    from humanoidtoolbench.sensors.config import CameraCfg


class Layout:
    def __init__(self) -> None:
        self.actors: dict[str, Actor] = {}
        self.cameras: dict[str, CameraEntity] = {}
        self.scene: Scene

    def _require_new_actor(self, name: str) -> None:
        if name in self.actors:
            raise ValueError(f"Actor with name {name!r} already exists in the layout")

    def add_object(self, name: str, asset: Asset) -> None:
        self._require_new_actor(name)
        self.actors[name] = ObjectActor(asset)

    def add_robot(self, robot: Robot) -> None:
        self._require_new_actor("robot")
        self.actors["robot"] = RobotActor(robot)

    def add_primitive(self, name: str, actor: Actor) -> None:
        self._require_new_actor(name)
        self.actors[name] = actor

    def add_camera(self, cam_id: str, cam_cfg: CameraCfg) -> None:
        from humanoidtoolbench.sensors.config import SensorCfg, StereoCameraCfg

        if cam_id in self.cameras:
            raise ValueError(f"Camera with ID {cam_id!r} already exists in the layout")

        if isinstance(cam_cfg, StereoCameraCfg):
            left = CameraEntity(cam_id, cam_cfg)
            right = CameraEntity(cam_id, cam_cfg)
            world_rotation = t3d.quaternions.quat2mat(left.pose.quaternion)
            right.pose.position = (
                np.asarray(left.pose.position, dtype=np.float32)
                - world_rotation[:3, 1] * cam_cfg.baseline
            ).tolist()
            self.cameras[f"{cam_id}_left"] = left
            self.cameras[f"{cam_id}_right"] = right
        elif isinstance(cam_cfg, SensorCfg):
            self.cameras[cam_id] = CameraEntity(cam_id, cam_cfg)
        else:
            raise TypeError(f"Unsupported camera config type: {type(cam_cfg)}")

    @property
    def robot(self) -> RobotActor:
        actor = self.actors["robot"]
        if not isinstance(actor, RobotActor):
            raise TypeError("layout robot actor has an invalid type")
        return actor

    def to_dict(self) -> dict[str, Any]:
        return {
            "actors": {
                name: actor.to_dict()
                for name, actor in self.actors.items()
                if name != "robot"
            },
            "cameras": {
                camera_id: camera.to_dict()
                for camera_id, camera in self.cameras.items()
            },
            "scene": self.scene.to_dict(),
        }
