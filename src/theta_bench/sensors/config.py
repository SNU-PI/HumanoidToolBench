"""
THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces


@dataclass
class SensorCfg:
    uid: str
    mount: str
    # parent_link: str

    @property
    def observation_space(self) -> spaces.Space: ...


@dataclass
class CameraCfg(SensorCfg):

    # uid: str

    width: int
    height: int

    fov: float
    near: float
    far: float

    focal_length: float

    pose: dict

    intrinsics: dict[str, Any] | None = field(default=None, kw_only=True)
    depth_intrinsics: dict[str, Any] | None = field(default=None, kw_only=True)
    depth_to_color_extrinsics: dict[str, Any] | None = field(default=None, kw_only=True)

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

    @property
    def resolution(self) -> tuple[int, int]:  # [W, H]
        return (self.width, self.height)

    @property
    def fy(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("fy") is not None:
            return float(self.intrinsics["fy"])
        fx = self.width / (2 * np.tan(self.fov / 2))
        fy = fx
        return fy

    @property
    def fx(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("fx") is not None:
            return float(self.intrinsics["fx"])
        return self.width / (2 * np.tan(self.fov / 2))

    @property
    def cx(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("cx") is not None:
            return float(self.intrinsics["cx"])
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        if self.intrinsics is not None and self.intrinsics.get("cy") is not None:
            return float(self.intrinsics["cy"])
        return 0.5 * self.height

    @property
    def position(self) -> list[float]:
        if "position" in self.pose:
            return self.pose["position"]

    @property
    def quaternion(self) -> list[float]:
        if "quaternion" in self.pose:
            return self.pose["quaternion"]


@dataclass
class StereoCameraCfg(CameraCfg):

    baseline: float

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "left": spaces.Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                ),
                "right": spaces.Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                ),
            }
        )
