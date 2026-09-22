"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from humanoidtoolbench.core.actor import Actor
from humanoidtoolbench.core.asset import Asset

# import os
from humanoidtoolbench.core.types import Pose


class Primitive(Asset, Actor):
    material: dict
    isaac_material: dict | None = None


class Box(Primitive):
    def __init__(self, size, position, quaternion) -> None:
        self.uid = "box"

        self.size = size
        # self.position = position
        # self.quaternion = quaternion
        self.pose = Pose(position, quaternion)

    def set_material(self, material: dict) -> None:
        self.material = material

    def set_isaac_material(self, material: dict | None) -> None:
        """An MDL material for the Isaac renderer.

        Kept apart from `material`, which is MuJoCo's and holds a texture file
        and rgba. The two describe the same surface to renderers that share no
        vocabulary, so neither may overwrite the other.
        """
        self.isaac_material = material

    def __repr__(self) -> str:
        return f"Box(size={self.size}, position={self.pose.position}, quaternion={self.pose.quaternion})"
