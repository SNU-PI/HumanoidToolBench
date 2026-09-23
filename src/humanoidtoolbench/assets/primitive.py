"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from humanoidtoolbench.core.actor import Actor
from humanoidtoolbench.core.asset import Asset
from humanoidtoolbench.core.types import Pose


class Primitive(Asset, Actor):
    material: dict


class Box(Primitive):
    def __init__(self, size, position, quaternion) -> None:
        self.uid = "box"

        self.size = size
        self.pose = Pose(position, quaternion)

    def set_material(self, material: dict) -> None:
        self.material = material

    def __repr__(self) -> str:
        return f"Box(size={self.size}, position={self.pose.position}, quaternion={self.pose.quaternion})"
