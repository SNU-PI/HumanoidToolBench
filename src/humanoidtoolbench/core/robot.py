"""Base identity shared by the active G1 robot implementation."""

from __future__ import annotations

from typing import Any


class Robot:
    uid: str
    label: str
    dof: int
    mjcf_path: str
    joint_names: list[str]

    def __init__(self, uid: str, dof: int) -> None:
        self.uid = uid
        self.dof = dof

    def reset(self, **kwargs: Any) -> None:
        del kwargs
