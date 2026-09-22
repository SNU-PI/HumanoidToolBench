"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from importlib import import_module
from typing import Type

from humanoidtoolbench.core.registry import RegistryMixin
from humanoidtoolbench.core.robot import Robot


class RobotRegistry(RegistryMixin[Robot]):
    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> Robot:
        if uid not in cls._registry:
            module_name = f"humanoidtoolbench.robots.{uid}"
            try:
                import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name == module_name:
                    raise ValueError(f"No robot registered under uid '{uid}'") from exc
                raise
        return super().make(uid, *args, **kwargs)

    @classmethod
    def _base_type(cls) -> Type:
        return Robot
