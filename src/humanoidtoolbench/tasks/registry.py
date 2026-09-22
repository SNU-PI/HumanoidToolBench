"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from importlib import import_module
from typing import Type

from humanoidtoolbench.core.registry import RegistryMixin
from humanoidtoolbench.core.task import Task
from humanoidtoolbench.scenario_names import TASK_UID_ALIASES


class TaskRegistry(RegistryMixin[Task]):

    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> Task:
        uid = TASK_UID_ALIASES.get(uid, uid)
        if uid not in cls._registry:
            module_name = f"humanoidtoolbench.tasks.{uid}"
            try:
                import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name == module_name:
                    raise ValueError(f"No task registered under uid '{uid}'") from exc
                raise
        return super().make(uid, *args, **kwargs)

    @classmethod
    def _base_type(cls) -> Type:
        return Task
