"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from humanoidtoolbench.runtime import validate_sim_mode

if TYPE_CHECKING:
    from humanoidtoolbench.engines.mujoco import MujocoSimulator
    from humanoidtoolbench.core.task import Task

import gymnasium as gym

from humanoidtoolbench.tasks.registry import TaskRegistry


class BaseDualSim(gym.Env):
    """
    Base class for MuJoCo-backed environments.

    MuJoCo owns the physics and draws every camera image.
    """

    task: Task
    sim_mode: str

    mujoco: MujocoSimulator

    def __init__(
        self, task: str | Task, sim_mode="mujoco", headless=True, *args, **kwargs
    ) -> None:
        sim_mode = validate_sim_mode(sim_mode)
        self.headless = headless
        self.sim_mode = sim_mode

        if isinstance(task, str):
            # FIXME dynamic import task
            self.task = TaskRegistry.make(task, *args, **kwargs)
        else:
            self.task = task

        from humanoidtoolbench.engines import MujocoSimulator

        make_renderers = kwargs.pop("make_renderers", True)
        self.mujoco = MujocoSimulator(
            self.task,
            headless=headless,
            make_renderers=make_renderers,
        )

        self.action_space = self.task.action_space

        self.observation_space = self.task.observation_space

    def _attach_task_outcome(
        self,
        info: dict,
        terminated: bool,
        *,
        success: bool | None = None,
        failure: bool = False,
        **kwargs,
    ) -> dict:
        """Merge the task's own outcome report into the gym `info` dict.

        `terminated` is what the environment already decided; a task that says
        nothing keeps HumanoidToolBench's existing meaning (`terminated == success`). A
        task that implements `task_info()` can override success, and add
        failure, stage and its own metrics. Nothing here interprets a reward.
        """

        info.setdefault(
            "success", bool(terminated) if success is None else bool(success)
        )
        info.setdefault("failure", bool(failure))
        report = getattr(self.task, "task_info", None)
        if callable(report):
            extra = report(info, **kwargs) or {}
            if extra:
                info.update(extra)
        return info

    def close(self):
        self.mujoco.close()
        super().close()
