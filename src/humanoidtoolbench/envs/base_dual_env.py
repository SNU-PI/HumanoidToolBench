"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from humanoidtoolbench.runtime import ISAAC_RENDER_SIM_MODES, validate_sim_mode

if TYPE_CHECKING:
    from humanoidtoolbench.engines.isaacsim import IsaacSimRenderer
    from humanoidtoolbench.engines.mujoco import MujocoSimulator
    from humanoidtoolbench.core.task import Task

import gymnasium as gym

from humanoidtoolbench.tasks.registry import TaskRegistry


class BaseDualSim(gym.Env):
    """
    Base class for MuJoCo-backed environments.

    MuJoCo always owns the physics. `sim_mode="isaacsim-mujoco"` additionally
    starts Isaac Sim as a renderer, which mirrors the MuJoCo scene onto a USD
    stage and draws it; it never steps anything. `mujoco_isaac` remains an alias.
    """

    task: Task
    sim_mode: str

    mujoco: MujocoSimulator
    isaac: IsaacSimRenderer | None

    def __init__(
        self, task: str | Task, sim_mode="mujoco", headless=True, *args, **kwargs
    ) -> None:
        sim_mode = validate_sim_mode(sim_mode)
        self.headless = headless
        self.sim_mode = sim_mode

        # Isaac renders for the headset, not for the desktop, so its own window
        # is off unless asked for. `headless` stays MuJoCo's viewer flag.
        isaac_headless = os.getenv("HUMANOIDTOOLBENCH_ISAAC_WINDOW", "0").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
            "",
        }

        if sim_mode in ISAAC_RENDER_SIM_MODES:
            # Start Kit before the task pulls its own machinery in, so the two
            # renderers are not fighting over a half-built process.
            # `start_simulation_app` preloads the native modules that have to
            # claim their symbols ahead of Kit's bundled libraries.
            from humanoidtoolbench.engines.isaac_app import start_simulation_app

            start_simulation_app(headless=isaac_headless)

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
            # Isaac supplies every camera image, so neither local OpenGL
            # contexts nor a MuJoCo render-owner connection are needed.
            make_renderers=make_renderers and sim_mode not in ISAAC_RENDER_SIM_MODES,
        )

        if sim_mode in ISAAC_RENDER_SIM_MODES:
            from humanoidtoolbench.engines.isaacsim import IsaacSimRenderer

            self.isaac = IsaacSimRenderer(self.task, headless=isaac_headless)
        else:
            self.isaac = None

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
        if self.isaac is not None:
            self.isaac.close()
            self.isaac = None
        super().close()
