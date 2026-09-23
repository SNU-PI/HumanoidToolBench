"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from humanoidtoolbench.core.task import Task
from humanoidtoolbench.envs.base_dual_env import BaseDualSim


class SonicLocoManipEnv(BaseDualSim):
    _success: bool

    def __init__(
        self,
        task: str | Task,
        sonic_config: dict,
        sim_mode: str = "mujoco",
        headless: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            task,
            sim_mode,
            headless,
            sonic_config=sonic_config,
            *args,
            **kwargs,
        )
        self.sonic_config = sonic_config
        # Whether an observation carries camera images. The evaluator turns
        # this off while the robot stabilizes, when nothing reads them.
        self.render_obs = True

    def _get_obs(self):
        qpos = np.asarray(list(self.mujoco.get_robot_qpos().values()), dtype=np.float32)
        if not self.render_obs:
            return {"joint_qpos": qpos}
        return {"joint_qpos": qpos, **self._render_frame()}

    def _get_info(self):
        info = {}
        for k, v in self.mujoco.mj_objects.items():
            info[str(k)] = np.concatenate([v.xpos, v.xquat])
        proprio = self.task.robot.prepare_obs()
        return {**info, "proprio": proprio}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
        super().reset(seed=seed, options=options)
        self.task.reset(seed)
        self.mujoco.update_layout(sonic_config=self.sonic_config)

        mujoco.mj_forward(self.mujoco.mjModel, self.mujoco.mjData)
        self.mujoco.step(render=False)

        self.control_decimal = int((1 / self.mujoco.physics_dt) / self.task.render_hz)
        self.step_count = 0
        obs = self._get_obs()
        info = self._get_info()
        self._success = False
        return obs, info

    def step(self, action):
        for _ in range(self.control_decimal):
            self.mujoco.apply_action(action)
            self.mujoco.step(render=False)

        self.step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        reward = self.task.compute_reward(info, mujoco_env=self.mujoco)
        success = self.task.check_success(info, mujoco_env=self.mujoco)
        failure = self.task.check_failure(info, mujoco_env=self.mujoco)
        terminated = success or failure
        truncated = False
        self._success = success
        info = self._attach_task_outcome(
            info,
            terminated,
            success=success,
            failure=failure,
            mujoco_env=self.mujoco,
        )
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        return self.mujoco.render()
