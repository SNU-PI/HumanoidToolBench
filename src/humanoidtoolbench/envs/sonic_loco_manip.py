"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

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
        mjviser_port: int | None = None,
        render_obs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            task,
            sim_mode,
            headless,
            sonic_config=sonic_config,
            # No observer for the pictures means no camera to draw them with.
            make_renderers=render_obs,
            *args,
            **kwargs,
        )
        self.sonic_config = sonic_config
        self.offscreen = headless
        self.onscreen = not self.offscreen

        if self.sonic_config.get("ENABLE_DDS", True):
            try:
                if self.sonic_config.get("INTERFACE"):
                    ChannelFactoryInitialize(
                        self.sonic_config["DOMAIN_ID"],
                        self.sonic_config["INTERFACE"],
                    )
                else:
                    ChannelFactoryInitialize(self.sonic_config["DOMAIN_ID"])
            except Exception as exc:
                print(f"Note: Channel factory initialization attempt: {exc}")
        self.viewer = None
        # Whether an observation carries camera images. A policy that reads the
        # simulator rather than pixels (the RL actors do) needs none of them,
        # and drawing them is not free: it is a full offscreen render on every
        # step, and on a card that is also running MuJoCo-Warp it aborts.
        self.render_obs = render_obs
        if not render_obs:
            # The space has to describe what `_get_obs` actually returns, or
            # gymnasium's passive checker rejects the first reset.
            from gymnasium import spaces

            kept = {
                name: space
                for name, space in self.observation_space.spaces.items()
                if name == "joint_qpos"
            }
            self.observation_space = spaces.Dict(kept)
        self._mjviser_port = mjviser_port
        self._mjviser = None

    def update_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()
        if self._mjviser is not None:
            self._mjviser.sync(self.mjData)

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
        self.task.reset(seed, options)
        self.mujoco.update_layout(sonic_config=self.sonic_config)

        # Public aliases retained for the viewer and debugging helpers.
        self.mjSpec = self.mujoco.mjSpec
        self.mjModel = self.mujoco.mjModel
        self.mjData = self.mujoco.mjData

        if self.onscreen:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                key_callback=self.task.robot.elastic_band.MujuocoKeyCallback,
                show_left_ui=False,
                show_right_ui=False,
            )
        else:
            mujoco.mj_forward(self.mjModel, self.mjData)
            self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 100
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 3
            self.viewer.cam.lookat = np.array([0, 0, 0.38])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        self.mujoco.step(render=False)

        if self._mjviser_port is not None:
            if self._mjviser is None:
                from humanoidtoolbench.viewers.mjviser import MjviserViewer

                self._mjviser = MjviserViewer(self._mjviser_port)
            self._mjviser.reset(self.mujoco)

        if self.isaac is not None:
            # Every reset compiles a new MuJoCo model, so the USD mirror of it
            # has to be rebuilt before the first frame is drawn.
            self.isaac.update_layout(self.mujoco)

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
        if self.isaac is None:
            return self.mujoco.render()
        # Isaac draws the same cameras under the same names, so the images
        # reaching the headset and the recorder swap engine without anything
        # downstream knowing. MuJoCo is not rendered at all in this mode.
        self.isaac.step(self.mujoco)
        return self.isaac.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self._mjviser is not None:
            self._mjviser.close()
            self._mjviser = None
        super().close()
