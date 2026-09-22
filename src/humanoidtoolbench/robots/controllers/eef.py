"""Dexterous-hand controller used by the active G1 robot."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from humanoidtoolbench.core.controller import Controller, ControllerCfg


class DexHandEEFController(Controller):
    cfg: DexHandEEFControllerCfg

    def __init__(self, cfg: DexHandEEFControllerCfg) -> None:
        super().__init__(cfg)

    @property
    def action_space(self) -> spaces.Space:
        dof = len(self.cfg.joint_names)
        joint_limits = np.array([[-np.pi, np.pi]] * dof, dtype=np.float32)
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        return spaces.Box(low, high, dtype=np.float32)

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        for joint_name, qpos in zip(self.cfg.joint_names, self.cfg.init_qpos):
            joints[joint_name].qpos = qpos
            joints[joint_name].qvel = 0
            joints[joint_name].qacc = 0
            actuators[joint_name].ctrl = qpos


class DexHandEEFControllerCfg(ControllerCfg):
    clazz: type[Controller] = DexHandEEFController

    def __init__(self, joint_names: list[str], init_qpos: list[float]) -> None:
        self.joint_names = joint_names
        self.init_qpos = init_qpos

    def __call__(self) -> DexHandEEFController:
        return DexHandEEFController(self)
