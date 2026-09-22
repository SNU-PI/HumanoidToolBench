"""Joint-position controller used to initialize the active G1 robot."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from humanoidtoolbench.core.controller import Controller, ControllerCfg


class PDJointPosController(Controller):
    def __init__(self, cfg: PDJointPosControllerCfg) -> None:
        super().__init__(cfg)
        self.dof = cfg.dof

    @property
    def action_space(self) -> spaces.Space:
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        return spaces.Box(low, high, dtype=np.float32)

    def _get_joint_limits(self) -> np.ndarray:
        return np.array([[-1.0, 1.0]] * self.dof, dtype=np.float32)

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        for joint_name, qpos in zip(self.cfg.joint_names, self.cfg.init_qpos):
            joints[joint_name].qpos = qpos
            joints[joint_name].qvel = 0
            joints[joint_name].qacc = 0
            actuators[joint_name].ctrl = qpos


class PDJointPosControllerCfg(ControllerCfg):
    clazz: type[Controller] = PDJointPosController

    def __init__(self, joint_names: list[str], init_qpos: list[float]) -> None:
        self.joint_names = joint_names
        self.dof = len(joint_names)
        self.init_qpos = init_qpos

    def __call__(self) -> PDJointPosController:
        return PDJointPosController(self)
