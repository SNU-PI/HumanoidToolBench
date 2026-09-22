"""Whole-body controller composition for the active G1 robot."""

from __future__ import annotations

from gymnasium import spaces

from humanoidtoolbench.core.controller import Controller, ControllerCfg

from .eef import DexHandEEFController, DexHandEEFControllerCfg
from .qpos import PDJointPosController, PDJointPosControllerCfg


class WholeBodyEEFController(Controller):
    cfg: WholeBodyEEFControllerCfg
    left_leg: PDJointPosController
    right_leg: PDJointPosController
    waist: PDJointPosController
    left_arm: PDJointPosController
    right_arm: PDJointPosController
    left_eef: DexHandEEFController
    right_eef: DexHandEEFController

    def __init__(self, cfg: WholeBodyEEFControllerCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.left_leg = cfg.left_leg_cfg()
        self.right_leg = cfg.right_leg_cfg()
        if cfg.waist_cfg is not None:
            self.waist = cfg.waist_cfg()
        self.left_arm = cfg.left_arm_cfg()
        self.right_arm = cfg.right_arm_cfg()
        self.left_eef = cfg.left_eef_cfg()
        self.right_eef = cfg.right_eef_cfg()

    def set_initial_qpos(self, actuators: dict, joints: dict) -> None:
        self.left_leg.set_initial_qpos(actuators, joints)
        self.right_leg.set_initial_qpos(actuators, joints)
        if self.cfg.waist_cfg is not None:
            self.waist.set_initial_qpos(actuators, joints)
        self.left_arm.set_initial_qpos(actuators, joints)
        self.right_arm.set_initial_qpos(actuators, joints)
        self.left_eef.set_initial_qpos(actuators, joints)
        self.right_eef.set_initial_qpos(actuators, joints)

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict(
            [
                ("left_leg", self.left_leg.action_space),
                ("right_leg", self.right_leg.action_space),
                ("waist", self.waist.action_space),
                ("left_arm", self.left_arm.action_space),
                ("right_arm", self.right_arm.action_space),
                ("left_eef", self.left_eef.action_space),
                ("right_eef", self.right_eef.action_space),
            ]
        )


class WholeBodyEEFControllerCfg(ControllerCfg):
    clazz: type[Controller] = WholeBodyEEFController

    def __init__(
        self,
        left_leg: PDJointPosControllerCfg,
        right_leg: PDJointPosControllerCfg,
        waist: PDJointPosControllerCfg | None,
        left_arm: PDJointPosControllerCfg,
        right_arm: PDJointPosControllerCfg,
        left_eef: DexHandEEFControllerCfg,
        right_eef: DexHandEEFControllerCfg,
    ) -> None:
        self.left_leg_cfg = left_leg
        self.right_leg_cfg = right_leg
        self.waist_cfg = waist
        self.left_arm_cfg = left_arm
        self.right_arm_cfg = right_arm
        self.left_eef_cfg = left_eef
        self.right_eef_cfg = right_eef
