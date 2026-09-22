"""Decoupled whole-body controller backend for the Unitree G1.

This module owns the robot-specific half of the legacy 36-D VLA path.  Task
policies only produce :class:`~humanoidtoolbench.actions.DecoupledGoal` objects; the
backend converts one goal into the 29 body and 14 Dex3 joint targets consumed
by the MuJoCo robot.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from humanoidtoolbench.actions import DecoupledGoal
from humanoidtoolbench.actions.g1 import (
    G1_LEFT_ARM_JOINTS,
    G1_LEFT_HAND_JOINTS,
    G1_RIGHT_ARM_JOINTS,
    G1_RIGHT_HAND_JOINTS,
)
from humanoidtoolbench.controllers.wbc_observation import build_wbc_observation
from humanoidtoolbench.core.action import ActionCmd


class DecoupledWbcBackend:
    """Run NVIDIA's decoupled WBC from typed 36-D high-level goals."""

    action_schema = DecoupledGoal.schema_id

    def __init__(self, robot: Any, sonic_config: Mapping[str, Any]) -> None:
        # Imports stay local so dataset/schema tooling remains usable without
        # the optional ``toolbench`` dependency group.
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.robot_model.instantiation.g1 import (
            instantiate_g1_robot_model,
        )

        self.robot = robot
        self._sonic_config = dict(sonic_config)
        self.sim_dt = float(self._sonic_config["SIMULATE_DT"])

        enable_waist = bool(self._sonic_config.get("enable_waist", False))
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"
        self._robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=bool(self._sonic_config.get("high_elbow_pose", False)),
        )

        supplemental = self._robot_model.supplemental_info
        if tuple(supplemental.body_actuated_joints) != tuple(robot.joint_names[:29]):
            raise ValueError("decoupled_wbc body joint order does not match G1Sonic")
        if tuple(supplemental.left_hand_actuated_joints) != tuple(robot.hand_names[:7]):
            raise ValueError("decoupled_wbc left-hand order does not match G1Sonic")
        if tuple(supplemental.right_hand_actuated_joints) != tuple(
            robot.hand_names[7:14]
        ):
            raise ValueError("decoupled_wbc right-hand order does not match G1Sonic")

        loop_config = ControlLoopConfig(
            enable_waist=enable_waist,
            high_elbow_pose=bool(self._sonic_config.get("high_elbow_pose", False)),
        )
        wbc_config = loop_config.load_wbc_yaml()
        if not np.isclose(float(wbc_config["SIMULATE_DT"]), self.sim_dt):
            raise ValueError(
                "decoupled_wbc and simulator time steps differ: "
                f"{wbc_config['SIMULATE_DT']} != {self.sim_dt}"
            )
        self._wbc_policy = get_wbc_policy(
            "g1",
            self._robot_model,
            wbc_config,
            # MuJoCo data is bound by env.reset(), after controller construction.
            init_time=0.0,
        )
        upper_indices = self._robot_model.get_joint_group_indices("upper_body")
        self._upper_joint_names = tuple(
            name
            for name, index in self._robot_model.joint_to_dof_index.items()
            if index in upper_indices
        )
        self._control_frequency = float(loop_config.control_frequency)
        self._cached_target_q: np.ndarray | None = None
        self._cached_left_hand_q: np.ndarray | None = None
        self._cached_right_hand_q: np.ndarray | None = None

    @property
    def wbc_policy(self) -> Any:
        """Underlying decoupled policy, exposed for diagnostics only."""

        return self._wbc_policy

    def _set_observation(self) -> None:
        proprio = self.robot.prepare_obs()
        self._wbc_policy.set_observation(
            build_wbc_observation(self._robot_model, proprio)
        )

    @staticmethod
    def _goal_joint_map(goal: DecoupledGoal) -> dict[str, float]:
        names = (
            *G1_LEFT_ARM_JOINTS,
            *G1_RIGHT_ARM_JOINTS,
            *G1_LEFT_HAND_JOINTS,
            *G1_RIGHT_HAND_JOINTS,
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
        )
        values = np.concatenate(
            (
                goal.left_arm_q,
                goal.right_arm_q,
                goal.left_hand_q,
                goal.right_hand_q,
                goal.torso_rpy,
            )
        )
        return dict(zip(names, (float(value) for value in values), strict=True))

    def _read_motor_targets(self, wbc_action: Mapping[str, Any]) -> ActionCmd:
        q = wbc_action["q"]
        self._cached_target_q = self._robot_model.get_body_actuated_joints(q)
        self._cached_left_hand_q = self._robot_model.get_hand_actuated_joints(
            q, side="left"
        )
        self._cached_right_hand_q = self._robot_model.get_hand_actuated_joints(
            q, side="right"
        )
        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )

    def step(
        self,
        goal: DecoupledGoal,
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> ActionCmd:
        del observation, kwargs
        if not isinstance(goal, DecoupledGoal):
            raise TypeError(
                f"DecoupledWbcBackend requires DecoupledGoal, got {type(goal).__name__}"
            )

        self._set_observation()
        now = float(self.robot.mjData.time)
        joint_map = self._goal_joint_map(goal)
        try:
            upper_body = np.asarray(
                [joint_map[name] for name in self._upper_joint_names],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unsupported upper-body joint requested by decoupled_wbc: {exc.args[0]}"
            ) from exc

        self._wbc_policy.set_goal(
            {
                "target_upper_body_pose": upper_body,
                "navigate_cmd": goal.navigate_cmd,
                "base_height_command": np.asarray(
                    [goal.base_height_command], dtype=np.float32
                ),
                "target_time": now + 1.0 / self._control_frequency,
                "interpolation_garbage_collection_time": (
                    now - 2.0 / self._control_frequency
                ),
                "timestamp": now,
            }
        )
        # Upstream set_goal stamps wall time; its timeout must use physics time too.
        self._wbc_policy.last_goal_time = now
        return self._read_motor_targets(self._wbc_policy.get_action(time=now))

    def get_stabilize_action(
        self, observation: dict[str, Any], **kwargs: Any
    ) -> ActionCmd:
        del observation, kwargs
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )

        self._set_observation()
        now = float(self.robot.mjData.time)
        first_step = self._cached_target_q is None
        self._wbc_policy.set_goal(
            {
                "target_upper_body_pose": (
                    self._robot_model.get_initial_upper_body_pose()
                ),
                "navigate_cmd": np.asarray(DEFAULT_NAV_CMD),
                "base_height_command": np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT)),
                "target_time": now
                + (2.0 if first_step else 1.0 / self._control_frequency),
                "interpolation_garbage_collection_time": (
                    now - 2.0 / self._control_frequency
                ),
                "timestamp": now,
            }
        )
        self._wbc_policy.last_goal_time = now
        return self._read_motor_targets(self._wbc_policy.get_action(time=now))

    def finish_stabilization(self) -> None:
        """Reset the gait phase after the simulator's stabilization loop."""

        lower_body = self._wbc_policy.lower_body_policy
        try:
            import torch

            lower_body.gait_indices = torch.zeros((1), dtype=torch.float32)
        except ImportError:  # pragma: no cover - controller dependency installs torch
            lower_body.gait_indices = np.zeros((1), dtype=np.float32)

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._wbc_policy.reset(init_time=float(self.robot.mjData.time))
        self._wbc_policy.lower_body_policy.use_policy_action = True
