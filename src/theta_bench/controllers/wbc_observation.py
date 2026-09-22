"""Convert G1 MuJoCo proprioception to the decoupled WBC observation layout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def hand_mjcf_to_natural(q: Any) -> np.ndarray:
    """Map MJCF thumb/middle/index order to WBC thumb/index/middle order."""
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(f"Expected a 7-DOF hand vector, got {values.shape}")
    return np.concatenate((values[:3], values[5:7], values[3:5]))


def build_wbc_observation(
    robot_model: Any, sim_obs: Mapping[str, Any]
) -> dict[str, Any]:
    """Build WBC q/dq/ddq/tau_est in the robot model's full joint layout.

    G1Sonic.prepare_obs reports hands in MJCF joint order. WBC's supplemental
    actuated-joint lists use the public thumb/index/middle order instead.
    Position and velocity are required; optional measurements retain zeros.
    """
    result: dict[str, Any] = {}
    for quantity in ("q", "dq", "ddq", "tau_est"):
        body_key = f"body_{quantity}"
        body_values = (
            sim_obs[body_key]
            if quantity in {"q", "dq"}
            else sim_obs.get(body_key, np.zeros(29))
        )
        result[quantity] = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_values,
            left_hand_actuated_joint_values=hand_mjcf_to_natural(
                sim_obs.get(f"left_hand_{quantity}", np.zeros(7))
            ),
            right_hand_actuated_joint_values=hand_mjcf_to_natural(
                sim_obs.get(f"right_hand_{quantity}", np.zeros(7))
            ),
        )
    result["floating_base_pose"] = sim_obs["floating_base_pose"]
    result["floating_base_vel"] = sim_obs["floating_base_vel"]
    result["floating_base_acc"] = sim_obs.get("floating_base_acc", np.zeros(6))
    result["torso_quat"] = sim_obs.get(
        "secondary_imu_quat", np.asarray((1.0, 0.0, 0.0, 0.0))
    )
    secondary_velocity = sim_obs.get("secondary_imu_vel")
    result["torso_ang_vel"] = (
        np.asarray(secondary_velocity)[3:6]
        if secondary_velocity is not None
        else np.zeros(3)
    )
    result["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))
    return result
