"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
from pathlib import Path
from typing import Any

import gear_sonic
import mujoco
import numpy as np
from gear_sonic.utils.mujoco_sim.robot import Robot as GearSonicRobot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand

from humanoidtoolbench.core.action import ActionCmd
from humanoidtoolbench.core.controller import ControllerCfg
from humanoidtoolbench.core.robot import Robot
from humanoidtoolbench.core.types import Pose
from humanoidtoolbench.robots.controllers.combo import (
    WholeBodyEEFController,
    WholeBodyEEFControllerCfg,
)
from humanoidtoolbench.robots.controllers.eef import DexHandEEFControllerCfg
from humanoidtoolbench.robots.controllers.qpos import PDJointPosControllerCfg
from humanoidtoolbench.robots.protocols import (
    Controllable,
    HeadCamMountable,
    WristCamMountable,
)
from humanoidtoolbench.robots.registry import RobotRegistry

LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_HAND_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

WHOLE_BODY_JOINTS = (
    LEFT_LEG_JOINTS
    + RIGHT_LEG_JOINTS
    + WAIST_JOINTS
    + LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
    + LEFT_HAND_JOINTS
    + RIGHT_HAND_JOINTS
)
STABILIZE_VEL_THRESHOLD: float = 1e-4  # max |qvel[0:6]| to consider robot stable
MIN_STABILIZE_STEPS: int = 100  # min steps before velocity check is valid (1s at 200Hz)


def hand_q_mjcf_to_natural(q: np.ndarray) -> np.ndarray:
    """Convert thumb/middle/index MJCF order to thumb/index/middle order."""
    values = np.asarray(q)
    if values.shape != (7,):
        raise ValueError(f"Expected a 7-DOF hand vector, got {values.shape}")
    return np.concatenate([values[:3], values[5:7], values[3:5]])


def hand_q_natural_to_mjcf(q: np.ndarray) -> np.ndarray:
    """Convert the public thumb/index/middle order to the MJCF actuator order."""
    # Swapping the two two-joint groups is its own inverse.
    return hand_q_mjcf_to_natural(q)


@RobotRegistry.register("g1_sonic")
class G1Sonic(Robot, Controllable, HeadCamMountable, WristCamMountable):
    uid: str = "g1_sonic"
    label: str = "Unitree G1 Wholebody"

    wholebody_dof: int = 43
    dof: int = 29

    # Reuse Gear Sonic's bundled MuJoCo model. This avoids downloading HumanoidToolBench's
    # robot archive for the MuJoCo-only path.
    mjcf_path: str = str(
        Path(gear_sonic.__file__).resolve().parent
        / "data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
    )
    # The Isaac renderer draws a USD instead of the MJCF's meshes, so the robot
    # looks the part. Downloaded on first use; the MuJoCo path never touches it.
    #
    # Two builds of the same robot are available and HUMANOIDTOOLBENCH_G1_USD picks
    # between them. They differ in one thing that matters here: SIMPLE's build
    # has no `{side}_hand_camera_base_link`, so a wrist camera has nothing to
    # hang off under Isaac, while Unitree's own build carries that link and
    # lines up with the mount this repo now builds in MuJoCo. SIMPLE's stays
    # the default because it is what the rest of the asset release was drawn
    # against; Unitree's has to be fetched separately, see the docs.
    usd_builds: dict[str, str] = {
        "simple": "robots/g1/g1_29dof_wholebody_dex3.usd",
        "unitree": "robots/g1/unitree/g1_29dof_with_dex3_rev_1_0.usd",
    }

    @property
    def usd_path(self) -> str:
        build = os.environ.get("HUMANOIDTOOLBENCH_G1_USD", "simple").strip().lower()
        if build not in self.usd_builds:
            raise ValueError(
                f"HUMANOIDTOOLBENCH_G1_USD must be one of {sorted(self.usd_builds)}, "
                f"got {build!r}"
            )
        return self.usd_builds[build]

    head_camera_orientation: list[float] = [1, 0, 0, 0]
    joint_names = WHOLE_BODY_JOINTS
    hand_names = LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS

    controller_cfg: ControllerCfg = WholeBodyEEFControllerCfg(
        left_leg=PDJointPosControllerCfg(
            joint_names=LEFT_LEG_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        right_leg=PDJointPosControllerCfg(
            joint_names=RIGHT_LEG_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        waist=PDJointPosControllerCfg(
            joint_names=WAIST_JOINTS,
            init_qpos=[0.0, 0.0, 0.0],
        ),
        left_arm=PDJointPosControllerCfg(
            joint_names=LEFT_ARM_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        right_arm=PDJointPosControllerCfg(
            joint_names=RIGHT_ARM_JOINTS,
            init_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        left_eef=DexHandEEFControllerCfg(
            joint_names=LEFT_HAND_JOINTS,
            init_qpos=[0, 0, 0, 0, 0, 0, 0],
        ),
        right_eef=DexHandEEFControllerCfg(
            joint_names=RIGHT_HAND_JOINTS,
            init_qpos=[0, 0, 0, 0, 0, 0, 0],
        ),
    )

    spawn_pose: Pose

    def __init__(self, sonic_config: dict, **kwargs) -> None:
        del kwargs
        Robot.__init__(self, self.uid, self.dof)
        self._controller = None
        self.sonic_config = sonic_config

        self.sim_dt = self.sonic_config["SIMULATE_DT"]
        sonic_robot = GearSonicRobot(self.sonic_config)

        self.num_body_dof = sonic_robot.NUM_JOINTS
        self.num_hand_dof = sonic_robot.NUM_HAND_JOINTS
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(sonic_robot.MOTOR_EFFORT_LIMIT_LIST)

    def reset(self, **kwargs):
        self.spawn_pose = kwargs["spawn_pose"]
        self._stabilized = False
        self._stabilize_step_count = 0

    @property
    def stabilized(self) -> bool:
        """True once max(|floating-base qvel[0:6]|) drops below STABILIZE_VEL_THRESHOLD.

        Requires MIN_STABILIZE_STEPS accesses before the velocity check is
        evaluated, preventing an immediate latch when qvel=0 right after reset.
        Latches to True and stays there until the next reset().
        """
        self._stabilize_step_count += 1
        if not self._stabilized and self._stabilize_step_count >= MIN_STABILIZE_STEPS:
            self._stabilized = bool(
                np.max(np.abs(self.mjData.qvel[0:6])) < STABILIZE_VEL_THRESHOLD
            )
        return self._stabilized

    def setup_control(
        self, mjData, mjModel, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Called after environment reset, ..."""
        del kwargs
        self.mjData = mjData
        self.mjModel = mjModel

        self.torso_index = mujoco.mj_name2id(
            mjModel, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            mjModel.joint(i).name for i in range(mjModel.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            mjModel.joint(i).name for i in range(mjModel.njnt)
        ]

        # Enable the elastic band. It stays None when disabled rather than being
        # left unset: QuestDecoupledAgent reads `robot.elastic_band` unguarded
        # (`if self.robot.elastic_band and ...`), so it already expects a falsy
        # value here: an absent attribute is an AttributeError the moment the
        # robot stabilises and the agent reaches that branch.
        self.elastic_band = None
        if self.sonic_config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand(
                point=np.array(self.spawn_pose.position[:2] + [1.0])
            )
            if "g1" in self.sonic_config["ROBOT_TYPE"]:
                if self.sonic_config["enable_waist"]:
                    self.band_attached_link = mjModel.body("pelvis").id
                else:
                    self.band_attached_link = mjModel.body("torso_link").id
            elif "h1" in self.sonic_config["ROBOT_TYPE"]:
                self.band_attached_link = mjModel.body("torso_link").id
            else:
                self.band_attached_link = mjModel.body("base_link").id

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        body_joint_index = []
        left_hand_index = []
        right_hand_index = []
        for i in range(self.mjModel.njnt):
            name = self.mjModel.joint(i).name
            if (
                any(
                    [
                        part_name in name
                        for part_name in [
                            "hip",
                            "knee",
                            "ankle",
                            "waist",
                            "shoulder",
                            "elbow",
                            "wrist",
                        ]
                    ]
                )
                and name in self.joint_names
            ):
                body_joint_index.append(i)
            elif "left_hand" in name:
                left_hand_index.append(i)
            elif "right_hand" in name:
                right_hand_index.append(i)

        assert len(body_joint_index) == self.num_body_dof
        assert len(left_hand_index) == self.num_hand_dof
        assert len(right_hand_index) == self.num_hand_dof

        self.body_joint_index = np.array(body_joint_index)
        self.left_hand_index = np.array(left_hand_index)
        self.right_hand_index = np.array(right_hand_index)

        actuators = {}
        joints = {}

        for name in self.joint_names:
            try:
                actuators[name] = mjData.actuator(name)
            except KeyError:
                # Gear Sonic's bundled MJCF names actuators without the
                # ``_joint`` suffix while the public data contract uses full
                # joint names.
                actuators[name] = mjData.actuator(name.removesuffix("_joint"))
            joints[name] = mjData.joint(name)

        self.joints = joints
        self.controller.set_initial_qpos(actuators, joints)

        return self.joints, actuators

    @property
    def pelvis_vz(self) -> float:
        return self.mjData.qvel[2]

    @property
    def pelvis_z(self) -> float:
        return self.mjData.qpos[2]

    def prepare_obs(self) -> dict[str, Any]:
        """Snapshot measured state independently of mutable simulator buffers."""
        obs = {}
        if self.use_floating_root_link:  # move to this class
            obs["floating_base_pose"] = self.mjData.qpos[:7].copy()
            obs["floating_base_vel"] = self.mjData.qvel[:6].copy()
            obs["floating_base_acc"] = self.mjData.qacc[:6].copy()
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mjData.xquat[self.torso_index].copy()

        pose = np.zeros(13)
        torso_link = self.mjModel.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mjModel,
            self.mjData,
            mujoco.mjtObj.mjOBJ_BODY,
            torso_link,
            pose[7:13],
            1,
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mjData.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mjData.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mjData.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mjData.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mjData.qpos[
                self.left_hand_index + self.qpos_offset - 1
            ]
            obs["left_hand_dq"] = self.mjData.qvel[
                self.left_hand_index + self.qvel_offset - 1
            ]
            obs["left_hand_ddq"] = self.mjData.qacc[
                self.left_hand_index + self.qvel_offset - 1
            ]
            obs["left_hand_tau_est"] = self.mjData.actuator_force[
                self.left_hand_index - 1
            ]
            obs["right_hand_q"] = self.mjData.qpos[
                self.right_hand_index + self.qpos_offset - 1
            ]
            obs["right_hand_dq"] = self.mjData.qvel[
                self.right_hand_index + self.qvel_offset - 1
            ]
            obs["right_hand_ddq"] = self.mjData.qacc[
                self.right_hand_index + self.qvel_offset - 1
            ]
            obs["right_hand_tau_est"] = self.mjData.actuator_force[
                self.right_hand_index - 1
            ]
        obs["time"] = self.mjData.time
        return obs  # joints in mjcf order (thumb, middle, index)

    def get_robot_qpos(self) -> dict[str, float]:
        """Get the current joint positions of the robot."""
        return {j: v.qpos[0] for j, v in self.joints.items()}

    def apply_action(self, action_cmd: ActionCmd) -> None:
        assert isinstance(self.controller, WholeBodyEEFController)

        match action_cmd.type:
            case "elastic_band":
                pose = np.concatenate(
                    [
                        self.mjData.xpos[self.band_attached_link],
                        self.mjData.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mjModel,
                    self.mjData,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mjData.xfrc_applied[self.band_attached_link] = (
                    self.elastic_band.Advance(pose)
                )

            case "gear_sonic":
                # Unified GEAR-SONIC decoders produce Unitree-style motor
                # targets (q/dq/tau/kp/kd) for the 29-DoF body.  Keeping this
                # ActionCmd distinct from ``decoupled_wbc`` prevents a 64-D
                # motion latent from accidentally being interpreted as the
                # legacy high-level navigation/upper-body goal.
                def motor_vector(name, value, default=None):
                    if value is None:
                        value = default
                    result = np.asarray(value, dtype=np.float64)
                    if result.shape != (self.num_body_dof,):
                        raise ValueError(
                            f"gear_sonic {name} must have shape "
                            f"({self.num_body_dof},), got {result.shape}"
                        )
                    if not np.all(np.isfinite(result)):
                        raise ValueError(
                            f"gear_sonic {name} must contain finite values"
                        )
                    return result

                target_q = motor_vector("target_q", action_cmd["target_q"])
                target_dq = motor_vector(
                    "target_dq",
                    action_cmd["target_dq"],
                    np.zeros(self.num_body_dof),
                )
                tau_ff = motor_vector(
                    "tau_ff",
                    action_cmd["tau_ff"],
                    np.zeros(self.num_body_dof),
                )
                kp = motor_vector(
                    "kp",
                    action_cmd["kp"],
                    self.sonic_config.get("MOTOR_KP", [100.0] * self.num_body_dof),
                )
                kd = motor_vector(
                    "kd",
                    action_cmd["kd"],
                    self.sonic_config.get("MOTOR_KD", [5.0] * self.num_body_dof),
                )

                q_cur = self.mjData.qpos[self.body_joint_index + self.qpos_offset - 1]
                dq_cur = self.mjData.qvel[self.body_joint_index + self.qvel_offset - 1]
                self.torques[self.body_joint_index - 1] = (
                    tau_ff + kp * (target_q - q_cur) + kd * (target_dq - dq_cur)
                )

                # GEAR-SONIC's official VLA action keeps the two Dex3 hands as
                # explicit 7-D channels next to the 64-D motion latent.  Those
                # vectors use the public thumb/index/middle order.
                if self.num_hand_dof > 0:
                    hand_kp = np.array([5.0, 5.0, 5.0, 2.5, 2.5, 2.5, 2.5])
                    hand_kd = 1.0
                    for hand_q, joint_index in (
                        (action_cmd["left_hand_q"], self.left_hand_index),
                        (action_cmd["right_hand_q"], self.right_hand_index),
                    ):
                        if hand_q is None:
                            continue
                        hand_q = hand_q_natural_to_mjcf(np.asarray(hand_q))
                        hand_q_cur = self.mjData.qpos[
                            joint_index + self.qpos_offset - 1
                        ]
                        hand_dq_cur = self.mjData.qvel[
                            joint_index + self.qvel_offset - 1
                        ]
                        self.torques[joint_index - 1] = hand_kp * (
                            hand_q - hand_q_cur
                        ) + hand_kd * (0 - hand_dq_cur)

                self.torques = np.clip(
                    self.torques, -self.torque_limit, self.torque_limit
                )
                if self.sonic_config["FREE_BASE"]:
                    self.mjData.ctrl = np.concatenate((np.zeros(6), self.torques))
                else:
                    self.mjData.ctrl = self.torques

            case "decoupled_wbc":
                target_q = action_cmd["target_q"]  # 29 body joints in actuator order
                # PD position control using per-joint gains from decoupled_wbc config
                kp = np.array(
                    self.sonic_config.get("MOTOR_KP", [100.0] * self.num_body_dof)
                )
                kd = np.array(
                    self.sonic_config.get("MOTOR_KD", [5.0] * self.num_body_dof)
                )

                q_cur = self.mjData.qpos[self.body_joint_index + self.qpos_offset - 1]
                dq_cur = self.mjData.qvel[self.body_joint_index + self.qvel_offset - 1]

                body_torques = kp * (target_q - q_cur) + kd * (0 - dq_cur)
                self.torques[self.body_joint_index - 1] = body_torques

                # Hand PD control (driven by trigger/grip via decoupled WBC teleop IK)
                # Joint order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
                # Index + middle work together against thumb in a power grip,
                # so their kp is halved to balance grip forces.
                if self.num_hand_dof > 0:
                    left_hand_q = action_cmd["left_hand_q"]
                    right_hand_q = action_cmd["right_hand_q"]
                    hand_kp = np.array([5.0, 5.0, 5.0, 2.5, 2.5, 2.5, 2.5])
                    hand_kd = 1.0
                    if left_hand_q is not None:
                        left_hand_q = hand_q_natural_to_mjcf(left_hand_q)
                        lh_q_cur = self.mjData.qpos[
                            self.left_hand_index + self.qpos_offset - 1
                        ]
                        lh_dq_cur = self.mjData.qvel[
                            self.left_hand_index + self.qvel_offset - 1
                        ]
                        self.torques[self.left_hand_index - 1] = hand_kp * (
                            left_hand_q - lh_q_cur
                        ) + hand_kd * (0 - lh_dq_cur)
                    if right_hand_q is not None:
                        right_hand_q = hand_q_natural_to_mjcf(right_hand_q)
                        rh_q_cur = self.mjData.qpos[
                            self.right_hand_index + self.qpos_offset - 1
                        ]
                        rh_dq_cur = self.mjData.qvel[
                            self.right_hand_index + self.qvel_offset - 1
                        ]
                        self.torques[self.right_hand_index - 1] = hand_kp * (
                            right_hand_q - rh_q_cur
                        ) + hand_kd * (0 - rh_dq_cur)

                self.torques = np.clip(
                    self.torques, -self.torque_limit, self.torque_limit
                )

                if self.sonic_config["FREE_BASE"]:
                    self.mjData.ctrl = np.concatenate((np.zeros(6), self.torques))
                else:
                    self.mjData.ctrl = self.torques

    @property
    def head_cam_link(self) -> str:
        """Get the head camera link name."""
        return "d435_link"

    def wrist_cam_link(self, side: str) -> str:
        """The body a wrist camera hangs off. The G1 has no camera link on the
        hand, so the camera rides the last wrist joint, which is what carries
        the palm and the fingers."""
        return f"{side}_wrist_yaw_link"
