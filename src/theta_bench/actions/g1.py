"""Stable policy-action schemas for the Unitree G1 with Dex3 hands.

The dataclasses in this module describe controller goals, while the codecs own
the flat arrays exchanged with policy servers and stored in training datasets.
Keeping the two concepts separate lets ACT, pi, Psi, and world-action models use
the same action contract without depending on a particular controller package.

``DecoupledActionCodec`` deliberately preserves the existing THETA-Bench 36-D
layout.  In that legacy flat layout the left hand is ordered
thumb/middle/index, whereas G1 joint metadata and ``DecoupledGoal.left_hand_q``
use the natural thumb/index/middle order.  The codec performs that permutation
explicitly so callers never need to reproduce the historical asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Iterable, Mapping, Sequence, Union

import numpy as np
from numpy.typing import NDArray

ArrayLike = Union[NDArray[np.floating], Sequence[float]]
Float32Vector = NDArray[np.float32]


# These names duplicate the dependency-free metadata from robots/g1_sonic.py.
# Importing that robot module here would pull in MuJoCo and GEAR-SONIC merely to
# decode a policy response, preventing dataset tooling from staying lightweight.
G1_LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
G1_RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_LEFT_HAND_JOINTS = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
G1_RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

_G1_TORSO_RPY_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)
_LEFT_HAND_LEGACY_ORDER = (
    G1_LEFT_HAND_JOINTS[0],
    G1_LEFT_HAND_JOINTS[1],
    G1_LEFT_HAND_JOINTS[2],
    G1_LEFT_HAND_JOINTS[5],
    G1_LEFT_HAND_JOINTS[6],
    G1_LEFT_HAND_JOINTS[3],
    G1_LEFT_HAND_JOINTS[4],
)

DECOUPLED_SCHEMA_ID = "decoupled_v1"
DECOUPLED_ACTION_DIM = 36
DECOUPLED_ACTION_SLICES: Mapping[str, slice] = MappingProxyType(
    {
        "left_hand_q": slice(0, 7),
        "right_hand_q": slice(7, 14),
        "left_arm_q": slice(14, 21),
        "right_arm_q": slice(21, 28),
        "torso_rpy": slice(28, 31),
        "base_height_command": slice(31, 32),
        "navigate_cmd": slice(32, 36),
    }
)
DECOUPLED_ACTION_ORDER = (
    *_LEFT_HAND_LEGACY_ORDER,
    *G1_RIGHT_HAND_JOINTS,
    *G1_LEFT_ARM_JOINTS,
    *G1_RIGHT_ARM_JOINTS,
    *_G1_TORSO_RPY_NAMES,
    "base_height_command",
    "navigate_vx",
    "navigate_vy",
    "navigate_vyaw",
    "navigate_target_yaw",
)

SONIC_LATENT_SCHEMA_ID = "sonic_latent_v1"
SONIC_LATENT_ACTION_DIM = 78
SONIC_LATENT_ACTION_SLICES: Mapping[str, slice] = MappingProxyType(
    {
        "motion_latent": slice(0, 64),
        "left_hand_q": slice(64, 71),
        "right_hand_q": slice(71, 78),
    }
)
SONIC_LATENT_ACTION_ORDER = (
    *(f"motion_latent_{index}" for index in range(64)),
    *G1_LEFT_HAND_JOINTS,
    *G1_RIGHT_HAND_JOINTS,
)

# Moving between natural thumb/index/middle order and the historical
# thumb/middle/index flat order is the same permutation in both directions.
_LEFT_HAND_PERMUTATION = np.asarray((0, 1, 2, 5, 6, 3, 4), dtype=np.intp)


def _validated_vector(name: str, value: ArrayLike, size: int) -> Float32Vector:
    """Return an owned, immutable float32 vector with the required shape."""

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric vector of shape ({size},)") from exc
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = array.copy()
    result.flags.writeable = False
    return result


def _validated_scalar(name: str, value: object) -> float:
    """Normalize a scalar or legacy one-element command vector to ``float``."""

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if array.shape == (1,):
        array = array.reshape(())
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_flat_action(
    value: ArrayLike, *, schema_id: str, dimension: int
) -> Float32Vector:
    return _validated_vector(f"{schema_id} action", value, dimension)


def _validated_flat_chunk(
    value: object, *, schema_id: str, dimension: int
) -> NDArray[np.float32]:
    try:
        chunk = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{schema_id} action chunk must be a numeric array of shape (T, {dimension})"
        ) from exc
    if chunk.ndim != 2 or chunk.shape[1] != dimension:
        raise ValueError(
            f"{schema_id} action chunk must have shape (T, {dimension}), got {chunk.shape}"
        )
    if not np.all(np.isfinite(chunk)):
        raise ValueError(f"{schema_id} action chunk must contain only finite values")
    return chunk


@dataclass(frozen=True, eq=False)
class DecoupledGoal:
    """One controller-neutral goal for the decoupled G1 WBC.

    All joint vectors use the G1 joint tuple exported next to this class.
    ``torso_rpy`` is roll, pitch, yaw. ``navigate_cmd`` is
    ``[vx, vy, vyaw, target_yaw]``.  Arrays are copied, converted to float32,
    checked for finite values, and made read-only during construction.
    """

    left_hand_q: Float32Vector
    right_hand_q: Float32Vector
    left_arm_q: Float32Vector
    right_arm_q: Float32Vector
    torso_rpy: Float32Vector
    base_height_command: float
    navigate_cmd: Float32Vector

    schema_id: ClassVar[str] = DECOUPLED_SCHEMA_ID
    action_dim: ClassVar[int] = DECOUPLED_ACTION_DIM
    left_hand_joint_order: ClassVar[tuple[str, ...]] = G1_LEFT_HAND_JOINTS
    right_hand_joint_order: ClassVar[tuple[str, ...]] = G1_RIGHT_HAND_JOINTS
    left_arm_joint_order: ClassVar[tuple[str, ...]] = G1_LEFT_ARM_JOINTS
    right_arm_joint_order: ClassVar[tuple[str, ...]] = G1_RIGHT_ARM_JOINTS
    torso_order: ClassVar[tuple[str, ...]] = _G1_TORSO_RPY_NAMES
    navigate_order: ClassVar[tuple[str, ...]] = (
        "vx",
        "vy",
        "vyaw",
        "target_yaw",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "left_hand_q", _validated_vector("left_hand_q", self.left_hand_q, 7)
        )
        object.__setattr__(
            self,
            "right_hand_q",
            _validated_vector("right_hand_q", self.right_hand_q, 7),
        )
        object.__setattr__(
            self, "left_arm_q", _validated_vector("left_arm_q", self.left_arm_q, 7)
        )
        object.__setattr__(
            self,
            "right_arm_q",
            _validated_vector("right_arm_q", self.right_arm_q, 7),
        )
        object.__setattr__(
            self, "torso_rpy", _validated_vector("torso_rpy", self.torso_rpy, 3)
        )
        object.__setattr__(
            self,
            "base_height_command",
            _validated_scalar("base_height_command", self.base_height_command),
        )
        object.__setattr__(
            self,
            "navigate_cmd",
            _validated_vector("navigate_cmd", self.navigate_cmd, 4),
        )

    def to_array(self) -> Float32Vector:
        """Encode this goal in the stable 36-D policy/dataset layout."""

        return DecoupledActionCodec.encode(self)

    @classmethod
    def from_array(cls, action: ArrayLike) -> "DecoupledGoal":
        """Decode one stable 36-D policy/dataset action."""

        return DecoupledActionCodec.decode(action)


@dataclass(frozen=True, eq=False)
class SonicLatentGoal:
    """One unified SONIC goal: 64-D motion latent and two Dex3 hands.

    The two hand vectors use ``G1_LEFT_HAND_JOINTS`` and
    ``G1_RIGHT_HAND_JOINTS`` order (thumb, index, middle).
    """

    motion_latent: Float32Vector
    left_hand_q: Float32Vector
    right_hand_q: Float32Vector

    schema_id: ClassVar[str] = SONIC_LATENT_SCHEMA_ID
    action_dim: ClassVar[int] = SONIC_LATENT_ACTION_DIM
    left_hand_joint_order: ClassVar[tuple[str, ...]] = G1_LEFT_HAND_JOINTS
    right_hand_joint_order: ClassVar[tuple[str, ...]] = G1_RIGHT_HAND_JOINTS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "motion_latent",
            _validated_vector("motion_latent", self.motion_latent, 64),
        )
        object.__setattr__(
            self, "left_hand_q", _validated_vector("left_hand_q", self.left_hand_q, 7)
        )
        object.__setattr__(
            self,
            "right_hand_q",
            _validated_vector("right_hand_q", self.right_hand_q, 7),
        )

    def to_array(self) -> Float32Vector:
        """Encode this goal in the stable 78-D policy/dataset layout."""

        return SonicLatentActionCodec.encode(self)

    @classmethod
    def from_array(cls, action: ArrayLike) -> "SonicLatentGoal":
        """Decode one stable 78-D policy/dataset action."""

        return SonicLatentActionCodec.decode(action)


class DecoupledActionCodec:
    """Codec for ``decoupled_v1`` flat actions and action chunks."""

    schema_id = DECOUPLED_SCHEMA_ID
    dimension = DECOUPLED_ACTION_DIM
    action_order = DECOUPLED_ACTION_ORDER
    action_slices = DECOUPLED_ACTION_SLICES

    @staticmethod
    def encode(goal: DecoupledGoal) -> Float32Vector:
        if not isinstance(goal, DecoupledGoal):
            raise TypeError(f"goal must be DecoupledGoal, got {type(goal).__name__}")

        action = np.empty(DECOUPLED_ACTION_DIM, dtype=np.float32)
        action[DECOUPLED_ACTION_SLICES["left_hand_q"]] = goal.left_hand_q[
            _LEFT_HAND_PERMUTATION
        ]
        action[DECOUPLED_ACTION_SLICES["right_hand_q"]] = goal.right_hand_q
        action[DECOUPLED_ACTION_SLICES["left_arm_q"]] = goal.left_arm_q
        action[DECOUPLED_ACTION_SLICES["right_arm_q"]] = goal.right_arm_q
        action[DECOUPLED_ACTION_SLICES["torso_rpy"]] = goal.torso_rpy
        action[DECOUPLED_ACTION_SLICES["base_height_command"]] = (
            goal.base_height_command
        )
        action[DECOUPLED_ACTION_SLICES["navigate_cmd"]] = goal.navigate_cmd
        return action

    @staticmethod
    def decode(action: ArrayLike) -> DecoupledGoal:
        flat = _validated_flat_action(
            action,
            schema_id=DECOUPLED_SCHEMA_ID,
            dimension=DECOUPLED_ACTION_DIM,
        )
        legacy_left_hand = flat[DECOUPLED_ACTION_SLICES["left_hand_q"]]
        return DecoupledGoal(
            left_hand_q=legacy_left_hand[_LEFT_HAND_PERMUTATION],
            right_hand_q=flat[DECOUPLED_ACTION_SLICES["right_hand_q"]],
            left_arm_q=flat[DECOUPLED_ACTION_SLICES["left_arm_q"]],
            right_arm_q=flat[DECOUPLED_ACTION_SLICES["right_arm_q"]],
            torso_rpy=flat[DECOUPLED_ACTION_SLICES["torso_rpy"]],
            base_height_command=flat[DECOUPLED_ACTION_SLICES["base_height_command"]],
            navigate_cmd=flat[DECOUPLED_ACTION_SLICES["navigate_cmd"]],
        )

    @classmethod
    def encode_chunk(cls, goals: Iterable[DecoupledGoal]) -> NDArray[np.float32]:
        encoded = [cls.encode(goal) for goal in goals]
        if not encoded:
            return np.empty((0, cls.dimension), dtype=np.float32)
        return np.stack(encoded, axis=0)

    @classmethod
    def decode_chunk(cls, actions: object) -> tuple[DecoupledGoal, ...]:
        chunk = _validated_flat_chunk(
            actions, schema_id=cls.schema_id, dimension=cls.dimension
        )
        return tuple(cls.decode(action) for action in chunk)


class SonicLatentActionCodec:
    """Codec for ``sonic_latent_v1`` flat actions and action chunks."""

    schema_id = SONIC_LATENT_SCHEMA_ID
    dimension = SONIC_LATENT_ACTION_DIM
    action_order = SONIC_LATENT_ACTION_ORDER
    action_slices = SONIC_LATENT_ACTION_SLICES

    @staticmethod
    def encode(goal: SonicLatentGoal) -> Float32Vector:
        if not isinstance(goal, SonicLatentGoal):
            raise TypeError(f"goal must be SonicLatentGoal, got {type(goal).__name__}")

        action = np.empty(SONIC_LATENT_ACTION_DIM, dtype=np.float32)
        action[SONIC_LATENT_ACTION_SLICES["motion_latent"]] = goal.motion_latent
        action[SONIC_LATENT_ACTION_SLICES["left_hand_q"]] = goal.left_hand_q
        action[SONIC_LATENT_ACTION_SLICES["right_hand_q"]] = goal.right_hand_q
        return action

    @staticmethod
    def decode(action: ArrayLike) -> SonicLatentGoal:
        flat = _validated_flat_action(
            action,
            schema_id=SONIC_LATENT_SCHEMA_ID,
            dimension=SONIC_LATENT_ACTION_DIM,
        )
        return SonicLatentGoal(
            motion_latent=flat[SONIC_LATENT_ACTION_SLICES["motion_latent"]],
            left_hand_q=flat[SONIC_LATENT_ACTION_SLICES["left_hand_q"]],
            right_hand_q=flat[SONIC_LATENT_ACTION_SLICES["right_hand_q"]],
        )

    @classmethod
    def encode_chunk(cls, goals: Iterable[SonicLatentGoal]) -> NDArray[np.float32]:
        encoded = [cls.encode(goal) for goal in goals]
        if not encoded:
            return np.empty((0, cls.dimension), dtype=np.float32)
        return np.stack(encoded, axis=0)

    @classmethod
    def decode_chunk(cls, actions: object) -> tuple[SonicLatentGoal, ...]:
        chunk = _validated_flat_chunk(
            actions, schema_id=cls.schema_id, dimension=cls.dimension
        )
        return tuple(cls.decode(action) for action in chunk)
