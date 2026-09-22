"""HTTP task policy for the decoupled G1 controller.

The adapter stops at a typed controller goal.  It sends the head image, the
instruction and the 32-D policy state to a policy server's ``/act`` endpoint
and decodes the returned ``(T, 36)`` chunk into ``DecoupledGoal`` values for
the decoupled whole-body controller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from humanoidtoolbench.actions import DECOUPLED_SCHEMA_ID, DecoupledActionCodec
from humanoidtoolbench.policies.base import ActionChunk


def _joint_qpos(observation: Mapping[str, Any]) -> np.ndarray:
    try:
        qpos = np.asarray(observation["joint_qpos"], dtype=np.float32)
    except KeyError as exc:
        raise KeyError("Humanoid policy observation is missing 'joint_qpos'") from exc
    if qpos.shape != (43,):
        raise ValueError(f"G1 joint_qpos must have shape (43,), got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("G1 joint_qpos must contain only finite values")
    return qpos


def _head_image(observation: Mapping[str, Any]) -> np.ndarray:
    try:
        image = np.asarray(observation["head_stereo_left"])
    except KeyError as exc:
        raise KeyError(
            "Humanoid policy observation is missing 'head_stereo_left'"
        ) from exc
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "head_stereo_left must be an RGB array with shape (H, W, 3), "
            f"got {image.shape}"
        )
    return image


def build_decoupled_policy_state(
    observation: Mapping[str, Any], last_base_height: float
) -> np.ndarray:
    """Build the 32-D policy state sent with every ``/act`` request."""

    qpos = _joint_qpos(observation)
    # Flat state order is the historical training contract.  In particular,
    # the left hand is thumb/middle/index even though public G1 goals are
    # thumb/index/middle.
    state = np.concatenate(
        (
            qpos[29:32],
            qpos[34:36],
            qpos[32:34],
            qpos[36:43],
            qpos[15:22],
            qpos[22:29],
            qpos[[13, 14, 12]],
            np.asarray((last_base_height,), dtype=np.float32),
        )
    ).astype(np.float32)
    if state.shape != (32,):  # defensive check when the layout changes
        raise AssertionError(f"Internal decoupled state shape error: {state.shape}")
    return state


def _normalize_prediction(raw: object) -> np.ndarray:
    """Normalize server output into a strict ``(T, 36)`` chunk."""

    action = np.asarray(raw, dtype=np.float32)
    while action.ndim > 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim == 1:
        action = action[None]
    if action.ndim != 2 or action.shape[1] != DecoupledActionCodec.dimension:
        raise ValueError(
            f"Policy response for {DECOUPLED_SCHEMA_ID!r} must have shape "
            f"(T, {DecoupledActionCodec.dimension}), got {action.shape}. Check that "
            "the checkpoint and controller use the same action schema."
        )
    return action


class RemoteChunkPolicy:
    """Codec, history, and chunk behavior of a remote policy transport."""

    def __init__(self, *, policy_name: str) -> None:
        self.policy_name = policy_name
        self._reset_history = True
        self._last_base_height = 0.74

    def _chunk(self, raw: object, *, transport: str) -> ActionChunk[Any]:
        flat = _normalize_prediction(raw)
        goals = list(DecoupledActionCodec.decode_chunk(flat))
        if goals:
            self._last_base_height = goals[-1].base_height_command
        return ActionChunk(
            schema=DECOUPLED_SCHEMA_ID,
            goals=goals,
            metadata={
                "policy": self.policy_name,
                "transport": transport,
                "raw_horizon": int(flat.shape[0]),
            },
        )

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        self._reset_history = True
        self._last_base_height = 0.74

    def _state(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        state = build_decoupled_policy_state(observation, self._last_base_height)
        return {"states": state[None]}


class HttpTaskPolicy(RemoteChunkPolicy):
    """Adapter for HumanoidToolBench's NumPy-over-HTTP ``/act`` protocol."""

    def __init__(self, host: str, port: int, *, client: Any | None = None) -> None:
        super().__init__(policy_name="http")
        if client is None:
            from humanoidtoolbench.policies.http_client import HttpActionClient

            client = HttpActionClient(host, port)
        self.client = client

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionChunk[Any]:
        del kwargs
        if not instruction:
            raise ValueError("The HTTP policy needs the task instruction")
        history = {"reset": True} if self._reset_history else {}
        self._reset_history = False
        raw = self.client.query_action(
            {"rgb_head_stereo_left": _head_image(observation)},
            instruction,
            self._state(observation),
            history=history,
        )
        return self._chunk(raw, transport="http_act")


def make_remote_task_policy(
    name: str,
    *,
    host: str,
    port: int,
    client: Any | None = None,
) -> RemoteChunkPolicy:
    """Construct the task-policy transport independently of the controller."""

    if name != "http":
        raise ValueError(f"Unsupported humanoid task policy {name!r}; choose http")
    return HttpTaskPolicy(host, port, client=client)
