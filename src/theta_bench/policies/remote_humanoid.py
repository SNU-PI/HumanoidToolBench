"""Remote task-policy adapters for composable humanoid control.

The adapters in this module stop at a typed controller goal.  They do not
import or run either decoupled WBC or GEAR-SONIC, so the same ACT, pi, Psi, or
world-action-model server can be paired with either body controller as long as
the checkpoint was trained for the selected action schema.
"""

from __future__ import annotations

import base64
import io
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from theta_bench.actions import (
    DECOUPLED_SCHEMA_ID,
    SONIC_LATENT_SCHEMA_ID,
    DecoupledActionCodec,
    DecoupledGoal,
    SonicLatentActionCodec,
)
from theta_bench.policies.base import ActionChunk

SUPPORTED_TASK_POLICIES = (
    "http",
    "act",
    "pi05",
    "psi0",
    "psix",
    "gr00t_n16",
    "groot",
    "intervla",
    "egovla",
    "hrdt",
    "dp",
    "cosmos3",
    "dreamzero",
    "unifolm",
    "fastwam",
    "molmoact2",
)


class ActionCodec(Protocol):
    schema_id: str
    dimension: int

    @staticmethod
    def decode_chunk(value: object) -> tuple[Any, ...]: ...


def action_codec_for_schema(schema: str) -> type[ActionCodec]:
    """Return the strict codec for a public controller action schema."""

    normalized = schema.strip().lower()
    if normalized == DECOUPLED_SCHEMA_ID:
        return DecoupledActionCodec
    if normalized == SONIC_LATENT_SCHEMA_ID:
        return SonicLatentActionCodec
    raise ValueError(
        f"Unknown humanoid action schema {schema!r}; expected "
        f"{DECOUPLED_SCHEMA_ID!r} or {SONIC_LATENT_SCHEMA_ID!r}"
    )


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


def _projected_gravity(kwargs: Mapping[str, Any]) -> np.ndarray:
    """Compute SONIC's pelvis-frame gravity vector from simulator proprio."""

    info = kwargs.get("info")
    proprio = info.get("proprio", {}) if isinstance(info, Mapping) else {}
    quat = proprio.get("base_quat")
    if quat is None:
        floating_base_pose = proprio.get("floating_base_pose")
        if floating_base_pose is not None:
            floating_base_pose = np.asarray(floating_base_pose, dtype=np.float64)
            if floating_base_pose.shape != (7,) or not np.all(
                np.isfinite(floating_base_pose)
            ):
                raise ValueError(
                    "floating_base_pose must be a finite xyz+wxyz vector of shape (7,)"
                )
            quat = floating_base_pose[3:7]
    if quat is None:
        raise KeyError(
            "Unified SONIC policy state requires proprio 'base_quat' or "
            "'floating_base_pose'; a torso IMU is not a pelvis-frame substitute"
        )

    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError(
            "base/IMU quaternion must be a finite wxyz vector of shape (4,)"
        )
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        raise ValueError("base/IMU quaternion has zero norm")
    w, x, y, z = quat / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return (rotation.T @ np.asarray((0.0, 0.0, -1.0))).astype(np.float32)


def build_decoupled_policy_state(
    observation: Mapping[str, Any], last_base_height: float
) -> np.ndarray:
    """Build the existing 32-D state used by decoupled ACT/pi/Psi/WAM runs."""

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


def build_sonic_policy_state(
    observation: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Build the official G1 SONIC state groups plus projected gravity."""

    qpos = _joint_qpos(observation)
    return {
        "left_leg": qpos[0:6][None],
        "right_leg": qpos[6:12][None],
        "waist": qpos[12:15][None],
        "left_arm": qpos[15:22][None],
        "right_arm": qpos[22:29][None],
        "left_hand": qpos[29:36][None],
        "right_hand": qpos[36:43][None],
        "projected_gravity": _projected_gravity(kwargs)[None],
        # A flat alias makes custom ACT/Psi/WAM servers easy to configure while
        # retaining the named fields expected by Isaac-GR00T-style adapters.
        "states": qpos[None],
    }


def _normalize_prediction(raw: object, codec: type[ActionCodec]) -> np.ndarray:
    """Normalize flat or named server output into a strict ``(T, D)`` chunk."""

    if isinstance(raw, Mapping):
        flat_key = next((key for key in ("actions", "action") if key in raw), None)
        if flat_key is not None:
            raw = raw[flat_key]
        elif codec.schema_id == SONIC_LATENT_SCHEMA_ID:

            def field(name: str) -> object:
                if name in raw:
                    return raw[name]
                prefixed = f"action.{name}"
                if prefixed in raw:
                    return raw[prefixed]
                raise KeyError(
                    f"Unified policy response is missing {name!r}; "
                    f"available keys: {sorted(raw)}"
                )

            motion = np.asarray(field("motion_token"), dtype=np.float32)
            left = np.asarray(field("left_hand_joints"), dtype=np.float32)
            right = np.asarray(field("right_hand_joints"), dtype=np.float32)
            while motion.ndim > 2 and motion.shape[0] == 1:
                motion = motion[0]
            while left.ndim > 2 and left.shape[0] == 1:
                left = left[0]
            while right.ndim > 2 and right.shape[0] == 1:
                right = right[0]
            if motion.ndim == 1:
                motion = motion[None]
            if left.ndim == 1:
                left = left[None]
            if right.ndim == 1:
                right = right[None]
            if not (motion.shape[0] == left.shape[0] == right.shape[0]):
                raise ValueError(
                    "Unified response horizons differ: "
                    f"motion={motion.shape}, left={left.shape}, right={right.shape}"
                )
            raw = np.concatenate((motion, left, right), axis=-1)
        else:
            raise KeyError(
                "Decoupled policy response must contain 'actions' or 'action'; "
                f"available keys: {sorted(raw)}"
            )

    action = np.asarray(raw, dtype=np.float32)
    while action.ndim > 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim == 1:
        action = action[None]
    if action.ndim != 2 or action.shape[1] != codec.dimension:
        raise ValueError(
            f"Policy response for {codec.schema_id!r} must have shape "
            f"(T, {codec.dimension}), got {action.shape}. Check that the "
            "checkpoint and controller use the same action schema."
        )
    return action


class RemoteChunkPolicy:
    """Shared codec, history, and chunk behavior for remote model transports."""

    def __init__(
        self,
        action_schema: str,
        *,
        policy_name: str,
        upsample_factor: int = 1,
    ) -> None:
        if upsample_factor <= 0:
            raise ValueError("upsample_factor must be > 0")
        self.codec = action_codec_for_schema(action_schema)
        self.policy_name = policy_name
        self.upsample_factor = int(upsample_factor)
        self._reset_history = True
        self._last_base_height = 0.74

    @property
    def action_schema(self) -> str:
        return self.codec.schema_id

    def _chunk(self, raw: object, *, transport: str) -> ActionChunk[Any]:
        flat = _normalize_prediction(raw, self.codec)
        goals = list(self.codec.decode_chunk(flat))
        if self.upsample_factor > 1:
            goals = [goal for goal in goals for _ in range(self.upsample_factor)]
        if goals and isinstance(goals[-1], DecoupledGoal):
            self._last_base_height = goals[-1].base_height_command
        return ActionChunk(
            schema=self.codec.schema_id,
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

    def invalidate_instruction_history(self) -> None:
        """Reset model history while preserving the last physical command."""

        self._reset_history = True

    def _state(
        self, observation: Mapping[str, Any], kwargs: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        if self.codec.schema_id == DECOUPLED_SCHEMA_ID:
            return {
                "states": build_decoupled_policy_state(
                    observation, self._last_base_height
                )[None]
            }
        return build_sonic_policy_state(observation, kwargs)


class HttpTaskPolicy(RemoteChunkPolicy):
    """Adapter for THETA-Bench's NumPy-over-HTTP ``/act`` protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        action_schema: str,
        *,
        policy_name: str,
        client: Any | None = None,
        image_key: str = "rgb_head_stereo_left",
        dataset: str = "simple",
        instruction_transform: Callable[[str], str] | None = None,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__(
            action_schema,
            policy_name=policy_name,
            upsample_factor=upsample_factor,
        )
        if client is None:
            from theta_bench.policies.http_client import HttpActionClient

            client = HttpActionClient(host, port)
        self.client = client
        self.image_key = image_key
        self.dataset = dataset
        self.instruction_transform = instruction_transform

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionChunk[Any]:
        prompt = instruction
        if prompt is not None and self.instruction_transform is not None:
            prompt = self.instruction_transform(prompt)
        prompt = prompt or "bend to pick up the object"
        history = {"reset": True} if self._reset_history else {}
        self._reset_history = False
        response = self.client.query_action(
            {self.image_key: _head_image(observation)},
            prompt,
            self._state(observation, kwargs),
            {},
            history=history,
            dataset=self.dataset,
        )
        raw = response[0] if isinstance(response, tuple) else response
        return self._chunk(raw, transport="http_act")


class OpenPiTaskPolicy(RemoteChunkPolicy):
    """Adapter for an OpenPI websocket policy server."""

    def __init__(
        self,
        host: str,
        port: int,
        action_schema: str,
        *,
        client: Any | None = None,
        policy_name: str = "pi05",
        upsample_factor: int = 1,
    ) -> None:
        super().__init__(
            action_schema,
            policy_name=policy_name,
            upsample_factor=upsample_factor,
        )
        if client is None:
            from openpi_client import websocket_client_policy

            client = websocket_client_policy.WebsocketClientPolicy(
                host=host, port=port, api_key=None
            )
        self.client = client

    def invalidate_instruction_history(self) -> None:
        """Use the next prompt without resetting server torso/base-height state."""

        # The pinned server processes each prompt afresh. Its reset flag also
        # resets physical command state, so it is reserved for episode resets.

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionChunk[Any]:
        state = self._state(observation, kwargs)
        if self.codec.schema_id == DECOUPLED_SCHEMA_ID:
            # Hands and arms only: Psi0's openpi server appends the torso RPY
            # and base height it tracks from its previous action chunk.
            policy_state: object = state["states"][0, :28]
        else:
            policy_state = state
        request = {
            "observation/image": _head_image(observation),
            "states": policy_state,
            "prompt": instruction or "As a smart robot agent, what to do next?",
            "reset": self._reset_history,
        }
        if self.codec.schema_id == DECOUPLED_SCHEMA_ID:
            # Preserve measured torso joints across the pinned server's append.
            request["theta/full_state"] = state["states"][0].copy()
        self._reset_history = False
        raw = self.client.infer(request)
        return self._chunk(raw, transport="openpi_websocket")


def _frame_to_png_b64(frame: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class CosmosTaskPolicy(RemoteChunkPolicy):
    """Adapter for the Cosmos world-action-model ``/predict`` protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        action_schema: str,
        *,
        client: Any | None = None,
        policy_name: str = "cosmos3",
        domain_name: str = "g1_simple",
        image_size: int = 256,
        timeout: int = 1200,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__(
            action_schema,
            policy_name=policy_name,
            upsample_factor=upsample_factor,
        )
        self.client = client
        self.server = f"http://{host}:{port}"
        self.domain_name = domain_name
        self.image_size = int(image_size)
        self.timeout = int(timeout)
        # Raw pixels reduce local CPU work; keep compression on remote links.
        self._numpy_images: bool | None = (
            None if host in ("127.0.0.1", "localhost") else False
        )

    def _encode_image(self, image: np.ndarray) -> object:
        import requests

        if self._numpy_images is None:
            self._numpy_images = False
            try:
                response = requests.get(self.server + "/info", timeout=5)
                response.raise_for_status()
                info = response.json()
                encodings = (
                    info.get("image_encodings") if isinstance(info, dict) else None
                )
                self._numpy_images = (
                    isinstance(encodings, list) and "numpy" in encodings
                )
            except (requests.RequestException, ValueError):
                pass  # Older Cosmos servers only accept PNG observations.
        if self._numpy_images:
            from theta_bench.policies.http_client import _encode

            return _encode(np.asarray(image, dtype=np.uint8))
        return _frame_to_png_b64(image)

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionChunk[Any]:
        state = self._state(observation, kwargs)
        flat_state = state["states"].reshape(-1)
        history = {"reset": True} if self._reset_history else {}
        self._reset_history = False
        if self.client is not None:
            raw = self.client.predict(
                _head_image(observation),
                instruction or "bend to pick up the object",
                flat_state,
                history,
            )
        else:
            import requests

            response = requests.post(
                self.server + "/predict",
                json={
                    "image": self._encode_image(_head_image(observation)),
                    "prompt": instruction or "bend to pick up the object",
                    "domain_name": self.domain_name,
                    "image_size": self.image_size,
                    "view_point": "ego_view",
                    "state": flat_state.tolist(),
                    **history,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json()
        return self._chunk(raw, transport="cosmos_http")


class DreamZeroTaskPolicy(HttpTaskPolicy):
    """History-aware world-action-model adapter with receding horizon."""

    def __init__(
        self,
        host: str,
        port: int,
        action_schema: str,
        *,
        client: Any | None = None,
        action_horizon: int = 24,
        video_frames_per_chunk: int = 8,
        video_stride: int | None = None,
        wait_for_server: bool | None = None,
        server_wait_timeout: float = 900.0,
        server_poll_interval: float = 5.0,
        upsample_factor: int = 1,
    ) -> None:
        client_was_injected = client is not None
        super().__init__(
            host,
            port,
            action_schema,
            policy_name="dreamzero",
            client=client,
            dataset="dreamzero_g1_simple",
            upsample_factor=upsample_factor,
        )
        self.host = host
        self.port = int(port)
        should_wait = (
            (not client_was_injected) if wait_for_server is None else wait_for_server
        )
        server_config: Mapping[str, Any] = {}
        if should_wait:
            self._wait_for_server(server_wait_timeout, server_poll_interval)
            server_config = self._query_server_config()
        self.server_config = dict(server_config)
        self.action_horizon = self._positive_int(
            "action_horizon", server_config.get("action_horizon", action_horizon)
        )
        self.video_frames_per_chunk = self._positive_int(
            "video_frames_per_chunk",
            server_config.get("video_frames_per_chunk", video_frames_per_chunk),
        )
        configured_stride = server_config.get("video_stride", video_stride)
        self.video_stride = self._positive_int(
            "video_stride",
            (
                configured_stride
                if configured_stride is not None
                else max(self.action_horizon // 8, 1)
            ),
        )
        self._frames: list[np.ndarray] = []
        self._observed_since_prediction = False
        self._session_index = 0
        self._step_index = 0
        self._session_id = self._make_session_id()

    def _make_session_id(self) -> str:
        return f"dreamzero-composed-{os.getpid()}-{self._session_index}"

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        result = int(str(value))
        if result <= 0:
            raise ValueError(f"DreamZero {name} must be > 0, got {result}")
        return result

    def _wait_for_server(self, timeout: float, poll_interval: float) -> None:
        if timeout < 0 or poll_interval <= 0:
            raise ValueError(
                "DreamZero server wait timeout must be >= 0 and interval > 0"
            )
        import requests

        url = f"http://{self.host}:{self.port}/health"
        started = time.monotonic()
        while True:
            try:
                if requests.get(url, timeout=3).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.monotonic() - started >= timeout:
                return
            time.sleep(poll_interval)

    def _query_server_config(self) -> dict[str, Any]:
        import requests

        try:
            response = requests.get(f"http://{self.host}:{self.port}/config", timeout=5)
            if response.status_code == 200:
                value = response.json()
                if not isinstance(value, Mapping):
                    raise TypeError("DreamZero /config response must be a mapping")
                return dict(value)
        except requests.RequestException:
            pass
        return {}

    def observe(self, observation: dict[str, Any], **kwargs: Any) -> None:
        """Buffer every executed control-step frame for the next WAM query."""

        del kwargs
        self._frames.append(_head_image(observation))
        self._observed_since_prediction = True

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionChunk[Any]:
        # Direct TaskPolicy callers do not go through HumanoidPolicyAgent's
        # optional observe hook, so retain a safe one-frame fallback.
        if not self._observed_since_prediction:
            self.observe(observation, **kwargs)
        if len(self._frames) <= 1:
            frames = list(self._frames)
        else:
            indices = [
                max(0, len(self._frames) - 1 - i * self.video_stride)
                for i in reversed(range(self.video_frames_per_chunk))
            ]
            frames = [self._frames[index] for index in indices]
        video = np.stack(frames, axis=0)
        self._frames = self._frames[-1:]
        self._observed_since_prediction = False
        info = kwargs.get("info")
        episode_index = (
            int(info.get("episode_index", -1)) if isinstance(info, Mapping) else -1
        )
        history = {
            "session_id": self._session_id,
            "episode_index": episode_index,
            "step_index": self._step_index,
        }
        if self._reset_history:
            history["reset"] = True
            self._reset_history = False
        response = self.client.query_action(
            {self.image_key: video},
            instruction or "perform the task",
            self._state(observation, kwargs),
            {},
            history=history,
            dataset=self.dataset,
        )
        raw = response[0] if isinstance(response, tuple) else response
        flat = _normalize_prediction(raw, self.codec)[: self.action_horizon]
        # The history index is expressed in simulator control steps.  Each raw
        # prediction is repeated ``upsample_factor`` times before the next
        # request, matching the legacy DreamZero agent's per-action counter.
        self._step_index += int(flat.shape[0]) * self.upsample_factor
        return self._chunk(flat, transport="dreamzero_http")

    def invalidate_instruction_history(self) -> None:
        """Start a fresh model session from the current image and physical state."""

        super().invalidate_instruction_history()
        # The next composed query sends one fresh frame, which also resets the
        # pinned action head's KV caches on every distributed rank.
        self._frames = []
        self._observed_since_prediction = False
        self._session_index += 1
        self._step_index = 0
        self._session_id = self._make_session_id()

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        try:
            import requests

            requests.post(f"http://{self.host}:{self.port}/flush", timeout=30)
        except Exception:
            # Flush is an optional server extension and must not prevent an
            # episode reset when the server is already gone.
            pass
        super().reset()
        self.invalidate_instruction_history()


def normalize_policy_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    for suffix in ("_decoupled_wbc", "_gear_sonic", "_sonic"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    aliases = {"pi": "pi05", "psi": "psi0", "cosmos": "cosmos3"}
    return aliases.get(normalized, normalized)


def make_remote_task_policy(
    name: str,
    *,
    action_schema: str,
    host: str,
    port: int,
    client: Any | None = None,
    upsample_factor: int = 1,
    **kwargs: Any,
) -> RemoteChunkPolicy:
    """Construct a task-policy transport independently of the controller."""

    normalized = normalize_policy_name(name)
    if normalized not in SUPPORTED_TASK_POLICIES:
        raise ValueError(
            f"Unsupported humanoid task policy {name!r}; choose one of "
            f"{', '.join(SUPPORTED_TASK_POLICIES)}"
        )
    if normalized == "groot" and action_schema != DECOUPLED_SCHEMA_ID:
        raise ValueError("THETA GR00T N1.7 requires the decoupled_v1 action schema")
    if normalized == "pi05":
        return OpenPiTaskPolicy(
            host,
            port,
            action_schema,
            client=client,
            upsample_factor=upsample_factor,
            **kwargs,
        )
    if normalized == "cosmos3":
        return CosmosTaskPolicy(
            host,
            port,
            action_schema,
            client=client,
            upsample_factor=upsample_factor,
            **kwargs,
        )
    if normalized == "dreamzero":
        return DreamZeroTaskPolicy(
            host,
            port,
            action_schema,
            client=client,
            upsample_factor=upsample_factor,
            **kwargs,
        )

    default_image_key = (
        "video.rs_view" if normalized == "psix" else "rgb_head_stereo_left"
    )
    default_transform = (
        (lambda prompt: f"Task: {prompt}") if normalized == "psix" else None
    )
    options = dict(kwargs)
    options.setdefault("image_key", default_image_key)
    options.setdefault("dataset", "simple")
    options.setdefault("instruction_transform", default_transform)
    options.setdefault("policy_name", normalized)
    return HttpTaskPolicy(
        host,
        port,
        action_schema,
        client=client,
        upsample_factor=upsample_factor,
        **options,
    )
