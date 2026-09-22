"""GEAR-SONIC controller backend for the 64-D latent action contract.

The public SONIC VLA contract is *not* a direct ``64 -> 29`` joint decoder.
The deployment decoder consumes a motion token together with a temporal window
of robot proprioception and previous policy actions.  This module mirrors that
contract for the official decoder ONNX export and deliberately has no
joint-angle or zero-action fallback.

The implementation follows the observation order and action post-processing in
NVlabs/GR00T-WholeBodyControl (``gear_sonic_deploy``).  A caller may also inject
another real runtime implementing :class:`GearSonicRuntime`, for example an
adapter around the official C++/TensorRT deployment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from theta_bench.actions import SONIC_LATENT_SCHEMA_ID, SonicLatentGoal
from theta_bench.core.action import ActionCmd

Float32Vector = NDArray[np.float32]
G1_BODY_DOF = 29
SONIC_LATENT_ABS_LIMIT = 1.25


class GearSonicError(RuntimeError):
    """Base error for the GEAR-SONIC backend."""


class GearSonicRuntimeUnavailable(GearSonicError):
    """Raised when no real SONIC decoder runtime can be constructed."""


class GearSonicInferenceError(GearSonicError):
    """Raised when a configured SONIC runtime cannot produce a valid command."""


def _vector(
    name: str,
    value: object,
    size: int,
    *,
    nonnegative: bool = False,
) -> Float32Vector:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric vector of shape ({size},)") from exc
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    result = array.copy()
    result.flags.writeable = False
    return result


def _load_initial_motion_token(value: object) -> Float32Vector:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"SONIC initial motion token file does not exist: {path}"
            )
        try:
            if path.suffix == ".npy":
                value = np.load(path, allow_pickle=False)
            elif path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix in {".txt", ".csv"}:
                value = np.loadtxt(
                    path, delimiter="," if path.suffix == ".csv" else None
                )
            else:
                raise ValueError(
                    "SONIC initial motion token path must use .npy, .json, .txt, or .csv"
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"failed to load SONIC initial motion token from {path}: {exc}"
            ) from exc
    token = _vector("initial_motion_token", value, 64)
    peak = float(np.max(np.abs(token)))
    if peak > SONIC_LATENT_ABS_LIMIT:
        raise ValueError(
            "initial_motion_token exceeds the official SONIC action bound: "
            f"max abs {peak:.6g} > {SONIC_LATENT_ABS_LIMIT}"
        )
    return token


def _hand_mjcf_to_natural(value: object, *, name: str) -> Float32Vector:
    """Convert G1Sonic proprio hand order to the public Dex3 goal order."""

    hand = _vector(name, value, 7)
    # MuJoCo: thumb/middle/index; public action contract: thumb/index/middle.
    return _vector(name, hand[[0, 1, 2, 5, 6, 3, 4]], 7)


@dataclass(frozen=True, eq=False)
class GearSonicMotorTargets:
    """One decoded 29-DoF G1 motor command in MuJoCo/hardware joint order.

    ``target_dq`` and ``tau_ff`` default to zero, matching the official SONIC
    deployment.  ``kp`` and ``kd`` may be omitted by an injected runtime so the
    robot adapter can use its checkpoint-specific configured gains.
    """

    target_q: Float32Vector
    target_dq: Float32Vector | None = None
    tau_ff: Float32Vector | None = None
    kp: Float32Vector | None = None
    kd: Float32Vector | None = None

    body_dof: ClassVar[int] = G1_BODY_DOF

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_q", _vector("target_q", self.target_q, G1_BODY_DOF)
        )
        object.__setattr__(
            self,
            "target_dq",
            _vector(
                "target_dq",
                (
                    np.zeros(G1_BODY_DOF, dtype=np.float32)
                    if self.target_dq is None
                    else self.target_dq
                ),
                G1_BODY_DOF,
            ),
        )
        object.__setattr__(
            self,
            "tau_ff",
            _vector(
                "tau_ff",
                (
                    np.zeros(G1_BODY_DOF, dtype=np.float32)
                    if self.tau_ff is None
                    else self.tau_ff
                ),
                G1_BODY_DOF,
            ),
        )
        if self.kp is not None:
            object.__setattr__(
                self, "kp", _vector("kp", self.kp, G1_BODY_DOF, nonnegative=True)
            )
        if self.kd is not None:
            object.__setattr__(
                self, "kd", _vector("kd", self.kd, G1_BODY_DOF, nonnegative=True)
            )


@runtime_checkable
class GearSonicRuntime(Protocol):
    """Runtime seam for a real SONIC decoder implementation."""

    def decode(
        self,
        motion_latent: Float32Vector,
        observation: Mapping[str, Any],
        **kwargs: Any,
    ) -> GearSonicMotorTargets:
        """Decode one latent using current robot proprioception."""

    def reset(self, **kwargs: Any) -> None:
        """Clear temporal state at an episode boundary."""


@dataclass(frozen=True)
class GearSonicAvailability:
    """Result of probing for an in-process official decoder runtime."""

    available: bool
    reason: str
    model_path: Path | None = None
    observation_config_path: Path | None = None


# Official gear_sonic_deploy policy_parameters.hpp constants.  The model emits
# IsaacLab-ordered residual actions; the simulator consumes MuJoCo/hardware
# order targets.
_ISAACLAB_TO_MUJOCO = np.asarray(
    (
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ),
    dtype=np.intp,
)
_MUJOCO_TO_ISAACLAB = np.asarray(
    (
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ),
    dtype=np.intp,
)
_DEFAULT_ANGLES = np.asarray(
    (
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float32,
)

_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTOR_TYPES = (
    "7520_22",
    "7520_22",
    "7520_14",
    "7520_22",
    "5020",
    "5020",
    "7520_22",
    "7520_22",
    "7520_14",
    "7520_22",
    "5020",
    "5020",
    "7520_14",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "4010",
    "4010",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "4010",
    "4010",
)
_STIFFNESS = np.asarray(
    [_ARMATURE[kind] * _NATURAL_FREQ**2 for kind in _MOTOR_TYPES], dtype=np.float32
)
_DAMPING = np.asarray(
    [2.0 * _DAMPING_RATIO * _ARMATURE[kind] * _NATURAL_FREQ for kind in _MOTOR_TYPES],
    dtype=np.float32,
)
_GAIN_MULTIPLIER = np.asarray(
    [1.0] * 4 + [2.0, 2.0] + [1.0] * 4 + [2.0, 2.0, 1.0, 2.0, 2.0] + [1.0] * 14,
    dtype=np.float32,
)
_KP = _STIFFNESS * _GAIN_MULTIPLIER
_KD = _DAMPING * _GAIN_MULTIPLIER
_ACTION_SCALE = np.asarray(
    [
        0.25 * _EFFORT[kind] / _STIFFNESS[index]
        for index, kind in enumerate(_MOTOR_TYPES)
    ],
    dtype=np.float32,
)
for _constant in (
    _ISAACLAB_TO_MUJOCO,
    _MUJOCO_TO_ISAACLAB,
    _DEFAULT_ANGLES,
    _KP,
    _KD,
    _ACTION_SCALE,
):
    _constant.flags.writeable = False


_OFFICIAL_OBSERVATIONS = (
    "token_state",
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
)
_TOKEN_DIM = 64
_PER_HISTORY_FRAME_DIM = 3 + G1_BODY_DOF * 3 + 3


def _active_observations(config_path: Path) -> tuple[str, ...]:
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GearSonicRuntimeUnavailable(
            f"cannot read SONIC observation config {config_path}: {exc}"
        ) from exc
    observation_section = re.split(
        r"^encoder\s*:\s*$", text, maxsplit=1, flags=re.MULTILINE
    )[0]
    name_matches = tuple(
        re.finditer(
            r'^\s*-\s*name\s*:\s*["\']?([^"\'\s#]+)',
            observation_section,
            flags=re.MULTILINE,
        )
    )
    enabled_names: list[str] = []
    for index, match in enumerate(name_matches):
        end = (
            name_matches[index + 1].start()
            if index + 1 < len(name_matches)
            else len(observation_section)
        )
        block = observation_section[match.end() : end]
        enabled_match = re.search(
            r"^\s*enabled\s*:\s*(true|false)\s*(?:#.*)?$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if enabled_match is None or enabled_match.group(1).lower() == "true":
            enabled_names.append(match.group(1))
    return tuple(enabled_names)


def _unwrap_proprio(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    if "body_q" in observation:
        return observation
    proprio = observation.get("proprio")
    if isinstance(proprio, Mapping):
        return proprio
    raise KeyError(
        "observation must contain body_q directly or under observation['proprio']"
    )


def _resolve_controller_observation(
    observation: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Attach eval-loop ``info['proprio']`` without copying image payloads."""

    try:
        _unwrap_proprio(observation)
    except KeyError:
        pass
    else:
        return observation

    info = kwargs.get("info")
    if isinstance(info, Mapping) and isinstance(info.get("proprio"), Mapping):
        resolved = dict(observation)
        resolved["proprio"] = info["proprio"]
        return resolved
    raise KeyError(
        "GEAR-SONIC requires proprioception: provide body_q/body_dq directly, "
        "observation['proprio'], or info['proprio']"
    )


def _runtime_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return only explicitly namespaced kwargs intended for the decoder."""

    value = kwargs.get("runtime_kwargs", {})
    if not isinstance(value, Mapping):
        raise TypeError("runtime_kwargs must be a mapping")
    return dict(value)


def _projected_gravity(base_quat_wxyz: Float32Vector) -> Float32Vector:
    quat = np.asarray(base_quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("base quaternion must have a finite, non-zero norm")
    quat = quat / norm
    # Rotate world gravity by the inverse base orientation.  Quaternions are
    # wxyz in both THETA-Bench and the official SONIC deployment.
    w = quat[0]
    xyz = -quat[1:]
    gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    cross = np.cross(xyz, gravity)
    rotated = gravity + 2.0 * (w * cross + np.cross(xyz, cross))
    return _vector("projected_gravity", rotated, 3)


@dataclass(frozen=True)
class _HistoryEntry:
    base_ang_vel: Float32Vector
    body_q: Float32Vector
    body_dq: Float32Vector
    last_action: Float32Vector
    gravity: Float32Vector


class GearSonicOnnxRuntime:
    """In-process runtime for an official ``model_decoder.onnx`` checkpoint.

    The latent is already encoded, so this runtime intentionally loads only the
    deployment decoder.  The matching ``observation_config.yaml`` is mandatory:
    accepting an arbitrary 29-output ONNX model here would silently apply the
    wrong observation layout and is unsafe for a humanoid controller.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        observation_config_path: str | os.PathLike[str],
        *,
        providers: Sequence[str] | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.observation_config_path = (
            Path(observation_config_path).expanduser().resolve()
        )
        if not self.model_path.is_file():
            raise GearSonicRuntimeUnavailable(
                f"SONIC decoder checkpoint does not exist: {self.model_path}"
            )
        if not self.observation_config_path.is_file():
            raise GearSonicRuntimeUnavailable(
                f"SONIC observation config does not exist: {self.observation_config_path}"
            )

        actual_observations = _active_observations(self.observation_config_path)
        if actual_observations != _OFFICIAL_OBSERVATIONS:
            raise GearSonicRuntimeUnavailable(
                "unsupported SONIC decoder observation contract in "
                f"{self.observation_config_path}; expected {_OFFICIAL_OBSERVATIONS}, "
                f"got {actual_observations}"
            )

        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise GearSonicRuntimeUnavailable(
                    "onnxruntime is required for in-process GEAR-SONIC decoding; "
                    "run `uv run --no-project scripts/setup_evaluation.py` "
                    "or inject a real GearSonicRuntime adapter"
                ) from exc
            options: dict[str, Any] = {}
            if providers is not None:
                options["providers"] = list(providers)
            try:
                session = ort.InferenceSession(str(self.model_path), **options)
            except Exception as exc:
                raise GearSonicRuntimeUnavailable(
                    f"failed to load SONIC decoder {self.model_path}: {exc}"
                ) from exc

        self._session = session
        try:
            inputs = list(session.get_inputs())
        except Exception as exc:
            raise GearSonicRuntimeUnavailable(
                f"SONIC runtime does not expose ONNX input metadata: {exc}"
            ) from exc
        if len(inputs) != 1:
            raise GearSonicRuntimeUnavailable(
                f"official SONIC decoder must have one observation input, got {len(inputs)}"
            )
        self._input = inputs[0]
        self._input_shape = tuple(self._input.shape)
        if not self._input_shape or not isinstance(self._input_shape[-1], int):
            raise GearSonicRuntimeUnavailable(
                "SONIC decoder must have a static final observation dimension, got "
                f"{self._input_shape}"
            )
        self._input_dim = int(self._input_shape[-1])
        history_payload = self._input_dim - _TOKEN_DIM
        if history_payload <= 0 or history_payload % _PER_HISTORY_FRAME_DIM:
            raise GearSonicRuntimeUnavailable(
                "SONIC decoder input dimension is incompatible with the official latent/history "
                f"contract: got {self._input_dim}"
            )
        self.history_frames = history_payload // _PER_HISTORY_FRAME_DIM
        self._history: deque[_HistoryEntry] = deque(maxlen=self.history_frames)
        self._last_action = np.zeros(G1_BODY_DOF, dtype=np.float32)
        self._lock = Lock()

    def _entry(self, observation: Mapping[str, Any]) -> _HistoryEntry:
        proprio = _unwrap_proprio(observation)
        body_q_mujoco = _vector("body_q", proprio.get("body_q"), G1_BODY_DOF)
        body_dq_mujoco = _vector("body_dq", proprio.get("body_dq"), G1_BODY_DOF)

        if "base_ang_vel" in proprio:
            base_ang_vel = _vector("base_ang_vel", proprio["base_ang_vel"], 3)
        else:
            floating_base_vel = _vector(
                "floating_base_vel", proprio.get("floating_base_vel"), 6
            )
            base_ang_vel = _vector("base_ang_vel", floating_base_vel[3:6], 3)

        if "base_quat" in proprio:
            base_quat = _vector("base_quat", proprio["base_quat"], 4)
        else:
            floating_base_pose = _vector(
                "floating_base_pose", proprio.get("floating_base_pose"), 7
            )
            # MuJoCo free-joint qpos is xyz + wxyz.  G1Sonic.prepare_obs()
            # forwards qpos[:7] without changing that quaternion convention.
            base_quat = _vector("base_quat", floating_base_pose[3:7], 4)

        body_q_isaac = (
            body_q_mujoco[_MUJOCO_TO_ISAACLAB] - _DEFAULT_ANGLES[_MUJOCO_TO_ISAACLAB]
        )
        body_dq_isaac = body_dq_mujoco[_MUJOCO_TO_ISAACLAB]
        return _HistoryEntry(
            base_ang_vel=base_ang_vel,
            body_q=_vector("body_q_isaaclab", body_q_isaac, G1_BODY_DOF),
            body_dq=_vector("body_dq_isaaclab", body_dq_isaac, G1_BODY_DOF),
            last_action=_vector("last_action", self._last_action, G1_BODY_DOF),
            gravity=_projected_gravity(base_quat),
        )

    def _history_array(self, attribute: str, width: int) -> Float32Vector:
        missing = self.history_frames - len(self._history)
        pieces: list[NDArray[np.float32]] = [
            np.zeros(width, dtype=np.float32) for _ in range(missing)
        ]
        pieces.extend(getattr(entry, attribute) for entry in self._history)
        return np.concatenate(pieces, dtype=np.float32)

    def _decoder_observation(self, motion_latent: Float32Vector) -> Float32Vector:
        result = np.concatenate(
            (
                motion_latent,
                self._history_array("base_ang_vel", 3),
                self._history_array("body_q", G1_BODY_DOF),
                self._history_array("body_dq", G1_BODY_DOF),
                self._history_array("last_action", G1_BODY_DOF),
                self._history_array("gravity", 3),
            ),
            dtype=np.float32,
        )
        return _vector("SONIC decoder observation", result, self._input_dim)

    def decode(
        self,
        motion_latent: Float32Vector,
        observation: Mapping[str, Any],
        **kwargs: Any,
    ) -> GearSonicMotorTargets:
        if kwargs:
            raise TypeError(f"unsupported SONIC ONNX runtime options: {sorted(kwargs)}")
        latent = _vector("motion_latent", motion_latent, _TOKEN_DIM)
        peak_latent = float(np.max(np.abs(latent)))
        if peak_latent > SONIC_LATENT_ABS_LIMIT:
            raise GearSonicInferenceError(
                "motion_latent exceeds the official SONIC action bound: "
                f"max abs {peak_latent:.6g} > {SONIC_LATENT_ABS_LIMIT}"
            )
        with self._lock:
            self._history.append(self._entry(observation))
            decoder_observation = self._decoder_observation(latent)
            rank = len(self._input_shape)
            feed_shape = (1,) * max(0, rank - 1) + (self._input_dim,)
            feed = decoder_observation.reshape(feed_shape)
            try:
                outputs = self._session.run(None, {self._input.name: feed})
            except Exception as exc:
                raise GearSonicInferenceError(
                    f"SONIC ONNX inference failed: {exc}"
                ) from exc
            if not outputs:
                raise GearSonicInferenceError(
                    "SONIC ONNX inference returned no outputs"
                )
            raw_array = np.asarray(outputs[0], dtype=np.float32)
            if raw_array.size != G1_BODY_DOF:
                raise GearSonicInferenceError(
                    "SONIC decoder action must contain exactly 29 values, got "
                    f"shape {raw_array.shape}"
                )
            raw_action = _vector(
                "SONIC decoder action", raw_array.reshape(G1_BODY_DOF), G1_BODY_DOF
            )
            self._last_action = raw_action.copy()

        target_q = _DEFAULT_ANGLES + raw_action[_ISAACLAB_TO_MUJOCO] * _ACTION_SCALE
        return GearSonicMotorTargets(
            target_q=target_q,
            target_dq=np.zeros(G1_BODY_DOF, dtype=np.float32),
            tau_ff=np.zeros(G1_BODY_DOF, dtype=np.float32),
            kp=_KP,
            kd=_KD,
        )

    def reset(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"unsupported SONIC ONNX reset options: {sorted(kwargs)}")
        with self._lock:
            self._history.clear()
            self._last_action = np.zeros(G1_BODY_DOF, dtype=np.float32)


def _candidate_roots(artifact_root: str | os.PathLike[str] | None) -> tuple[Path, ...]:
    if artifact_root is not None:
        return (Path(artifact_root).expanduser().resolve(),)

    roots: list[Path] = []
    configured_root = os.environ.get("THETA_GEAR_SONIC_DIR")
    if configured_root:
        roots.append(Path(configured_root).expanduser().resolve())

    repository_root = Path(__file__).resolve().parents[3]
    roots.extend((repository_root, repository_root / "third_party" / "gear_sonic"))
    spec = importlib.util.find_spec("gear_sonic")
    if spec is not None and spec.origin:
        package_root = Path(spec.origin).resolve().parent
        roots.extend((package_root, package_root.parent))

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def _find_artifact_pair(roots: Sequence[Path]) -> tuple[Path, Path] | None:
    relative_pairs = (
        ("model_decoder.onnx", "observation_config.yaml"),
        (
            "policy/sonic_v1_1/model_decoder.onnx",
            "policy/sonic_v1_1/observation_config.yaml",
        ),
        (
            "policy/low_latency/model_decoder.onnx",
            "policy/low_latency/observation_config.yaml",
        ),
        ("policy/release/model_decoder.onnx", "policy/release/observation_config.yaml"),
        (
            "gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx",
            "gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml",
        ),
        (
            "gear_sonic_deploy/policy/low_latency/model_decoder.onnx",
            "gear_sonic_deploy/policy/low_latency/observation_config.yaml",
        ),
        (
            "gear_sonic_deploy/policy/release/model_decoder.onnx",
            "gear_sonic_deploy/policy/release/observation_config.yaml",
        ),
    )
    for root in roots:
        if root.is_file() and root.suffix == ".onnx":
            config = root.with_name("observation_config.yaml")
            if config.is_file():
                return root, config
            continue
        for model_relative, config_relative in relative_pairs:
            model = root / model_relative
            config = root / config_relative
            if model.is_file() and config.is_file():
                return model, config
    return None


class GearSonicBackend:
    """Turn :class:`SonicLatentGoal` values into executable robot commands."""

    def __init__(
        self,
        runtime: GearSonicRuntime | None = None,
        *,
        artifact_root: str | os.PathLike[str] | None = None,
        providers: Sequence[str] | None = None,
        initial_motion_token: object | None = None,
    ) -> None:
        if runtime is None:
            availability = self.probe(artifact_root)
            if not availability.available:
                raise GearSonicRuntimeUnavailable(availability.reason)
            assert availability.model_path is not None
            assert availability.observation_config_path is not None
            runtime = GearSonicOnnxRuntime(
                availability.model_path,
                availability.observation_config_path,
                providers=providers,
            )
        if not callable(getattr(runtime, "decode", None)) or not callable(
            getattr(runtime, "reset", None)
        ):
            raise TypeError(
                "runtime must implement GearSonicRuntime.decode() and reset()"
            )
        self._runtime = runtime
        self._initial_motion_token = (
            None
            if initial_motion_token is None
            else _load_initial_motion_token(initial_motion_token)
        )

    @classmethod
    def probe(
        cls, artifact_root: str | os.PathLike[str] | None = None
    ) -> GearSonicAvailability:
        roots = _candidate_roots(artifact_root)
        artifacts = _find_artifact_pair(roots)
        if artifacts is None:
            searched = ", ".join(str(path) for path in roots)
            return GearSonicAvailability(
                available=False,
                reason=(
                    "GEAR-SONIC decoder runtime is unavailable: no matching "
                    "model_decoder.onnx + observation_config.yaml pair was found. "
                    f"Searched: {searched}. The vendored gear_sonic fork contains "
                    "simulation/teleoperation utilities but no unified SONIC decoder. "
                    "Download the official deployment artifacts, set "
                    "THETA_GEAR_SONIC_DIR, or inject a real GearSonicRuntime; a 64-D "
                    "motion token cannot be treated as joint angles."
                ),
            )
        model_path, config_path = artifacts
        try:
            actual_observations = _active_observations(config_path)
        except GearSonicRuntimeUnavailable as exc:
            return GearSonicAvailability(
                available=False,
                reason=str(exc),
                model_path=model_path,
                observation_config_path=config_path,
            )
        if actual_observations != _OFFICIAL_OBSERVATIONS:
            return GearSonicAvailability(
                available=False,
                reason=(
                    f"found SONIC artifacts at {model_path.parent}, but the enabled "
                    "decoder observation contract is unsupported: "
                    f"{actual_observations}; expected {_OFFICIAL_OBSERVATIONS}"
                ),
                model_path=model_path,
                observation_config_path=config_path,
            )
        if importlib.util.find_spec("onnxruntime") is None:
            return GearSonicAvailability(
                available=False,
                reason=(
                    f"found SONIC decoder artifacts at {model_path.parent}, but "
                    "onnxruntime is not installed; run "
                    "`uv run --no-project scripts/setup_evaluation.py` "
                    "or inject a real GearSonicRuntime"
                ),
                model_path=model_path,
                observation_config_path=config_path,
            )
        return GearSonicAvailability(
            available=True,
            reason="official SONIC decoder artifacts and onnxruntime are available",
            model_path=model_path,
            observation_config_path=config_path,
        )

    @property
    def action_schema(self) -> str:
        return SONIC_LATENT_SCHEMA_ID

    @property
    def runtime(self) -> GearSonicRuntime:
        return self._runtime

    @property
    def initial_motion_token(self) -> Float32Vector | None:
        return self._initial_motion_token

    @staticmethod
    def _action(
        targets: GearSonicMotorTargets,
        left_hand_q: object,
        right_hand_q: object,
    ) -> ActionCmd:
        if not isinstance(targets, GearSonicMotorTargets):
            raise TypeError(
                "GEAR-SONIC runtime must return GearSonicMotorTargets, got "
                f"{type(targets).__name__}"
            )
        left = _vector("left_hand_q", left_hand_q, 7)
        right = _vector("right_hand_q", right_hand_q, 7)
        return ActionCmd(
            "gear_sonic",
            target_q=targets.target_q,
            target_dq=targets.target_dq,
            tau_ff=targets.tau_ff,
            kp=targets.kp,
            kd=targets.kd,
            left_hand_q=left,
            right_hand_q=right,
        )

    def step(
        self,
        goal: SonicLatentGoal,
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> ActionCmd:
        if not isinstance(goal, SonicLatentGoal):
            raise TypeError(f"goal must be SonicLatentGoal, got {type(goal).__name__}")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        peak_latent = float(np.max(np.abs(goal.motion_latent)))
        if peak_latent > SONIC_LATENT_ABS_LIMIT:
            raise GearSonicInferenceError(
                "motion_latent exceeds the official SONIC action bound: "
                f"max abs {peak_latent:.6g} > {SONIC_LATENT_ABS_LIMIT}"
            )
        try:
            runtime_observation = _resolve_controller_observation(observation, kwargs)
            targets = self._runtime.decode(
                goal.motion_latent,
                runtime_observation,
                **_runtime_kwargs(kwargs),
            )
            return self._action(targets, goal.left_hand_q, goal.right_hand_q)
        except GearSonicError:
            raise
        except Exception as exc:
            raise GearSonicInferenceError(f"GEAR-SONIC decode failed: {exc}") from exc

    def get_stabilize_action(
        self,
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> ActionCmd:
        """Delegate stabilization to a runtime that explicitly supports it.

        An arbitrary latent cannot safely stand in for a checkpoint-specific
        initial-pose token, so the ONNX runtime intentionally does not invent
        one.  C++/TensorRT adapters can expose ``stabilize`` and own that token.
        """

        stabilize = getattr(self._runtime, "stabilize", None)
        if not callable(stabilize) and self._initial_motion_token is None:
            raise GearSonicRuntimeUnavailable(
                "the configured GEAR-SONIC runtime has no stabilize() method; "
                "configure initial_motion_token with the checkpoint-specific 64-D "
                "initial-pose token or provide a runtime stabilize() method; a zero "
                "latent will not be fabricated"
            )
        try:
            runtime_observation = _resolve_controller_observation(observation, kwargs)
            proprio = _unwrap_proprio(runtime_observation)
            left_hand_q = kwargs.get("left_hand_q")
            if left_hand_q is None:
                left_hand_q = _hand_mjcf_to_natural(
                    proprio.get("left_hand_q"), name="left_hand_q"
                )
            right_hand_q = kwargs.get("right_hand_q")
            if right_hand_q is None:
                right_hand_q = _hand_mjcf_to_natural(
                    proprio.get("right_hand_q"), name="right_hand_q"
                )
            runtime_options = _runtime_kwargs(kwargs)
            if callable(stabilize):
                targets = stabilize(runtime_observation, **runtime_options)
            else:
                assert self._initial_motion_token is not None
                targets = self._runtime.decode(
                    self._initial_motion_token,
                    runtime_observation,
                    **runtime_options,
                )
            return self._action(targets, left_hand_q, right_hand_q)
        except GearSonicError:
            raise
        except Exception as exc:
            raise GearSonicInferenceError(
                f"GEAR-SONIC stabilization failed: {exc}"
            ) from exc

    def reset(self, **kwargs: Any) -> None:
        try:
            self._runtime.reset(**_runtime_kwargs(kwargs))
        except GearSonicError:
            raise
        except Exception as exc:
            raise GearSonicInferenceError(f"GEAR-SONIC reset failed: {exc}") from exc


__all__ = [
    "G1_BODY_DOF",
    "SONIC_LATENT_ABS_LIMIT",
    "GearSonicAvailability",
    "GearSonicBackend",
    "GearSonicError",
    "GearSonicInferenceError",
    "GearSonicMotorTargets",
    "GearSonicOnnxRuntime",
    "GearSonicRuntime",
    "GearSonicRuntimeUnavailable",
]
