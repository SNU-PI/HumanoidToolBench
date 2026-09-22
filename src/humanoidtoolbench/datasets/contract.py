"""Task-agnostic recording contract for HumanoidToolBench's LeRobot datasets.

The recorders in this repository know how to move bytes; they must not know
what a task *means*. This module is the boundary between the two: it defines

* the ``info`` contract a task fills in during ``step()``,
* the dataset features that carry the outcome of every frame,
* how a task declares its own metrics so that adding a task never requires
  touching a recorder,
* the episode-level metadata that says whether an episode succeeded and why it
  ended.

Nothing here computes success, failure or any metric. Everything is taken from
what the task reported.

Frame convention (unchanged from the existing datasets)::

    observation.*   state actually measured at s_t
    action, teleop.*  what was commanded at s_t
    next.*, metric.*  the outcome of applying that command, i.e. state s_t+1

Conventions recorded in the metadata: poses are ``[x, y, z, qw, qx, qy, qz]``
with scalar-first quaternions, joint values are radians in the robot's own
joint order, and lengths are metres.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CONTRACT_VERSION = "simple-lerobot-contract-v1"

# --- the keys a task may put into the gym `info` dict -----------------------
SUCCESS_KEY = "success"
FAILURE_KEY = "failure"
STAGE_KEY = "stage"
METRICS_KEY = "metrics"

#: Outcome of one frame. Declared for every dataset, written on every frame.
OUTCOME_FEATURES: dict[str, dict[str, Any]] = {
    "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
    "next.success": {"dtype": "bool", "shape": (1,), "names": ["success"]},
    "next.failure": {"dtype": "bool", "shape": (1,), "names": ["failure"]},
    "next.terminated": {"dtype": "bool", "shape": (1,), "names": ["terminated"]},
    "next.truncated": {"dtype": "bool", "shape": (1,), "names": ["truncated"]},
}

_STAGE_FEATURE = {"dtype": "int64", "shape": (1,), "names": ["stage"]}
_DEFAULT_METRIC_FEATURE = {"dtype": "float32", "shape": (1,)}


# ---------------------------------------------------------------------------
# metric declaration
# ---------------------------------------------------------------------------
def normalize_metric_spec(
    spec: Mapping[str, Any] | Sequence[str] | None
) -> dict[str, dict[str, Any]]:
    """Turn a task's metric declaration into feature definitions.

    Accepted forms::

        ["tip_to_head_distance", "axis_angle_error"]        # all float32 scalars
        {"peg_depth": "float32", "stage": "int64"}
        {"coverage": {"dtype": "float32", "shape": (1,)}}

    ``stage`` is always an int64 scalar. Returns ``{metric_name: feature}``
    keyed by the *bare* metric name, not by the dataset field name.
    """

    if not spec:
        return {}

    if isinstance(spec, Mapping):
        items: Iterable[tuple[str, Any]] = spec.items()
    else:
        items = ((str(name), None) for name in spec)

    normalized: dict[str, dict[str, Any]] = {}
    for name, declaration in items:
        if name == STAGE_KEY:
            normalized[name] = dict(_STAGE_FEATURE)
            continue
        if declaration is None:
            feature = dict(_DEFAULT_METRIC_FEATURE)
        elif isinstance(declaration, str):
            feature = {"dtype": declaration, "shape": (1,)}
        elif isinstance(declaration, Mapping):
            feature = {
                "dtype": declaration.get("dtype", "float32"),
                "shape": tuple(declaration.get("shape", (1,))),
            }
        else:
            raise TypeError(f"metric '{name}': unsupported declaration {declaration!r}")
        feature.setdefault("names", [name])
        normalized[name] = feature
    return normalized


def resolve_metric_spec(task: Any) -> dict[str, dict[str, Any]]:
    """Read the metric declaration off a task, if it has one.

    A task opts in by implementing ``metric_spec()``. Tasks that do not are
    recorded without any ``metric.*`` field -- absent, rather than filled with
    placeholder values.
    """

    declaration = getattr(task, "metric_spec", None)
    if declaration is None:
        return {}
    return normalize_metric_spec(
        declaration() if callable(declaration) else declaration
    )


def metric_features(
    metric_spec: Mapping[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Dataset features for a normalized metric spec, prefixed with ``metric.``."""

    return {f"metric.{name}": dict(feature) for name, feature in metric_spec.items()}


def outcome_features() -> dict[str, dict[str, Any]]:
    return {name: dict(feature) for name, feature in OUTCOME_FEATURES.items()}


def contract_features(
    metric_spec: Mapping[str, dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    """Every feature this contract adds to a dataset."""

    features = outcome_features()
    features.update(metric_features(metric_spec or {}))
    return features


# ---------------------------------------------------------------------------
# per-frame values
# ---------------------------------------------------------------------------
def success_source(info: Mapping[str, Any] | None) -> str:
    """Where ``success`` came from, so the dataset can be audited later."""

    if info is not None and SUCCESS_KEY in info:
        return "task_info"
    return "terminated_fallback"


def resolve_success(info: Mapping[str, Any] | None, terminated: bool) -> bool:
    """Success as reported by the task.

    Never inferred from the reward. Tasks that do not report it fall back to
    ``terminated``, which is what HumanoidToolBench's environments already mean by it
    (``terminated = task.check_success(...)``); the fallback is recorded in the
    dataset metadata under ``success_source``.
    """

    if info is not None and SUCCESS_KEY in info:
        return bool(info[SUCCESS_KEY])
    return bool(terminated)


def resolve_failure(info: Mapping[str, Any] | None) -> bool:
    if info is None:
        return False
    return bool(info.get(FAILURE_KEY, False))


def outcome_frame(
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """The ``next.*`` values for one frame."""

    return {
        "next.reward": np.array([float(reward)], dtype=np.float32),
        "next.success": np.array([resolve_success(info, terminated)], dtype=bool),
        "next.failure": np.array([resolve_failure(info)], dtype=bool),
        "next.terminated": np.array([bool(terminated)], dtype=bool),
        "next.truncated": np.array([bool(truncated)], dtype=bool),
    }


def metric_frame(
    info: Mapping[str, Any] | None,
    metric_spec: Mapping[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """The ``metric.*`` values for one frame, taken verbatim from the task.

    Every declared metric must be present on every frame: a dataset with holes
    in it is worse than one that fails loudly while it is being recorded.
    """

    if not metric_spec:
        return {}

    reported: Mapping[str, Any] = {}
    if info is not None:
        reported = dict(info.get(METRICS_KEY, {}) or {})
        if STAGE_KEY in info and STAGE_KEY not in reported:
            reported[STAGE_KEY] = info[STAGE_KEY]

    missing = [name for name in metric_spec if name not in reported]
    if missing:
        raise KeyError(
            f"the task declared metrics {sorted(metric_spec)} but did not report {missing} "
            f"in info['{METRICS_KEY}']"
        )

    frame: dict[str, np.ndarray] = {}
    for name, feature in metric_spec.items():
        value = np.asarray(reported[name], dtype=np.dtype(feature["dtype"]))
        frame[f"metric.{name}"] = value.reshape(tuple(feature["shape"]))
    return frame


def contract_frame(
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any] | None,
    metric_spec: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    frame = outcome_frame(reward, terminated, truncated, info)
    frame.update(metric_frame(info, metric_spec or {}))
    return frame


# ---------------------------------------------------------------------------
# temporal alignment
# ---------------------------------------------------------------------------
class TransitionAligner:
    """Holds ``(observation_t, info_t)`` until the step that consumed it returns.

    A gym wrapper only sees an observation *after* ``env.step()`` has already
    produced the next one. Writing that observation into the same frame as the
    action pairs ``a_t`` with ``s_t+1``: the recorded policy input is the
    result of the action it is supposed to predict. This keeps the state the
    action was actually chosen from, so every frame is

        observation_t, action_t, outcome_t+1

    which is what ``next.*`` and ``metric.*`` already mean.
    """

    def __init__(self) -> None:
        self._observation: Any = None
        self._info: Any = None
        self._started = False

    def reset(self, observation: Any, info: Any = None) -> None:
        self._observation = observation
        self._info = info
        self._started = True

    def current(self) -> tuple[Any, Any]:
        if not self._started:
            raise RuntimeError("reset() must run before a transition can be aligned")
        return self._observation, self._info

    def advance(self, observation: Any, info: Any = None) -> None:
        self._observation = observation
        self._info = info


# ---------------------------------------------------------------------------
# episode level
# ---------------------------------------------------------------------------
def termination_reason(
    *, success: bool, failure: bool, terminated: bool, truncated: bool
) -> str:
    if success:
        return "success"
    if failure:
        return "failure"
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    return "unfinished"


class EpisodeOutcome:
    """Accumulates the episode-level view of what the task reported."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.length = 0
        self.episode_return = 0.0
        self.success = False
        self.failure = False
        self.terminated = False
        self.truncated = False
        self.last_stage: int | None = None

    def update(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        self.length += 1
        self.episode_return += float(reward)
        self.success = self.success or resolve_success(info, terminated)
        self.failure = self.failure or resolve_failure(info)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        if info is not None and STAGE_KEY in info:
            self.last_stage = int(info[STAGE_KEY])

    def as_metadata(self) -> dict[str, Any]:
        metadata = {
            "episode_success": bool(self.success),
            "episode_failure": bool(self.failure),
            "episode_return": float(self.episode_return),
            "episode_length": int(self.length),
            "termination_reason": termination_reason(
                success=self.success,
                failure=self.failure,
                terminated=self.terminated,
                truncated=self.truncated,
            ),
        }
        if self.last_stage is not None:
            metadata["final_stage"] = self.last_stage
        return metadata


def episode_metadata(
    outcome: EpisodeOutcome,
    *,
    task_uid: str | None = None,
    task_spec_version: Any = None,
    controller_name: str | None = None,
    controller_version: str | None = None,
    robot_name: str | None = None,
    simulator: str | None = None,
    seed: Any = None,
    environment_config: Any = None,
    success_source: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Episode-level metadata, on top of the ``environment_config`` HumanoidToolBench already writes."""

    metadata: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "task_uid": task_uid,
        "task_spec_version": task_spec_version,
        "controller_name": controller_name,
        "controller_version": controller_version,
        "robot_name": robot_name,
        "simulator": simulator,
        "seed": seed,
        "success_source": success_source,
        **outcome.as_metadata(),
    }
    if environment_config is not None:
        metadata["environment_config"] = environment_config
    if extra:
        metadata.update(dict(extra))
    return {key: value for key, value in metadata.items() if value is not None}


# ---------------------------------------------------------------------------
# dataset level
# ---------------------------------------------------------------------------
def conventions_metadata(
    *,
    joint_order: Mapping[str, Sequence[str]] | None = None,
    controller_specific_fields: Sequence[str] = (),
    unavailable_fields: Mapping[str, str] | None = None,
    command_fields: Sequence[str] = (),
    frames: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The semantics a consumer needs in order to read the dataset correctly."""

    metadata: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "frame_convention": {
            "observation.*": "state measured at s_t",
            "action, action.*, teleop.*": "command issued at s_t",
            "next.*, metric.*": "outcome of that command, measured at s_t+1",
        },
        "temporal_alignment": (
            "each row is (observation_t, action_t, outcome_t+1); the observation is the "
            "one the action was chosen from, and the terminal observation of an episode "
            "is not recorded"
        ),
        "quaternion_convention": "scalar-first [w, x, y, z]",
        "units": {
            "joint_position": "radian",
            "joint_velocity": "radian/second",
            "torque": "newton_meter",
            "position": "meter",
            "angle": "radian",
            "time": "second",
        },
        "success_definition": (
            "reported by the task through info['success']; never inferred from the reward"
        ),
    }
    if joint_order:
        metadata["joint_order"] = {
            key: list(value) for key, value in joint_order.items()
        }
    if command_fields:
        metadata["command_fields"] = list(command_fields)
    if controller_specific_fields:
        metadata["controller_specific_fields"] = {
            "note": (
                "internal state of the whole-body controller, not a generic robot "
                "observation; kept under its historical name for schema compatibility"
            ),
            "fields": list(controller_specific_fields),
        }
    if unavailable_fields:
        metadata["unavailable_fields"] = dict(unavailable_fields)
    return metadata
