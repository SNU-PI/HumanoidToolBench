"""Base contract shared by the active tool-use tasks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from gymnasium import spaces

if TYPE_CHECKING:
    from humanoidtoolbench.core.randomizer import RandomizerCfg
    from humanoidtoolbench.dr.manager import DRManager

from humanoidtoolbench.core.layout import Layout
from humanoidtoolbench.core.robot import Robot
from humanoidtoolbench.dr.scene import TabletopSceneDR
from humanoidtoolbench.robots.protocols import HeadCamMountable
from humanoidtoolbench.sensors.config import CameraCfg, SensorCfg
from humanoidtoolbench.tasks.compat import normalize_task_state_uid


def _validate_task_success_criteria(task: Any, value: float, source: str) -> float:
    try:
        valid = math.isfinite(value) and 0.0 <= value <= 1.0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{source} must be a finite value between 0 and 1")
    fixed = getattr(task, "fixed_success_criteria", None)
    if fixed is not None and value != fixed:
        raise ValueError(
            f"{source} must be {fixed} for {task.uid}: "
            "this task uses a fixed success predicate"
        )
    return float(value)


def set_task_success_criteria(task: Any, value: float | None) -> None:
    """Remember an explicit threshold override for this task's future episodes."""
    if value is None:
        return
    value = _validate_task_success_criteria(task, value, "--success-criteria")
    if not hasattr(task, "_configured_success_criteria"):
        task._configured_success_criteria = task.metadata.get("success_criteria")
    task._success_criteria_override = value
    task.success_criteria = value
    task.metadata = {**task.metadata, "success_criteria": value}


class Task(ABC):
    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 30,
        "split": "train",  # train, test
    }

    uid: str
    label: str
    description: str

    robot_cfg: dict[str, Any]

    sensor_cfgs: dict[str, SensorCfg]

    dr_cfgs: dict[str, RandomizerCfg]

    # None permits a reward threshold override. Geometric tasks declare the
    # fixed value that their success predicate and recording metadata use.
    fixed_success_criteria: float | None = None

    # Stable semantic columns used by dataset recorders. Concrete tasks map
    # each slot to the MuJoCo object name for the current episode.
    recording_object_slots: tuple[str, ...] = ()

    def __init__(
        self,
        dr: DRManager,
        split: str | None = None,
        render_hz: int | None = None,
        dr_level: int | None = None,
        physics_dt: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.metadata = dict(type(self).metadata)
        self.metadata.update(
            {
                k: v
                for k, v in {
                    "split": split,
                    "render_hz": render_hz,
                    "dr_level": dr_level,
                    "physics_dt": physics_dt,
                }.items()
                if v is not None
            }
        )
        self.dr = dr
        self.split = split

    @property
    def render_hz(self) -> int:
        return self.metadata.get("render_hz", 30)

    @property
    def robot(self) -> Robot:
        return self._robot

    @robot.setter
    def robot(self, value: Robot) -> None:
        self._robot = value

    @property
    @abstractmethod
    def layout(self) -> Layout: ...

    @property
    @abstractmethod
    def instruction(self) -> str:
        """A natural language instruction describing the task."""
        # return self.description

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space contributed by configured sensors."""
        obs: dict[str, spaces.Space] = {}
        for key, sensor_cfg in self.sensor_cfgs.items():
            sense_obs = sensor_cfg.observation_space
            if isinstance(sense_obs, spaces.Dict):
                for sub_key, sub_space in sense_obs.spaces.items():
                    obs[f"{key}_{sub_key}"] = sub_space
            else:
                obs[key] = sense_obs
        return spaces.Dict(obs)

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space: ...

    # -- recording contract -------------------------------------------------
    def task_info(self, info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Report the outcome of the current step to whoever is recording.

        Recorders must not know what a task means, so a task that wants its
        outcome in the dataset returns it here::

            {"success": bool, "failure": bool, "stage": int,
             "metrics": {"<name>": float, ...}}

        Every key is optional. Whatever is returned is merged into the gym
        ``info`` dict and written verbatim; nothing is inferred from the reward.
        The keys under ``metrics`` must match :meth:`metric_spec`.
        """

        del info, kwargs
        return {}

    def metric_spec(self) -> dict[str, Any] | list[str]:
        """Declare the task-specific metrics that :meth:`task_info` will report.

        Datasets have a fixed schema, so the names have to be known before the
        first frame. Return either ``["name", ...]`` (float32 scalars) or
        ``{"name": "float32", "stage": "int64"}``. Tasks that declare nothing
        are recorded without any ``metric.*`` column instead of with padding.
        """

        return {}

    def recording_object_map(self) -> dict[str, str | None]:
        """Map semantic recording slots to current MuJoCo object names.

        ``None`` means that the slot is intentionally absent. Recorders keep
        the corresponding pose column and mark it absent rather than changing
        their dataset schema between episodes. Recorders ask every frame, so a
        task whose scene gains or loses a body mid-episode says so here.
        """

        return {slot: None for slot in self.recording_object_slots}

    def show_recording_object(self, model, slot: str, visible: bool) -> None:
        """Switch one recording slot's body on or off in a compiled model.

        The inverse of a slot going absent in :meth:`recording_object_map`, for
        a replay that poses a scene from a recording rather than simulating it.
        A task whose slots are all there for the whole episode has nothing to
        do here.
        """

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
        """Build a new randomized layout or replay a recorded DR state."""
        split = self.metadata.get("split", "train")

        if (
            options is not None
            and "state_dict" in options
            and options["state_dict"] is not None
        ):
            state_dict = normalize_task_state_uid(options["state_dict"], self.uid)
            self._restore_success_criteria(state_dict)
            dr_level = options.get("dr_level", None)
            # Seed before loading so any domains omitted by a DR-level replay
            # are redrawn reproducibly.
            self.dr.reset(seed=seed)
            self.dr.load_state_dict(state_dict, dr_level=dr_level)
        else:
            self._restore_success_criteria(None)
            self.dr.reset(seed=seed)

        self._layout = Layout()
        self._layout.add_robot(self.robot)

        scene_dr = self.dr.get_randomizer("scene")
        if isinstance(scene_dr, TabletopSceneDR):
            scene = scene_dr(split)
            self._layout.scene = scene
            self._layout.add_primitive("table", scene.table)

            if scene.table2 is not None:
                self._layout.add_primitive("table2", scene.table2)
            if scene.tool_table is not None:
                self._layout.add_primitive("tool_table", scene.tool_table)
            table_height = scene.table.pose.position[2] + 0.5 * scene.table.size[2]
        else:
            table_height = 0.0

        # Robot spawn randomization. Task-specific randomizers place all tools
        # and targets after this shared setup has completed.
        spatial_dr = self.dr.get_randomizer("spatial")
        if spatial_dr is not None:
            spatial_dr(split, self._layout, table_height=table_height)

        camera_dr = self.dr.get_randomizer("camera")
        if camera_dr is not None:
            for cam_id, cam_info in self.sensor_cfgs.items():
                if isinstance(cam_info, CameraCfg):
                    cam_cfg = camera_dr(split, cam_info)
                    if (
                        cam_cfg.quaternion is None
                        and cam_cfg.mount == "eye_in_head"
                        and isinstance(self.robot, HeadCamMountable)
                    ):
                        cam_cfg.pose["quaternion"] = self.robot.head_camera_orientation

                    self._layout.add_camera(cam_id, cam_cfg)

    def _restore_success_criteria(self, state_dict: dict[str, Any] | None) -> None:
        """Resolve each episode independently without turning replay into an override."""
        configured = getattr(
            self, "_configured_success_criteria", self.metadata.get("success_criteria")
        )
        value = getattr(self, "_success_criteria_override", None)
        source = "--success-criteria"
        if value is None:
            value = (state_dict or {}).get("metadata", {}).get("success_criteria")
            source = "recorded success_criteria"
        if value is None:
            value = configured
            source = "configured success_criteria"
        if value is not None:
            value = _validate_task_success_criteria(self, value, source)

        self._configured_success_criteria = configured
        self.success_criteria = value
        self.metadata = dict(self.metadata)
        if value is None:
            self.metadata.pop("success_criteria", None)
        else:
            self.metadata["success_criteria"] = value

    def state_dict(self) -> dict[str, Any]:
        """Dump the current state of the task.

        Returns:
            A dictionary containing the current state of the task.
        """
        rand_state_dict = self.dr.state_dict()
        return {
            "uid": self.uid,
            "label": self.label,
            "description": self.description,
            "metadata": self.metadata,
            "robot_cfg": self.robot_cfg,
            "sensor_cfgs": {k: v.__dict__ for k, v in self.sensor_cfgs.items()},
            "dr_cfgs": {k: v.to_dict() for k, v in self.dr_cfgs.items()},
            "dr_state_dict": rand_state_dict,
            "layout": self._layout.to_dict() if self._layout is not None else {},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Clone the current layout with given options."""
        return self.reset(options={"state_dict": state_dict})

    def check_success(self, *args, **kwargs) -> bool:
        """Check if the task is successfully completed."""
        raise NotImplementedError

    def check_failure(self, *args: Any, **kwargs: Any) -> bool:
        """Return whether an explicit task failure has occurred."""
        del args, kwargs
        return False

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """Compute the reward for the current state of the task."""
        return 0.0
