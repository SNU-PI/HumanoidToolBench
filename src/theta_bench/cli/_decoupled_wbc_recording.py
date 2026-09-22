from __future__ import annotations

import json
import math
import shutil
import sys
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import numpy as np
import transforms3d as t3d

from theta_bench.controllers.wbc_observation import hand_mjcf_to_natural
from theta_bench.datasets import contract
from theta_bench.instructions import (
    dataset_env_id,
    dataset_instruction,
    normalize_dataset_instructions,
)

EGO_VIEW_FEATURE = "observation.images.ego_view"
# The two wrist cameras, keyed by the observation they come from. The head
# pair cannot tell a grasp from a near miss at bench range; these can.
WRIST_VIEW_FEATURES = {
    "wrist_left": "observation.images.wrist_left",
    "wrist_right": "observation.images.wrist_right",
}
RECORDING_FPS = 50


def validate_recording_timing(render_hz: int, physics_dt: float) -> float:
    """Require the 50 Hz cadence supported by decoupled WBC recordings."""
    if render_hz != RECORDING_FPS:
        raise ValueError(
            f"Decoupled WBC recording/replay requires {RECORDING_FPS} Hz; "
            f"got {render_hz} Hz"
        )
    control_dt = 1.0 / RECORDING_FPS
    if not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("physics_dt must be finite and positive")
    substeps = int((1.0 / physics_dt) / render_hz)
    if substeps < 1 or not math.isclose(
        substeps * physics_dt, control_dt, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"physics_dt={physics_dt} must divide the {control_dt}s control interval"
        )
    return control_dt


# The exporter writes meta/info.json, meta/modality.json AND meta/tasks.jsonl
# when it creates a dataset, so none of those can tell a crashed skeleton from
# a real one. meta/episodes.jsonl only appears once the first episode is saved,
# and its absence is exactly what the exporter's resume path dies on.
EPISODE_MARKER = Path("meta") / "episodes.jsonl"
RECORDED_DATA_GLOBS = ("*.parquet", "*.mp4")


def _normalize_shape(ego_view_shape) -> list[int] | None:
    if ego_view_shape is None:
        return None

    shape = [int(dim) for dim in ego_view_shape]
    if len(shape) != 3:
        raise ValueError(f"Expected ego-view image shape to be HWC, got {tuple(shape)}")
    return shape


def set_ego_view_feature_shape(features: dict, ego_view_shape) -> None:
    """Match the LeRobot video metadata to the actual egocentric image shape."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    features[EGO_VIEW_FEATURE]["shape"] = shape


def add_wrist_view_features(features: dict, wrist_view_shape) -> None:
    """Declare the wrist streams alongside the ego view, in its own format."""
    shape = _normalize_shape(wrist_view_shape)
    if shape is None:
        return
    for name in WRIST_VIEW_FEATURES.values():
        feature = deepcopy(features[EGO_VIEW_FEATURE])
        feature["shape"] = shape
        info = feature.get("info")
        if isinstance(info, dict):
            info = dict(info)
            info["video.height"] = shape[0]
            info["video.width"] = shape[1]
            feature["info"] = info
        features[name] = feature


def validate_existing_ego_view_feature_shape(
    save_dir: str | Path, ego_view_shape
) -> None:
    """Refuse to append 480p frames to a dataset initialized with old 360p metadata."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    info_path = Path(save_dir) / "meta" / "info.json"
    if not info_path.exists():
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    existing_shape = info.get("features", {}).get(EGO_VIEW_FEATURE, {}).get("shape")
    if existing_shape is None:
        return

    existing_shape = [int(dim) for dim in existing_shape]
    if existing_shape != shape:
        raise ValueError(
            f"Existing LeRobot dataset at {save_dir} declares {EGO_VIEW_FEATURE} "
            f"shape {existing_shape}, but the current ego-view frame is {shape}. "
            "Use a clean --save-dir or remove the old dataset before recording."
        )


def discard_incomplete_dataset(save_dir: str | Path) -> bool:
    """Delete a dataset directory that was created but never received an episode.

    A teleop run that is stopped before the first episode is saved leaves the
    metadata skeleton behind. The next run then takes the exporter's "resume"
    path, fails to read the local metadata, falls back to the HuggingFace Hub
    for the placeholder repo id and dies with a 401 and a misleading
    "corrupted dataset" error. Removing the skeleton is safe because it holds
    no recorded data; anything that does contain data is left alone.

    Returns True when a skeleton was removed.
    """

    root = Path(save_dir)
    if not root.exists() or (root / EPISODE_MARKER).exists():
        return False

    recorded = [path for pattern in RECORDED_DATA_GLOBS for path in root.rglob(pattern)]
    if recorded:
        raise RuntimeError(
            f"The dataset at {root} has no {EPISODE_MARKER} but still holds "
            f"{len(recorded)} recorded file(s), starting with {recorded[0]}. "
            "Refusing to delete it; inspect it manually or record into a clean --save-dir."
        )

    shutil.rmtree(root)
    print(
        f"[Record] Removed the empty dataset skeleton at {root} "
        "(a previous run stopped before saving an episode); starting a fresh dataset."
    )
    return True


def load_episodes(data_dir: str):
    """Load all episodes from a LeRobot dataset directory, keyed by episode index."""
    import pyarrow.parquet as pq

    data_path = Path(data_dir) / "data"
    parquet_files = sorted(data_path.rglob("*.parquet"))
    if not parquet_files:
        # The recorder writes `<save-dir>/<env id>/level-<n>`, so the dataset
        # root always ends in a level. Pointing one directory short of it is
        # the usual way to land here, and the bare path says nothing about it.
        levels = sorted(p.name for p in Path(data_dir).glob("level-*") if p.is_dir())
        hint = f" Did you mean {Path(data_dir) / levels[0]}?" if levels else ""
        raise FileNotFoundError(f"No parquet files found in {data_path}.{hint}")

    episodes = {}
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        for ep_idx in df["episode_index"].unique():
            episodes[int(ep_idx)] = df[df["episode_index"] == ep_idx].reset_index(
                drop=True
            )

    return episodes


def load_episode_configs(data_dir: str):
    """Load environment_config per episode from meta/episodes.jsonl."""
    meta_file = Path(data_dir) / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return {}

    configs = {}
    with open(meta_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            ep_idx = entry.get("episode_index", None)
            env_conf_str = entry.get("environment_config", None)
            if ep_idx is not None and env_conf_str is not None:
                configs[int(ep_idx)] = json.loads(env_conf_str)
    return configs


def cleanup_on_exit(close):
    """ExitStack callback that keeps an active error if resource cleanup fails."""

    def cleanup(exc_type, exc, traceback):
        try:
            close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            print(f"[Cleanup] {close}: {cleanup_error}", file=sys.stderr)
        return False

    return cleanup


# The outcome goes right after the index, so a glance down episodes.jsonl
# says which episodes succeeded; environment_config, the largest field, goes
# last. Everything else keeps the order it was written in.
OUTCOME_FIRST = (
    "episode_index",
    "episode_success",
    "termination_reason",
    "failure_source",
    "episode_failure",
    "final_stage",
    "episode_return",
    "episode_length",
)


def order_episode_entry(entry: dict) -> dict:
    ordered = {key: entry[key] for key in OUTCOME_FIRST if key in entry}
    ordered.update(
        (key, value)
        for key, value in entry.items()
        if key not in ordered and key != "environment_config"
    )
    if "environment_config" in entry:
        ordered["environment_config"] = entry["environment_config"]
    return ordered


def save_episode_metadata(
    exporter,
    task,
    episode_index: int,
    outcome=None,
    seed=None,
    environment_config=None,
    success_source=None,
    controller_name=None,
    controller_version=None,
    task_uid=None,
    render_backend=None,
    extra_metadata=None,
):
    """Write environment_config and the episode outcome into episodes.jsonl.

    `environment_config` keeps its existing shape; the outcome is written next
    to it so a failed episode can be recognised without replaying it.
    """
    from theta_bench.utils import NumpyArrayEncoder

    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    if episode_index >= len(lines):
        return
    env_conf = task.state_dict() if environment_config is None else environment_config
    entry = {"environment_config": json.dumps(env_conf, cls=NumpyArrayEncoder)}
    if outcome is not None:
        entry.update(
            contract.episode_metadata(
                outcome,
                task_uid=task_uid or getattr(task, "uid", None),
                task_spec_version=task.metadata.get("version"),
                controller_name=controller_name,
                controller_version=controller_version,
                robot_name=getattr(task.robot, "uid", None),
                simulator="mujoco",
                seed=seed,
                success_source=success_source,
                extra=extra_metadata,
            )
        )
    if render_backend is not None:
        entry["render_backend"] = render_backend
    lines[episode_index] = order_episode_entry({**lines[episode_index], **entry})
    with open(meta_file, "w") as f:
        for line_entry in lines:
            f.write(json.dumps(line_entry, cls=NumpyArrayEncoder) + "\n")


def _git_commit(path) -> str | None:
    """Short HEAD hash of the git work tree at *path*, or None if unavailable."""
    import subprocess

    try:
        done = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def _controller_provenance(agent, robot_model) -> dict:
    """What produced this dataset: code versions, WBC config, retargeting scale.

    Every value is read from the objects already driving the run, so it cannot
    drift from what actually executed. Missing pieces are recorded as None
    rather than omitted, so a consumer can tell "not applicable" from "lost".
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    cfg = getattr(agent, "_dwbc_config", None)
    supp = getattr(robot_model, "supplemental_info", None)
    return {
        "theta_bench_commit": _git_commit(root),
        "decoupled_wbc_commit": _git_commit(root / "third_party" / "decoupled_wbc"),
        "gear_sonic_commit": _git_commit(root / "third_party" / "gear_sonic"),
        "wbc_version": getattr(cfg, "wbc_version", None),
        "wbc_model_path": getattr(cfg, "wbc_model_path", None),
        "wbc_policy_class": getattr(cfg, "wbc_policy_class", None),
        "wbc_control_frequency": getattr(cfg, "control_frequency", None),
        "enable_waist": getattr(cfg, "enable_waist", None),
        "high_elbow_pose": getattr(cfg, "high_elbow_pose", None),
        "lower_body_policy": getattr(cfg, "lower_body_policy", None),
        "teleop_upper_body_motion_scale": getattr(
            supp, "teleop_upper_body_motion_scale", None
        ),
        "controller_name": type(agent).__name__ if agent is not None else None,
    }


def controller_version_string(provenance: dict) -> str | None:
    """Compact identifier for the controller stack, for episodes.jsonl."""
    parts = [
        f"{key}={provenance[key]}"
        for key in ("theta_bench_commit", "decoupled_wbc_commit", "wbc_version")
        if provenance.get(key)
    ]
    return "; ".join(parts) or None


_OBJECT_POSE_SUFFIXES = (
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)


def object_recording_features(object_slots: list[str]) -> dict[str, dict]:
    """Build fixed pose and presence features for semantic object slots."""
    if not object_slots:
        return {}
    if len(set(object_slots)) != len(object_slots):
        raise ValueError(f"recording object slots must be unique: {object_slots}")

    pose_names = [
        f"{slot}.{suffix}" for slot in object_slots for suffix in _OBJECT_POSE_SUFFIXES
    ]
    return {
        "observation.object_poses": {
            "dtype": "float64",
            "shape": (len(object_slots) * len(_OBJECT_POSE_SUFFIXES),),
            "names": pose_names,
        },
        "observation.object_present": {
            "dtype": "bool",
            "shape": (len(object_slots),),
            "names": list(object_slots),
        },
    }


def init_exporter(
    save_dir: str,
    task_prompt: str,
    robot_model,
    object_slots: list[str],
    joint_names: list[str],
    ego_view_shape=None,
    wrist_view_shape=None,
    overwrite_existing: bool = False,
    metric_spec=None,
    hand_joint_names=None,
    agent=None,
    env_id: str | None = None,
):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    recorded_env_id = dataset_env_id(Path(save_dir), env_id=env_id)
    task_prompt = dataset_instruction(Path(save_dir), env_id=env_id) or task_prompt

    if not overwrite_existing:
        # A skeleton left behind by a run that never saved an episode would send
        # the exporter down its resume path and fail against the Hub.
        discard_incomplete_dataset(save_dir)
        info_path = Path(save_dir) / "meta" / "info.json"
        if info_path.exists():
            existing_fps = json.loads(info_path.read_text()).get("fps")
            if existing_fps != RECORDING_FPS:
                raise ValueError(
                    f"Cannot append {RECORDING_FPS} Hz frames to a dataset "
                    f"declaring fps={existing_fps!r}"
                )

    features = get_dataset_features(robot_model)
    set_ego_view_feature_shape(features, ego_view_shape)
    validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    add_wrist_view_features(features, wrist_view_shape)
    features["observation.state"]["names"] = joint_names  # state joint names
    modality_config = get_modality_config(robot_model)
    _reslice_modality_state(modality_config, robot_model, joint_names)

    # # Add torso RPY command feature (3D: roll, pitch, yaw)
    # features["observation.torso_rpy_command"] = {
    #     "dtype": "float64",
    #     "shape": (3,),
    #     "names": ["roll", "pitch", "yaw"],
    # }

    # Actual end-effector observation, measured by forward kinematics on the
    # joint state. Kept apart from the teleoperator's target below: an
    # observation that is a copy of the command leaks the label.
    wrist_names = ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    for side in ("left", "right"):
        features[f"observation.eef.{side}_pose"] = {
            "dtype": "float64",
            "shape": (7,),
            "names": wrist_names,
        }
        features[f"teleop.eef.{side}_target"] = {
            "dtype": "float64",
            "shape": (7,),
            "names": wrist_names,
        }
        side_hand_names = (hand_joint_names or {}).get(side) or [
            f"{side}_hand_{i}" for i in range(7)
        ]
        features[f"teleop.{side}_hand_command"] = {
            "dtype": "float64",
            "shape": (7,),
            "names": list(side_hand_names),
        }

    # Measured joint velocity and estimated torque, sharing observation.state's
    # joint order. Recorded now because the values already exist; a torque-level
    # controller added later cannot recover them retroactively.
    for name in ("observation.velocity", "observation.effort"):
        features[name] = {
            "dtype": "float64",
            "shape": (len(joint_names),),
            "names": list(joint_names),
        }

    # Operator's raw XR poses, before the retargeting IK. `teleop.eef.*_target`
    # above is the same motion after motion scaling and yaw compensation, so it
    # cannot be replayed through a controller that does its own retargeting.
    for key in ("head", "left", "right"):
        features[f"teleop.xr.{key}_pose"] = {
            "dtype": "float64",
            "shape": (7,),
            "names": wrist_names,
        }

    # Outcome of every frame plus whatever metrics the task declares. Computed
    # by the task, never here.
    features.update(contract.contract_features(metric_spec or {}))

    features.update(object_recording_features(object_slots))

    from decoupled_wbc.data.exporter import DataCollectionInfo

    provenance = _controller_provenance(agent, robot_model)
    if not overwrite_existing and (Path(save_dir) / "meta/tasks.jsonl").exists():
        # Load the updated task table before the exporter resumes, so appending
        # an episode reuses the recorded task index instead of adding an alias.
        normalize_dataset_instructions(Path(save_dir), env_id=env_id)
    exporter = Gr00tDataExporter.create(
        save_root=save_dir,
        fps=RECORDING_FPS,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
        overwrite_existing=overwrite_existing,
        script_config=provenance,
        data_collection_info=DataCollectionInfo(
            lower_body_policy=provenance["lower_body_policy"],
            wbc_model_path=provenance["wbc_model_path"],
            robot_type=provenance["controller_name"],
        ),
    )

    with ExitStack() as resources:
        resources.push(cleanup_on_exit(exporter.stop_video_writers))
        if recorded_env_id is not None:
            exporter.meta.info["env_id"] = recorded_env_id
            (Path(save_dir) / "meta/info.json").write_text(
                json.dumps(exporter.meta.info, indent=4) + "\n", encoding="utf-8"
            )
        # After the exporter, never before: it decides "resume vs new" purely on
        # whether save_dir exists, and writing the sidecar first recreates the very
        # directory `discard_incomplete_dataset` just removed. The exporter then
        # resumes a dataset with no meta/info.json, falls back to the Hub for a
        # placeholder repo id, and dies with "corrupted dataset".
        _write_contract_metadata(
            save_dir, features, metric_spec or {}, joint_names, provenance
        )
        # Read back by the per-episode metadata writer.
        exporter.theta_provenance = provenance
        resources.pop_all()
        return exporter


def _reslice_modality_state(modality_config, robot_model, joint_names):
    """Point the modality state limb slices at the recorded joint order.

    ``get_modality_config`` derives every slice from the robot model's joint
    order.  The action column does use that order, but ``observation.state``
    is recorded in ``joint_names`` order (overridden in ``init_exporter``),
    which interleaves the limb groups differently.
    """
    positions = {name: index for index, name in enumerate(joint_names)}
    state = modality_config["state"]
    for group in (
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
    ):
        group_names = [
            robot_model.joint_names[index]
            for index in robot_model.get_joint_group_indices(group)
        ]
        missing = [name for name in group_names if name not in positions]
        if missing:
            raise ValueError(
                f"observation.state joint names are missing {missing} for {group}"
            )
        indices = sorted(positions[name] for name in group_names)
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(
                f"observation.state does not keep the {group} joints contiguous, "
                "so a modality slice cannot describe them"
            )
        state[group] = {"start": indices[0], "end": indices[-1] + 1}


def _write_contract_metadata(
    save_dir, features, metric_spec, joint_names, provenance=None
):
    """Describe field semantics next to the dataset instead of in tribal knowledge."""
    import os

    metadata = contract.conventions_metadata(
        # The action column keeps the robot model's joint order (see
        # build_frame), which is not the observation.state order.
        joint_order={
            "observation.state": list(joint_names),
            "action": list(features["action"]["names"]),
        },
        command_fields=[
            "action",
            "action.eef",
            "teleop.eef.left_target",
            "teleop.eef.right_target",
            "teleop.xr.head_pose",
            "teleop.xr.left_pose",
            "teleop.xr.right_pose",
            "teleop.left_hand_command",
            "teleop.right_hand_command",
            "teleop.navigate_command",
            "teleop.base_height_command",
        ],
        unavailable_fields={
            "observation.img_state_delta": "not measured; written as 0.0 for schema compatibility",
        },
        # Meaning depends on the controller that produced the row; a consumer
        # must read `controller_provenance` before comparing these across runs.
        controller_specific_fields=["action", "action.eef"],
    )
    metadata["action_semantics"] = {
        "action": "executed 43-DoF joint target produced by the decoupled WBC (alias: executed_joint_target)",
        "action.eef": "teleoperated wrist target before the WBC, identical to teleop.eef.*_target",
    }
    metadata["controller_provenance"] = provenance or {}
    metadata["humanoid_action_contract"] = {
        "recording_controller": "decoupled_wbc",
        "training_schema": "decoupled_v1",
        "training_action_dim": 36,
        "requires_preparation": True,
        "sonic_motion_token_recorded": False,
        "preparation_command": (
            "prepare-humanoid-actions INPUT --schema decoupled_v1 --output OUTPUT"
        ),
    }
    metadata["declared_metrics"] = sorted(metric_spec)
    object_slots = features.get("observation.object_present", {}).get("names", [])
    metadata["object_pose_contract"] = {
        "slots": list(object_slots),
        "pose_order": list(_OBJECT_POSE_SUFFIXES),
        "presence_field": "observation.object_present",
        "missing_pose": list(_IDENTITY_POSE),
    }
    metadata["schema"] = {
        name: {"dtype": feature["dtype"], "shape": list(feature["shape"])}
        for name, feature in features.items()
    }
    path = os.path.join(str(save_dir), "meta", "simple_contract.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


_IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # pos + quat wxyz


def build_frame(
    agent,
    object_slots: list[str],
    object_map: dict[str, str | None],
    observation,
    privileged_info,
    action,
    target_waist,
    outcome=None,
    metric_spec=None,
):
    """Assemble one recording frame from current sim state and agent caches.

    `observation`/`action` describe s_t; `outcome`, when given as
    `(reward, terminated, truncated, info)`, is the result of applying that
    action and is written into the `next.*` and `metric.*` fields.
    """
    proprio = privileged_info["proprio"]
    rm = agent._dwbc_robot_model

    # Raw XR poses come from the Quest agent's per-step cache. Agents that do not
    # drive from a headset (replay, scripted) have none, and record identity.
    cached_xr = getattr(agent, "_last_raw_xr_pose", None) or {}
    raw_xr = {
        key: np.asarray(cached_xr.get(key, _IDENTITY_POSE), dtype=np.float64)
        for key in ("head", "left", "right")
    }

    # action: 43-DOF target q in robot_model joint order
    # NOTE: joint matches with rm.joint_names
    action_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=action["target_q"],
        left_hand_actuated_joint_values=action[
            "left_hand_q"
        ],  # natural order: thumb/index/middle
        right_hand_actuated_joint_values=action["right_hand_q"],
    )

    # # Extract torso RPY command directly from waist joints (waist_yaw, waist_roll, waist_pitch)
    # # These three joints fully determine the torso orientation relative to pelvis
    # waist_indices = rm.get_joint_group_indices("waist")
    # torso_rpy_command = action_q[waist_indices]

    proprio_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=proprio["body_q"],
        left_hand_actuated_joint_values=hand_mjcf_to_natural(proprio["left_hand_q"]),
        right_hand_actuated_joint_values=hand_mjcf_to_natural(proprio["right_hand_q"]),
    )
    proprio_joints = dict(zip(rm.joint_names, proprio_q))
    from theta_bench.robots.g1_sonic import WHOLE_BODY_JOINTS

    assert np.allclose(
        observation["joint_qpos"],
        np.array(
            [proprio_joints[joint] for joint in WHOLE_BODY_JOINTS], dtype=np.float32
        ),
    )

    # Joint velocity and estimated torque, in the joint order of
    # `observation.state` (not the robot model's) so the three columns index
    # alike. `prepare_obs` already measures both; nothing here is derived.
    def _theta_order(body, left_hand, right_hand):
        values = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=np.asarray(body, dtype=np.float64),
            left_hand_actuated_joint_values=hand_mjcf_to_natural(
                np.asarray(left_hand, dtype=np.float64)
            ),
            right_hand_actuated_joint_values=hand_mjcf_to_natural(
                np.asarray(right_hand, dtype=np.float64)
            ),
        )
        by_joint = dict(zip(rm.joint_names, values))
        return np.array(
            [by_joint[joint] for joint in WHOLE_BODY_JOINTS], dtype=np.float64
        )

    joint_velocity = _theta_order(
        proprio.get("body_dq", np.zeros(29)),
        proprio.get("left_hand_dq", np.zeros(7)),
        proprio.get("right_hand_dq", np.zeros(7)),
    )
    joint_effort = _theta_order(
        proprio.get("body_tau_est", np.zeros(29)),
        proprio.get("left_hand_tau_est", np.zeros(7)),
        proprio.get("right_hand_tau_est", np.zeros(7)),
    )

    # override waist joints in action_q with target_waist from IK, to reflect the actual input to lower-body policy
    action_q[12:15] = (
        target_waist  # waist joints are indices 12,13,14 in robot_model order
    )
    # Actual wrist poses: forward kinematics on the *measured* configuration.
    # `action["action_eef"]` is the teleoperator's target and must not be
    # recorded as an observation.
    rm.cache_forward_kinematics(proprio_q)
    measured_eef = {}
    for side in ("left", "right"):
        placement = rm.frame_placement(rm.supplemental_info.hand_frame_names[side])
        measured_eef[side] = np.concatenate(
            [
                np.asarray(placement.translation[:3]),
                t3d.quaternions.mat2quat(placement.rotation),
            ]
        ).astype(np.float64)

    commanded_eef = np.asarray(action["action_eef"], dtype=np.float64)

    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        # Only when the scene carries them, so a task with a head camera alone
        # still records.
        **{
            name: observation[key]
            for key, name in WRIST_VIEW_FEATURES.items()
            if key in observation
        },
        "observation.state": np.asarray(
            observation["joint_qpos"], dtype=np.float64
        ),  # obs_state
        # legacy field name, now carrying the measured pose instead of the command
        "observation.eef_state": np.concatenate(
            [measured_eef["left"], measured_eef["right"]]
        ),
        "observation.velocity": joint_velocity,
        "observation.effort": joint_effort,
        "observation.eef.left_pose": measured_eef["left"],
        "observation.eef.right_pose": measured_eef["right"],
        "teleop.eef.left_target": commanded_eef[:7],
        "teleop.eef.right_target": commanded_eef[7:],
        "teleop.xr.head_pose": raw_xr["head"],
        "teleop.xr.left_pose": raw_xr["left"],
        "teleop.xr.right_pose": raw_xr["right"],
        "teleop.left_hand_command": np.asarray(action["left_hand_q"], dtype=np.float64),
        "teleop.right_hand_command": np.asarray(
            action["right_hand_q"], dtype=np.float64
        ),
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": np.asarray(
            action["action_eef"], dtype=np.float64
        ),  # 1-cycle delayed teleop eef
        "observation.img_state_delta": np.array([0.0], dtype=np.float32),  # FIXME
        "teleop.navigate_command": np.asarray(action["navigate_cmd"], dtype=np.float64),
        "teleop.base_height_command": np.asarray(
            action["base_height_command"], dtype=np.float64
        ),
        "observation.base_pose": np.asarray(
            proprio["floating_base_pose"], dtype=np.float64
        ),
        "observation.base_vel": np.asarray(
            proprio["floating_base_vel"], dtype=np.float64
        ),
    }

    if object_slots:
        object_poses = []
        object_present = []
        for slot in object_slots:
            object_name = object_map.get(slot)
            if object_name is None:
                object_poses.append(np.asarray(_IDENTITY_POSE, dtype=np.float64))
                object_present.append(False)
                continue
            if object_name not in privileged_info:
                raise KeyError(
                    f"recording slot {slot!r} maps to missing object {object_name!r}"
                )
            pose = np.asarray(privileged_info[object_name], dtype=np.float64)
            if pose.shape != (7,):
                raise ValueError(
                    f"recording object {object_name!r} pose must have shape (7,), "
                    f"got {pose.shape}"
                )
            object_poses.append(pose)
            object_present.append(True)
        frame["observation.object_poses"] = np.concatenate(object_poses)
        frame["observation.object_present"] = np.asarray(object_present, dtype=np.bool_)

    if outcome is not None:
        reward, terminated, truncated, step_info = outcome
        frame.update(
            contract.contract_frame(
                reward, terminated, truncated, step_info, metric_spec or {}
            )
        )

    return frame
