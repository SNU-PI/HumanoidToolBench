"""Opt-in measured MuJoCo trajectories and strict initial-scene pairing."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

SCHEMA = "humanoidtoolbench_body_trajectory_v1"


def _json(value):
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _array_digest(named_arrays):
    digest = hashlib.sha256()
    for name, value in sorted(named_arrays):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


# Pairing is exact for the compiled model, physics options, integration state,
# layout and the 32-D policy state. GPU rasterisation is not bit-reproducible
# across processes (2026-09-15 canary: identical physics and state, 2.6% of
# head-camera values differing, mean 0.08 per value), so initial images are
# compared with recorded deviation statistics and a gross-failure bound.
FINGERPRINT_POLICY = "physical_exact_image_tolerance_v2"
PHYSICAL_COMPONENTS = (
    "model",
    "physics_options",
    "initial_state",
    "layout",
    "initial_policy_state",
)
IMAGE_DEVIATION_LIMITS = {"pixel_fraction": 0.2, "mean_abs": 1.0}


def compare_initial_images(reference, candidate):
    """Return head-image deviation statistics, rejecting gross rendering failures."""
    key = "initial_head_rgb"
    if (key in reference) != (key in candidate):
        raise ValueError("Paired initial policy input differs: initial_head_rgb")
    if key not in reference:
        return None
    left, right = np.asarray(reference[key]), np.asarray(candidate[key])
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Paired initial image shape or dtype differs")
    delta = np.abs(left.astype(np.int64) - right.astype(np.int64))
    pixels = (
        delta.reshape(-1, left.shape[-1]).max(axis=1) > 0
        if left.ndim == 3
        else delta.reshape(-1) > 0
    )
    stats = {
        "policy": FINGERPRINT_POLICY,
        "values_differing": int((delta > 0).sum()),
        "value_count": int(delta.size),
        "pixel_fraction": float(pixels.mean()),
        "max_abs": int(delta.max()),
        "mean_abs": float(delta.mean()),
        "limits": dict(IMAGE_DEVIATION_LIMITS),
        "identical": bool(delta.max() == 0),
    }
    if (
        stats["pixel_fraction"] > IMAGE_DEVIATION_LIMITS["pixel_fraction"]
        or stats["mean_abs"] > IMAGE_DEVIATION_LIMITS["mean_abs"]
    ):
        raise ValueError(
            "Paired initial image deviates beyond rendering tolerance: "
            f"{stats['pixel_fraction']:.3f} of pixels, mean {stats['mean_abs']:.3f}"
        )
    return stats


def match_initial_scene(reference, candidate):
    """Reject changed initial states, visual scenes, or paired-run settings."""
    keys = (
        "schema",
        "env_id",
        "seed",
        "policy_id",
        "controller",
        "control_dt",
        "scene_fingerprint",
        "runtime_settings",
        "controller_state",
    )
    for key in keys:
        if (
            key not in reference
            or key not in candidate
            or reference[key] != candidate[key]
        ):
            raise ValueError(f"Paired trajectory mismatch: {key}")
    if reference["schema"] != SCHEMA or not reference.get("policy_id"):
        raise ValueError(
            "Paired trajectories require the supported schema and policy identity"
        )
    return {"matched": True, "scene_fingerprint": reference["scene_fingerprint"]}


def verify_pair(reference_path, candidate_path):
    """Validate two completed recordings and their bound NPZ payloads."""
    result = match_initial_scene(
        _load_completed(reference_path), _load_completed(candidate_path)
    )
    with (
        np.load(Path(reference_path).with_suffix(".npz"), allow_pickle=False) as left,
        np.load(Path(candidate_path).with_suffix(".npz"), allow_pickle=False) as right,
    ):
        for key in (
            "joint_qpos",
            "base_pose",
            "body_positions",
            "body_quaternions",
            "object_positions",
            "object_quaternions",
        ):
            if not np.array_equal(left[key][0], right[key][0]):
                raise ValueError(f"Paired initial frame differs: {key}")
        key = "initial_policy_state"
        if (key in left) != (key in right) or (
            key in left and not np.array_equal(left[key], right[key])
        ):
            raise ValueError(f"Paired initial policy input differs: {key}")
        result["initial_image_deviation"] = compare_initial_images(left, right)
    return result


def controller_snapshot(agent):
    """Capture decoupled controller history and interpolation without advancing it."""
    from humanoidtoolbench.controllers.decoupled_wbc import DecoupledWbcBackend

    controller = getattr(agent, "body_controller", None)
    if controller is None:
        return None
    if not isinstance(controller, DecoupledWbcBackend):
        raise ValueError(
            "Trajectory controller-state capture supports decoupled_wbc only"
        )
    wbc = controller.wbc_policy
    lower = wbc.lower_body_policy
    upper = wbc.upper_body_policy
    lower_keys = (
        "config",
        "observation",
        "obs_history",
        "obs_buffer",
        "obs_tensor",
        "counter",
        "use_policy_action",
        "use_teleop_policy_cmd",
        "action",
        "target_dof_pos",
        "cmd",
        "height_cmd",
        "freq_cmd",
        "roll_cmd",
        "pitch_cmd",
        "yaw_cmd",
        "target_yaw_cmd",
        "gait_indices",
        "foot_indices",
        "clock_inputs",
        "nav_cmd",
    )
    snapshot = {
        "configuration": controller._sonic_config,
        "frequency": controller._control_frequency,
        "cached_targets": [
            controller._cached_target_q,
            controller._cached_left_hand_q,
            controller._cached_right_hand_q,
        ],
        "last_goal_time": wbc.last_goal_time,
        "teleop_mode": wbc.is_in_teleop_mode,
        "lower": {
            key: getattr(lower, key) for key in lower_keys if hasattr(lower, key)
        },
        "upper": {
            key: getattr(upper, key)
            for key in (
                "last_action",
                "concat_order",
                "concat_dims",
                "init_values_concat",
                "max_change_rate",
                "last_waypoint_time",
            )
        },
        "interpolation": {"times": upper.interp.times, "poses": upper.interp.poses},
    }
    return _json(snapshot)


def _load_completed(path):
    path = Path(path)
    metadata = json.loads(path.read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"Trajectory is not complete: {path}")
    payload = path.with_suffix(".npz")
    if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata.get(
        "arrays_sha256"
    ):
        raise ValueError(f"Trajectory payload hash differs: {payload}")
    return metadata


def initial_policy_inputs(observation, agent):
    """Retain frame-zero inputs only, using the deployed decoupled state builder."""
    from humanoidtoolbench.actions import DECOUPLED_SCHEMA_ID
    from humanoidtoolbench.policies.remote_humanoid import (
        RemoteChunkPolicy,
        build_decoupled_policy_state,
    )

    arrays = {}
    if "head_stereo_left" in observation:
        arrays["initial_head_rgb"] = np.asarray(observation["head_stereo_left"]).copy()
    policy = getattr(agent, "task_policy", None)
    if (
        isinstance(policy, RemoteChunkPolicy)
        and policy.action_schema == DECOUPLED_SCHEMA_ID
    ):
        arrays["initial_policy_state"] = build_decoupled_policy_state(
            observation, policy._last_base_height
        )
    return arrays


class TrajectoryRecorder:
    """Capture actual poses, including frame zero, without modifying live physics."""

    def __init__(
        self,
        raw_env,
        directory,
        episode,
        observation,
        *,
        env_id,
        seed,
        policy_id,
        controller,
        control_dt,
        instruction,
        pair_reference=None,
        policy_inputs=None,
        controller_state=None,
        runtime_settings=None,
    ):
        import mujoco

        self.env = raw_env.unwrapped
        self.model = self.env.mujoco.mjModel
        self.data = self.env.mujoco.mjData
        self.scratch = mujoco.MjData(self.model)
        self.path = Path(directory) / f"{episode}.json"
        self.observation = observation
        self.reference = pair_reference
        self.policy_inputs = policy_inputs or {}
        self.frames = []
        self.start_time = float(self.data.time)
        self.done = False
        self.metadata = {
            "schema": SCHEMA,
            "episode": episode,
            "env_id": env_id,
            "seed": seed,
            "policy_id": policy_id,
            "controller": controller,
            "controller_state": _json(controller_state),
            "runtime_settings": _json(runtime_settings or {}),
            "control_dt": control_dt,
            "instruction": instruction,
            "coordinate_frame": "world",
            "quaternion_order": "wxyz",
            "initial_sim_time": self.start_time,
            "mujoco_version": mujoco.__version__,
            "policy_identity_source": "caller-supplied checkpoint identity",
            "initial_policy_inputs": {
                "frame": 0,
                "image_array": (
                    "initial_head_rgb"
                    if "initial_head_rgb" in self.policy_inputs
                    else None
                ),
                "state_array": (
                    "initial_policy_state"
                    if "initial_policy_state" in self.policy_inputs
                    else None
                ),
                "state_description": (
                    "32-D decoupled state before server preprocessing"
                    if "initial_policy_state" in self.policy_inputs
                    else None
                ),
                "instruction": instruction,
            },
        }
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if root < 0 or not policy_id:
            raise ValueError(
                "Trajectory recording requires G1 pelvis and a checkpoint identity"
            )
        self.body_ids = []
        for body in range(1, self.model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == root:
                self.body_ids.append(body)
        self.root = root
        self.joint_ids = [
            j
            for j in range(self.model.njnt)
            if int(self.model.jnt_bodyid[j]) in self.body_ids
            and int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        ]
        self.joint_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.body_names = [self.model.body(i).name for i in self.body_ids]
        actors = self.env.task.layout.actors
        self.object_names, self.object_ids = [], []
        for slot, actor in sorted(actors.items()):
            if slot == "robot":
                continue
            asset = getattr(actor, "asset", None)
            name = getattr(asset, "label", None) or getattr(asset, "uid", slot)
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(
                    f"Cannot record actual pose of scene actor {slot}: {name}"
                )
            self.object_names.append(slot)
            self.object_ids.append(body)
        self.metadata.update(
            {
                "joint_names": [self.model.joint(i).name for i in self.joint_ids],
                "body_names": self.body_names,
                "object_names": self.object_names,
                "object_body_names": [self.model.body(i).name for i in self.object_ids],
                "body_masses": self.model.body_mass[self.body_ids].tolist(),
                "scene": _json(self.env.task.layout.to_dict()),
            }
        )

    def __enter__(self):
        if self.path.exists() or self.path.with_suffix(".npz").exists():
            raise FileExistsError(f"Refusing to overwrite trajectory: {self.path}")
        # Hash compiled physical/visual arrays, including mesh and texture data.
        arrays = [
            (name, getattr(self.model, name))
            for name in dir(self.model)
            if isinstance(getattr(self.model, name), np.ndarray)
        ]
        arrays.append(("names", np.frombuffer(self.model.names, dtype=np.uint8)))
        options = {
            name: _json(getattr(self.model.opt, name))
            for name in dir(self.model.opt)
            if not name.startswith("_")
            and isinstance(getattr(self.model.opt, name), (int, float, np.ndarray))
        }
        import mujoco

        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial_state = np.empty(mujoco.mj_stateSize(self.model, state_spec))
        mujoco.mj_getState(self.model, self.data, initial_state, state_spec)
        initial = [("mjSTATE_INTEGRATION", initial_state)]
        images = [
            (key, value)
            for key, value in self.observation.items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        ]
        policy_items = list(self.policy_inputs.items())
        components = {
            "model": _array_digest(arrays),
            "physics_options": _digest(options),
            "initial_state": _array_digest(initial),
            "layout": _digest(self.metadata["scene"]),
            "initial_policy_state": _array_digest(
                [(k, v) for k, v in policy_items if k != "initial_head_rgb"]
            ),
            # Image digests are recorded for provenance but excluded from the
            # fingerprint; see FINGERPRINT_POLICY and compare_initial_images.
            "initial_images": _array_digest(images),
            "initial_policy_image": _array_digest(
                [(k, v) for k, v in policy_items if k == "initial_head_rgb"]
            ),
        }
        self.metadata["scene_components"] = components
        self.metadata["scene_fingerprint_policy"] = FINGERPRINT_POLICY
        self.metadata["fingerprint_components"] = list(PHYSICAL_COMPONENTS)
        self.metadata["scene_fingerprint"] = _digest(
            {
                "policy": FINGERPRINT_POLICY,
                "components": {key: components[key] for key in PHYSICAL_COMPONENTS},
            }
        )
        if self.reference:
            reference = Path(self.reference)
            paths = (
                [reference]
                if reference.is_file()
                else list(reference.glob(self.path.name))
                + list(reference.rglob(f"trajectories/{self.path.name}"))
            )
            if len(paths) != 1:
                raise ValueError(
                    f"Expected exactly one paired reference for {self.path.name}"
                )
            previous = _load_completed(paths[0])
            match_initial_scene(previous, self.metadata)
            with np.load(paths[0].with_suffix(".npz"), allow_pickle=False) as arrays:
                self.metadata["pair_image_deviation"] = compare_initial_images(
                    arrays, self.policy_inputs
                )
            self.metadata["pair_reference"] = str(paths[0].resolve())
        self.capture(0, self.metadata["instruction"])
        return self

    def capture(self, step, instruction, *, terminated=False, truncated=False):
        import mujoco

        self.scratch.qpos[:] = self.data.qpos
        self.scratch.mocap_pos[:] = self.data.mocap_pos
        self.scratch.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        frame = {
            "step": step,
            "time_s": float(self.data.time) - self.start_time,
            "joint_qpos": self.data.qpos[self.joint_addresses].copy(),
            "base_pose": np.r_[
                self.scratch.xpos[self.root], self.scratch.xquat[self.root]
            ],
            "body_positions": self.scratch.xpos[self.body_ids].copy(),
            "body_quaternions": self.scratch.xquat[self.body_ids].copy(),
            "body_com_positions": self.scratch.xipos[self.body_ids].copy(),
            "object_positions": self.scratch.xpos[self.object_ids].copy(),
            "object_quaternions": self.scratch.xquat[self.object_ids].copy(),
        }
        if any(not np.isfinite(value).all() for value in frame.values()):
            raise ValueError("Nonfinite measured trajectory state")
        self.frames.append(frame)
        self.done = bool(terminated or truncated)
        if instruction != self.metadata["instruction"]:
            raise ValueError(
                "Paired instruction recording requires a fixed instruction per episode"
            )

    def __exit__(self, exc_type, exc, tb):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: np.asarray([frame[key] for frame in self.frames])
            for key in self.frames[0]
        }
        arrays.update(self.policy_inputs)
        payload = self.path.with_suffix(".npz")
        temporary = payload.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(payload)
        self.metadata.update(
            {
                "status": "completed" if exc is None and self.done else "incomplete",
                "frames": len(self.frames),
                "arrays": payload.name,
                "arrays_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "success": bool(getattr(self.env, "_success", False)),
                "error": None if exc is None else f"{exc_type.__name__}: {exc}",
            }
        )
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(self.path)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify paired measured trajectories")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_pair(args.reference, args.candidate), indent=2))


if __name__ == "__main__":
    main()
