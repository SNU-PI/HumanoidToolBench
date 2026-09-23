from __future__ import annotations

import json
import math
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import humanoidtoolbench.envs as _  # noqa: F401
from humanoidtoolbench.cli._eval_common import _execute_run, _rollout_episode
from humanoidtoolbench.evals.api import EvalConfig, EvalResult
from humanoidtoolbench.runtime import validate_sim_mode

# Allow 30 seconds at 50 Hz; seed 10069 first stabilizes at step 1044.
_MAX_STABILIZATION_STEPS = 1500

# The decoupled WBC runs its policy and camera loop at this rate.
CONTROL_HZ = 50


def validate_control_timing(render_hz: int, physics_dt: float) -> float:
    """Require the 50 Hz control cadence; return the control interval in seconds."""
    if render_hz != CONTROL_HZ:
        raise ValueError(
            f"The decoupled WBC evaluator requires {CONTROL_HZ} Hz; got {render_hz} Hz"
        )
    control_dt = 1.0 / CONTROL_HZ
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


@contextmanager
def _stabilization_images(sonic_env):
    """Skip images while the proprioceptive WBC stabilizes; nothing reads them."""
    suppress = getattr(sonic_env, "render_obs", False) is True
    if suppress:
        sonic_env.render_obs = False
    try:
        yield suppress
    finally:
        if suppress:
            sonic_env.render_obs = True


def _fetch_policy_info(host: str, port: int, timeout: float = 5.0) -> dict[str, Any]:
    """GET http://<host>:<port>/info from the policy server.

    Returns the parsed JSON (containing e.g. "policy" and "timestamp"). The
    endpoint is optional and not guaranteed to exist; if it is missing or the
    request fails, returns an empty dict. Metadata never determines output paths.
    """
    import urllib.request

    url = f"http://{host}:{port}/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"No policy info from {url} ({e}); continuing without server metadata.")
        return {}


def _make_sonic_config() -> dict[str, Any]:
    """The pinned G1 WBC configuration the robot and the controller read.

    This is gear_sonic's g1_29dof_sonic_model12.yaml with the values its
    simulation-loop defaults (SimLoopConfig.load_wbc_yaml) write on top.
    """
    import gear_sonic
    import yaml

    path = (
        Path(gear_sonic.__file__).resolve().parent
        / "utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
    )
    with path.open() as file:
        sonic_config = yaml.safe_load(file)
    sonic_config.update(
        {
            "INTERFACE": "lo",
            "ENV_TYPE": "sim",
            "VERSION": "sonic_model12",
            "SIMULATOR": "mujoco",
            "SIMULATE_DT": 1 / 200.0,
            "ENABLE_OFFSCREEN": False,
            "ENABLE_ONSCREEN": True,
            "model_path": "policy/stand.onnx,policy/walk.onnx",
            # False would silently hand the waist to the lower-body policy.
            "enable_waist": True,
            "with_hands": True,
            "verbose": False,
            "verbose_timing": False,
            "upper_body_max_joint_speed": 1000,
            "keyboard_dispatcher_type": "raw",
            "enable_gravity_compensation": False,
            "gravity_compensation_joints": ["arms"],
            "high_elbow_pose": False,
            "joint_safety_mode": "kill",
            "arm_velocity_limit": 25.0,
            "hand_velocity_limit": 1000.0,
            "lower_body_velocity_limit": 20.0,
            "waist_pitch_limit": 15.0,
            "hand_torque_limit": 0.1,
            "enable_natural_walk": False,
            "ENV_NAME": "humanoidtoolbench",
        }
    )
    return sonic_config


def _run_episodes(
    config: EvalConfig,
    sonic_config: dict[str, Any],
    report: Callable[..., None],
) -> dict[str, bool]:
    """Evaluate every episode of ``config`` in this process."""
    env_id = config.env_id
    sim_mode = config.sim_mode
    eval_dir = config.eval_dir

    # Keep OpenCV/video dependencies off the import path of runs without videos.
    VideoRecorder: type[Any] | None = None
    if config.save_video:
        from humanoidtoolbench.envs.wrappers.video_recorder import VideoRecorder

    sim_dt = sonic_config["SIMULATE_DT"]

    eval_output_dir = str(Path(eval_dir) / "videos")

    if config.episode_start < 0:
        raise ValueError(f"episode_start must be >= 0, got {config.episode_start}")
    # The episode set is the seed range: episode i is whatever reset(seed=i)
    # builds, so a run is defined by the env id and the seed range alone.
    episodes = range(config.episode_start, config.episode_start + config.num_episodes)
    print(f"Evaluating {len(episodes)} episodes.")
    report("worker_init", total_episodes=len(episodes), status="creating_env")

    setup_start_time = time.perf_counter()
    print(f"Creating environment: {env_id}")
    raw_env = gym.make(
        env_id,
        split=config.split,
        sim_mode=sim_mode,
        physics_dt=sim_dt,
        headless=config.headless,
        sonic_config=sonic_config,
    )
    try:
        sonic_env = raw_env.unwrapped  # type: ignore
        task = raw_env.unwrapped.task  # type: ignore[attr-defined]

        # The task's own declaration is the source of truth for the frame rate.
        render_hz = task.metadata.get("render_hz", 30)
        control_dt = validate_control_timing(render_hz, sim_dt)

        max_episode_steps = config.max_episode_steps
        raw_env = TimeLimit(raw_env, max_episode_steps=max_episode_steps)

        robot = task.robot

        from humanoidtoolbench.policies import make_humanoid_policy_agent

        agent = make_humanoid_policy_agent(
            robot=task.robot,
            policy=config.policy,
            controller=config.controller,
            host=config.host,
            port=config.port,
            sonic_config=sonic_config,
        )
        setup_seconds = time.perf_counter() - setup_start_time
        report(
            "worker_init",
            total_episodes=len(episodes),
            status="ready",
            setup_seconds=setup_seconds,
            max_episode_steps=max_episode_steps,
            video_path=eval_output_dir if config.save_video else None,
        )

        stats: dict[str, bool] = {}

        for eps_idx in episodes:
            task_id = f"episode_{eps_idx}"
            report("episode_start", episode=task_id)

            if VideoRecorder is not None:
                env = VideoRecorder(
                    env=raw_env,
                    video_folder=eval_output_dir,
                    name_prefix=task_id,
                    framerate=render_hz,
                    write_png=False,
                    start_on_reset=False,
                )
            else:
                env = raw_env

            try:
                observation, info = env.reset(seed=eps_idx)

                # Reset the agent BEFORE stabilization so the WBC pipeline starts the
                # episode fresh and the upper body gets the smooth 2-second ramp to the
                # default pose on every episode (not just the first).
                agent.reset()

                # DecoupledWbcBackend.reset() engages its lower-body policy.

                # --- Wait for robot to stabilize (velocity-based) ---
                stabilization_start = time.perf_counter()
                sim_cnt = 0
                stabilized = robot.stabilized
                with _stabilization_images(sonic_env) as images_suppressed:
                    while not stabilized and sim_cnt < _MAX_STABILIZATION_STEPS:
                        action = agent.get_stabilize_action(observation, info=info)
                        # Bypass TimeLimit so startup keeps the full policy budget.
                        observation, _reward, _terminated, _truncated, info = (
                            sonic_env.step(action)
                        )
                        sim_cnt += 1
                        stabilized = robot.stabilized

                final_observation_refreshed = False
                if images_suppressed and stabilized:
                    # This is policy input and video frame zero at the final state.
                    observation = sonic_env._get_obs()
                    final_observation_refreshed = True

                report(
                    "stabilization_end",
                    episode=task_id,
                    stabilized=stabilized,
                    control_steps=sim_cnt,
                    max_control_steps=_MAX_STABILIZATION_STEPS,
                    control_dt=control_dt,
                    elapsed_seconds=time.perf_counter() - stabilization_start,
                    image_observations_suppressed=images_suppressed,
                    image_observations_skipped=sim_cnt if images_suppressed else 0,
                    final_observation_refreshed=final_observation_refreshed,
                )
                if not stabilized:
                    raise RuntimeError(
                        f"{task_id}: robot did not stabilize within {sim_cnt} "
                        "simulation steps; policy rollout was not started."
                    )

                if VideoRecorder is not None and isinstance(env, VideoRecorder):
                    # Start once, with the observation after stabilization as frame zero.
                    env._init_writers(observation)

                instruction = task.instruction

                print(
                    f"Robot stabilized after {sim_cnt} simulation steps. Engaging Policy Now!"
                )

                agent.body_controller.finish_stabilization()

                _rollout_episode(
                    env,
                    raw_env,
                    agent,
                    task_id,
                    instruction,
                    observation,
                    info,
                    stats,
                    eval_dir,
                    report,
                )

                if VideoRecorder is not None and isinstance(env, VideoRecorder):
                    env.release()
            except BaseException:
                if VideoRecorder is not None and isinstance(env, VideoRecorder):
                    try:
                        env.discard()
                    except Exception as cleanup_error:
                        print(f"Video cleanup failed: {cleanup_error}")
                raise
        report("worker_status", status="closing")
    except BaseException:
        try:
            raw_env.close()
        except Exception as cleanup_error:
            print(f"Environment cleanup failed: {cleanup_error}")
        raise
    else:
        raw_env.close()

    report(
        "worker_done",
        completed_episodes=len(stats),
        successes=sum(stats.values()),
    )
    return stats


def run_eval(config: EvalConfig) -> EvalResult:
    config = replace(config, sim_mode=validate_sim_mode(config.sim_mode))
    sonic_config = _make_sonic_config()

    eval_root = Path(config.eval_dir).expanduser().resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=eval_root))
    config = replace(config, eval_dir=str(run_dir))
    (run_dir / "evaluation_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    policy_info = _fetch_policy_info(config.host, config.port)
    # Live server state (current session, idle time) is not provenance.
    policy_info.pop("server_state", None)
    if policy_info:
        (run_dir / "policy_info.json").write_text(
            json.dumps(policy_info, indent=2) + "\n"
        )

    result = _execute_run(
        config, lambda report: _run_episodes(config, sonic_config, report)
    )
    result_path = run_dir / "evaluation_result.json"
    result_tmp = result_path.with_suffix(".json.tmp")
    result_tmp.write_text(
        json.dumps(
            {
                "status": "completed",
                "episodes": len(result.stats),
                "stats": {
                    episode: bool(success) for episode, success in result.stats.items()
                },
                "success_rate": float(result.success_rate),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    result_tmp.replace(result_path)
    return result
