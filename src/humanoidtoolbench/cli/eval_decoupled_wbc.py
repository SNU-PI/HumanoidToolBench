from __future__ import annotations

import json
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import humanoidtoolbench.envs as _  # noqa: F401
from humanoidtoolbench.cli._decoupled_wbc_recording import validate_recording_timing
from humanoidtoolbench.cli._eval_common import _execute_run, _rollout_episode
from humanoidtoolbench.evals.api import EvalConfig, EvalResult
from humanoidtoolbench.runtime import validate_sim_mode

# Allow 30 seconds at 50 Hz; seed 10069 first stabilizes at step 1044.
_MAX_STABILIZATION_STEPS = 1500


@contextmanager
def _stabilization_images(sonic_env, agent, *, enabled):
    """Skip unused images only for the known proprioceptive WBC backend."""
    from humanoidtoolbench.controllers.decoupled_wbc import DecoupledWbcBackend

    suppress = (
        enabled
        and type(getattr(agent, "body_controller", None)) is DecoupledWbcBackend
        and getattr(sonic_env, "render_obs", False) is True
    )
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
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(
        SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[]
    )
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "humanoidtoolbench"
    return sonic_config


def _run_episodes(
    config: EvalConfig,
    sonic_config: dict[str, Any],
    report: Callable[..., None],
) -> dict[str, bool]:
    """Evaluate every episode of ``config`` in this process."""
    env_id = config.env_id
    controller = config.controller
    sim_mode = config.sim_mode
    headless = config.headless
    eval_dir = config.eval_dir

    # Keep OpenCV/video dependencies off the import path of runs without videos.
    VideoRecorder: type[Any] | None = None
    if config.save_video:
        from humanoidtoolbench.envs.wrappers.video_recorder import VideoRecorder

    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt  # = 0.02 s (50 Hz)

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
        headless=headless,
        sonic_config=sonic_config,
    )
    try:
        sonic_env = raw_env.unwrapped  # type: ignore
        task = raw_env.unwrapped.task  # type: ignore[attr-defined]

        # There is no dataset to read a frame rate from, so the task's own
        # declaration is the source of truth.
        render_hz = task.metadata.get("render_hz", 30)
        if controller == "decoupled_wbc":
            control_dt = validate_recording_timing(render_hz, sim_dt)

        max_episode_steps = config.max_episode_steps
        raw_env = TimeLimit(raw_env, max_episode_steps=max_episode_steps)

        robot = task.robot

        from humanoidtoolbench.policies import make_humanoid_policy_agent

        agent = make_humanoid_policy_agent(
            robot=task.robot,
            policy=config.policy,
            controller=controller,
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

                # DecoupledWbcBackend.reset() engages its lower-body policy.  Unified
                # SONIC owns the equivalent episode state inside its decoder runtime.

                # --- Wait for robot to stabilize (velocity-based) ---
                realtime_stabilization = (
                    not headless
                    or getattr(agent, "controller_name", controller) != "decoupled_wbc"
                    or getattr(sonic_env, "viewer", None) is not None
                    or getattr(sonic_env, "_mjviser", None) is not None
                )
                stabilization_start = time.perf_counter()
                stabilization_sleep_seconds = 0.0
                sim_cnt = 0
                stabilized = robot.stabilized
                with _stabilization_images(
                    sonic_env,
                    agent,
                    enabled=not realtime_stabilization,
                ) as images_suppressed:
                    while not stabilized and sim_cnt < _MAX_STABILIZATION_STEPS:
                        if realtime_stabilization:
                            step_start = time.monotonic()
                        # Unified SONIC retains its usual observation and info.
                        action = agent.get_stabilize_action(observation, info=info)
                        # Bypass TimeLimit so startup keeps the full policy budget.
                        observation, _reward, _terminated, _truncated, info = (
                            sonic_env.step(action)
                        )
                        sonic_env.update_viewer()
                        if realtime_stabilization:
                            elapsed = time.monotonic() - step_start
                            sleep_time = control_dt - elapsed
                            if sleep_time > 0:
                                sleep_start = time.perf_counter()
                                time.sleep(sleep_time)
                                stabilization_sleep_seconds += (
                                    time.perf_counter() - sleep_start
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
                    realtime_pacing=realtime_stabilization,
                    elapsed_seconds=time.perf_counter() - stabilization_start,
                    sleep_seconds=stabilization_sleep_seconds,
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

                finish_stabilization = getattr(
                    getattr(agent, "body_controller", None),
                    "finish_stabilization",
                    None,
                )
                if callable(finish_stabilization):
                    finish_stabilization()

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
