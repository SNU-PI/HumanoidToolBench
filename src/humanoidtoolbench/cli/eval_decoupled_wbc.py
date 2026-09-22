from __future__ import annotations

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import typer
from gymnasium.wrappers import TimeLimit
from typing_extensions import Annotated

import humanoidtoolbench.envs as _  # noqa: F401
from humanoidtoolbench.cli._decoupled_wbc_recording import validate_recording_timing
from humanoidtoolbench.cli._eval_common import (
    _execute_run,
    _finish_worker,
    _make_worker_helpers,
    _prepare_episodes,
    _rollout_episode,
    _setup_run,
)
from humanoidtoolbench.cli._wandb_checkpoint import wandb_url_for_checkpoint
from humanoidtoolbench.cli.slack_alerts import RunNotification, _valid_wandb_url
from humanoidtoolbench.core.task import set_task_success_criteria
from humanoidtoolbench.evals.api import EvalConfig, EvalResult
from humanoidtoolbench.policies.reasoner import (
    DEFAULT_REASONER_MODEL,
    ReasonerConfig,
    VisionReasoner,
)
from humanoidtoolbench.runtime import ISAAC_RENDER_SIM_MODES, validate_sim_mode
from humanoidtoolbench.scenario_names import canonicalize_env_id
from humanoidtoolbench.tasks.compat import normalize_task_state_uid

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


def _validate_reasoner_policy(
    policy: str, rl_checkpoint: str | None, reasoner: ReasonerConfig | None
) -> None:
    if reasoner is None:
        return
    if rl_checkpoint is not None:
        raise ValueError("--rl-checkpoint actors do not condition on reasoner text")

    from humanoidtoolbench.policies.remote_humanoid import normalize_policy_name

    if normalize_policy_name(policy) in {"act", "dp"}:
        raise ValueError(
            "ACT/DP baselines do not condition on language; text reasoner has no effect"
        )


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


def _run_eval_worker(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    split: Annotated[str, typer.Argument()] = "train",
    controller: Annotated[str, typer.Option()] = "decoupled_wbc",
    gear_sonic_artifact_root: Annotated[str | None, typer.Option()] = None,
    gear_sonic_initial_token: Annotated[str | None, typer.Option()] = None,
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "seeds",
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals_decoupled_wbc",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = 16,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    success_criteria: Annotated[float | None, typer.Option()] = None,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    record_trajectory: bool = False,
    trajectory_policy_id: str | None = None,
    instruction_override: str | None = None,
    pair_reference: str | None = None,
    scene_seed: int | None = None,
    rl_checkpoint: str | None = None,
    reasoner: ReasonerConfig | None = None,
    sonic_config: dict[str, Any] | None = None,
    worker_id: int = 0,
    num_workers: int = 1,
    worker_result_path: str | None = None,
    progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    episode_cursor: Any | None = None,
):
    env_id = canonicalize_env_id(env_id)
    sim_mode = validate_sim_mode(sim_mode)
    _validate_reasoner_policy(policy, rl_checkpoint, reasoner)
    vision_reasoner = VisionReasoner(reasoner) if reasoner is not None else None

    # Keep OpenCV/video dependencies off both the CLI import/help path and
    # headless metric-only runs. The Toolbench controller stack can therefore be
    # evaluated with ``--no-save-video`` without installing an encoder.
    VideoRecorder: type[Any] | None = None
    if save_video:
        from humanoidtoolbench.envs.wrappers.video_recorder import VideoRecorder

    persist_payload, report = _make_worker_helpers(
        worker_result_path, worker_id, progress_reporter
    )

    if sonic_config is None:
        sonic_config = _make_sonic_config()

    sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = 4 * sim_dt  # = 0.02 s (50 Hz)

    eval_output_dir = str(Path(eval_dir) / "videos")

    get_episode, render_hz, global_episode_indices, claim_next_episode = (
        _prepare_episodes(
            data_format=data_format,
            eval_dir=eval_dir,
            env_id=env_id,
            split=split,
            data_dir=data_dir,
            num_episodes=num_episodes,
            episode_start=episode_start,
            worker_id=worker_id,
            num_workers=num_workers,
            episode_cursor=episode_cursor,
            report=report,
        )
    )

    setup_start_time = time.perf_counter()
    print(f"Creating environment: {env_id}")
    make_kwargs = dict(
        split=split,
        sim_mode=sim_mode,
        physics_dt=sim_dt,
        render_hz=render_hz,
        headless=headless,
        sonic_config=sonic_config,
        # An RL actor reads the simulator, never the pictures, so drawing them
        # every step buys nothing. It also costs: the offscreen render is what
        # makes this unable to share a card with MuJoCo-Warp training. Videos
        # still render when asked for.
        render_obs=save_video or rl_checkpoint is None,
    )
    raw_env = gym.make(env_id, **make_kwargs)
    try:
        sonic_env = raw_env.unwrapped  # type: ignore
        task = raw_env.unwrapped.task  # type: ignore[attr-defined]

        # The seeds format has no dataset to read a frame rate from, so the task's
        # own declaration is the source of truth.
        if render_hz is None:
            render_hz = task.metadata.get("render_hz", 30)
        if controller == "decoupled_wbc":
            control_dt = validate_recording_timing(render_hz, sim_dt)

        # Use provided max_episode_steps or fall back to task's metadata
        if max_episode_steps is None:
            max_episode_steps = task.metadata.get("max_episode_steps")

        # Apply TimeLimit wrapper if max_episode_steps is specified
        if max_episode_steps is not None:
            raw_env = TimeLimit(raw_env, max_episode_steps=max_episode_steps)

        set_task_success_criteria(task, success_criteria)

        robot = task.robot

        from humanoidtoolbench.policies import make_humanoid_policy_agent

        controller_options: dict[str, Any] = {}
        if gear_sonic_initial_token is not None:
            controller_options["initial_motion_token"] = gear_sonic_initial_token
        task_policy: Any = policy
        agent = None
        if rl_checkpoint is not None:
            from humanoidtoolbench.controllers.factory import normalize_controller_name
            from humanoidtoolbench.policies import HumanoidPolicyAgent
            from humanoidtoolbench_rl.eval.benchmark_policy import (
                BallMoveDecoupledPolicy,
                BallMoveSonicPolicy,
                make_benchmark_policy,
            )

            task_policy = make_benchmark_policy(rl_checkpoint, sonic_env.mujoco)
            if isinstance(task_policy, (BallMoveSonicPolicy, BallMoveDecoupledPolicy)):
                # The RL adapters own their training-matched motor-target path.
                agent = HumanoidPolicyAgent(
                    task_policy, task_policy.make_body_controller()
                )
                agent.policy_name = task_policy.policy_name
                agent.controller_name = (
                    "decoupled_rl"
                    if isinstance(task_policy, BallMoveDecoupledPolicy)
                    else "sonic_joint_rl"
                )
                agent.action_schema = agent.body_controller.action_schema
            elif normalize_controller_name(controller) != "decoupled_wbc":
                raise ValueError("--rl-checkpoint requires --controller decoupled_wbc")
        if agent is None:
            agent = make_humanoid_policy_agent(
                robot=task.robot,
                policy=task_policy,
                controller=controller,
                host=host,
                port=port,
                sonic_config=sonic_config,
                gear_sonic_artifact_root=gear_sonic_artifact_root,
                controller_options=controller_options,
            )
        setup_seconds = time.perf_counter() - setup_start_time
        report(
            "worker_init",
            total_episodes=0,
            status="ready",
            setup_seconds=setup_seconds,
            max_episode_steps=max_episode_steps,
            video_path=eval_output_dir if save_video else None,
            global_total=len(global_episode_indices),
        )

        stats = defaultdict(bool)
        claimed = 0

        while True:
            eps_idx = claim_next_episode()
            if eps_idx is None:
                break
            # "assigned" is only known as we go under dynamic dispatch, so grow the
            # worker's denominator each time it claims another episode.
            claimed += 1
            report("worker_init", total_episodes=claimed, status="running")

            env_conf, episode = get_episode(eps_idx)
            task_id = f"episode_{eps_idx}"
            report("episode_start", episode=task_id)

            if env_conf is not None:
                env_conf = normalize_task_state_uid(env_conf, task.uid)

            if save_video:
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
                # The seed matters in both modes: in seeds format it *is* the episode,
                # and in replay it pins the randomizers this dr_level re-draws.
                reset_options = None if env_conf is None else {"state_dict": env_conf}
                # A scene seed rebuilds one fixed scene for every episode; the
                # episode index still names the episode and its policy requests.
                reset_seed = eps_idx if scene_seed is None else scene_seed
                observation, info = env.reset(seed=reset_seed, options=reset_options)

                # Reset the agent BEFORE stabilization so the WBC pipeline starts the
                # episode fresh and the upper body gets the smooth 2-second ramp to the
                # default pose on every episode (not just the first).
                agent.reset()

                # DecoupledWbcBackend.reset() engages its lower-body policy.  Unified
                # SONIC owns the equivalent episode state inside its decoder runtime.

                # --- Wait for robot to stabilize (velocity-based) ---
                isaac_renderer = getattr(sonic_env, "isaac", None)
                realtime_stabilization = (
                    not headless
                    or getattr(agent, "controller_name", controller) != "decoupled_wbc"
                    or getattr(sonic_env, "viewer", None) is not None
                    or getattr(sonic_env, "_mjviser", None) is not None
                    or (
                        isaac_renderer is not None
                        and (
                            not isaac_renderer.headless
                            or os.getenv("HUMANOIDTOOLBENCH_ISAAC_WEBRTC", "0")
                            .strip()
                            .lower()
                            not in {"0", "false", "no", "off"}
                        )
                    )
                )
                stabilization_start = time.perf_counter()
                stabilization_sleep_seconds = 0.0
                sim_cnt = 0
                stabilized = robot.stabilized
                with _stabilization_images(
                    sonic_env,
                    agent,
                    enabled=(
                        headless
                        and not realtime_stabilization
                        and sim_mode == "mujoco"
                        and isaac_renderer is None
                    ),
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

                instruction = instruction_override or task.instruction
                instruction_update = None
                if vision_reasoner is not None:
                    reasoning = vision_reasoner.reason(observation, instruction)
                    reasoning_dir = Path(eval_dir) / "reasoning"

                    def save_reasoning(result, control_step):
                        path = reasoning_dir / f"{task_id}.json"
                        if control_step:
                            path = (
                                reasoning_dir
                                / task_id
                                / f"step_{control_step:06d}.json"
                            )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            json.dumps(
                                {
                                    **asdict(result),
                                    "episode": task_id,
                                    "episode_index": eps_idx,
                                    "env_id": env_id,
                                    "policy": policy,
                                    "split": split,
                                    "control_step": control_step,
                                },
                                indent=2,
                            )
                            + "\n"
                        )

                    save_reasoning(reasoning, 0)
                    instruction = reasoning.policy_instruction
                    if reasoner.replan_interval_steps is not None:
                        original_instruction = task.instruction
                        previous_observation = {
                            "head_stereo_left": observation["head_stereo_left"].copy()
                        }
                        last_review_step = 0

                        def instruction_update(control_step, current_observation):
                            nonlocal reasoning, previous_observation, last_review_step
                            elapsed = control_step - last_review_step
                            if (
                                elapsed >= reasoner.replan_interval_steps
                                and agent.pending_goals == 0
                            ):
                                reviewed = vision_reasoner.reason(
                                    current_observation,
                                    original_instruction,
                                    previous_result=reasoning,
                                    previous_observation=previous_observation,
                                    steps_since_plan=elapsed,
                                )
                                save_reasoning(reviewed, control_step)
                                if (
                                    reviewed.policy_instruction
                                    != reasoning.policy_instruction
                                ):
                                    agent.invalidate_policy_plan()
                                reasoning = reviewed
                                previous_observation = {
                                    "head_stereo_left": current_observation[
                                        "head_stereo_left"
                                    ].copy()
                                }
                                last_review_step = control_step
                            return reasoning.policy_instruction

                print(
                    f"Robot stabilized after {sim_cnt} simulation steps. Engaging Policy Now!"
                )

                # # TEST: stop here and move on to the next episode (skip policy eval)
                # if save_video and isinstance(env, VideoRecorder):
                #     env.release()
                # continue

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
                    instruction_update=instruction_update,
                    trajectory_options=(
                        dict(
                            env_id=env_id,
                            seed=eps_idx,
                            policy_id=trajectory_policy_id,
                            controller=controller,
                            control_dt=control_dt,
                            pair_reference=pair_reference,
                            runtime_settings={
                                "max_episode_steps": max_episode_steps,
                                "success_criteria": success_criteria,
                                "split": split,
                                "sim_mode": sim_mode,
                                "data_format": data_format,
                                "scene_seed": scene_seed,
                            },
                        )
                        if record_trajectory
                        else None
                    ),
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

    _finish_worker(stats, persist_payload, report)
    return stats


def run_eval(
    config: EvalConfig,
    *,
    sonic_config: dict[str, Any] | None = None,
    show_progress: bool = True,
    notify: bool = True,
) -> EvalResult:
    config = replace(config, sim_mode=validate_sim_mode(config.sim_mode))
    episode_range = (
        f"[{config.episode_start}, {config.episode_start + config.num_episodes})"
    )
    data = (
        f"simulator seeds {episode_range}, split={config.split}"
        if config.data_format == "seeds"
        else f"{config.data_format}: {config.data_dir}, "
        f"episodes {episode_range}, split={config.split}"
    )
    notification = RunNotification(
        enabled=notify,
        kind="evaluation",
        model=config.policy,
        task=config.env_id,
        data=data,
        output=config.eval_dir,
        checkpoint=config.rl_checkpoint,
        wandb_url=wandb_url_for_checkpoint(config.rl_checkpoint),
    )
    isaac = config.sim_mode in ISAAC_RENDER_SIM_MODES
    previous_defer = os.environ.get("HUMANOIDTOOLBENCH_ISAAC_DEFER_CLOSE")
    if isaac:
        # Kit may exit from app.close(). Keep it alive until worker results,
        # the final evaluation receipt, and notifications have been written.
        os.environ["HUMANOIDTOOLBENCH_ISAAC_DEFER_CLOSE"] = "1"
    try:
        notification.start()
        try:
            result = _run_eval(
                config,
                sonic_config=sonic_config,
                show_progress=show_progress,
                notification=notification,
            )
        except BaseException as error:
            notification.log_path = getattr(
                error, "alert_log_path", notification.log_path
            )
            notification.finish(False, error=getattr(error, "alert_reason", error))
            raise
        notification.finish(
            True,
            summary=f"success={result.success_rate:.1%} "
            f"({sum(result.stats.values())}/{len(result.stats)} episodes)",
        )
    finally:
        if isaac:
            if previous_defer is None:
                os.environ.pop("HUMANOIDTOOLBENCH_ISAAC_DEFER_CLOSE", None)
            else:
                os.environ["HUMANOIDTOOLBENCH_ISAAC_DEFER_CLOSE"] = previous_defer
    # Leave failed runs to propagate their exception, since Kit shutdown can
    # terminate Python with a successful exit status and hide that failure.
    if isaac and (
        previous_defer is None
        or previous_defer.strip().lower() in {"0", "false", "no", "off"}
    ):
        from humanoidtoolbench.engines.isaac_app import close_simulation_app

        close_simulation_app()
    return result


def _run_eval(
    config: EvalConfig,
    *,
    sonic_config: dict[str, Any] | None,
    show_progress: bool,
    notification: RunNotification,
) -> EvalResult:
    _validate_reasoner_policy(config.policy, config.rl_checkpoint, config.reasoner)
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {config.num_workers}")
    if sonic_config is None:
        sonic_config = _make_sonic_config()

    eval_root = Path(config.eval_dir).expanduser().resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=eval_root))
    config = replace(config, eval_dir=str(run_dir))
    notification.output = run_dir
    (run_dir / "evaluation_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    if config.reasoner is not None:
        (run_dir / "reasoner_config.json").write_text(
            json.dumps(asdict(config.reasoner), indent=2) + "\n"
        )
    if config.rl_checkpoint is not None:
        from humanoidtoolbench_rl.eval.checkpoints import (
            checkpoint_control,
            resolve_checkpoint,
        )

        checkpoint = resolve_checkpoint(Path(config.rl_checkpoint).expanduser())
        control = checkpoint_control(checkpoint)
        policy_info = {
            "policy": config.policy,
            "checkpoint": str(checkpoint.resolve()),
            "rl_control": control,
            "controller": {
                "sonic": "sonic_joint_rl",
                "decoupled": "decoupled_rl",
                "balance": "decoupled_wbc",
            }[control],
            "observation_type": "state",
        }
    else:
        policy_info = _fetch_policy_info(config.host, config.port)
    if policy_info:
        (run_dir / "policy_info.json").write_text(
            json.dumps(policy_info, indent=2) + "\n"
        )
        if isinstance(policy_info, dict):
            model = policy_info.get("policy")
            checkpoint = policy_info.get("checkpoint")
            if isinstance(model, str) and model:
                notification.model = model
            if isinstance(checkpoint, str) and checkpoint:
                notification.checkpoint = checkpoint
                notification.wandb_url = wandb_url_for_checkpoint(checkpoint)
            wandb_url = _valid_wandb_url(policy_info.get("wandb_url"))
            if wandb_url:
                notification.wandb_url = wandb_url

    console, terminal_stream, worker_states, log_path, worker_log_paths = _setup_run(
        config, show_progress
    )
    notification.log_path = log_path
    notification.stage = "rollouts"

    worker_kwargs = dict(
        env_id=config.env_id,
        policy=config.policy,
        controller=config.controller or "decoupled_wbc",
        gear_sonic_artifact_root=config.gear_sonic_artifact_root,
        gear_sonic_initial_token=config.gear_sonic_initial_token,
        split=config.split,
        host=config.host,
        port=config.port,
        data_format=config.data_format,
        sim_mode=config.sim_mode,
        headless=config.headless,
        eval_dir=config.eval_dir,
        max_episode_steps=config.max_episode_steps,
        num_episodes=config.num_episodes,
        episode_start=config.episode_start,
        data_dir=config.data_dir,
        success_criteria=config.success_criteria,
        save_video=config.save_video,
        record_trajectory=config.record_trajectory,
        trajectory_policy_id=config.trajectory_policy_id,
        instruction_override=config.instruction_override,
        pair_reference=config.pair_reference,
        rl_checkpoint=config.rl_checkpoint,
        reasoner=config.reasoner,
        sonic_config=sonic_config,
    )

    result = _execute_run(
        config,
        _run_eval_worker,
        worker_kwargs,
        console=console,
        terminal_stream=terminal_stream,
        worker_states=worker_states,
        log_path=log_path,
        worker_log_paths=worker_log_paths,
        show_progress=show_progress,
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


def main(
    env_id: Annotated[str, typer.Argument()],
    policy: Annotated[str, typer.Argument()],
    split: Annotated[str, typer.Argument()] = "train",
    controller: Annotated[
        str,
        typer.Option(help="Body controller: decoupled_wbc (36D) or gear_sonic (78D)."),
    ] = "decoupled_wbc",
    gear_sonic_artifact_root: Annotated[
        str | None,
        typer.Option(
            help="Directory containing matching model_decoder.onnx and observation_config.yaml."
        ),
    ] = None,
    gear_sonic_initial_token: Annotated[
        str | None,
        typer.Option(
            help="Checkpoint-specific 64D .npy/.json/.txt token used for safe stabilization."
        ),
    ] = None,
    host: Annotated[str, typer.Option()] = "172.17.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    data_format: Annotated[str, typer.Option()] = "seeds",
    sim_mode: Annotated[
        str,
        typer.Option(
            help="Render backend: mujoco or isaacsim-mujoco (alias: mujoco_isaac)."
        ),
    ] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    eval_dir: Annotated[str, typer.Option()] = "data/evals_decoupled_wbc",
    max_episode_steps: Annotated[int | None, typer.Option()] = None,
    num_episodes: Annotated[int, typer.Option()] = 20,
    episode_start: Annotated[int, typer.Option()] = 0,
    data_dir: Annotated[str, typer.Option()] = "data/datagen",
    success_criteria: Annotated[float | None, typer.Option()] = None,
    save_video: Annotated[bool, typer.Option("--save-video/--no-save-video")] = True,
    record_trajectory: Annotated[
        bool, typer.Option(help="Save measured full-body poses per control step.")
    ] = False,
    trajectory_policy_id: Annotated[
        str | None,
        typer.Option(help="Exact checkpoint identity for paired trajectory records."),
    ] = None,
    instruction_override: Annotated[
        str | None,
        typer.Option(
            help="Fixed replacement policy instruction; task success criteria stay unchanged."
        ),
    ] = None,
    pair_reference: Annotated[
        str | None,
        typer.Option(
            help="Reference trajectory JSON or run directory; reject different initial scenes."
        ),
    ] = None,
    num_workers: Annotated[int, typer.Option()] = 1,
    rl_checkpoint: Annotated[
        str | None,
        typer.Option(
            help=(
                "Score a BallMove RL run (legacy L0, decoupled or SONIC L0/L1/L2) "
                "in-process instead of a policy server; POLICY is then its label."
            )
        ),
    ] = None,
    reasoner_enabled: Annotated[
        bool,
        typer.Option(
            "--reasoner",
            help="Enable one visual plan per episode using Cosmos3-Nano by default.",
        ),
    ] = False,
    reasoner_model: Annotated[
        str | None,
        typer.Option(help="Override the reasoner model and enable visual planning."),
    ] = None,
    reasoner_provider: Annotated[
        str, typer.Option(help="Reasoner API provider: openai or gemini.")
    ] = "openai",
    reasoner_base_url: Annotated[
        str | None,
        typer.Option(help="Reasoner API base URL, including /v1 or /v1beta."),
    ] = None,
    reasoner_api_key_env: Annotated[
        str | None,
        typer.Option(help="Environment variable containing the reasoner API key."),
    ] = None,
    reasoner_timeout: Annotated[float, typer.Option()] = 120.0,
    reasoner_max_tokens: Annotated[int, typer.Option()] = 4096,
    reasoner_replan_steps: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Review progress every N control steps "
                "at the next action-chunk boundary."
            ),
        ),
    ] = None,
):
    run_eval(
        EvalConfig(
            env_id=env_id,
            policy=policy,
            controller=controller,
            gear_sonic_artifact_root=gear_sonic_artifact_root,
            gear_sonic_initial_token=gear_sonic_initial_token,
            split=split,
            host=host,
            port=port,
            data_format=data_format,
            sim_mode=sim_mode,
            headless=headless,
            eval_dir=eval_dir,
            max_episode_steps=max_episode_steps,
            num_episodes=num_episodes,
            episode_start=episode_start,
            data_dir=data_dir,
            success_criteria=success_criteria,
            save_video=save_video,
            record_trajectory=record_trajectory,
            trajectory_policy_id=trajectory_policy_id,
            instruction_override=instruction_override,
            pair_reference=pair_reference,
            num_workers=num_workers,
            rl_checkpoint=rl_checkpoint,
            reasoner=(
                ReasonerConfig(
                    model=(
                        reasoner_model
                        if reasoner_model is not None
                        else DEFAULT_REASONER_MODEL
                    ),
                    provider=reasoner_provider,
                    base_url=reasoner_base_url,
                    api_key_env=reasoner_api_key_env,
                    timeout=reasoner_timeout,
                    max_tokens=reasoner_max_tokens,
                    replan_interval_steps=reasoner_replan_steps,
                )
                if (
                    reasoner_enabled
                    or reasoner_model is not None
                    or reasoner_replan_steps is not None
                )
                else None
            ),
        ),
        show_progress=True if not os.environ.get("DEBUG", 0) else False,
    )


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
