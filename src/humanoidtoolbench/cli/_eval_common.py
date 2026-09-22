"""Worker, rollout, and reporting logic for humanoid evaluation."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from contextlib import contextmanager, nullcontext
from multiprocessing.connection import wait

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import typer
from rich.live import Live

from humanoidtoolbench.evals.api import EvalConfig, EvalResult
from humanoidtoolbench.evals.tui import (
    WorkerProgress,
    ensure_cursor_restored_at_exit,
    make_console,
    render_progress,
    restore_cursor,
    update_progress,
)


def _append_eval_stats_line(eval_dir: str, line: str) -> None:
    path = Path(eval_dir) / "eval_stats.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def _redirect_stdio_to_log(log_path: str | None):
    if not log_path:
        yield
        return

    log_file = open(log_path, "a", buffering=1)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    redirected_stdout = None
    redirected_stderr = None

    try:
        os.dup2(log_file.fileno(), 1)
        os.dup2(log_file.fileno(), 2)
        redirected_stdout = os.fdopen(os.dup(1), "w", buffering=1)
        redirected_stderr = os.fdopen(os.dup(2), "w", buffering=1)
        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if redirected_stdout is not None:
            redirected_stdout.close()
        if redirected_stderr is not None:
            redirected_stderr.close()

        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        log_file.close()


def _make_worker_helpers(
    worker_result_path: str | None,
    worker_id: int,
    progress_reporter: Callable[[dict[str, Any]], None] | None,
):
    def persist_payload(kind: str, payload: Any):
        if not worker_result_path:
            return
        with open(worker_result_path, "wb") as f:
            pickle.dump((kind, worker_id, payload), f)

    def report(event: str, **payload: Any) -> None:
        if progress_reporter is not None:
            progress_reporter({"event": event, **payload})

    return persist_payload, report


def _run_eval_worker_entry(
    worker_fn: Callable[..., Any],
    worker_result_path: str,
    worker_id: int,
    num_workers: int,
    worker_kwargs: dict[str, Any],
    log_path: str | None,
    progress_conn: Any | None,
    episode_cursor: Any | None = None,
):
    def report(payload: dict[str, Any]) -> None:
        if progress_conn is not None:
            progress_conn.send((worker_id, payload))

    try:
        with _redirect_stdio_to_log(log_path):
            try:
                worker_fn(
                    **worker_kwargs,
                    worker_id=worker_id,
                    num_workers=num_workers,
                    worker_result_path=worker_result_path,
                    progress_reporter=report,
                    episode_cursor=episode_cursor,
                )
            except Exception:
                with open(worker_result_path, "wb") as f:
                    pickle.dump(("err", worker_id, traceback.format_exc()), f)
                try:
                    report({"event": "worker_error", "message": "see eval log"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
    finally:
        if progress_conn is not None:
            try:
                progress_conn.close()
            except Exception:
                pass


def _prepare_episodes(
    data_format: str,
    eval_dir: str,
    env_id: str,
    split: str,
    data_dir: str,
    num_episodes: int,
    episode_start: int,
    worker_id: int,
    num_workers: int,
    episode_cursor: Any | None,
    report: Callable[..., None],
):
    if episode_start < 0:
        raise ValueError(f"episode_start must be >= 0, got {episode_start}")

    if data_format == "lerobot":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from humanoidtoolbench.datasets.lerobot import get_episode_lerobot

        dataset = LeRobotDataset(repo_id=env_id, root=data_dir, video_backend="pyav")
        dataset_size = dataset.num_episodes
        render_hz = dataset.meta.fps
        print(f"loaded dataset with {dataset_size} episodes.")

        def get_episode(idx):
            return get_episode_lerobot(dataset, idx)

    elif data_format == "seeds":
        # Benchmark mode: the episode set IS the seed range -- no recorded
        # dataset, no --data-dir, no datagen pass to produce one first.
        # Episode i is whatever `reset(seed=i)` builds, which DRManager.reset
        # now makes reproducible, so a run is defined by the env id and the
        # seed range alone.
        dataset_size = episode_start + num_episodes
        render_hz = None  # no dataset fps; taken from the task by the caller

        def get_episode(idx):
            return None, None

    else:
        raise ValueError(
            f"Unsupported data format {data_format!r}; choose 'seeds' or 'lerobot'."
        )

    global_episode_indices = list(range(episode_start, dataset_size))
    global_episode_indices = global_episode_indices[:num_episodes]

    # Episode durations vary wildly (an early success finishes in seconds, a
    # timeout runs the full step budget), so a fixed per-worker slice leaves fast
    # workers idle at the end while one straggler finishes. When the parent hands
    # us a shared cursor, pull the next episode on demand instead: every worker
    # stays busy until the global list is exhausted. Falls back to the static
    # round-robin slice when running solo or when no cursor was supplied.
    if episode_cursor is None:
        _static_iter = iter(global_episode_indices[worker_id::num_workers])

        def claim_next_episode() -> int | None:
            return next(_static_iter, None)

    else:

        def claim_next_episode() -> int | None:
            with episode_cursor.get_lock():
                cursor = episode_cursor.value
                if cursor >= len(global_episode_indices):
                    return None
                episode_cursor.value = cursor + 1
            return global_episode_indices[cursor]

    print(
        f"Evaluating up to {len(global_episode_indices)} episodes "
        f"(worker={worker_id}/{num_workers}, requested_total={num_episodes}, "
        f"dispatch={'dynamic' if episode_cursor is not None else 'static'})."
    )
    report(
        "worker_init",
        total_episodes=0,
        status="creating_env",
        global_total=len(global_episode_indices),
    )
    return get_episode, render_hz, global_episode_indices, claim_next_episode


def _json_metric_value(value: Any) -> Any:
    """Convert simulator metrics to strict JSON, retaining missing/nonfinite as null."""
    if isinstance(value, (np.ndarray, np.generic)):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _json_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metric_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rollout_episode(
    env: Any,
    raw_env: Any,
    agent: Any,
    task_id: str,
    instruction: Any,
    observation: Any,
    info: Any,
    stats: dict[str, bool],
    eval_dir: str,
    report: Callable[..., None],
    *,
    instruction_update: Callable[[int, Any], str] | None = None,
    trajectory_options: dict[str, Any] | None = None,
) -> None:
    step_update_every = 5
    frame_idx = 0
    initial_instruction = instruction
    episode_start_time = time.perf_counter()
    episode_over = False
    terminated = truncated = False
    final_metrics: dict[str, Any] = {}
    ever_true: dict[str, bool] = {}
    first_true_step: dict[str, int | None] = {}
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}
    metrics_path = Path(eval_dir) / "metrics" / f"{task_id}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = None
    if trajectory_options is not None:
        from humanoidtoolbench.evals.trajectory import (
            TrajectoryRecorder,
            controller_snapshot,
            initial_policy_inputs,
        )

        trajectory = TrajectoryRecorder(
            raw_env,
            Path(eval_dir) / "trajectories",
            task_id,
            observation,
            instruction=instruction,
            policy_inputs=initial_policy_inputs(observation, agent),
            controller_state=controller_snapshot(agent),
            **trajectory_options,
        )
    with (
        metrics_path.open("w", buffering=1) as metrics_file,
        trajectory if trajectory is not None else nullcontext(),
    ):
        while not episode_over:
            if instruction_update is not None:
                instruction = instruction_update(frame_idx, observation)
            try:
                action = agent.get_action(
                    observation, info=info, instruction=instruction
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                frame_idx += 1
                if trajectory is not None:
                    trajectory.capture(
                        frame_idx,
                        instruction,
                        terminated=terminated,
                        truncated=truncated,
                    )
                final_metrics = _json_metric_value(info.get("metrics", {}))
                metrics_file.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "episode": task_id,
                            "step": frame_idx,
                            "reward": _json_metric_value(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "metrics": final_metrics,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                for name, value in final_metrics.items():
                    if isinstance(value, bool):
                        ever_true[name] = ever_true.get(name, False) or value
                        first_true_step.setdefault(name, None)
                        if value and first_true_step[name] is None:
                            first_true_step[name] = frame_idx
                    elif isinstance(value, float):
                        # Integer metrics include categorical roles and modes.
                        # Preserve them verbatim instead of treating them as ranges.
                        numeric_min[name] = min(numeric_min.get(name, value), value)
                        numeric_max[name] = max(numeric_max.get(name, value), value)
                if frame_idx == 1 or frame_idx % step_update_every == 0 or episode_over:
                    report("episode_step", episode=task_id, step=frame_idx)
            except StopIteration:
                episode_over = True
                print("Episode finished.")

    is_success = raw_env.unwrapped._success  # type: ignore[attr-defined]
    stats[task_id] = is_success
    episode_seconds = time.perf_counter() - episode_start_time
    if is_success:
        outcome = "success"
    elif terminated:
        outcome = "terminated_failure"
    elif truncated:
        outcome = "timeout"
    else:
        outcome = "agent_exhausted"
    task_metrics = {
        "schema_version": 1,
        "episode": task_id,
        "instruction": initial_instruction,
        "step": frame_idx,
        "success": bool(is_success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "outcome": outcome,
        "episode_seconds": episode_seconds,
        "final_metrics": final_metrics,
        "ever_true": ever_true,
        "first_true_step": first_true_step,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
    }
    summary_path = Path(eval_dir) / "summaries" / f"{task_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = summary_path.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(task_metrics, indent=2, allow_nan=False) + "\n")
    summary_tmp.replace(summary_path)
    _append_eval_stats_line(eval_dir, f"{task_id}: {is_success} \n")
    report(
        "episode_end",
        episode=task_id,
        step=frame_idx,
        completed_episodes=len(stats),
        successes=sum(stats.values()),
        episode_seconds=episode_seconds,
        task_metrics=task_metrics,
        steps_per_second=(
            (frame_idx / episode_seconds) if episode_seconds > 0 else 0.0
        ),
    )


def _finish_worker(
    stats: dict[str, bool],
    persist_payload: Callable[[str, Any], None],
    report: Callable[..., None],
) -> None:
    persist_payload("ok", dict(stats))
    report(
        "worker_done",
        completed_episodes=len(stats),
        successes=sum(stats.values()),
    )


def _setup_run(config: EvalConfig, show_progress: bool):
    num_workers = config.num_workers
    eval_dir_path = Path(config.eval_dir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = str(eval_dir_path / "eval_latest.log")
    Path(log_path).write_text("")
    worker_log_paths = {
        wid: str(eval_dir_path / f"eval_worker_{wid}.log") for wid in range(num_workers)
    }
    for worker_log_path in worker_log_paths.values():
        Path(worker_log_path).write_text("")

    terminal_stream = None
    if num_workers == 1 and show_progress:
        terminal_stream = os.fdopen(os.dup(2), "w", buffering=1)
    console = make_console(terminal_stream)
    # Workers are joined after the Live display stops and may write to this
    # terminal while shutting down; make sure we restore the cursor last.
    ensure_cursor_restored_at_exit(console)
    worker_states = {wid: WorkerProgress() for wid in range(num_workers)}
    _append_eval_stats_line(config.eval_dir, "================\n")
    _append_eval_stats_line(
        config.eval_dir, f"run: {config.env_id} - {config.policy}\n"
    )
    return console, terminal_stream, worker_states, log_path, worker_log_paths


def _cancel_eval_workers(procs: list[Any], *, grace_seconds: float = 5.0) -> None:
    """Stop every started worker, then reap it within bounded grace periods."""
    for method in ("terminate", "kill"):
        for proc in procs:
            try:
                if proc.is_alive():
                    getattr(proc, method)()
            except BaseException as exc:
                print(f"Worker {method} failed: {exc}", file=sys.stderr)
        deadline = time.monotonic() + grace_seconds
        for proc in procs:
            try:
                proc.join(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:
                print(f"Worker join failed: {exc}", file=sys.stderr)
    if any(proc.is_alive() for proc in procs):
        raise RuntimeError("Evaluation workers did not stop after terminate and kill")


def _execute_run(
    config: EvalConfig,
    worker_fn: Callable[..., Any],
    worker_kwargs: dict[str, Any],
    *,
    console: Any,
    terminal_stream: Any,
    worker_states: dict[int, WorkerProgress],
    log_path: str,
    worker_log_paths: dict[int, str],
    show_progress: bool,
) -> EvalResult:
    # The console writes to terminal_stream, so the stream must outlive the
    # summary lines printed after the workers finish.
    try:
        return _run_eval_workers(
            config,
            worker_fn,
            worker_kwargs,
            console=console,
            worker_states=worker_states,
            log_path=log_path,
            worker_log_paths=worker_log_paths,
            show_progress=show_progress,
        )
    finally:
        if terminal_stream is not None:
            try:
                terminal_stream.close()
            except BaseException as exc:
                print(f"Terminal cleanup failed: {exc}", file=sys.stderr)


def _run_eval_workers(
    config: EvalConfig,
    worker_fn: Callable[..., Any],
    worker_kwargs: dict[str, Any],
    *,
    console: Any,
    worker_states: dict[int, WorkerProgress],
    log_path: str,
    worker_log_paths: dict[int, str],
    show_progress: bool,
) -> EvalResult:
    env_id = config.env_id
    policy = config.policy
    eval_dir = config.eval_dir
    num_workers = config.num_workers

    console.print(f"Writing eval logs to [bold]{log_path}[/bold]")

    if num_workers == 1:
        stats: dict[str, bool]
        if show_progress:
            try:
                with (
                    Live(
                        render_progress(env_id, policy, worker_states, log_path),
                        console=console,
                        refresh_per_second=4,
                    ) as live,
                    _redirect_stdio_to_log(log_path),
                ):

                    def report(payload: dict[str, Any]) -> None:
                        update_progress(worker_states, 0, payload)
                        live.update(
                            render_progress(env_id, policy, worker_states, log_path),
                            refresh=False,
                        )

                    stats = worker_fn(
                        **worker_kwargs,
                        worker_id=0,
                        num_workers=1,
                        progress_reporter=report,
                    )
            finally:
                restore_cursor(console)
        else:
            with _redirect_stdio_to_log(log_path):
                stats = worker_fn(
                    **worker_kwargs,
                    worker_id=0,
                    num_workers=1,
                )
    else:
        ctx = mp.get_context("spawn")
        progress_readers: dict[int, Any] = {}
        progress_connections: list[Any] = []
        procs: list[mp.Process] = []
        result_dir = Path(eval_dir) / "worker_results"
        result_dir.mkdir()
        # Shared hand-out point into the global episode list. Workers claim the
        # next index under its lock, so whoever frees up first takes the next
        # episode rather than sitting idle on an exhausted static slice.
        episode_cursor = ctx.Value("i", 0)
        # Total is only known once a worker has opened the dataset; it reports it.
        global_total: int | None = None

        try:
            with Live(
                render_progress(env_id, policy, worker_states, log_path),
                console=console,
                refresh_per_second=4,
                auto_refresh=show_progress,
            ) as live:
                for wid in range(num_workers):
                    worker_result_path = str(result_dir / f"worker_{wid}.pkl")
                    recv_conn, send_conn = ctx.Pipe(duplex=False)
                    progress_connections.extend((recv_conn, send_conn))
                    progress_readers[wid] = recv_conn
                    p = ctx.Process(
                        target=_run_eval_worker_entry,
                        args=(
                            worker_fn,
                            worker_result_path,
                            wid,
                            num_workers,
                            worker_kwargs,
                            worker_log_paths[wid],
                            send_conn,
                            episode_cursor,
                        ),
                        name=f"eval-worker-{wid}",
                    )
                    p.start()
                    procs.append(p)
                    send_conn.close()

                while any(p.is_alive() for p in procs) or progress_readers:
                    worker_failed = False
                    ready = (
                        wait(list(progress_readers.values()), timeout=0.2)
                        if progress_readers
                        else []
                    )
                    for conn in ready:
                        try:
                            wid, payload = conn.recv()
                        except EOFError:
                            for key, value in list(progress_readers.items()):
                                if value is conn:
                                    value.close()
                                    del progress_readers[key]
                                    break
                            continue
                        if payload.get("global_total") is not None:
                            global_total = int(payload["global_total"])
                        worker_failed |= payload.get("event") == "worker_error"
                        update_progress(worker_states, wid, payload)
                        if show_progress:
                            live.update(
                                render_progress(
                                    env_id,
                                    policy,
                                    worker_states,
                                    log_path,
                                    total_target=global_total,
                                ),
                                refresh=False,
                            )
                    if worker_failed or any(p.exitcode not in (None, 0) for p in procs):
                        _cancel_eval_workers(procs)
                        break
        except BaseException:
            try:
                _cancel_eval_workers(procs)
            except BaseException as exc:
                print(f"Worker cleanup failed: {exc}", file=sys.stderr)
            raise
        finally:
            try:
                restore_cursor(console)
            except BaseException as exc:
                print(f"Cursor cleanup failed: {exc}", file=sys.stderr)
            for conn in progress_connections:
                try:
                    conn.close()
                except BaseException as exc:
                    print(f"Progress pipe cleanup failed: {exc}", file=sys.stderr)

        stats = {}
        results_by_worker: dict[int, dict[str, bool]] = {}
        errors_by_worker: dict[int, str] = {}
        failed_exit: list[tuple[int, int]] = []

        for wid, p in enumerate(procs):
            worker_result_path = result_dir / f"worker_{wid}.pkl"
            p.join()
            exit_code = p.exitcode if p.exitcode is not None else 999
            if exit_code != 0:
                failed_exit.append((wid, exit_code))
            if worker_result_path.exists():
                try:
                    with open(worker_result_path, "rb") as f:
                        kind, got_wid, payload = pickle.load(f)
                    if got_wid != wid:
                        errors_by_worker[wid] = (
                            f"mismatched worker payload id={got_wid}"
                        )
                    elif kind == "ok":
                        results_by_worker[wid] = payload
                    else:
                        errors_by_worker[wid] = payload
                except Exception:
                    errors_by_worker[wid] = traceback.format_exc()

        for wid in sorted(results_by_worker):
            worker_stats = results_by_worker[wid]
            overlap = set(worker_stats).intersection(stats)
            if overlap:
                raise RuntimeError(
                    f"Duplicate episode ids across workers: {sorted(overlap)}"
                )
            stats.update(worker_stats)

        if errors_by_worker or failed_exit:
            summary = []
            if errors_by_worker:
                summary.append(f"python_errors={sorted(errors_by_worker)}")
            if failed_exit:
                summary.append(f"nonzero_exit={failed_exit}")
            for wid in sorted(errors_by_worker):
                console.print(
                    f"[red]worker {wid} traceback[/red]\n{errors_by_worker[wid]}\n"
                    f"[dim]log: {worker_log_paths.get(wid, log_path)}[/dim]"
                )
            if failed_exit:
                console.print("[yellow]worker native exits[/yellow]")
                for wid, exit_code in failed_exit:
                    console.print(
                        f"[yellow]worker {wid} exit {exit_code}[/yellow] "
                        f"[dim]log: {worker_log_paths.get(wid, log_path)}[/dim]"
                    )
            console.print(
                "[red]Parallel eval worker failure[/red] "
                + "("
                + ", ".join(summary)
                + f"). See {log_path}"
            )
            error = typer.Exit(code=1)
            if errors_by_worker:
                wid = min(errors_by_worker)
                lines = errors_by_worker[wid].strip().splitlines()
                reason = lines[-1] if lines else "unknown worker error"
                error.alert_reason = f"worker {wid}: {reason}"
                error.alert_log_path = worker_log_paths.get(wid, log_path)
            else:
                error.alert_reason = f"worker exits: {failed_exit}"
                error.alert_log_path = worker_log_paths.get(failed_exit[0][0], log_path)
            raise error

        missing = sorted(
            set(range(num_workers)) - set(results_by_worker) - set(errors_by_worker)
        )
        if missing:
            console.print(
                "[red]Parallel eval worker missing result payload[/red] "
                f"(workers={missing}, exits={failed_exit}). See {log_path}"
            )
            error = typer.Exit(code=1)
            error.alert_reason = f"workers {missing}: missing result payload"
            error.alert_log_path = worker_log_paths.get(missing[0], log_path)
            raise error

    sr = sum(stats.values()) / len(stats) if stats else 0.0
    console.print(f"Success rate {env_id} - {policy}: {sr:.2%}")
    console.print(f"Eval log: {log_path}")

    _append_eval_stats_line(eval_dir, f"success rate: {sr:.2f} \n")
    return EvalResult(
        env_id=env_id,
        policy=policy,
        split=config.split,
        stats=dict(stats),
        success_rate=sr,
        log_path=log_path,
        eval_dir=eval_dir,
    )
