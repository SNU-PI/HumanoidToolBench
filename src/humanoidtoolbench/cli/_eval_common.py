"""Rollout, progress display and reporting logic for humanoid evaluation."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
def _redirect_stdio_to_log(log_path: str):
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
) -> None:
    step_update_every = 5
    frame_idx = 0
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
    with metrics_path.open("w", buffering=1) as metrics_file:
        while not episode_over:
            try:
                action = agent.get_action(
                    observation, info=info, instruction=instruction
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                frame_idx += 1
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
        "instruction": instruction,
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


def _execute_run(
    config: EvalConfig,
    run_episodes: Callable[[Callable[..., None]], dict[str, bool]],
) -> EvalResult:
    """Run the episodes in this process under a live progress display.

    ``run_episodes(report)`` returns the success of each episode. Everything it
    prints goes to ``eval_latest.log``; the terminal shows the progress panel.
    """
    env_id = config.env_id
    policy = config.policy
    eval_dir = config.eval_dir
    eval_dir_path = Path(eval_dir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = str(eval_dir_path / "eval_latest.log")
    Path(log_path).write_text("")

    # A duplicate of the terminal's stderr stays on the terminal while the
    # process's own stdout and stderr go to the log.
    terminal_stream = os.fdopen(os.dup(2), "w", buffering=1)
    console = make_console(terminal_stream)
    # Anything printed while the process shuts down could hide the cursor
    # again; make sure we restore it last.
    ensure_cursor_restored_at_exit(console)
    progress = {0: WorkerProgress()}
    _append_eval_stats_line(eval_dir, "================\n")
    _append_eval_stats_line(eval_dir, f"run: {env_id} - {policy}\n")

    # The console writes to terminal_stream, so the stream must outlive the
    # summary lines printed after the episodes finish.
    try:
        console.print(f"Writing eval logs to [bold]{log_path}[/bold]")
        try:
            with (
                Live(
                    render_progress(env_id, policy, progress, log_path),
                    console=console,
                    refresh_per_second=4,
                ) as live,
                _redirect_stdio_to_log(log_path),
            ):

                def report(event: str, **payload: Any) -> None:
                    update_progress(progress, 0, {"event": event, **payload})
                    live.update(
                        render_progress(env_id, policy, progress, log_path),
                        refresh=False,
                    )

                stats = run_episodes(report)
        finally:
            restore_cursor(console)

        sr = sum(stats.values()) / len(stats) if stats else 0.0
        console.print(f"Success rate {env_id} - {policy}: {sr:.2%}")
        console.print(f"Eval log: {log_path}")
    finally:
        try:
            terminal_stream.close()
        except BaseException as exc:
            print(f"Terminal cleanup failed: {exc}", file=sys.stderr)

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
