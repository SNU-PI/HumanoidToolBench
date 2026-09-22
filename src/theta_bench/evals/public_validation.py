"""Validate outcome coverage and every recorded frame before publishing a score."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

from theta_bench.evals.api import EvalConfig

CAMERAS = ("head_stereo_left", "head_stereo_right", "wrist_left", "wrist_right")


def source_identity() -> dict[str, str]:
    source = Path(__file__).resolve().parents[1]
    paths = set(source.rglob("*.py")) | set(source.rglob("*.c"))
    paths.update(path for path in (source / "resources").rglob("*") if path.is_file())
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    for name in ("run_mujoco.sh", "select_mujoco_device.py"):
        path = source.parents[1] / "scripts" / name
        hashes[f"scripts/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_video(path: Path, steps: int) -> dict:
    import av
    import numpy as np

    decoded = hashlib.sha256()
    count = 0
    reformat = {"threads": 1} if int(av.__version__.split(".")[0]) >= 17 else {}
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if (
            stream.average_rate != 50
            or (stream.width, stream.height) != (640, 360)
            or stream.codec_context.name != "h264"
        ):
            raise ValueError(f"Expected H.264 640x360 video at 50 Hz: {path}")
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts * frame.time_base != Fraction(count, 50):
                raise ValueError(
                    f"Invalid presentation timestamp: {path}, frame {count}"
                )
            pixels = frame.to_ndarray(format="rgb24", **reformat)
            if pixels.shape != (360, 640, 3) or frame.is_corrupt:
                raise ValueError(f"Invalid decoded RGB frame: {path}")
            if np.max(pixels) <= 1:
                raise ValueError(f"Black camera frame: {path}, frame {count}")
            decoded.update(pixels.tobytes())
            count += 1
    if count != steps + 1:
        raise ValueError(f"Expected {steps + 1} frames, got {count}: {path}")
    return {
        "path": str(path),
        "frames": count,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_sha256": decoded.hexdigest(),
    }


def validate_run(root: Path, config: EvalConfig) -> dict:
    from theta_bench.cli.public_eval import ENV_IDS

    if config.env_id not in ENV_IDS or not config.save_video:
        raise ValueError("Validation requires a canonical environment and four videos")
    expected_config = {**asdict(config), "eval_dir": str(root)}
    if json.loads((root / "evaluation_config.json").read_text()) != expected_config:
        raise ValueError("Recorded evaluation configuration differs from the request")
    result = json.loads((root / "evaluation_result.json").read_text())
    seeds = list(
        range(config.episode_start, config.episode_start + config.num_episodes)
    )
    expected = {f"episode_{seed}" for seed in seeds}
    stats = result.get("stats", {})
    if (
        result.get("status") != "completed"
        or result.get("episodes") != len(expected)
        or set(stats) != expected
        or any(type(value) is not bool for value in stats.values())
        or result.get("success_rate") != sum(stats.values()) / len(expected)
    ):
        raise ValueError(
            "Evaluation result does not cover the requested seeds and outcomes"
        )
    if {path.stem for path in (root / "summaries").glob("*.json")} != expected:
        raise ValueError("Episode summaries do not match the requested seed set")
    videos = []
    for episode in sorted(expected):
        summary = json.loads((root / "summaries" / f"{episode}.json").read_text())
        steps = summary.get("step")
        if (
            summary.get("episode") != episode
            or summary.get("success") is not stats[episode]
            or type(steps) is not int
            or not 1 <= steps <= config.max_episode_steps
            or not (
                summary.get("terminated") is True or summary.get("truncated") is True
            )
            or summary.get("outcome") == "agent_exhausted"
        ):
            raise ValueError(f"Invalid completed episode summary: {episode}")
        suffix = "success" if stats[episode] else "failed"
        folder = root / "videos" / episode
        paths = [folder / f"{camera}_{suffix}.mp4" for camera in CAMERAS]
        if set(folder.glob("*.mp4")) != set(paths):
            raise ValueError(f"Expected all four camera recordings: {episode}")
        group = [validate_video(path, steps) for path in paths]
        if len({video["decoded_sha256"] for video in group}) != 4:
            raise ValueError(f"Duplicated camera recordings: {episode}")
        videos.extend(group)
    policy_path = root / "policy_info.json"
    policy_info = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    return {
        "protocol": "theta_canonical_evaluation_v1",
        "status": "validated",
        "reportable": (
            config.num_episodes == 100
            and config.episode_start == 10000
            and config.max_episode_steps == 3000
            and config.split == "test"
            and config.data_format == "seeds"
            and config.controller == "decoupled_wbc"
            and config.sim_mode == "mujoco"
            and config.reasoner is None
            and config.instruction_override is None
            and config.success_criteria is None
            and policy_info.get("diagnostic") is not True
        ),
        "env_id": config.env_id,
        "configuration": expected_config,
        "policy_info": policy_info,
        "episodes": len(stats),
        "seeds": seeds,
        "successes": sum(stats.values()),
        "success_rate": sum(stats.values()) / len(stats),
        "control_hz": 50,
        "action_schema": "decoupled_v1",
        "videos": videos,
        "source_sha256": source_identity(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("mujoco", "numpy", "av")},
        "execution_environment": {
            key: os.environ.get(key)
            for key in (
                "MUJOCO_GL",
                "CUDA_VISIBLE_DEVICES",
                "MUJOCO_EGL_DEVICE_ID",
                "THETA_BENCH_POLICY_DEVICE",
                "LIBGL_ALWAYS_SOFTWARE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }
