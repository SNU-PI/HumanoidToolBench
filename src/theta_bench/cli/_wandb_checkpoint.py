"""Recover a training W&B link from metadata belonging to one checkpoint."""

from __future__ import annotations

import re
from pathlib import Path

from theta_bench.cli.slack_alerts import (
    _read_metadata,
    _valid_wandb_url,
    wandb_url_from_log,
)


def _checkpoint_key(path: Path) -> Path:
    return path.parent if path.name == "model.safetensors" else path


def _result_matches(result: dict, checkpoint: Path, run: Path) -> bool:
    saved = result.get("checkpoint")
    if isinstance(saved, str):
        return _checkpoint_key(Path(saved).resolve()) == _checkpoint_key(checkpoint)
    if result.get("run_dir") != str(run):
        return False
    match = re.fullmatch(
        r"(?:ckpt_|step_|model_)?(\d+)(?:\.pt)?", _checkpoint_key(checkpoint).name
    )
    return match is not None and result.get("checkpoint_step") == int(match[1])


def _fastwam_log(root: Path, checkpoint: Path, run: Path) -> Path | None:
    launches = list(root.glob("launch-resume-*.json"))
    if not launches:
        return root / "train.log"
    step = re.fullmatch(r"step_(\d+)\.pt", checkpoint.name)
    if step is None:
        return None
    candidates: dict[int, list[Path]] = {}
    for path in launches:
        start = re.fullmatch(r"launch-resume-(\d+)-.+\.json", path.name)
        if start is None:
            return None
        start_step = int(start[1])
        if start_step >= int(step[1]):
            continue
        launch = _read_metadata(path)
        state = launch.get("resume_state")
        if (
            launch.get("model") != "fastwam"
            or launch.get("run_dir") != str(run)
            or not isinstance(state, dict)
            or state.get("global_step") != start_step
        ):
            return None
        log = root / path.name.replace("launch-", "train-", 1).replace(".json", ".log")
        candidates.setdefault(start_step, []).append(log)
    if not candidates:
        return root / "train.log"
    logs = candidates[max(candidates)]
    # Repeated attempts at the same step cannot identify the checkpoint producer.
    return logs[0] if len(logs) == 1 else None


def wandb_url_for_checkpoint(checkpoint: Path | str | None) -> str | None:
    """Read only the checkpoint's own training root, never other experiments."""
    if checkpoint is None:
        return None
    try:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            return None
        for root in (path, *path.parents):
            if (root / "params/agent.yaml").is_file():
                if path == root or (
                    path.parent == root and re.fullmatch(r"model_\d+\.pt", path.name)
                ):
                    with (root / "wandb_url.txt").open() as stream:
                        return _valid_wandb_url(stream.read(2048).strip())
                return None
            if not (root / "launch.json").is_file():
                continue
            launch = _read_metadata(root / "launch.json")
            if not launch:
                return None
            run_dir = launch.get("run_dir")
            if isinstance(run_dir, str):
                run = Path(run_dir).resolve()
                if not run.is_relative_to(root) or not path.is_relative_to(run):
                    return None
            else:
                native_name = {
                    "act": "act-g1",
                    "dp": "diffusion-policy-g1",
                }.get(launch.get("model"))
                parts = path.relative_to(root).parts
                if len(parts) < 2 or parts[0] != native_name:
                    return None
                run = root / parts[0] / parts[1]
            result = _read_metadata(root / "result.json")
            if not isinstance(run_dir, str) and result.get("run_dir") not in (
                None,
                str(run),
            ):
                return None
            if _result_matches(result, path, run):
                log = result.get("log")
                if isinstance(log, str):
                    log_path = Path(log).resolve()
                    if log_path.parent == root and log_path.name.startswith("train"):
                        return wandb_url_from_log(log_path)
            if launch.get("model") == "fastwam":
                return wandb_url_from_log(_fastwam_log(root, path, run))
            return wandb_url_from_log(root / "train.log")
    except (OSError, RuntimeError, TypeError, ValueError):
        # A missing or malformed link must never change evaluation behavior.
        return None
    return None
