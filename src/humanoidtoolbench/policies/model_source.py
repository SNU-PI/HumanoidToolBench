"""Resolve HumanoidToolBench checkpoints from disk or a pinned Hub snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


def resolve_model(reference: str, revision: str | None = None) -> tuple[Path, dict]:
    local = Path(reference).expanduser()
    if local.exists():
        if revision:
            raise ValueError("--revision applies to Hugging Face models only")
        # Keep a Hub-cache weight symlink beside its native run configuration.
        return local.absolute(), {"model_source": str(local.absolute())}
    if reference.startswith(("/", ".", "~")) or local.suffix in {
        ".safetensors",
        ".pt",
        ".pth",
    }:
        raise FileNotFoundError(f"Local checkpoint does not exist: {reference}")
    if "://" in reference:
        url = urlparse(reference)
        if url.scheme != "https" or url.netloc != "huggingface.co":
            raise ValueError("Use a Hugging Face model ID, model URL or local path")
        reference = url.path.strip("/")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import validate_repo_id

    from humanoidtoolbench.policies.pretrained import checkpoint_family

    validate_repo_id(reference)
    info = HfApi().model_info(reference, revision=revision)
    files = {item.rfilename for item in info.siblings}
    config_name = next(
        (name for name in ("run/run_config.json", "run_config.json") if name in files),
        None,
    )
    if config_name is None:
        raise ValueError(
            "Automatic loading supports HumanoidToolBench ACT/DP checkpoints "
            "with run_config.json. "
            "See docs/PUBLIC_EVALUATION.md for other policy formats."
        )
    config_path = hf_hub_download(reference, config_name, revision=info.sha)
    checkpoint_family(json.loads(Path(config_path).read_text()))
    prefix = str(Path(config_name).parent)
    prefix = "" if prefix == "." else prefix + "/"
    selected = {
        name
        for name in files
        if name
        in {
            prefix + "run_config.json",
            prefix + "argv.txt",
            prefix + "dataset_statistics.json",
            "launch.json",
            "action_contract.json",
        }
        or (
            name.startswith(prefix + "checkpoints/ckpt_")
            and Path(name).name in {"model.safetensors", "ema_net.pth"}
        )
    }
    snapshot = snapshot_download(
        reference, revision=info.sha, allow_patterns=sorted(selected), max_workers=4
    )
    return Path(snapshot), {"model_source": reference, "model_revision": info.sha}
