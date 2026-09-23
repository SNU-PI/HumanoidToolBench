"""Load HumanoidToolBench ACT and DP checkpoints without a training backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from humanoidtoolbench.policies import _language


def checkpoint_family(config: dict) -> str:
    """Identify supported native configurations without importing model libraries."""
    model = config.get("model", {})
    if "chunk_size" in model and "dim_model" in model:
        return "act"
    if "num_diffusion_iters" in model and "action_chunk_size" in model:
        return "dp"
    raise ValueError(
        "Direct model loading supports HumanoidToolBench ACT and DP checkpoints."
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(path: Path) -> tuple[Path, Path, dict]:
    """Accept a downloaded repository, native run, checkpoint directory or file."""
    # HF cache weights are symlinks to blobs; preserve the surrounding run layout.
    path = path.expanduser().absolute()
    selected = path if path.is_file() else None
    directory = path.parent if selected else path
    if (directory / "run/run_config.json").is_file():
        run = directory / "run"
    elif (directory / "run_config.json").is_file():
        run = directory
    elif (directory.parent.parent / "run_config.json").is_file():
        run = directory.parent.parent
        if selected is None:
            selected = directory / "model.safetensors"
    else:
        raise ValueError("Model path must include its native run_config.json.")
    config = json.loads((run / "run_config.json").read_text())
    family = checkpoint_family(config)
    if selected is None:
        checkpoints = [
            item
            for item in (run / "checkpoints").glob("ckpt_*")
            if item.is_dir() and item.name.removeprefix("ckpt_").isdigit()
        ]
        if not checkpoints:
            raise ValueError("No checkpoint found under run/checkpoints/ckpt_STEP.")
        selected = (
            max(checkpoints, key=lambda item: int(item.name[5:])) / "model.safetensors"
        )
    if (
        family == "dp"
        and not path.is_file()
        and (selected.parent / "ema_net.pth").is_file()
    ):
        selected = selected.parent / "ema_net.pth"
    if not selected.is_file():
        raise ValueError(f"Missing model weights: {selected}")
    return run, selected, config


class Bounds:
    """Min/max ("bounds") normalization as the checkpoints were trained with it.

    It follows Psi0, whose ACT and DP model code is vendored in `_vendor`,
    including its handling of state dimensions with a constant range.
    """

    def __init__(self, field: dict):
        if (
            field.get("action_norm_type") != "bounds"
            or field.get("normalize_state") is not True
            or field.get("pad_state_dim") != 36
            or field.get("use_norm_mask", False)
        ):
            raise ValueError(
                "Expected the HumanoidToolBench normalized 36D action contract."
            )
        self.values = {}
        for name in ("action_min", "action_max", "state_min", "state_max"):
            value = np.asarray(field[name], dtype=np.float32)
            if value.shape != (36,) or not np.isfinite(value).all():
                raise ValueError(f"Expected 36 finite values for {name}.")
            self.values[name] = value
        for kind in ("state", "action"):
            if (self.values[f"{kind}_max"] < self.values[f"{kind}_min"]).any():
                raise ValueError(f"Invalid {kind} normalization bounds.")

    def state(self, state):
        low, high = self.values["state_min"], self.values["state_max"].copy()
        ill = np.abs(high - low) < 1e-4 * (np.abs(high) + np.abs(low) + 1e-8)
        high[ill] = 1.0
        value = np.pad(state, ((0, 0), (0, 4)))
        value = np.where(ill, 0, (value - low) / (high - low) * 2 - 1)
        return np.clip(value, -1, 1).astype(np.float32)

    def action(self, action):
        import torch

        low, high = self.values["action_min"], self.values["action_max"]
        if torch.is_tensor(action):
            low = torch.tensor(low, dtype=action.dtype, device=action.device)
            high = torch.tensor(high, dtype=action.dtype, device=action.device)
        return 0.5 * (action + 1) * (high - low) + low


class PretrainedPolicy:
    def __init__(self, family, model, config, device, bounds, encoder):
        import torch
        from torchvision.transforms import v2

        transform = config["data"]["transform"]["model"]
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(
                    **transform["resize"], interpolation=v2.InterpolationMode.NEAREST
                ),
                v2.CenterCrop(**transform["center_crop"]),
                v2.Normalize(**transform["normalize"]),
            ]
        )
        self.family, self.model, self.device = family, model, device
        self.bounds, self.encoder = bounds, encoder
        self.horizon = (
            model.config.chunk_size if family == "act" else model.pred_horizon
        )
        self.metadata = {}

    def __call__(self, request: dict) -> np.ndarray:
        import torch
        from PIL import Image

        images = request["image"]
        if not isinstance(images, dict) or len(images) != 1:
            raise ValueError("ACT/DP expects one egocentric image.")
        image = np.asarray(next(iter(images.values())))
        state = np.asarray(request["state"]["states"], dtype=np.float32)
        if image.dtype != np.uint8 or image.shape != (360, 640, 3):
            raise ValueError("Expected a 360x640 RGB uint8 image.")
        if state.shape != (1, 32) or not np.isfinite(state).all():
            raise ValueError("Expected finite state shaped (1, 32).")
        instruction = request["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Expected a nonempty instruction.")
        with torch.inference_mode():
            image = self.transform(Image.fromarray(image))[None].to(self.device)
            state = torch.from_numpy(self.bounds.state(state)).to(self.device)
            language = self.encoder.encode([instruction], device=self.device)
            if self.family == "act":
                action = self.model.predict_action(
                    {
                        "observation.images": image[:, None],
                        "observation.state": state,
                        "language_embedding": language,
                    }
                )
            else:
                episode, index = (
                    request["humanoidtoolbench_episode_seed"],
                    request["humanoidtoolbench_request_index"],
                )
                if any(
                    type(value) is not int or value < 0 for value in (episode, index)
                ):
                    raise ValueError(
                        "Episode seed and request index must be nonnegative integers."
                    )
                seed = int.from_bytes(
                    hashlib.sha256(
                        # A historical domain prefix for the DP sampling seed. It
                        # must never change: other bytes give every DP episode
                        # different noise and so different results.
                        bytes.fromhex("74686574612d706f6c6963792d76313a")
                        + f"{episode}:{index}".encode()
                    ).digest()[:8],
                    "little",
                ) % (2**63 - 1)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                features = self.model.vision_encoder(image)[:, None]
                features = self.model.condition_features(features, language)[:, 0]
                condition = torch.cat([features, state], dim=-1)
                action = torch.randn(
                    (1, self.horizon, 36), device=self.device, generator=generator
                )
                scheduler = self.model.noise_scheduler
                scheduler.set_timesteps(self.model.num_diffusion_iters)
                steps = scheduler.timesteps.to(self.device).reshape(-1, 1)
                for step, model_step in zip(scheduler.timesteps, steps):
                    noise = self.model.noise_pred_net(
                        sample=action,
                        timestep=model_step,
                        global_cond=condition,
                    )
                    action = scheduler.step(
                        model_output=noise,
                        timestep=step,
                        sample=action,
                        generator=[generator],
                    ).prev_sample
                action = action.cpu().numpy()
            action = self.bounds.action(action)
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            result = np.asarray(action[0], dtype=np.float32)
        if result.shape != (self.horizon, 36) or not np.isfinite(result).all():
            raise ValueError("Policy returned invalid action values.")
        return result


def load_policy(checkpoint: Path, device: str) -> PretrainedPolicy:
    """Load trained tensors strictly and preserve the saved inference contract."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    run, weight, config = checkpoint_files(checkpoint)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    family = checkpoint_family(config)
    model_config = config["model"]
    if (
        model_config.get("action_dim") != 36
        or not _language.config_receipt(model_config)["enabled"]
    ):
        raise ValueError(
            "Expected a language-conditioned HumanoidToolBench 36D simulation policy."
        )
    for directory in (run, run.parent, run.parent.parent):
        launch = directory / "launch.json"
        if launch.is_file():
            metadata = json.loads(launch.read_text())
            if metadata.get("action_schema", "decoupled_v1") != "decoupled_v1":
                raise ValueError(
                    "This checkpoint uses a hardware action contract, not simulation."
                )
            break
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bounds = Bounds(config["data"]["transform"]["field"])
    if family == "act":
        from humanoidtoolbench.policies._vendor.act import ACTConfig, ACTPolicy

        names = {field.name for field in fields(ACTConfig)}
        values = {key: value for key, value in model_config.items() if key in names}
        values.update(
            language_dim=model_config["language_projection_dim"],
            pretrained_backbone_weights=None,
        )
        if (
            values.get("state_dim") != 36
            or values.get("n_obs_steps") != 1
            or type(values.get("chunk_size")) is not int
            or values["chunk_size"] < 1
            or values.get("chunk_size") != values.get("n_action_steps")
            or values.get("temporal_ensemble_coeff") is not None
        ):
            raise ValueError(
                "ACT requires one observation and full stateless action chunks."
            )
        model = ACTPolicy(ACTConfig(**values))
    else:
        from humanoidtoolbench.policies._vendor.diffusion_policy import (
            DiffusionPolicyModel,
        )

        if (
            model_config.get("obs_dim") != 36
            or model_config.get("obs_horizon") != 1
            or model_config.get("num_diffusion_iters") != 100
            or model_config.get("action_chunk_size") != 16
        ):
            raise ValueError(
                "Expected native HumanoidToolBench DP with a 16-action, "
                "100-step diffusion horizon."
            )
        model = DiffusionPolicyModel(
            vision_feature_dim=512,
            lowdim_obs_dim=36,
            action_dim=36,
            obs_horizon=1,
            pred_horizon=16,
            num_diffusion_iters=100,
            language_dim=model_config["language_projection_dim"],
        )
    tensors = (
        load_file(weight)
        if weight.suffix == ".safetensors"
        else torch.load(weight, map_location="cpu", weights_only=True)
    )
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in tensors.values()
    ):
        raise ValueError("Checkpoint contains nonfinite model weights.")
    model.load_state_dict(tensors, strict=True)
    model.to(device).eval()
    snapshot_download(
        _language.MODEL_ID,
        revision=_language.REVISION,
        allow_patterns=list(_language.CACHE_FILES),
    )
    encoder = _language.FrozenCLIPInstructionCache()
    encoder._load()
    policy = PretrainedPolicy(family, model, config, device, bounds, encoder)
    source = Path(__file__).parent
    policy.metadata = {
        "policy": f"humanoidtoolbench-{family}",
        "family": family,
        "checkpoint": str(weight),
        "checkpoint_sha256": _sha(weight),
        "config_sha256": _sha(run / "run_config.json"),
        "chunk_size": policy.horizon,
        "action_schema": "decoupled_v1",
        "state_dim": 32,
        "action_dim": 36,
        "device": device,
        "diagnostic": False,
        "source_sha256": {
            str(path.relative_to(source)): _sha(path)
            for path in (
                Path(__file__),
                source / "_language.py",
                source / "_vendor/act.py",
                source / "_vendor/diffusion_policy.py",
                source / "_vendor/provenance.json",
            )
        },
        "language": _language.language_receipt(True),
        "precision": {
            "weight_dtype": "torch.float32",
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    return policy
