"""Public model discovery, native normalization and trained-state loading."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanoidtoolbench.policies import pretrained


def config(family="act"):
    model = {
        "language_conditioning": True,
        "language_model_id": pretrained._language.MODEL_ID,
        "language_revision": pretrained._language.REVISION,
        "language_projection_dim": 768,
        "action_dim": 36,
    }
    if family == "act":
        model.update(
            n_obs_steps=1,
            chunk_size=2,
            n_action_steps=2,
            state_dim=36,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
            n_vae_encoder_layers=1,
            temporal_ensemble_coeff=None,
        )
    else:
        model.update(
            num_diffusion_iters=100, action_chunk_size=16, obs_dim=36, obs_horizon=1
        )
    return {
        "model": model,
        "data": {
            "transform": {
                "field": {
                    "action_norm_type": "bounds",
                    "normalize_state": True,
                    "pad_state_dim": 36,
                    "use_norm_mask": False,
                    "action_min": [-2.0] * 36,
                    "action_max": [2.0] * 36,
                    "state_min": [-1.0] * 32 + [0.0] * 4,
                    "state_max": [1.0] * 32 + [0.0] * 4,
                },
                "model": {
                    "resize": {"size": [256, 480]},
                    "center_crop": {"size": [224, 224]},
                    "normalize": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            }
        },
    }


def write_layout(root, family="act"):
    run = root / "run"
    (run / "checkpoints/ckpt_10").mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps(config(family)))
    weight = run / "checkpoints/ckpt_10/model.safetensors"
    weight.write_bytes(b"discovery fixture")
    return run, weight


@pytest.mark.parametrize("kind", ["act", "dp"])
def test_family_and_unsupported_configuration(kind):
    assert pretrained.checkpoint_family(config(kind)) == kind
    with pytest.raises(ValueError, match="ACT and DP"):
        pretrained.checkpoint_family({"model": {"model_type": "generic"}})


@pytest.mark.parametrize("selection", ["repository", "run", "checkpoint", "file"])
def test_native_paths_and_hf_weight_symlink(tmp_path, selection):
    run, weight = write_layout(tmp_path)
    blob = tmp_path / "blobs/weight"
    blob.parent.mkdir()
    weight.rename(blob)
    weight.symlink_to(blob)
    paths = {
        "repository": tmp_path,
        "run": run,
        "checkpoint": weight.parent,
        "file": weight,
    }
    found_run, found_weight, found_config = pretrained.checkpoint_files(
        paths[selection]
    )
    assert found_run == run
    assert found_weight == weight
    assert found_config == config()


def test_latest_numeric_checkpoint_and_explicit_selection(tmp_path):
    run, weight = write_layout(tmp_path)
    older = run / "checkpoints/ckpt_9/model.safetensors"
    older.parent.mkdir()
    older.write_bytes(b"older")
    assert pretrained.checkpoint_files(run)[1] == weight
    assert pretrained.checkpoint_files(older)[1] == older


def test_dp_prefers_native_ema_weights(tmp_path):
    run, weight = write_layout(tmp_path, "dp")
    ema = weight.with_name("ema_net.pth")
    ema.write_bytes(b"EMA")
    assert pretrained.checkpoint_files(run)[1] == ema
    assert pretrained.checkpoint_files(weight)[1] == weight


def test_bounds_constant_dimensions_and_clipping():
    field = config()["data"]["transform"]["field"]
    field["state_min"][0] = field["state_max"][0] = 3.0
    bounds = pretrained.Bounds(field)
    state = np.full((1, 32), 3.0, dtype=np.float32)
    expected = np.ones((1, 36), np.float32)
    expected[0, [0, 32, 33, 34, 35]] = 0
    np.testing.assert_array_equal(bounds.state(state), expected)
    np.testing.assert_array_equal(
        bounds.action(np.zeros((2, 36), np.float32)), np.zeros((2, 36))
    )


def test_image_pixels_match_native_nearest_resize_and_center_crop():
    from PIL import Image

    policy = pretrained.PretrainedPolicy(
        "act",
        SimpleNamespace(config=SimpleNamespace(chunk_size=2)),
        config(),
        "cpu",
        None,
        None,
    )
    image = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    y = np.floor((np.arange(224) + 16) * 360 / 256).astype(int)
    x = np.floor((np.arange(224) + 128) * 640 / 480).astype(int)
    expected = image[y[:, None], x[None, :]].astype(np.float32).transpose(2, 0, 1)
    expected *= np.float32(1 / 255)
    expected -= np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
    expected /= np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]
    np.testing.assert_array_equal(
        policy.transform(Image.fromarray(image)).numpy(), expected
    )


@pytest.mark.parametrize(
    "field,value",
    [("action_norm_type", "mean_std"), ("pad_state_dim", 32), ("use_norm_mask", True)],
)
def test_reject_changed_normalization_contract(field, value):
    values = config()["data"]["transform"]["field"]
    values[field] = value
    with pytest.raises(ValueError, match="contract"):
        pretrained.Bounds(values)


@pytest.fixture
def trained_act(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from humanoidtoolbench.policies._vendor.act import ACTConfig, ACTPolicy

    torch.set_num_threads(2)
    run, weight = write_layout(tmp_path)
    values = config()["model"].copy()
    for name in tuple(values):
        if name.startswith("language_"):
            del values[name]
    model = ACTPolicy(
        ACTConfig(**values, language_dim=768, pretrained_backbone_weights=None)
    ).eval()
    save_file(model.state_dict(), weight)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pretrained._language.FrozenCLIPInstructionCache, "_load", lambda self: None
    )
    monkeypatch.setattr(
        pretrained._language.FrozenCLIPInstructionCache,
        "encode",
        lambda self, texts, device: torch.zeros((len(texts), 768), device=device),
    )
    return run, weight, model


def test_strict_act_loading_and_actual_action(trained_act):
    from torchvision.transforms import v2

    run, weight, model = trained_act
    policy = pretrained.load_policy(run, "cpu")
    assert policy.transform.transforms[2].interpolation == v2.InterpolationMode.NEAREST
    request = {
        "image": {"egocentric": np.zeros((360, 640, 3), np.uint8)},
        "state": {"states": np.zeros((1, 32), np.float32)},
        "instruction": "Move the ball.",
    }
    action = policy(request)
    assert action.shape == (2, 36)
    assert np.isfinite(action).all()
    np.testing.assert_array_equal(action, policy(request))
    assert policy.metadata["checkpoint_sha256"] == pretrained._sha(weight)
    assert policy.metadata["diagnostic"] is False


def test_missing_trained_tensor_is_rejected(trained_act):
    from safetensors.torch import save_file

    run, weight, model = trained_act
    tensors = model.state_dict()
    tensors.pop(next(iter(tensors)))
    save_file(tensors, weight)
    with pytest.raises(RuntimeError, match="Missing key"):
        pretrained.load_policy(run, "cpu")


def test_hardware_checkpoint_contract_is_rejected(trained_act):
    run, _, _ = trained_act
    (run.parent / "launch.json").write_text(
        json.dumps({"action_schema": "real_g1_executed_joint_targets_v1"})
    )
    with pytest.raises(ValueError, match="hardware"):
        pretrained.load_policy(run, "cpu")


def test_vendor_sources_match_their_recorded_provenance():
    import hashlib
    import re

    root = Path(pretrained.__file__).parent / "_vendor"
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["repository"] == (
        "https://github.com/physical-superintelligence-lab/Psi0"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["revision"])
    vendored = {path.name for path in root.glob("*.py")} - {"__init__.py"}
    assert set(provenance["files"]) == vendored
    for filename, record in provenance["files"].items():
        assert record["upstream_path"].endswith("/" + filename)
        assert re.fullmatch(r"[0-9a-f]{64}", record["upstream_sha256"])
        assert record["modifications"]
        # The shipped file is exactly the one recorded, byte for byte.
        assert (
            hashlib.sha256((root / filename).read_bytes()).hexdigest()
            == record["vendored_sha256"]
        )
