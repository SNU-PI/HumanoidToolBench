"""Hub resolution pins downloads and excludes unrelated model artifacts."""

import json
from types import SimpleNamespace

import pytest

from humanoidtoolbench.policies.model_source import resolve_model


def test_local_checkpoint_is_used_without_hub_access(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    path, receipt = resolve_model(str(checkpoint))
    assert path == checkpoint
    assert receipt == {"model_source": str(checkpoint)}
    with pytest.raises(ValueError, match="revision"):
        resolve_model(str(checkpoint), "main")


def test_missing_local_checkpoint_is_not_treated_as_hub_id(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model(str(tmp_path / "missing"))


def test_hub_metadata_and_weights_use_one_immutable_revision(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from humanoidtoolbench.policies import pretrained

    files = [
        "run/run_config.json",
        "run/argv.txt",
        "run/dataset_statistics.json",
        "run/checkpoints/ckpt_40000/model.safetensors",
        "run/checkpoints/ckpt_40000/optimizer.pt",
        "source/untrusted.py",
        "launch.json",
        "action_contract.json",
    ]
    calls = []
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {"chunk_size": 100}}))

    def model_info(repo, revision):
        calls.append((repo, revision))
        return SimpleNamespace(
            sha="a" * 40, siblings=[SimpleNamespace(rfilename=p) for p in files]
        )

    def download(repo, name, revision):
        assert name == "run/run_config.json" and revision == "a" * 40
        return str(config)

    def snapshot(repo, revision, allow_patterns, max_workers):
        assert revision == "a" * 40 and max_workers == 4
        assert "run/checkpoints/ckpt_40000/model.safetensors" in allow_patterns
        assert not any(
            p.startswith("source/") or p.endswith("optimizer.pt")
            for p in allow_patterns
        )
        return str(tmp_path)

    monkeypatch.setattr(hub, "HfApi", lambda: SimpleNamespace(model_info=model_info))
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    monkeypatch.setattr(pretrained, "checkpoint_family", lambda config: "act")
    path, receipt = resolve_model("https://huggingface.co/lab/model", "published")
    assert path == tmp_path and receipt["model_revision"] == "a" * 40
    assert calls == [("lab/model", "published")]


def test_unsupported_hub_format_fails_before_weight_download(monkeypatch):
    import huggingface_hub as hub

    monkeypatch.setattr(
        hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *args, **kwargs: SimpleNamespace(
                sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json")]
            )
        ),
    )
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda *args, **kwargs: pytest.fail("downloaded weights"),
    )
    with pytest.raises(ValueError, match="ACT/DP"):
        resolve_model("lab/model")
