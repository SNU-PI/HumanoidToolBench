"""Canonical evaluations must use the same complete collision geometry."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from theta_bench.assets import evaluation


def _manifest(tmp_path: Path) -> dict:
    package = tmp_path / "objects/objaverse/tool"
    package.mkdir(parents=True)
    files = {}
    for name in ("tool_visual.obj", "tool_collider0.obj", "tool_collider1.obj"):
        content = f"mesh:{name}".encode()
        (package / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    return {
        "repository": "allenai/molmospaces",
        "revision": "pinned",
        "collision_mode": "shipped",
        "assets": {"tool": {"files": files}},
        "wbc_weights": {},
    }


def test_canonical_colliders_ignore_local_decomposition(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(evaluation, "load_manifest", lambda: manifest)
    package = tmp_path / "objects/objaverse/tool"
    (package / "tool_acd0.obj").write_text("different geometry")
    colliders = evaluation.canonical_colliders(package, "tool")
    assert [path.name for path in colliders] == [
        "tool_collider0.obj",
        "tool_collider1.obj",
    ]


def test_missing_collider_does_not_fall_back_to_coacd(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(evaluation, "load_manifest", lambda: manifest)
    package = tmp_path / "objects/objaverse/tool"
    (package / "tool_acd0.obj").write_text("different geometry")
    (package / "tool_collider1.obj").unlink()
    with pytest.raises(evaluation.EvaluationAssetError, match="Missing canonical"):
        evaluation.canonical_colliders(package, "tool")


def test_checker_detects_changed_geometry(tmp_path):
    manifest = _manifest(tmp_path)
    receipt = evaluation.verify_evaluation_assets(tmp_path, manifest=manifest)
    assert receipt["verified_files"] == 3
    (tmp_path / "objects/objaverse/tool/tool_collider0.obj").write_text("changed")
    with pytest.raises(evaluation.EvaluationAssetError, match="missing or changed"):
        evaluation.verify_evaluation_assets(tmp_path, manifest=manifest)


def test_checker_rejects_unhydrated_controller_weight(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["wbc_weights"]["Balance.onnx"] = hashlib.sha256(b"model").hexdigest()
    (tmp_path / "Balance.onnx").write_text("version https://git-lfs.github.com/spec/v1")
    with pytest.raises(evaluation.EvaluationAssetError, match="Balance.onnx"):
        evaluation.verify_evaluation_assets(
            tmp_path, repo_root=tmp_path, manifest=manifest
        )


def test_manifest_covers_exact_canonical_pools():
    from theta_bench.assets.distractors import DISTRACTOR_POOL
    from theta_bench.assets.tools import ASSET_POOLS

    required = {uid for pool in ASSET_POOLS.values() for uid in pool} | set(
        DISTRACTOR_POOL
    )
    manifest = evaluation.load_manifest()
    assert set(manifest["assets"]) == required
    assert len(manifest["revision"]) == 40
    assert len(manifest["wbc_weights"]) == 2
    for uid, asset in manifest["assets"].items():
        assert f"{uid}_visual.obj" in asset["files"]
        assert any(name.startswith(f"{uid}_collider") for name in asset["files"])


def test_checker_rejects_modified_pinned_controller(tmp_path):
    manifest = _manifest(tmp_path)
    checkout = tmp_path / "controller"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    controller = checkout / "policy.py"
    controller.write_text("original controller")
    subprocess.run(["git", "-C", str(checkout), "add", "policy.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "test: initialize controller fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest["submodules"] = {
        "controller": {"url": "https://example.invalid", "revision": revision}
    }
    receipt = evaluation.verify_evaluation_assets(
        tmp_path, repo_root=tmp_path, manifest=manifest
    )
    assert receipt["submodules"]["controller"]["tracked_files_clean"]
    controller.write_text("changed controller")
    with pytest.raises(
        evaluation.EvaluationAssetError, match="changes to pinned controller"
    ):
        evaluation.verify_evaluation_assets(
            tmp_path, repo_root=tmp_path, manifest=manifest
        )


def test_downloader_checks_all_hashes_before_writing(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/fetch_evaluation_assets.py"
    spec = importlib.util.spec_from_file_location("fetch_evaluation_assets", script)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w") as archive:
        member = tarfile.TarInfo("tool/tool_collider0.obj")
        content = b"changed geometry"
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(
        downloader, "urlopen", lambda *args, **kwargs: io.BytesIO(b"zstd")
    )
    monkeypatch.setattr(
        downloader.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=packed.getvalue()),
    )
    asset = {
        "offset": 0,
        "size": 4,
        "shard": "shard.tar",
        "files": {"tool_collider0.obj": hashlib.sha256(b"original").hexdigest()},
    }
    with pytest.raises(RuntimeError, match="Pinned asset hash mismatch"):
        downloader.fetch_asset("tool", asset, tmp_path, "https://example.invalid")
    assert not (tmp_path / "objects").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_downloader_requires_every_asset(tmp_path, monkeypatch, capsys, fail):
    script = Path(__file__).resolve().parents[1] / "scripts/fetch_evaluation_assets.py"
    spec = importlib.util.spec_from_file_location("fetch_evaluation_assets", script)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    manifest = {
        "repository": "example",
        "revision": "pinned",
        "assets": {str(index): {} for index in range(12)},
    }
    checked = []

    def fetch(uid, asset, destination, base_url):
        checked.append(uid)
        if fail and uid == "3":
            raise RuntimeError("asset mismatch")
        return True

    monkeypatch.setattr(downloader, "load_manifest", lambda: manifest)
    monkeypatch.setattr(downloader, "fetch_asset", fetch)
    monkeypatch.setenv("THETA_BENCH_MS_ASSETS", str(tmp_path))
    if fail:
        with pytest.raises(RuntimeError, match="asset mismatch"):
            downloader.main()
        assert "12 verified" not in capsys.readouterr().out
    else:
        assert downloader.main() == 0
        assert "12 downloaded, 12 verified" in capsys.readouterr().out
    assert set(checked) == set(manifest["assets"])
