"""Tests for the Git LFS hydration guard used by bootstrap and packaging."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_runtime_assets.py"
_SPEC = importlib.util.spec_from_file_location("verify_runtime_assets", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)


def _write_catalog(root: Path, *, visual_mesh: str | None = None) -> None:
    directory = root / "stick"
    directory.mkdir(parents=True)
    (directory / "stick.xml").write_text("<mujoco/>", encoding="utf-8")
    catalog = [
        {
            "uid": "stick",
            "dir": "benchmark_assets/stick",
            "mjcf": "stick.xml",
            "collision_meshes_mujoco": [],
            "visual_mesh": visual_mesh,
            "texture": None,
        }
    ]
    (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")


def test_guard_accepts_hydrated_catalog(tmp_path: Path) -> None:
    root = tmp_path / "benchmark_assets"
    _write_catalog(root)

    assert guard.verify_runtime_assets(root) == (2, 1)


def test_guard_rejects_any_lfs_pointer(tmp_path: Path) -> None:
    root = tmp_path / "benchmark_assets"
    _write_catalog(root)
    pointer = root / "unreferenced.obj"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(guard.RuntimeAssetError, match="Git LFS pointer") as exc_info:
        guard.verify_runtime_assets(root)

    assert "git lfs pull" in str(exc_info.value)


def test_guard_rejects_missing_catalog_reference(tmp_path: Path) -> None:
    root = tmp_path / "benchmark_assets"
    _write_catalog(root, visual_mesh="stick/missing.obj")

    with pytest.raises(guard.RuntimeAssetError, match="missing runtime asset"):
        guard.verify_runtime_assets(root)


def test_rebuild_removes_packages_left_by_a_previous_release(tmp_path, monkeypatch):
    import setuptools

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    spec = importlib.util.spec_from_file_location(
        "release_setup", _SCRIPT.parents[1] / "setup.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "_verify_runtime_assets", lambda: None)

    source = tmp_path / "src/humanoidtoolbench"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# evaluation package\n")
    distribution = setuptools.Distribution(
        {"packages": ["humanoidtoolbench"], "package_dir": {"": str(source.parent)}}
    )
    distribution.script_name = str(_SCRIPT.parents[1] / "setup.py")
    command = build.VerifiedBuildPy(distribution)
    command.ensure_finalized()
    output = tmp_path / "build/lib"
    command.build_lib = str(output)
    old_module = output / "humanoidtoolbench_rl/train.py"
    old_module.parent.mkdir(parents=True)
    old_module.write_text("# obsolete private training module\n")

    command.run()

    assert not old_module.exists()
    assert (output / "humanoidtoolbench/__init__.py").read_bytes() == (
        source / "__init__.py"
    ).read_bytes()
