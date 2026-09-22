"""Regression checks for installs on headless machines and fresh Git clones."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("conflict", [None, "opencv-python", "opencv-contrib-python-headless"])
def test_manifest_rejects_conflicting_cv2_wheels(tmp_path, monkeypatch, capsys, conflict):
    spec = importlib.util.spec_from_file_location(
        "verify_manifest", ROOT / "scripts/verify_mujoco_manifest.py"
    )
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    lock = (ROOT / "uv.lock").read_text()
    if conflict:
        lock += f'\n[[package]]\nname = "{conflict}"\nversion = "4.11.0.86"\n'
    (tmp_path / "uv.lock").write_text(lock)
    monkeypatch.setattr(manifest, "ROOT", tmp_path)

    assert manifest.main() == (1 if conflict else 0)
    if conflict:
        assert f"remove locked packages: {conflict}" in capsys.readouterr().err


def test_missing_lfs_has_an_actionable_setup_error(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/bootstrap_evaluation.sh", scripts / "bootstrap_evaluation.sh")
    (scripts / "verify_runtime_assets.py").write_text("raise SystemExit(1)\n")
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    for name, target in (("python3", sys.executable), ("dirname", shutil.which("dirname"))):
        (command_dir / name).symlink_to(target)
    for name in ("git", "uv", "zstd"):
        command = command_dir / name
        command.write_text("#!/bin/sh\nexit 1\n")
        command.chmod(0o755)

    result = subprocess.run(
        [shutil.which("bash"), str(scripts / "bootstrap_evaluation.sh"), "--install"],
        env={**os.environ, "PATH": str(command_dir)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Git LFS is required" in result.stderr
    assert "https://git-lfs.com/" in result.stderr
    assert "rerun setup" in result.stderr
