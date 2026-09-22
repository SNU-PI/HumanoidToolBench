"""Regression checks for installs on headless machines."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

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
