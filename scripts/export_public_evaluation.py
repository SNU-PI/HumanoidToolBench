#!/usr/bin/env python3
"""Export a standalone evaluation source snapshot into an empty directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# New release files must be reviewed and added here explicitly. In particular,
# neither the repository root nor third-party working trees are copied.
FILES = (
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "README.ko.md",
    "MANIFEST.in",
    "setup.py",
    "pyproject.toml",
    "pytest.ini",
    "uv.lock",
    "data/README.md",
    "docs/PUBLIC_EVALUATION.md",
    "docs/assets/paper-overview.png",
    "examples/serve_policy.py",
    "scripts/bootstrap_evaluation.sh",
    "scripts/setup_evaluation.py",
    "scripts/export_public_evaluation.py",
    "scripts/fetch_evaluation_assets.py",
    "scripts/run_mujoco.sh",
    "scripts/run_mujoco_cpu.sh",
    "scripts/select_mujoco_device.py",
    "scripts/verify_evaluation_assets.py",
    "scripts/verify_mujoco_manifest.py",
    "scripts/verify_runtime_assets.py",
    "src/theta_bench/__init__.py",
    "src/theta_bench/instructions.py",
    "src/theta_bench/runtime.py",
    "src/theta_bench/scenario_names.py",
    "src/theta_bench/utils.py",
    "src/theta_bench/cli/public_eval.py",
    "src/theta_bench/cli/eval_decoupled_wbc.py",
    "src/theta_bench/cli/_eval_common.py",
    "src/theta_bench/cli/_decoupled_wbc_recording.py",
    "src/theta_bench/cli/_wandb_checkpoint.py",
    "src/theta_bench/cli/slack_alerts.py",
    "src/theta_bench/datasets/contract.py",
    "src/theta_bench/datasets/lerobot.py",
    "src/theta_bench/evals/__init__.py",
    "src/theta_bench/evals/api.py",
    "src/theta_bench/evals/tui.py",
    "src/theta_bench/evals/trajectory.py",
    "src/theta_bench/evals/public_validation.py",
    "src/theta_bench/resources/evaluation_assets.json",
    "src/theta_bench/policies/_vendor/provenance.json",
    "src/theta_bench/resources/vMaterials_2/material_split.npy",
    "tests/test_canonical_environments.py",
    "tests/test_registries.py",
    "tests/test_public_evaluation.py",
    "tests/test_pretrained_policy.py",
    "tests/test_model_source.py",
    "tests/test_export_public_evaluation.py",
    "tests/test_evaluation_assets.py",
    "tests/test_runtime_asset_guard.py",
    "tests/test_public_install.py",
    "tests/test_public_gpu_selection.py",
)
SOURCE_DIRS = tuple(
    f"src/theta_bench/{name}"
    for name in (
        "actions",
        "assets",
        "controllers",
        "core",
        "dr",
        "engines",
        "envs",
        "policies",
        "robots",
        "sensors",
        "tasks",
    )
)
ASSET_DIR = "src/theta_bench/resources/benchmark_assets"
ASSET_SUFFIXES = {".json", ".xml", ".obj", ".mtl", ".png", ".jpg", ".jpeg"}
LICENSE_NAMES = {"LICENSE", "COPYING", "NOTICE", "AUTHORS"}


def selected_files(source: Path) -> list[Path]:
    selected = {Path(name) for name in FILES}
    for name in (*SOURCE_DIRS, ASSET_DIR):
        directory = source / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Release directory is missing or a symlink: {name}")
        for path in directory.rglob("*"):
            relative = path.relative_to(source)
            if any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"Release source contains a symlink: {relative}")
            suffixes = ASSET_SUFFIXES if name == ASSET_DIR else {".py", ".c"}
            if path.is_file() and (
                path.suffix in suffixes or path.name.split(".")[0] in LICENSE_NAMES
            ):
                selected.add(relative)
    for relative in selected:
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Release file is missing or a symlink: {relative}")
        for parent in path.parents:
            if parent == source:
                break
            if parent.is_symlink():
                raise ValueError(f"Release source contains a symlink: {relative}")
    return sorted(selected)


def export_evaluation(output: Path, *, source: Path = ROOT) -> dict:
    """Copy only the allowlist and record hashes of the copied bytes."""
    source = source.resolve()
    output = output.absolute()
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("Output directory must not be a symlink")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            "Output directory must be empty; existing files are never replaced"
        )
    if source.is_relative_to(output) or any(
        output.is_relative_to(source / name) for name in (*SOURCE_DIRS, ASSET_DIR)
    ):
        raise ValueError("Output directory overlaps the release source")
    files = selected_files(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "files": {}}
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        content = destination.read_bytes()
        manifest["files"][relative.as_posix()] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest["file_count"] = len(files)
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new or empty destination directory")
    args = parser.parse_args(argv)
    try:
        manifest = export_evaluation(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(
        f"Exported {manifest['file_count']} files ({manifest['total_bytes']} bytes) "
        f"to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
