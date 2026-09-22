#!/usr/bin/env python3
"""Keep the default install MuJoCo-only with headless OpenCV."""

from __future__ import annotations

import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TORCH_PACKAGES = ("torch", "torchvision")
TORCH_INDEXES = {
    "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
    "pytorch-cu128": "https://download.pytorch.org/whl/cu128",
}


def dependency_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split(";", 1)[0].split("=", 1)[0].strip()


def main() -> int:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    base_names = {dependency_name(item) for item in project["dependencies"]}
    optional = project.get("optional-dependencies", {})

    errors: list[str] = []
    if "isaacsim" in base_names:
        errors.append("isaacsim must not be a default dependency")
    if "openpi-client" in base_names:
        errors.append("openpi-client must remain opt-in for the core smoke test")
    if "isaac" in optional or any(
        "isaacsim" in item for items in optional.values() for item in items
    ):
        errors.append(
            "Isaac packages must not appear in any extra (the Isaac backend was removed)"
        )

    # Both Torch flavours stay declared so either runtime mode is installable
    # without editing the manifest: GPU by default, CPU under THETA_BENCH_FORCE_CPU=1.
    indexes = config.get("tool", {}).get("uv", {}).get("index", [])
    for name, url in TORCH_INDEXES.items():
        matching = [index for index in indexes if index.get("name") == name]
        if len(matching) != 1:
            errors.append(f"exactly one {name} index is required")
        elif matching[0].get("url") != url:
            errors.append(f"{name} index has an unexpected URL")

    sources = config.get("tool", {}).get("uv", {}).get("sources", {})
    pinned = {
        package: sources.get(package, {}).get("index") for package in TORCH_PACKAGES
    }
    for package, index_name in pinned.items():
        if index_name not in TORCH_INDEXES:
            errors.append(
                f"{package} must be pinned to one of: {', '.join(sorted(TORCH_INDEXES))}"
            )
    if len(set(pinned.values())) != 1:
        errors.append(f"torch and torchvision disagree on their index: {pinned}")

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {package["name"] for package in lock["package"]}
    conflicting_opencv = packages & {
        "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"
    }
    if conflicting_opencv:
        errors.append(
            "only opencv-python-headless may provide cv2; remove locked packages: "
            + ", ".join(sorted(conflicting_opencv))
        )
    if "opencv-python-headless" not in packages:
        errors.append("opencv-python-headless must be included in uv.lock")

    if errors:
        for error in errors:
            print(f"[mujoco-manifest] ERROR: {error}", file=sys.stderr)
        return 1

    print("[mujoco-manifest] PASS: default dependencies contain no Isaac package")
    print(
        "[mujoco-manifest] PASS: Torch sources resolve to "
        f"{pinned['torch']} ({TORCH_INDEXES[pinned['torch']]})"
    )
    print("[mujoco-manifest] PASS: only headless OpenCV is locked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
