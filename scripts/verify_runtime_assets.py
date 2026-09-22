#!/usr/bin/env python3
"""Fail fast when packaged Toolbench assets were not hydrated by Git LFS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
DEFAULT_ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "theta_bench"
    / "resources"
    / "benchmark_assets"
)
LFS_FIX = (
    "Install Git LFS and hydrate the runtime assets:\n"
    "  git lfs install\n"
    '  git lfs pull --include="src/theta_bench/resources/benchmark_assets/**"'
)


class RuntimeAssetError(RuntimeError):
    """The packaged Toolbench runtime assets are missing or incomplete."""


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_HEADER)).startswith(LFS_POINTER_HEADER)


def _resolve_inside(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeAssetError(f"{label} must be a non-empty relative path")

    value = Path(relative)
    if value.is_absolute():
        raise RuntimeAssetError(f"{label} must be relative, got {relative!r}")

    resolved_root = root.resolve()
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeAssetError(f"{label} escapes the asset directory: {relative!r}")
    return resolved


def verify_runtime_assets(
    asset_root: Path | str = DEFAULT_ASSET_ROOT,
) -> tuple[int, int]:
    """Validate hydration and catalog references, returning file and entry counts."""

    root = Path(asset_root)
    if not root.is_dir():
        raise RuntimeAssetError(
            f"runtime asset directory is missing: {root}\n{LFS_FIX}"
        )

    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeAssetError(f"runtime asset directory is empty: {root}\n{LFS_FIX}")

    pointers: list[Path] = []
    unreadable: list[tuple[Path, OSError]] = []
    for path in files:
        try:
            if _is_lfs_pointer(path):
                pointers.append(path)
        except OSError as exc:
            unreadable.append((path, exc))

    if unreadable:
        examples = "\n".join(f"  {path}: {exc}" for path, exc in unreadable[:10])
        raise RuntimeAssetError(f"runtime assets could not be read:\n{examples}")

    if pointers:
        examples = "\n".join(f"  {path.relative_to(root)}" for path in pointers[:10])
        remainder = len(pointers) - 10
        if remainder > 0:
            examples += f"\n  ... and {remainder} more"
        raise RuntimeAssetError(
            f"found {len(pointers)} Git LFS pointer file(s) instead of asset data:\n"
            f"{examples}\n{LFS_FIX}"
        )

    catalog_path = root / "catalog.json"
    if not catalog_path.is_file():
        raise RuntimeAssetError(f"runtime asset catalog is missing: {catalog_path}")

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAssetError(f"runtime asset catalog is invalid: {exc}") from exc

    if not isinstance(catalog, list) or not catalog:
        raise RuntimeAssetError("runtime asset catalog must be a non-empty JSON list")

    missing: list[str] = []
    for index, entry in enumerate(catalog):
        if not isinstance(entry, dict):
            raise RuntimeAssetError(f"catalog entry {index} must be a JSON object")
        uid = entry.get("uid", f"entry {index}")

        directory = _resolve_inside(
            root.parent,
            entry.get("dir"),
            label=f"catalog entry {uid!r} dir",
        )
        if not directory.is_relative_to(root.resolve()):
            raise RuntimeAssetError(
                f"catalog entry {uid!r} dir is outside benchmark_assets"
            )

        mjcf = _resolve_inside(
            directory,
            entry.get("mjcf"),
            label=f"catalog entry {uid!r} mjcf",
        )
        if not mjcf.is_file():
            missing.append(str(mjcf.relative_to(root)))

        references = list(entry.get("collision_meshes_mujoco") or [])
        references.extend(
            value for value in (entry.get("visual_mesh"), entry.get("texture")) if value
        )
        for reference in references:
            path = _resolve_inside(
                root,
                reference,
                label=f"catalog entry {uid!r} asset",
            )
            if not path.is_file():
                missing.append(str(path.relative_to(root)))

    if missing:
        unique = sorted(set(missing))
        examples = "\n".join(f"  {path}" for path in unique[:10])
        remainder = len(unique) - 10
        if remainder > 0:
            examples += f"\n  ... and {remainder} more"
        raise RuntimeAssetError(
            f"catalog references {len(unique)} missing runtime asset(s):\n"
            f"{examples}\n{LFS_FIX}"
        )

    return len(files), len(catalog)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="benchmark_assets directory to validate",
    )
    args = parser.parse_args(argv)

    try:
        file_count, entry_count = verify_runtime_assets(args.asset_root)
    except RuntimeAssetError as exc:
        print(f"[runtime-assets] ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"[runtime-assets] verified {file_count} files "
        f"for {entry_count} catalog entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
