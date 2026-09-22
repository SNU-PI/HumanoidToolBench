"""Pinned assets and collision geometry for the public canonical benchmark."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "resources/evaluation_assets.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


class EvaluationAssetError(RuntimeError):
    """An evaluation asset is missing or differs from the release manifest."""


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def canonical_colliders(package: Path, uid: str) -> list[Path] | None:
    """Select every frozen shipped collider, without a local CoACD fallback."""
    asset = load_manifest()["assets"].get(uid)
    if asset is None:
        return None
    colliders = sorted(
        package / name
        for name in asset["files"]
        if name.startswith(f"{uid}_collider") and name.endswith(".obj")
    )
    if not colliders or any(not path.is_file() for path in colliders):
        raise EvaluationAssetError(
            f"Missing canonical shipped colliders for {uid}. "
            "Run scripts/bootstrap_evaluation.sh --install."
        )
    return colliders


def file_matches(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def verify_evaluation_assets(
    ms_assets: Path | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check all canonical meshes, textures, colliders and both WBC weights."""
    if ms_assets is None:
        from theta_bench.assets.tools import MS_ASSETS_DIR

        ms_assets = MS_ASSETS_DIR
    release_manifest = manifest is None
    manifest = load_manifest() if release_manifest else manifest
    if manifest["collision_mode"] != "shipped":
        raise EvaluationAssetError("Canonical evaluation requires shipped colliders")
    verified_dependencies = {}
    for relative, dependency in manifest.get("submodules", {}).items():
        checkout = repo_root / relative
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if revision.returncode or revision.stdout.strip() != dependency["revision"]:
            raise EvaluationAssetError(
                f"{relative} is missing or is not at {dependency['revision']}. "
                "Run scripts/bootstrap_evaluation.sh --install."
            )
        clean = subprocess.run(
            ["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--"],
            capture_output=True,
            check=False,
        )
        if clean.returncode:
            raise EvaluationAssetError(
                f"{relative} has changes to pinned controller files. "
                "Use a clean checkout for canonical evaluation."
            )
        verified_dependencies[relative] = {
            "url": dependency["url"],
            "revision": revision.stdout.strip(),
            "tracked_files_clean": True,
        }
    expected = {
        ms_assets / "objects/objaverse" / uid / name: digest
        for uid, asset in manifest["assets"].items()
        for name, digest in asset["files"].items()
    }
    expected.update(
        {repo_root / name: digest for name, digest in manifest["wbc_weights"].items()}
    )
    invalid = [
        str(path) for path, digest in expected.items() if not file_matches(path, digest)
    ]
    if invalid:
        examples = "\n".join(f"  {name}" for name in invalid[:10])
        raise EvaluationAssetError(
            f"{len(invalid)} missing or changed evaluation asset(s):\n{examples}\n"
            "Run scripts/bootstrap_evaluation.sh --install."
        )
    return {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "collision_mode": manifest["collision_mode"],
        "assets": len(manifest["assets"]),
        "verified_files": len(expected),
        "manifest_sha256": (
            hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
            if release_manifest
            else hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
        ),
        "submodules": verified_dependencies,
        "wbc_weights": manifest["wbc_weights"],
    }
