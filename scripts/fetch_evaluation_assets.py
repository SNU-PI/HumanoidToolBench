#!/usr/bin/env python3
"""Download only the pinned canonical meshes, textures and shipped colliders."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoidtoolbench.assets.evaluation import file_matches, load_manifest  # noqa: E402


def fetch_asset(uid: str, asset: dict, destination: Path, base_url: str) -> bool:
    package = destination / "objects/objaverse" / uid
    if all(
        file_matches(package / name, digest) for name, digest in asset["files"].items()
    ):
        return False
    offset, size = asset["offset"], asset["size"]
    request = Request(
        f"{base_url}/{asset['shard']}",
        headers={"Range": f"bytes={offset}-{offset + size - 1}"},
    )
    with urlopen(request, timeout=120) as response:
        blob = response.read(size + 1)
    if len(blob) != size:
        raise RuntimeError(
            f"Incorrect HTTP range length for {uid}: {len(blob)} != {size}"
        )
    unpacked = subprocess.run(
        ["zstd", "-dc"], input=blob, capture_output=True, check=True
    ).stdout
    # Read only declared ordinary files. Never extract archive paths or links.
    with tarfile.open(fileobj=io.BytesIO(unpacked)) as archive:
        members = {
            Path(member.name).name: member
            for member in archive.getmembers()
            if member.isfile()
        }
        payloads = {}
        for name, digest in asset["files"].items():
            stream = archive.extractfile(members[name])
            assert stream is not None
            content = stream.read()
            if hashlib.sha256(content).hexdigest() != digest:
                raise RuntimeError(f"Pinned asset hash mismatch: {uid}/{name}")
            payloads[name] = content
    package.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        target = package / name
        temporary = package / f".{name}.download"
        temporary.write_bytes(content)
        temporary.replace(target)
    return True


def main() -> int:
    manifest = load_manifest()
    destination = Path(
        os.environ.get("HUMANOIDTOOLBENCH_MS_ASSETS", ROOT / "data/ms_assets")
    ).expanduser()
    base_url = (
        f"https://huggingface.co/datasets/{manifest['repository']}/resolve/"
        f"{manifest['revision']}"
    )
    downloaded = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {
            executor.submit(fetch_asset, uid, asset, destination, base_url): uid
            for uid, asset in manifest["assets"].items()
        }
        for future in as_completed(pending):
            if future.result():
                downloaded += 1
                print(f"[evaluation-assets] downloaded {pending[future]}", flush=True)
    print(
        f"[evaluation-assets] {downloaded} downloaded, {len(manifest['assets'])} verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
