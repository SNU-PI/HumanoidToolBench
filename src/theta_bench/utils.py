"""
THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import enum
import importlib.resources as res
import os
import re
import zipfile
from importlib.resources import as_file
from json import JSONEncoder

import numpy as np
from huggingface_hub import snapshot_download


def resolve_res_path(rel_path=None) -> str:
    res_dir = get_res_dir()
    if not rel_path:
        return res_dir
    with as_file(res_dir / rel_path) as res_path:
        if not res_path.exists():
            raise FileNotFoundError(res_path)
    return str(res_path)


def get_res_dir() -> str:
    return res.files("theta_bench") / "resources"


def get_data_dir() -> str:
    return res.files("theta_bench").parent.parent / "data"  # type: ignore


def _parse_zip_file_from_rel_path(rel_path: str) -> str:
    rel_path = rel_path.rstrip("/")

    pattern = r"^vMaterials_2/(?:[^/]+/)*[^/]+\.mdl$"
    match = re.match(pattern, rel_path)
    if match:
        return "vMaterials_2.zip"

    pattern = r"^robots/([^/]+)/.+\.(?:yml|xml|usd|usda|usdc)$"
    match = re.match(pattern, rel_path)
    if match:
        robot_name = match.group(1)
        return f"robots_{robot_name}.zip"

    pattern = r"^assets/([^/]+)/([^/]+)/.*\.(xml|usd)$"

    match = re.search(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        asset_name = match.group(2)
        return f"assets_{asset_category}.zip"

    pattern = r"^assets/([^/]+)/([^/]+)/([^/]+)/(.+)$"

    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        dex_category = match.group(2)
        hand_uid = match.group(3)
        asset_label = match.group(4)

        return f"assets_{asset_category}_{dex_category}_{hand_uid}_{asset_label}.zip"

    pattern = r"^assets/([^/]+)/([^/]+)$"
    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        asset_name = match.group(2)
        return f"assets_{asset_category}_{asset_name}.zip"

    pattern = r"^assets/([^/]+)/(.+)$"
    match = re.match(pattern, rel_path)

    if match:
        asset_category = match.group(1)
        asset_name = match.group(2).replace("/", "_")
        return f"assets_{asset_category}.zip"

    raise FileNotFoundError(f"{rel_path}")


def resolve_data_path(
    rel_path=None, create_if_not_exist=False, auto_download=False
) -> str:
    data_dir = get_data_dir()
    if not rel_path:
        return data_dir
    with as_file(data_dir / rel_path) as res_path:
        # assert res_path.exists(), f"Data not found: {res_path}"
        if not res_path.exists():
            if not auto_download:
                raise FileNotFoundError(res_path)

            if create_if_not_exist:
                os.makedirs(res_path, exist_ok=True)
            else:
                zip_file = _parse_zip_file_from_rel_path(rel_path)
                zip_path = os.path.join(data_dir, zip_file)
                snapshot_download(
                    repo_id="USC-PSI-Lab/SIMPLE",
                    allow_patterns=[zip_file],
                    local_dir=data_dir,
                    repo_type="dataset",
                    token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
                )

                if not os.path.exists(zip_path):
                    raise FileNotFoundError(
                        f"Download did not materialize {zip_path} for {rel_path}"
                    )
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    print(f"Deleted {zip_path}")
                return str(res_path)

    return str(res_path)


def class_to_str(cls: type) -> str:
    """Get fully qualified class path: module + class name."""
    return f"{cls.__module__}.{cls.__qualname__}"


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, enum.Enum):
            return obj.value
        elif type(obj).__module__ == "mujoco._enums":
            return int(obj)
        return JSONEncoder.default(self, obj)
