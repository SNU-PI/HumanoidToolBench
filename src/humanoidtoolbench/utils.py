"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import importlib.resources as res
from importlib.resources import as_file


def resolve_res_path(rel_path=None) -> str:
    res_dir = get_res_dir()
    if not rel_path:
        return res_dir
    with as_file(res_dir / rel_path) as res_path:
        if not res_path.exists():
            raise FileNotFoundError(res_path)
    return str(res_path)


def get_res_dir() -> str:
    return res.files("humanoidtoolbench") / "resources"
