"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import enum
import importlib.resources as res
from importlib.resources import as_file
from json import JSONEncoder

import numpy as np


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
