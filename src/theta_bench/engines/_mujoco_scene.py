"""Version-matched byte transport for native rigid MuJoCo render scenes."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import mujoco


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix(".c")
    include = Path(mujoco.__file__).parent / "include"
    header = include / "mujoco/mjvisualize.h"
    identity = hashlib.sha256(
        source.read_bytes() + header.read_bytes() + mujoco.__version__.encode()
    ).hexdigest()
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "theta-mujoco-scene"
    )
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{identity}.so"
    if not target.is_file():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / "scene.so"
            subprocess.run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-std=c99",
                    "-I",
                    str(include),
                    str(source),
                    "-o",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(output, target)
    library = ctypes.CDLL(str(target))
    library.theta_scene_flags_offset.argtypes = []
    library.theta_scene_flags_offset.restype = ctypes.c_size_t
    library.theta_scene_size.argtypes = [ctypes.c_void_p]
    library.theta_scene_size.restype = ctypes.c_size_t
    library.theta_pack_scene.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.theta_unpack_scene.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.theta_pack_scene.restype = library.theta_unpack_scene.restype = ctypes.c_int
    return library


def pack_scene(scene: mujoco.MjvScene) -> bytes:
    library = _library()
    address = scene.flags.ctypes.data - library.theta_scene_flags_offset()
    size = library.theta_scene_size(address)
    if not size:
        raise ValueError(
            "Render-owner scene transport requires a rigid scene without flex or skin"
        )
    buffer = ctypes.create_string_buffer(size)
    if library.theta_pack_scene(address, buffer, size):
        raise ValueError("Could not serialize the native render scene")
    return buffer.raw


def unpack_scene(scene: mujoco.MjvScene, packet: bytes) -> None:
    library = _library()
    address = scene.flags.ctypes.data - library.theta_scene_flags_offset()
    if library.theta_unpack_scene(address, packet, len(packet)):
        raise ValueError("Render scene packet does not match the allocated rigid scene")
