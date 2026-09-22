"""CPU scene preparation with RGB output from a single remote EGL owner."""

from __future__ import annotations

import mmap
import os
import socket
import tempfile
from multiprocessing.connection import Connection
from pathlib import Path

import mujoco
import numpy as np

from theta_bench.engines._mujoco_scene import pack_scene


class RenderClient:
    def __init__(self, address: str, model, cameras: dict[str, tuple[int, int]]):
        if model.nflex or model.nskin:
            raise ValueError(
                "The shared renderer currently supports rigid THETA scenes"
            )
        self.model = model
        self.cameras = cameras
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in cameras
        }
        if min(self.camera_ids.values()) < 0:
            raise ValueError("Shared renderer camera is missing from the model")
        self.fd, filename = tempfile.mkstemp(prefix="theta-render-", dir="/dev/shm")
        self.path = Path(filename)
        total = sum(width * height * 3 for width, height in cameras.values())
        os.ftruncate(self.fd, total)
        self.memory = mmap.mmap(self.fd, total)
        self.frames = {}
        offset = 0
        for name, (width, height) in cameras.items():
            self.frames[name] = np.ndarray(
                (height, width, 3), dtype=np.uint8, buffer=self.memory, offset=offset
            )
            offset += width * height * 3
        self.connection = None
        try:
            endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            endpoint.connect(address)
            self.connection = Connection(endpoint.detach())
            self.connection.send(
                {
                    "op": "open",
                    "version": mujoco.__version__,
                    "model": model,
                    "cameras": cameras,
                    "output": str(self.path),
                }
            )
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(60):
            raise TimeoutError("MuJoCo render owner did not respond within 60 seconds")
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "MuJoCo render owner failed"))
        return response

    def render(self, data, option) -> dict[str, np.ndarray]:
        packets = {}
        for name, camera_id in self.camera_ids.items():
            self.camera.fixedcamid = camera_id
            mujoco.mjv_updateScene(
                self.model,
                data,
                option,
                None,
                self.camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            packets[name] = pack_scene(self.scene)
        self.connection.send({"op": "render", "scenes": packets})
        self._receive()
        # Observations retain their native lifetime after the next render call.
        return {name: frame.copy() for name, frame in self.frames.items()}

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.frames.clear()
        if self.memory is not None:
            self.memory.close()
            self.memory = None
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def owner_health(address: str) -> dict:
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.settimeout(5)
    endpoint.connect(address)
    endpoint.settimeout(None)
    with Connection(endpoint.detach()) as connection:
        connection.send({"op": "health"})
        # Session creation shares the owner's request queue with health checks.
        if not connection.poll(60):
            raise TimeoutError("MuJoCo render owner health check timed out")
        return connection.recv()
