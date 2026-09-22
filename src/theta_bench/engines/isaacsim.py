"""Isaac Sim as a renderer for a scene MuJoCo is simulating.

This engine deliberately implements only half of `Simulator`: it draws, and it
never advances anything. `update_layout` mirrors the compiled MuJoCo model onto
a USD stage, `step` copies the current `mjData` poses across and asks
Replicator for one frame, and `render` hands back the RGB buffers under the
same camera names `MujocoSimulator.render` uses, so the Quest streamer and the
recorder cannot tell which engine produced the images.

Isaac Sim is not a dependency of THETA(θ)-Bench. Every import below happens after
`start_simulation_app`, which is the only place allowed to fail with an
installation message.

THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from theta_bench.core.task import Task
    from theta_bench.engines.mujoco import MujocoSimulator

from theta_bench.core.simulator import Simulator
from theta_bench.engines.isaac_app import (
    _env_flag,
    close_simulation_app,
    start_simulation_app,
)
from theta_bench.utils import resolve_data_path


class _FrameNotReadyError(RuntimeError):
    """A render product has not supplied the requested RGB frame."""


class IsaacSimRenderer(Simulator):
    """Draws the MuJoCo scene with Isaac Sim's RTX renderer."""

    def __init__(self, task: Task, headless: bool = True) -> None:
        self.task = task
        self.headless = headless

        start_simulation_app(headless=headless)

        import omni.replicator.core as rep  # noqa: PLC0415 - needs the running app
        import omni.usd  # noqa: PLC0415 - needs the running app

        from theta_bench.engines.isaac_scene import IsaacSceneBuilder

        self._rep = rep
        # Real-time RTX rather than path tracing: teleop needs a frame every
        # control tick far more than it needs a converged one.
        rep.settings.set_render_rtx_realtime()

        self._texture_dir = tempfile.TemporaryDirectory(prefix="theta-bench-isaac-")
        self._scene = IsaacSceneBuilder(
            omni.usd.get_context().get_stage(), self._texture_dir.name
        )
        self._render_products: dict[str, Any] = {}
        self._annotators: dict[str, Any] = {}
        self._resolutions: dict[str, tuple[int, int]] = {}

    # -- Simulator ---------------------------------------------------------

    def update_layout(self, mujoco_sim: MujocoSimulator, **kwargs: Any) -> None:
        """Rebuild the stage for the episode MuJoCo has just compiled."""
        del kwargs
        self._scene.build(
            mujoco_sim.mjModel,
            self._camera_resolutions(),
            layout=self.task.layout,
            robot_usd=self._robot_usd(),
        )
        self._attach_render_products()
        self._scene.sync(mujoco_sim.mjData)
        if _env_flag("THETA_BENCH_ISAAC_WEBRTC", default=False):
            from isaacsim.core.utils.viewports import set_camera_view  # noqa: PLC0415
            from omni.kit.viewport.utility import get_active_viewport  # noqa: PLC0415

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Isaac WebRTC requires an active viewport")
            viewport.camera_path = "/OmniverseKit_Persp"
            # Keep the initial view inside the Toolbench room's walls.
            set_camera_view(
                eye=np.array([-1.8, -1.8, 1.8]),
                target=np.array([-0.4, 0.0, 0.8]),
                camera_prim_path=viewport.camera_path,
            )
        # Author first, render second: the products below need the camera prims
        # to have reached Hydra before they can bind to them.
        # Initial products can take several frames to become available. Keep
        # physics fixed while warming every camera, and never substitute black
        # pixels into an evaluation observation or recording.
        for attempt in range(30):
            self._rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
            try:
                self.render()
                break
            except _FrameNotReadyError:
                if attempt == 29:
                    raise

    def step(self, mujoco_sim: MujocoSimulator, **kwargs: Any) -> None:
        """Copy the current MuJoCo poses across and draw one frame."""
        del kwargs
        self._scene.sync(mujoco_sim.mjData)
        self._rep.orchestrator.step(rt_subframes=1, pause_timeline=False)

    def render(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        """Return the most recently drawn frame for every camera."""
        del args, kwargs
        images: dict[str, np.ndarray] = {}
        for name, annotator in self._annotators.items():
            width, height = self._resolutions[name]
            frame = np.asarray(annotator.get_data())
            if (
                frame.ndim != 3
                or frame.shape[:2] != (height, width)
                or frame.shape[2] not in (3, 4)
            ):
                raise _FrameNotReadyError(
                    f"Isaac RGB camera {name!r} returned shape {frame.shape}; "
                    f"expected ({height}, {width}, 3 or 4). "
                    "No complete current frame is available."
                )
            images[name] = frame[..., :3].astype(np.uint8, copy=False)
        return images

    def close(self) -> None:
        for name, annotator in self._annotators.items():
            annotator.detach([self._render_products[name].path])
        for product in self._render_products.values():
            product.destroy()
        self._annotators.clear()
        self._render_products.clear()
        self._resolutions.clear()
        self._texture_dir.cleanup()
        # Kit may exit the process inside close(). Evaluation workers defer
        # this final call until their result and validation receipts are saved.
        if not _env_flag("THETA_BENCH_ISAAC_DEFER_CLOSE", default=False):
            close_simulation_app()

    # -- internals ---------------------------------------------------------

    def _robot_usd(self) -> str | None:
        """The robot's USD, downloaded on first use, or None if it has none.

        A robot without a USD falls back to the meshes in its MJCF, which is
        worse looking but is not a reason to refuse to render.
        """
        relative = getattr(self.task.robot, "usd_path", None)
        if not relative:
            return None
        try:
            return resolve_data_path(relative, auto_download=True)
        except Exception as exc:
            print(f"[isaac] robot USD unavailable ({exc}); drawing MJCF meshes")
            return None

    def _camera_resolutions(self) -> dict[str, tuple[int, int]]:
        """The (width, height) each camera is rendered at, by camera name."""
        return {
            name: (int(camera.resolution[0]), int(camera.resolution[1]))
            for name, camera in self.task.layout.cameras.items()
        }

    def _attach_render_products(self) -> None:
        """Bind one Replicator RGB annotator per camera, once per process.

        Products survive resets because the mirror rebuilds the scene under a
        different root than the cameras; only a camera appearing, disappearing
        or changing resolution forces a rebuild.
        """
        wanted = self._camera_resolutions()
        for name, resolution in wanted.items():
            existing = self._resolutions.get(name)
            if existing == resolution:
                continue
            if existing is not None:
                self._annotators[name].detach([self._render_products[name].path])
                self._render_products[name].destroy()

            camera_path = self._scene.camera_path(name)
            assert camera_path is not None, f"no MuJoCo camera named {name!r}"
            width, height = int(resolution[0]), int(resolution[1])
            product = self._rep.create.render_product(camera_path, (width, height))
            annotator = self._rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])

            self._render_products[name] = product
            self._annotators[name] = annotator
            self._resolutions[name] = (width, height)

        for name in set(self._resolutions) - set(wanted):
            product = self._render_products.pop(name)
            self._annotators.pop(name).detach([product.path])
            product.destroy()
            self._resolutions.pop(name)
