"""Isaac Sim `SimulationApp` bootstrap for the render-only backend.

HumanoidToolBench never asks Isaac Sim to simulate anything: MuJoCo owns the
physics and Isaac only draws what MuJoCo has already computed. The app is
therefore started with the RTX renderer and with capture-on-play disabled, so
frames are produced when the engine asks for them rather than on a timeline.

The SimulationApp has to exist before any other `isaacsim`/`omni` module is
imported, which is why this module is deliberately import-light.

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
from typing import Any

# Waiting for Hydra to go idle is what makes a rendered frame correspond to the
# state that was just written, instead of trailing it by a frame or two.
HYDRA_WAIT_IDLE = "/app/hydraEngine/waitIdle"
HYDRA_RENDER_COMPLETE = "/app/updateOrder/checkForHydraRenderComplete"
THROTTLING_ENABLE_ASYNC = "/exts/isaacsim.core.throttling/enable_async"
# RTX real-time treats an opacity below one as a cutout mask unless this is on:
# the surface is either drawn solid or not drawn, never blended. The ice block
# is authored at 0.65 and comes out as a solid white box without it.
FRACTIONAL_CUTOUT = "/rtx/raytracing/fractionalCutoutOpacity"

_SIMULATION_APP: Any = None

# Native extensions that have to claim their shared-library symbols before Kit
# starts. Kit ships its own copies of several system libraries and loads them
# into the process during startup; a native module imported afterwards binds
# against Kit's copy instead of its own. `pinocchio` is the one that bites:
# its `libhpp-fcl.so` wants `Assimp::IOSystem::CurrentDirectory`, which Kit's
# assimp does not export, so the decoupled-WBC controller fails to import with
# an `undefined symbol` error part-way through teleop startup. Importing these
# first resolves them against their own libraries, and Kit starts fine after.
_PRELOAD_MODULES = ("torch", "pinocchio")


def _preload_native_runtime() -> None:
    """Import the native modules that must outrank Kit's bundled libraries."""
    import importlib  # noqa: PLC0415 - only needed on the Isaac path

    for name in _PRELOAD_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            # Only the toolbench extra pulls some of these in. A missing one
            # cannot clash with Kit, so there is nothing to preload.
            pass


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _import_simulation_app():
    """Return the `SimulationApp` class across Isaac Sim's module reshuffles."""
    try:  # Isaac Sim 4.5 and newer
        from isaacsim import SimulationApp  # type: ignore
    except ImportError:
        try:  # Isaac Sim 5.x
            from isaacsim.simulation_app import SimulationApp  # type: ignore
        except ImportError:
            try:  # Isaac Sim 4.2
                from omni.isaac.kit import SimulationApp  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sim_mode='mujoco_isaac' needs Isaac Sim on the Python path. "
                    "Install it into a separate environment and run through "
                    "scripts/run_isaac.sh; see docs/toolbench_isaac_render.md."
                ) from exc
    return SimulationApp


def simulation_app() -> Any:
    """Return the running SimulationApp, or None when Isaac was never started."""
    return _SIMULATION_APP


def _disable_default_viewport_updates() -> None:
    """Stop the unused UI viewport without touching Replicator cameras."""
    try:
        from omni.kit.viewport.utility import get_active_viewport  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        # A minimal headless experience can omit the viewport extensions.
        if exc.name in {
            "omni",
            "omni.kit",
            "omni.kit.viewport",
            "omni.kit.viewport.utility",
        }:
            return
        raise
    viewport = get_active_viewport()
    if viewport is not None:
        # This runtime API is available in Isaac Sim 4.5, unlike the newer
        # SimulationApp disable_viewport_updates configuration option.
        viewport.updates_enabled = False


def start_simulation_app(
    *,
    headless: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> Any:
    """Start Isaac Sim once per process and return the SimulationApp.

    A second call returns the app already running: Kit is a singleton and
    starting it twice aborts the process.
    """
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return _SIMULATION_APP

    _preload_native_runtime()
    SimulationApp = _import_simulation_app()

    zero_delay = _env_flag("HUMANOIDTOOLBENCH_ISAAC_ZERO_DELAY", default=True)
    disable_throttling_async = _env_flag(
        "HUMANOIDTOOLBENCH_ISAAC_DISABLE_THROTTLING_ASYNC", default=True
    )
    webrtc = _env_flag("HUMANOIDTOOLBENCH_ISAAC_WEBRTC", default=False)
    cpu_threads = os.getenv(
        "HUMANOIDTOOLBENCH_CPU_THREADS", os.getenv("OMP_NUM_THREADS", "4")
    ).strip()
    if not cpu_threads.isdecimal() or int(cpu_threads) < 1:
        raise ValueError(
            "HUMANOIDTOOLBENCH_CPU_THREADS (or OMP_NUM_THREADS) must be a positive integer"
        )

    # Each entry is (setting, launch argument value, runtime value): Kit reads
    # some of these only at launch and others only after the app exists.
    settings: list[tuple[str, object, object]] = []
    if zero_delay:
        settings.append((HYDRA_WAIT_IDLE, 1, True))
        settings.append((HYDRA_RENDER_COMPLETE, 1000, 1000))
    if disable_throttling_async:
        settings.append((THROTTLING_ENABLE_ASYNC, "false", False))
    settings.append((FRACTIONAL_CUTOUT, "true", True))
    if webrtc:
        settings.extend(
            [
                ("/app/livestream/publicEndpointAddress", "127.0.0.1", "127.0.0.1"),
                ("/app/livestream/port", 49100, 49100),
                ("/app/livestream/allowResize", "false", False),
            ]
        )

    config: dict[str, Any] = {
        "headless": headless,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
        "multi_gpu": False,
    }
    # Vulkan does not honor CUDA_VISIBLE_DEVICES; this is Kit's GPU index.
    active_gpu = os.getenv("HUMANOIDTOOLBENCH_ISAAC_GPU", "").strip()
    if active_gpu:
        if not active_gpu.isdecimal():
            raise ValueError("HUMANOIDTOOLBENCH_ISAAC_GPU must be a non-negative integer")
        config["active_gpu"] = int(active_gpu)
    if width is not None:
        config["width"] = width
    if height is not None:
        config["height"] = height
    if webrtc:
        # Headless Kit still needs a visible viewport for WebRTC capture.
        config.update(
            hide_ui=False, width=1280, height=720, window_width=1280, window_height=720
        )
    experience = os.getenv("HUMANOIDTOOLBENCH_ISAAC_EXPERIENCE", "").strip()
    if webrtc and not experience:
        # Importing SimulationApp sets EXP_PATH for pip and standalone installs.
        experience = os.path.join(
            os.getenv("EXP_PATH", ""), "isaacsim.exp.full.streaming.kit"
        )
        if not os.path.isfile(experience):
            raise RuntimeError(
                "HUMANOIDTOOLBENCH_ISAAC_WEBRTC=1 needs isaacsim.exp.full.streaming.kit "
                "in Isaac Sim's EXP_PATH. Install Isaac Sim with streaming support "
                "or set HUMANOIDTOOLBENCH_ISAAC_EXPERIENCE to its streaming experience."
            )
    # These pools are independent of OpenMP and read their limits at startup.
    config["extra_args"] = [f"--{key}={value}" for key, value, _ in settings] + [
        f"--/plugins/carb.tasking.plugin/threadCount={int(cpu_threads)}",
        f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={int(cpu_threads)}",
    ]
    cache_root = os.getenv("HUMANOIDTOOLBENCH_ISAAC_CACHE_ROOT", "").strip()
    if cache_root:
        cache_root = os.path.abspath(os.path.expanduser(cache_root))
        config["extra_args"].extend(
            [
                f"--/rtx/shaderDb/shaderCachePath={os.path.join(cache_root, 'shadercache')}",
                f"--/rtx/shaderDb/driverShaderCachePath={os.path.join(cache_root, 'nv_shadercache')}",
            ]
        )
    user_root = os.getenv("HUMANOIDTOOLBENCH_ISAAC_USER_ROOT", "").strip()
    if user_root:
        config["extra_args"].extend(
            ["--portable-root", os.path.abspath(os.path.expanduser(user_root))]
        )

    app = (
        SimulationApp(config, experience=experience)
        if experience
        else SimulationApp(config)
    )
    for key, _, runtime_value in settings:
        app.set_setting(key, runtime_value)

    # Replicator drives every frame explicitly through `orchestrator.step`, so
    # nothing may be captured off the timeline.
    import carb.settings  # noqa: PLC0415 - only importable after SimulationApp

    carb_settings = carb.settings.get_settings()
    carb_settings.set("/omni/replicator/captureOnPlay", False)
    carb_settings.set("/omni/replicator/asyncRendering", False)
    carb_settings.set("/app/asyncRendering", False)

    # Retain ownership if viewport setup fails so callers can close Kit after
    # writing their failure receipt.
    _SIMULATION_APP = app
    if headless and not webrtc:
        _disable_default_viewport_updates()
    return app


def close_simulation_app() -> None:
    """Shut Isaac Sim down; a no-op when it was never started."""
    global _SIMULATION_APP
    if _SIMULATION_APP is None:
        return
    app, _SIMULATION_APP = _SIMULATION_APP, None
    # Kit ends the process from inside `close()`, so anything Python still has
    # buffered never reaches the terminal. Without this an exception raised
    # during an episode disappears: the `finally` that closes the app runs
    # first, and the traceback is lost with the buffer.
    import sys  # noqa: PLC0415 - kept next to the flush it exists for

    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
