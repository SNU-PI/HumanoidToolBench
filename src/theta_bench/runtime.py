"""Runtime mode safeguards shared by THETA(θ)-Bench entry points."""

from __future__ import annotations

import os

VALID_SIM_MODES = frozenset({"mujoco", "isaacsim-mujoco", "mujoco_isaac"})
# Modes that start Isaac Sim. Isaac never simulates anything in THETA(θ)-Bench: it
# renders a scene MuJoCo owns, so these modes still run MuJoCo physics.
ISAAC_RENDER_SIM_MODES = frozenset({"isaacsim-mujoco", "mujoco_isaac"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def mujoco_only_enabled() -> bool:
    """Return whether THETA_BENCH_MUJOCO_ONLY is set (the default).

    MuJoCo owns the physics in every mode. This flag additionally keeps Isaac
    Sim out of the process entirely, which is what the managed-runtime entry
    points, the wrapper scripts and the dependency audits assume; clearing it is
    the explicit opt-in for the Isaac renderer.
    """

    value = os.getenv("THETA_BENCH_MUJOCO_ONLY", "1").strip().lower()
    return value not in _FALSE_VALUES


def isaac_install_allowed() -> bool:
    """Return whether Isaac may share an environment with the MuJoCo runtime.

    Installing the render backend puts Isaac wheels next to THETA(θ)-Bench's own.
    The guarantee the audits exist to protect is that the MuJoCo path never
    *loads* Isaac, which they check directly by inspecting `sys.modules`; the
    weaker "nothing Isaac is installed" rule is what this flag lifts.
    """

    value = os.getenv("THETA_BENCH_ALLOW_ISAAC_INSTALL", "0").strip().lower()
    return value in _TRUE_VALUES


def force_cpu_enabled() -> bool:
    """Return whether this process must stay off every GPU.

    The GPU-free envelope is no longer the default: MuJoCo-only mode now
    renders and trains on the GPU unless THETA_BENCH_FORCE_CPU=1 is set, which is
    what scripts/run_mujoco_cpu.sh does to reproduce the original smoke test.
    """

    return os.getenv("THETA_BENCH_FORCE_CPU", "0").strip().lower() in _TRUE_VALUES


def gpu_hidden() -> bool:
    """Return whether CUDA devices are explicitly hidden from this process.

    An unset CUDA_VISIBLE_DEVICES exposes every device, so only an explicitly
    empty value counts as hidden.
    """

    value = os.getenv("CUDA_VISIBLE_DEVICES")
    return value is not None and value.strip() == ""


def require_managed_runtime(entry_point: str) -> None:
    """Fail fast when a MuJoCo-only entry point runs outside the wrapper.

    The wrapper is what pins the GPU, the renderer, the thread pools, and the
    scheduling priority, so these entry points refuse to guess those settings.
    """

    if not mujoco_only_enabled():
        raise RuntimeError(f"{entry_point} requires THETA_BENCH_MUJOCO_ONLY=1")
    if os.getenv("THETA_BENCH_RUNTIME_WRAPPER") != "1":
        raise RuntimeError(f"Run {entry_point} through scripts/run_mujoco.sh")
    if force_cpu_enabled() and not gpu_hidden():
        raise RuntimeError(
            "THETA_BENCH_FORCE_CPU=1 requires an empty CUDA_VISIBLE_DEVICES; "
            "use scripts/run_mujoco_cpu.sh for the GPU-free envelope."
        )


def validate_sim_mode(sim_mode: str) -> str:
    """Validate a mode before simulator imports and return its canonical name."""

    if sim_mode not in VALID_SIM_MODES:
        choices = ", ".join(sorted(VALID_SIM_MODES))
        raise ValueError(f"Invalid sim_mode {sim_mode!r}; choose one of: {choices}")

    if sim_mode in ISAAC_RENDER_SIM_MODES and mujoco_only_enabled():
        raise RuntimeError(
            f"sim_mode {sim_mode!r} starts Isaac Sim, which THETA_BENCH_MUJOCO_ONLY=1 "
            "forbids; run it through scripts/run_isaac.sh"
        )

    return "mujoco_isaac" if sim_mode == "isaacsim-mujoco" else sim_mode
