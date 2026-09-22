"""Factory for the humanoid body controller."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from humanoidtoolbench.controllers.decoupled_wbc import DecoupledWbcBackend


def make_body_controller(
    name: str,
    *,
    robot: Any,
    sonic_config: Mapping[str, Any],
) -> DecoupledWbcBackend:
    """Construct the decoupled WBC, the controller the benchmark runs."""

    if name != "decoupled_wbc":
        raise ValueError(f"Unsupported body controller {name!r}; choose decoupled_wbc")
    return DecoupledWbcBackend(robot, sonic_config)
