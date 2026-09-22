"""Lazy public exports for active domain randomizers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ToolReasoningDR": (".tool_reasoning", "ToolReasoningDR"),
    "ToolReasoningDRCfg": (".tool_reasoning", "ToolReasoningDRCfg"),
    "CameraDR": (".camera", "CameraDR"),
    "CameraDRCfg": (".camera", "CameraDRCfg"),
    "MujocoLightingDR": (".mujoco_lighting", "MujocoLightingDR"),
    "MujocoLightingDRCfg": (".mujoco_lighting", "MujocoLightingDRCfg"),
    "RoomDR": (".room", "RoomDR"),
    "RoomDRCfg": (".room", "RoomDRCfg"),
    "SceneDR": (".scene", "SceneDR"),
    "SpatialDR": (".spatial", "SpatialDR"),
    "SpatialDRCfg": (".spatial", "SpatialDRCfg"),
    "TabletopSceneDR": (".scene", "TabletopSceneDR"),
    "TabletopSceneDRCfg": (".scene", "TabletopSceneDRCfg"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
