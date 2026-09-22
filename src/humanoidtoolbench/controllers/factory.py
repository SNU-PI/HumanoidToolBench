"""Factory and naming helpers for humanoid body-controller backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from humanoidtoolbench.actions import DECOUPLED_SCHEMA_ID, SONIC_LATENT_SCHEMA_ID

SUPPORTED_BODY_CONTROLLERS = ("decoupled_wbc", "gear_sonic")


def normalize_controller_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "decoupled": "decoupled_wbc",
        "dwbc": "decoupled_wbc",
        "sonic": "gear_sonic",
        "unified": "gear_sonic",
        "unified_sonic": "gear_sonic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_BODY_CONTROLLERS:
        raise ValueError(
            f"Unsupported body controller {name!r}; choose "
            f"{', '.join(SUPPORTED_BODY_CONTROLLERS)}"
        )
    return normalized


def action_schema_for_controller(name: str) -> str:
    normalized = normalize_controller_name(name)
    if normalized == "decoupled_wbc":
        return DECOUPLED_SCHEMA_ID
    return SONIC_LATENT_SCHEMA_ID


def make_body_controller(
    name: str,
    *,
    robot: Any,
    sonic_config: Mapping[str, Any],
    gear_sonic_runtime: Any | None = None,
    gear_sonic_artifact_root: str | PathLike[str] | None = None,
    gear_sonic_providers: Sequence[str] | None = None,
    initial_motion_token: Any | None = None,
) -> Any:
    """Construct one backend without importing the other optional stack."""

    normalized = normalize_controller_name(name)
    if normalized == "decoupled_wbc":
        from humanoidtoolbench.controllers.decoupled_wbc import DecoupledWbcBackend

        return DecoupledWbcBackend(robot, sonic_config)

    from humanoidtoolbench.controllers.gear_sonic import GearSonicBackend

    return GearSonicBackend(
        runtime=gear_sonic_runtime,
        artifact_root=gear_sonic_artifact_root,
        providers=gear_sonic_providers,
        initial_motion_token=initial_motion_token,
    )
