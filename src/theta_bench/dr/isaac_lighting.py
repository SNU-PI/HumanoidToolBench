"""Randomize the Isaac renderer's light rig.

This is the RTX counterpart of `mujoco_lighting`, and the two are deliberately
separate. MuJoCo's light model has no emitter size and no colour temperature,
so a rig authored for RTX cannot be expressed as `mujoco_lights`; a task that
runs under both engines carries one of each and every engine reads its own.

The rig is a ceiling grid of cylinder lights, which is what a room lit from
overhead strip lighting looks like, and it is the rig SIMPLE used.

THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import transforms3d as t3d

from theta_bench.core.actor import Light
from theta_bench.core.randomizer import Randomizer, RandomizerCfg
from theta_bench.core.types import Pose
from theta_bench.dr.types import Box

# The fixed rig, used when a domain is not randomized. Warm-white, bright
# enough that a camera can read the bench without blowing out its markings.
DEFAULT_COLOR_TEMPERATURE = 6000.0
DEFAULT_INTENSITY = 6000.0
DEFAULT_RADIUS = 0.21
DEFAULT_LENGTH = 2.1
DEFAULT_SPACING = [2.0, 2.0]
DEFAULT_CENTRE = [0.0, 0.0, 2.6]


class IsaacLightingDR(Randomizer):
    """Draw a grid of ceiling cylinder lights for the Isaac renderer."""

    def apply(self, split: str = "train") -> list[Light]:
        """The name every other THETA(θ)-Bench randomizer is driven by."""
        return self(split)

    def __call__(self, split: str, *args: Any, **kwargs: Any) -> list[Light]:
        del split, args, kwargs
        if self._inner_state is not None:
            return self._inner_state

        cfg = self.cfg
        if cfg.light_mode not in {"fixed", "random"}:
            raise ValueError(f"Invalid Isaac lighting mode {cfg.light_mode!r}")
        fixed = cfg.light_mode == "fixed"

        def draw(domain: Box | None, default: Any) -> Any:
            return default if fixed or domain is None else domain.sample()

        temperature = draw(cfg.light_color_temperature, DEFAULT_COLOR_TEMPERATURE)
        intensity = draw(cfg.light_intensity, DEFAULT_INTENSITY)
        radius = draw(cfg.light_radius, DEFAULT_RADIUS)
        length = draw(cfg.light_length, DEFAULT_LENGTH)
        spacing = draw(cfg.light_spacing, DEFAULT_SPACING)
        centre = draw(cfg.light_position, DEFAULT_CENTRE)
        eulers = draw(cfg.light_eulers, [0.0, 0.0, 0.0])
        orientation = t3d.euler.euler2quat(*eulers).tolist()

        rows, columns = cfg.light_num
        lights: list[Light] = []
        for row in range(rows):
            for column in range(columns):
                light = Light(uid=f"Light_{row}_{column}", type="CylinderLight")
                # Positions are relative to the rig's centre, so the whole
                # grid moves and rotates as one.
                light.pose = Pose(
                    position=[
                        spacing[1] * (column - 0.5 * (columns - 1)),
                        spacing[0] * (row - 0.5 * (rows - 1)),
                        0.0,
                    ]
                )
                light.light_radius = radius
                light.light_length = length
                light.light_intensity = intensity
                light.light_color_temperature = temperature
                light.center_light_position = list(centre)
                light.center_light_orientation = list(orientation)
                lights.append(light)

        return super()._transient(lights)

    def state_dict(self) -> dict[str, Any]:
        if self._inner_state is None:
            return {}
        return {light.uid: light.to_dict() for light in self._inner_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            self._inner_state = None
            return
        lights = []
        for entry in state_dict.values():
            light = Light(uid=entry["uid"], type=entry["type"])
            light.pose = Pose(position=entry["pose"]["position"])
            light.light_radius = entry["light_radius"]
            light.light_length = entry["light_length"]
            light.light_intensity = entry["light_intensity"]
            light.light_color_temperature = entry["light_color_temperature"]
            light.center_light_position = entry["center_light_position"]
            light.center_light_orientation = entry["center_light_orientation"]
            lights.append(light)
        self._inner_state = lights


@dataclass
class IsaacLightingDRCfg(RandomizerCfg):
    light_mode: str = "fixed"  # fixed, random
    light_num: tuple[int, int] = (2, 3)
    light_color_temperature: Box | None = None
    light_intensity: Box | None = None
    light_radius: Box | None = None
    light_length: Box | None = None
    light_spacing: Box | None = None
    light_position: Box | None = None
    light_eulers: Box | None = None
    randmizer_class: type[Randomizer] = IsaacLightingDR
