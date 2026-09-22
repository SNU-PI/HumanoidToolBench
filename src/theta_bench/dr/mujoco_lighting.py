"""Lighting randomization in MuJoCo-native terms.

This randomizer draws a rig with a directional key that casts shadows, two
fills so the far side of an object is not black, and an uplight off the back
wall. It hands the result to the task as `mujoco_lights`.

Colour temperature is kept, because it is the part worth varying: a scene lit at
3000 K and one at 8000 K look genuinely different to a camera. Intensity varies
too, but around a floor: an episode a policy cannot see is not a hard episode,
it is a wasted one.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from theta_bench.core.randomizer import Randomizer, RandomizerCfg


def kelvin_to_rgb(k: float) -> np.ndarray:
    """Approximate white point of a black body at `k` kelvin, normalised to 1.

    Tanner Helland's piecewise fit, which is close enough for a light colour and
    needs no table. Normalising to the brightest channel keeps colour and
    brightness independent, so the intensity draw is the only thing that decides
    how bright the scene is.
    """
    t = np.clip(k, 1000.0, 40000.0) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    rgb = np.clip([r, g, b], 0.0, 255.0)
    return rgb / max(rgb.max(), 1e-6)


class MujocoLightingDR(Randomizer):
    """Draws a key/fill/uplight rig per episode."""

    def __init__(self, cfg: "MujocoLightingDRCfg") -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self._lights: list[dict[str, Any]] = []

    @property
    def lights(self) -> list[dict[str, Any]]:
        return list(self._lights)

    @staticmethod
    def _encode_lights(lights: list[dict[str, Any]]) -> list[dict[str, Any]]:
        encoded = deepcopy(lights)
        for light in encoded:
            if "type" in light:
                light["type"] = mujoco.mjtLightType(light["type"]).name
        return encoded

    @staticmethod
    def _decode_lights(lights: list[dict[str, Any]]) -> list[dict[str, Any]]:
        decoded = deepcopy(lights)
        for light in decoded:
            kind = light.get("type")
            if isinstance(kind, str):
                light["type"] = getattr(mujoco.mjtLightType, kind)
            elif kind is not None:
                light["type"] = mujoco.mjtLightType(kind)
        return decoded

    def _rig(self, colour, gain, key_pos, key_dir, fill_ratio, fill_skew):
        """Assemble one key / two fills / one uplight from drawn parameters."""
        cfg = self.cfg
        key = dict(
            pos=list(key_pos),
            dir=list(key_dir),
            # Directional on purpose: were shadows on, a positional caster
            # would leave its shadow-map frustum on the floor as a bright
            # rectangle. `shadowscale` does not apply to directional lights.
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=cfg.shadows,
            diffuse=(colour * gain).tolist(),
            specular=[0.16, 0.16, 0.16],
            ambient=(colour * cfg.ambient).tolist(),
        )
        fill = gain * fill_ratio
        return [
            key,
            dict(
                pos=[-1.9, -1.4, 2.40],
                dir=[0.55, 0.45, -1.0],
                castshadow=False,
                diffuse=(colour * fill).tolist(),
                specular=[0.05, 0.05, 0.05],
            ),
            dict(
                pos=[1.9, -1.4, 2.40],
                dir=[-0.55, 0.45, -1.0],
                castshadow=False,
                diffuse=(colour * fill * fill_skew).tolist(),
                specular=[0.05, 0.05, 0.05],
            ),
            dict(
                pos=[0.0, 1.9, 1.30],
                dir=[0.0, -0.6, -0.35],
                castshadow=False,
                diffuse=(colour * gain * cfg.uplight_ratio).tolist(),
                specular=[0.0, 0.0, 0.0],
            ),
        ]

    def fixed(self) -> list[dict[str, Any]]:
        """The middle of every range: what a task uses before its first reset."""
        cfg = self.cfg
        self._lights = self._rig(
            kelvin_to_rgb(float(np.mean(cfg.temperature))),
            float(np.mean(cfg.key_intensity)),
            [0.0, 0.35, 2.55],
            [0.0, 0.10, -1.0],
            float(np.mean(cfg.fill_ratio)),
            1.0,
        )
        return self.lights

    def apply(self, layout=None, split: str = "train") -> list[dict[str, Any]]:
        del layout, split
        if self._inner_state is not None and "lights" in self._inner_state:
            self._lights = self._decode_lights(self._inner_state["lights"])
            return self.lights
        if (
            self._inner_state is not None
            and {
                "key_gain",
                "colour",
            }
            <= self._inner_state.keys()
        ):
            # Early recordings stored only the two independent lighting draws.
            # Rebuild the old fixed-position rig deterministically instead of
            # sampling new lights each time that episode is replayed.
            self._lights = self._rig(
                np.asarray(self._inner_state["colour"], dtype=float),
                float(self._inner_state["key_gain"]),
                [0.0, 0.35, 2.55],
                [0.0, 0.10, -1.0],
                float(np.mean(self.cfg.fill_ratio)),
                1.0,
            )
            return self.lights

        cfg = self.cfg
        if cfg.light_mode not in {"fixed", "random"}:
            raise ValueError(f"Invalid MuJoCo lighting mode {cfg.light_mode!r}")
        if cfg.light_mode == "fixed":
            self.fixed()
            self._transient(
                {
                    "key_gain": float(np.mean(cfg.key_intensity)),
                    "colour": kelvin_to_rgb(float(np.mean(cfg.temperature))).tolist(),
                    "lights": self._encode_lights(self._lights),
                }
            )
            return self.lights

        # Drawn from the global stream, like the other randomisers here, so
        # `DRManager.reset(seed)` reproduces the lighting too.
        rng = np.random.default_rng(int(np.random.randint(0, 2**32, dtype=np.uint32)))

        colour = kelvin_to_rgb(float(rng.uniform(*cfg.temperature)))
        gain = float(rng.uniform(*cfg.key_intensity))
        tilt = rng.uniform(-cfg.key_tilt, cfg.key_tilt, size=2)
        jitter = rng.uniform(-cfg.key_jitter, cfg.key_jitter, size=2)
        self._lights = self._rig(
            colour,
            gain,
            [float(jitter[0]), float(0.35 + jitter[1]), 2.55],
            [float(tilt[0]), float(0.10 + tilt[1]), -1.0],
            float(rng.uniform(*cfg.fill_ratio)),
            float(rng.uniform(0.7, 1.3)),
        )
        self._transient(
            {
                "key_gain": gain,
                "colour": colour.tolist(),
                "lights": self._encode_lights(self._lights),
            }
        )
        return self.lights

    def __call__(self, split: str = "train", **kwargs) -> list[dict[str, Any]]:
        del split, kwargs
        return self.lights


@dataclass
class MujocoLightingDRCfg(RandomizerCfg):
    """Ranges for the rig. Defaults sit around the hand-tuned toolbench values."""

    temperature: tuple[float, float] = (3200.0, 7600.0)  # kelvin
    key_intensity: tuple[float, float] = (0.42, 0.62)  # MuJoCo diffuse, per channel
    fill_ratio: tuple[float, float] = (0.5, 0.75)  # of the key
    uplight_ratio: float = 0.3
    ambient: float = 0.34  # the floor: never a scene a camera cannot read
    key_jitter: float = 0.35  # m, where the key hangs
    key_tilt: float = 0.12  # how far off vertical it points
    # Off by default. The head camera looks down at the floor from a metre away,
    # so the robot's own shadow lands across most of the frame as a hard-edged
    # silhouette: measured over four episodes, shadows push 6.9% of the image
    # below 25/255 and 3.1% above 250, against 1.8% and 0.3% with them off.
    # It is the robot's own body occluding its own light, and it tells a policy
    # nothing the proprioception does not already say.
    shadows: bool = False
    light_mode: str = "random"  # fixed, random

    randmizer_class: Any = MujocoLightingDR


__all__ = ["MujocoLightingDR", "MujocoLightingDRCfg", "kelvin_to_rgb"]
