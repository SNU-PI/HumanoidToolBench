"""Domain-randomizer lifecycle management."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np

from theta_bench.core.randomizer import Randomizer, RandomizerCfg


class DRManager:
    """Build, seed, serialize, and look up a task's randomizers."""

    def __init__(self, level: int, **configs: RandomizerCfg) -> None:
        self._dr_level = level
        self.randomizers: dict[str, Randomizer] = {}
        self.seed: int | None = None

        for name, config in configs.items():
            if not isinstance(config, RandomizerCfg):
                raise TypeError(
                    f"Expected RandomizerCfg for {name}, got {type(config).__name__}"
                )
            self.randomizers[name] = config.build()

        self.set_level(level)

    @property
    def level(self) -> int:
        return self._dr_level

    @level.setter
    def level(self, value: int) -> None:
        if value != self._dr_level:
            self.set_level(value)

    def set_level(self, dr_level: int) -> None:
        self._dr_level = dr_level

    def get_randomizer(self, name: str) -> Randomizer | None:
        return self.randomizers.get(name)

    def state_dict(self) -> dict[str, Any]:
        return {
            name: randomizer.state_dict()
            for name, randomizer in self.randomizers.items()
        }

    def load_state_dict(
        self, state_dict: dict[str, Any], dr_level: int | None = None
    ) -> None:
        """Restore recorded draws, optionally re-randomizing selected domains."""
        recorded = dict(state_dict["dr_state_dict"])
        # Retired render-only randomizers may still appear in old recordings.
        recorded.pop("material", None)
        if dr_level is not None:
            if dr_level not in (0, 1, 2):
                raise ValueError(f"Invalid DR level {dr_level}")
            recorded.pop("distractors", None)
            if dr_level >= 1:
                recorded.pop("lighting", None)
                recorded.pop("isaac_lighting", None)
                recorded.pop("isaac_material", None)
            if dr_level == 2 and "spatial" in recorded:
                spatial = recorded["spatial"]
                robot_state = next(
                    (
                        {uid: spatial[uid]}
                        for uid in ("g1_sonic", "g1_wholebody")
                        if uid in spatial
                    ),
                    None,
                )
                recorded["spatial"] = robot_state

        for name, randomizer_state in recorded.items():
            randomizer = self.get_randomizer(name)
            if randomizer is None:
                available = ", ".join(sorted(self.randomizers))
                raise KeyError(
                    f"Randomizer {name!r} is not configured. Available: {available}"
                )
            randomizer.load_state_dict(randomizer_state)

        for name, randomizer in self.randomizers.items():
            if name not in recorded:
                randomizer._inner_state = None

    def reset(self, seed: int | None = None) -> None:
        """Clear draws and optionally seed both random streams."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
        for randomizer in self.randomizers.values():
            randomizer._inner_state = None


class ToolbenchDRManager(DRManager):
    """Apply the benchmark DR-level policy to active randomizers."""

    def __init__(self, level: int, **configs: RandomizerCfg) -> None:
        self._initial_configs = deepcopy(configs)
        super().__init__(level, **configs)

    def set_level(self, dr_level: int) -> None:
        # Reapply policy from the original settings so lowering a level restores
        # its random domains. Recorded draws remain authoritative until reset.
        for name, randomizer in self.randomizers.items():
            randomizer.cfg = deepcopy(self._initial_configs[name])
        super().set_level(dr_level)
        if dr_level <= 0:
            return

        self._replace_cfg("lighting", "light_mode", "fixed")
        self._replace_cfg("isaac_lighting", "light_mode", "fixed")
        self._replace_cfg("isaac_material", "material_mode", "fixed")
        self._replace_cfg("scene", "scene_mode", "fixed")

        if dr_level > 1:
            self._replace_cfg("distractors", "number_of_distractors", 0)
        if dr_level > 2:
            self._replace_cfg("spatial", "spatial_mode", "fixed")

    def _replace_cfg(self, name: str, field: str, value: Any) -> None:
        randomizer = self.randomizers.get(name)
        if randomizer is None:
            return
        if not hasattr(randomizer.cfg, field):
            raise ValueError(f"Randomizer {name!r} has no config field {field!r}")
        setattr(randomizer.cfg, field, value)
