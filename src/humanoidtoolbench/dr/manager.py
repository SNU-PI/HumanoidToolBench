"""Domain-randomizer lifecycle management."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np

from humanoidtoolbench.core.randomizer import Randomizer, RandomizerCfg


class DRManager:
    """Build, seed, and look up a task's randomizers."""

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
        # its random domains.
        for name, randomizer in self.randomizers.items():
            randomizer.cfg = deepcopy(self._initial_configs[name])
        super().set_level(dr_level)
        if dr_level <= 0:
            return

        self._replace_cfg("lighting", "light_mode", "fixed")
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
