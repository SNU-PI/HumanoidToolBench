"""Domain-randomizer lifecycle management."""

from __future__ import annotations

import random

import numpy as np

from humanoidtoolbench.core.randomizer import Randomizer, RandomizerCfg


class DRManager:
    """Build, seed, and look up a task's randomizers."""

    def __init__(self, **configs: RandomizerCfg) -> None:
        self.randomizers: dict[str, Randomizer] = {}
        self.seed: int | None = None

        for name, config in configs.items():
            if not isinstance(config, RandomizerCfg):
                raise TypeError(
                    f"Expected RandomizerCfg for {name}, got {type(config).__name__}"
                )
            self.randomizers[name] = config.build()

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
