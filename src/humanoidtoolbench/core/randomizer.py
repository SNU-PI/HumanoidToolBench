"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from dataclasses import dataclass
from typing import Any, Protocol, Type, runtime_checkable


# @runtime_checkable
@dataclass
class Randomizer:  # (Protocol)
    seed: int | None
    cfg: "RandomizerCfg"

    _inner_state: Any

    def __init__(self, cfg: "RandomizerCfg", seed: int | None = None) -> None:
        self.cfg = cfg
        self.seed = seed
        self._inner_state = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Applies the domain randomization to the specified split."""

    def _transient(self, rand_state) -> Any:
        """Keep this episode's draw and return it.

        A draw already kept since the last `DRManager.reset` wins, so a second
        call within one episode returns the first draw instead of a new one.
        """
        if self._inner_state is not None:
            return self._inner_state

        self._inner_state = rand_state
        return self._inner_state

    # def apply(self, split:str, *args, **kwargs) -> Any:
    #     """Applies the domain randomization to the specified split."""

    # def fixed(self) -> Any:
    #     """Apllied a fixed set of parameters."""


# @dataclass
@runtime_checkable
@dataclass
class RandomizerCfg(Protocol):
    randmizer_class: Type[Randomizer]

    def build(self) -> Randomizer:
        return self.randmizer_class(self)
