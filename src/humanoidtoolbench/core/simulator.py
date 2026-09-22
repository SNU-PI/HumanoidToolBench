"""Simulator interface shared by the environment and MuJoCo engine."""

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np


class Simulator(ABC):
    def update_layout(self, **kwargs: Any) -> None: ...

    def step(self, **kwargs: Any) -> dict[str, np.ndarray] | None: ...

    def render(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...
