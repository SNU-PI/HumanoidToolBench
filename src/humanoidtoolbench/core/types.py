"""Small value objects shared by scene construction code."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pose:
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
