"""Common interface for humanoid body-controller backends.

A body controller owns the robot-specific, low-level control loop.  It accepts
one goal in its declared action schema and turns that goal into the
``ActionCmd`` consumed by a THETA-Bench environment.  Task policies therefore
do not need to know whether the selected backend is decoupled WBC, GEAR-SONIC,
or a test double.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from theta_bench.core.action import ActionCmd

_ControllerGoalT_contra = TypeVar("_ControllerGoalT_contra", contravariant=True)


@runtime_checkable
class BodyController(Protocol[_ControllerGoalT_contra]):
    """Structural interface implemented by a humanoid controller backend."""

    @property
    def action_schema(self) -> str:
        """Stable schema tag accepted by :meth:`step`."""

    def step(
        self,
        goal: _ControllerGoalT_contra,
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> ActionCmd:
        """Convert one policy goal into an executable simulator command."""

    def reset(self, **kwargs: Any) -> None:
        """Reset controller state at an episode boundary."""


@runtime_checkable
class StabilizingBodyController(Protocol):
    """Optional capability for controllers that need a startup pose ramp."""

    def get_stabilize_action(
        self,
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> ActionCmd:
        """Produce an action that moves or holds the robot in a safe pose."""
