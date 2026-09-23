"""Task-policy contract shared by the policy transport and the controller."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

GoalT = TypeVar("GoalT")
_PolicyGoalT_co = TypeVar("_PolicyGoalT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class ActionChunk(Generic[GoalT]):
    """A non-empty sequence of controller goals tagged with its flat schema.

    A model adapter is responsible for decoding model output into controller
    goals before constructing the chunk.  Keeping the schema tag on the chunk
    prevents a goal from being sent to a controller that accepts a different
    schema merely because both happen to be represented by arrays.
    """

    schema: str
    goals: tuple[GoalT, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        schema: str,
        goals: Iterable[GoalT],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_schema = schema.strip() if isinstance(schema, str) else ""
        if not normalized_schema:
            raise ValueError("ActionChunk.schema must be a non-empty string")

        materialized_goals = tuple(goals)
        if not materialized_goals:
            raise ValueError("ActionChunk must contain at least one controller goal")

        object.__setattr__(self, "schema", normalized_schema)
        object.__setattr__(self, "goals", materialized_goals)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @classmethod
    def single(
        cls,
        schema: str,
        goal: GoalT,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionChunk[GoalT]:
        """Build a one-step chunk without special-casing it in a policy."""

        return cls(schema=schema, goals=(goal,), metadata=metadata)

    @property
    def actions(self) -> tuple[GoalT, ...]:
        """Alias for callers that use model-centric ``actions`` terminology."""

        return self.goals

    def __iter__(self) -> Iterator[GoalT]:
        return iter(self.goals)

    def __len__(self) -> int:
        return len(self.goals)


@runtime_checkable
class TaskPolicy(Protocol[_PolicyGoalT_co]):
    """A high-level model that predicts chunks in one controller schema."""

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionChunk[_PolicyGoalT_co]:
        """Predict the next non-empty chunk of controller goals."""

    def reset(self, **kwargs: Any) -> None:
        """Reset recurrent/history state at an episode boundary."""
