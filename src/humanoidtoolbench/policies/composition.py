"""Generic task-policy and humanoid-controller composition."""

from __future__ import annotations

from collections import deque
from typing import Any, Generic, TypeVar

from humanoidtoolbench.controllers import BodyController
from humanoidtoolbench.core.action import ActionCmd
from humanoidtoolbench.policies.base import ActionChunk, TaskPolicy

GoalT = TypeVar("GoalT")


class ActionSchemaMismatchError(ValueError):
    """Raised before a chunk is sent to an incompatible body controller."""


class HumanoidPolicyAgent(Generic[GoalT]):
    """Run any chunk-producing task policy through any compatible controller.

    The class intentionally implements the small ``get_action``/``reset``
    protocol used by the humanoid evaluation loop; it does not depend on
    a concrete robot class.  One prediction request fills a FIFO queue, which
    is consumed before the policy is queried again.
    """

    def __init__(
        self,
        task_policy: TaskPolicy[GoalT],
        body_controller: BodyController[GoalT],
    ) -> None:
        self.task_policy = task_policy
        self.body_controller = body_controller
        self._goal_queue: deque[GoalT] = deque()
        self._last_action_chunk: ActionChunk[GoalT] | None = None

    @property
    def pending_goals(self) -> int:
        """Number of goals left from the current prediction chunk."""

        return len(self._goal_queue)

    @property
    def last_action_chunk(self) -> ActionChunk[GoalT] | None:
        return self._last_action_chunk

    def __len__(self) -> int:
        return self.pending_goals

    def _request_chunk(
        self,
        observation: dict[str, Any],
        instruction: str | None,
        **kwargs: Any,
    ) -> None:
        chunk = self.task_policy.predict(
            observation,
            instruction=instruction,
            **kwargs,
        )
        if not isinstance(chunk, ActionChunk):
            raise TypeError(
                "TaskPolicy.predict() must return ActionChunk, got "
                f"{type(chunk).__name__}"
            )

        controller_schema = self.body_controller.action_schema
        if chunk.schema != controller_schema:
            raise ActionSchemaMismatchError(
                "Task policy produced action schema "
                f"{chunk.schema!r}, but controller accepts {controller_schema!r}"
            )

        self._last_action_chunk = chunk
        self._goal_queue.extend(chunk.goals)

    def get_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        **kwargs: Any,
    ) -> ActionCmd:
        """Return one simulator action, requesting a new chunk only if needed."""

        if not self._goal_queue:
            self._request_chunk(observation, instruction, **kwargs)

        # Remove a goal only after the controller accepts it.  A recoverable
        # controller exception can therefore be retried without skipping time.
        goal = self._goal_queue[0]
        action = self.body_controller.step(goal, observation, **kwargs)
        if not isinstance(action, ActionCmd):
            raise TypeError(
                "BodyController.step() must return ActionCmd, got "
                f"{type(action).__name__}"
            )
        self._goal_queue.popleft()
        return action

    def reset(self, **kwargs: Any) -> None:
        """Clear queued actions and reset both halves of the composition."""

        self._goal_queue.clear()
        self._last_action_chunk = None
        self.task_policy.reset(**kwargs)
        self.body_controller.reset(**kwargs)

    def get_stabilize_action(
        self,
        observation: dict[str, Any],
        **kwargs: Any,
    ) -> ActionCmd:
        """Forward the optional controller startup/stabilization capability."""

        stabilize = getattr(self.body_controller, "get_stabilize_action", None)
        if not callable(stabilize):
            raise NotImplementedError(
                f"{type(self.body_controller).__name__} does not provide "
                "get_stabilize_action()"
            )
        action = stabilize(observation, **kwargs)
        if not isinstance(action, ActionCmd):
            raise TypeError(
                "BodyController.get_stabilize_action() must return ActionCmd, got "
                f"{type(action).__name__}"
            )
        return action
