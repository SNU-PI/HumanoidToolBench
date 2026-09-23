"""Compose a task policy with the humanoid body controller."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from humanoidtoolbench.controllers.factory import make_body_controller
from humanoidtoolbench.policies.composition import HumanoidPolicyAgent
from humanoidtoolbench.policies.remote_humanoid import make_remote_task_policy


def make_humanoid_policy_agent(
    *,
    robot: Any,
    policy: str | Any,
    controller: str,
    host: str,
    port: int,
    sonic_config: Mapping[str, Any],
    policy_client: Any | None = None,
) -> HumanoidPolicyAgent[Any]:
    """Build the selected ``task policy x controller`` composition.

    A string policy selects the remote HTTP transport.  An object may instead
    implement ``TaskPolicy`` directly, which is useful for in-process policies
    and tests.
    """

    body_controller = make_body_controller(
        controller, robot=robot, sonic_config=sonic_config
    )
    if isinstance(policy, str):
        task_policy = make_remote_task_policy(
            policy, host=host, port=port, client=policy_client
        )
        policy_name = task_policy.policy_name
    else:
        task_policy = policy
        policy_name = type(policy).__name__

    agent = HumanoidPolicyAgent(task_policy, body_controller)
    # These labels are intentionally informational; protocol behavior remains
    # entirely structural and does not depend on model-specific subclasses.
    agent.policy_name = policy_name
    agent.controller_name = controller
    agent.action_schema = body_controller.action_schema
    return agent
