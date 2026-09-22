"""Compose a task model and humanoid controller from independent choices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from humanoidtoolbench.controllers.factory import (
    action_schema_for_controller,
    make_body_controller,
    normalize_controller_name,
)
from humanoidtoolbench.policies.composition import HumanoidPolicyAgent
from humanoidtoolbench.policies.remote_humanoid import make_remote_task_policy


def controller_from_legacy_policy_name(policy: str) -> str | None:
    """Read only explicit legacy suffixes; unsuffixed models stay ambiguous."""

    normalized = policy.strip().lower().replace("-", "_")
    if normalized.endswith("_decoupled_wbc"):
        return "decoupled_wbc"
    if normalized.endswith(("_gear_sonic", "_sonic")):
        return "gear_sonic"
    return None


def make_humanoid_policy_agent(
    *,
    robot: Any,
    policy: str | Any,
    controller: str | None,
    host: str,
    port: int,
    sonic_config: Mapping[str, Any],
    policy_client: Any | None = None,
    gear_sonic_runtime: Any | None = None,
    gear_sonic_artifact_root: str | None = None,
    policy_options: Mapping[str, Any] | None = None,
    controller_options: Mapping[str, Any] | None = None,
) -> HumanoidPolicyAgent[Any]:
    """Build the selected ``task policy x controller`` composition.

    A string policy selects one of the repository's remote ACT/pi/Psi/WAM
    transports.  An object may instead implement ``TaskPolicy`` directly,
    which is useful for in-process policies and tests.
    """

    if controller is None:
        if not isinstance(policy, str):
            raise ValueError("controller is required for an injected task policy")
        controller = controller_from_legacy_policy_name(policy) or "decoupled_wbc"
    controller_name = normalize_controller_name(controller)
    action_schema = action_schema_for_controller(controller_name)

    controller_kwargs = dict(controller_options or {})
    body_controller = make_body_controller(
        controller_name,
        robot=robot,
        sonic_config=sonic_config,
        gear_sonic_runtime=gear_sonic_runtime,
        gear_sonic_artifact_root=gear_sonic_artifact_root,
        **controller_kwargs,
    )
    if isinstance(policy, str):
        task_policy = make_remote_task_policy(
            policy,
            action_schema=action_schema,
            host=host,
            port=port,
            client=policy_client,
            **dict(policy_options or {}),
        )
        policy_name = task_policy.policy_name
    else:
        task_policy = policy
        policy_name = type(policy).__name__

    agent = HumanoidPolicyAgent(task_policy, body_controller)
    # These labels are intentionally informational; protocol behavior remains
    # entirely structural and does not depend on model-specific subclasses.
    agent.policy_name = policy_name
    agent.controller_name = controller_name
    agent.action_schema = action_schema
    return agent
