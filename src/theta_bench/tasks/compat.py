"""Compatibility checks for recorded task state identifiers."""

from __future__ import annotations

from typing import Any

from theta_bench.scenario_names import TASK_UID_ALIASES


def normalize_task_state_uid(
    state_dict: dict[str, Any], expected_uid: str
) -> dict[str, Any]:
    """Validate scenario identity and adapt a legacy UID without changing its input."""

    recorded_uid = state_dict.get("uid")
    canonical_uid = TASK_UID_ALIASES.get(recorded_uid, recorded_uid)
    if canonical_uid != TASK_UID_ALIASES.get(expected_uid, expected_uid):
        raise ValueError(
            f"Recorded task uid {recorded_uid!r} is incompatible with "
            f"environment task {expected_uid!r}"
        )
    if recorded_uid == expected_uid:
        return state_dict
    return {**state_dict, "uid": expected_uid}


__all__ = ["normalize_task_state_uid"]
