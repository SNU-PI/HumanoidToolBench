"""Canonical scenario names and compatibility with previously saved identities."""

import re

SCENARIO_ALIASES = {
    "G1StickMove": "G1BallMove",
    "G1HookRetrieve": "G1BallRetrieve",
}
TASK_UID_ALIASES = {
    "g1_stick_move_teleop": "g1_ball_move_teleop",
    "g1_hook_retrieve_teleop": "g1_ball_retrieve_teleop",
}
SCENARIO_RENAMES = (
    ("StickMove", "BallMove"),
    ("HookRetrieve", "BallRetrieve"),
    ("stick_move", "ball_move"),
    ("hook_retrieve", "ball_retrieve"),
    ("stick-move", "ball-move"),
    ("hook-retrieve", "ball-retrieve"),
    ("stickmove", "ballmove"),
    ("hookretrieve", "ballretrieve"),
)


def canonicalize_scenario_names(value: str) -> str:
    """Rename scenario identifiers inside paths or serialized metadata strings."""
    for previous, canonical in SCENARIO_RENAMES:
        value = value.replace(previous, canonical)
    return value


_ENV_ID = re.compile(
    r"^(theta_bench/)?"
    r"(G1(?:BallMove|BallRetrieve|IceBreak)"
    r"|G1(?:StickMove|HookRetrieve))-L([012])-([SR])(?:-v0)?$"
)


def canonicalize_env_id(value: str) -> str:
    """Normalize canonical THETA IDs while preserving namespaces.

    Previously registered version-zero IDs remain readable in saved metadata.
    Other namespaces, tasks and version suffixes are not rewritten.
    """
    match = _ENV_ID.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return value
    namespace, scenario, level, mode = match.groups()
    scenario = SCENARIO_ALIASES.get(scenario, scenario)
    return f"{namespace or ''}{scenario}-L{level}-{mode}"


def canonicalize_env_mapping(values: dict) -> dict:
    """Normalize identity keys without accepting duplicate legacy/current IDs."""
    result = {canonicalize_env_id(key): value for key, value in values.items()}
    if len(result) != len(values):
        raise ValueError("Duplicate THETA environment identities after normalization")
    return result
