"""The canonical scenarios, levels, modes and the instruction for every cell."""

from __future__ import annotations

LEVELS = (0, 1, 2)
MODES = ("S", "R")
SCENARIO_UIDS = {
    "G1BallMove": "g1_ball_move_teleop",
    "G1BallRetrieve": "g1_ball_retrieve_teleop",
    "G1IceBreak": "g1_ice_break_teleop",
}

INSTRUCTIONS: dict[tuple[str, int, str], str] = {
    (uid, level, mode): wording[0 if level == 0 else 1]
    for uid, wording in {
        "g1_ball_move_teleop": (
            "Pick the tool for moving the ball to the target.",
            "Pick the tool and move the ball to the target.",
        ),
        "g1_ball_retrieve_teleop": (
            "Pick the tool for retrieving the ball to the target.",
            "Pick the tool and retrieve the ball to the target.",
        ),
        "g1_ice_break_teleop": (
            "Pick the tool for breaking the ice blocks.",
            "Pick the tool and break the ice blocks.",
        ),
    }.items()
    for level in LEVELS
    for mode in MODES
}
