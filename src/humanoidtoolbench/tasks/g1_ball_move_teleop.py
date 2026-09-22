"""Push a ball to a marked spot with a stick. Spatial.

The ball sits further from the robot than a short stick can reach and closer
than a long one can. That distance is the whole scenario: both sticks push
equally well, and the only thing that decides the task is whether the tool is
long enough to touch the ball at all.

The two candidates are the same mesh at two scales, which is what this axis
asks for. They weigh the same, they look the same, and a policy that answers
from appearance has nothing to go on: it has to compare how far away the ball
is with how far each stick reaches.

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from humanoidtoolbench.core.randomizer import RandomizerCfg

from humanoidtoolbench.assets.tools import (
    SHORT_STICK_LENGTH,
    SPATIAL_TOOLS,
    push_ball,
    ring_marker,
)
from humanoidtoolbench.dr.tool_reasoning import GOAL_STRIP_DEPTH, ToolReasoningDRCfg
from humanoidtoolbench.tasks.registry import TaskRegistry
from humanoidtoolbench.tasks.tool_reasoning import ToolReasoningTask

# How far the ball sits from the robot, along the bench's depth axis. Measured
# from the robot rather than from the bench centre, so the spawn jitter cannot
# change it. Far enough that the short stick falls short even at full
# extension, near enough that the long one arrives.
BALL_REACH = 0.87
# The marked spot, and the whole of the success condition. The placer keeps
# the ball at least MIN_SEPARATION from it, so reaching the ring is always a
# push worth making.
TARGET_RADIUS = 0.12
# Where along the bench the ball and the spot may go, measured from the
# centre line: the bench's half width less the edge margin and the object's
# own radius. The spot goes to the ball's right, every episode, so the push
# runs one way and the two never swap places.
BALL_LATERAL = 0.36
SPOT_LATERAL = 0.27
# Drawing the two independently left them anywhere from 0.20 to 0.58 apart,
# so half the episodes asked for a 0.46 m push and half for a nudge. The gap
# is drawn instead, which holds the job to 0.08 to 0.22 m of travel.
MIN_SEPARATION = 0.22
MAX_SEPARATION = 0.32
# At L2 the ball has to cross to the other half, so it starts on the seam side
# of its own half rather than anywhere along it.
L2_BALL_LATERAL = (0.15, BALL_LATERAL)
# L2 puts the spot on the bench the tool row is on, so it is redrawn until it
# clears every tool by its own radius and this margin. The draw is bounded: a
# scene whose bench leaves no clear spot keeps the last one rather than looping.
SPOT_TRIES = 24
SPOT_CLEARANCE = 0.02


def _place(dr, layout, bench, goal_bench, rng) -> None:
    """The ball out past the short stick's reach, and the spot it goes to.

    `goal_bench` is the same bench below L2 and the left one at L2, where the
    spot crosses the seam and the push has to carry the ball over to it.
    """
    centre, top, depth, lateral, half, base = bench
    gap = float(rng.uniform(MIN_SEPARATION, MAX_SEPARATION))
    if goal_bench is bench:
        # The ball goes to the spot's left, so it needs a spot's worth of bench
        # to its right: it is drawn from what is left once the gap has its room.
        ball_u = float(
            rng.uniform(max(-BALL_LATERAL, gap - SPOT_LATERAL), BALL_LATERAL)
        )
    else:
        ball_u = float(rng.uniform(*L2_BALL_LATERAL))
    spot_u = ball_u - gap

    ball = push_ball(rng)
    dr._add(
        layout,
        "target",
        ball,
        dr._clamp(dr._at(bench, BALL_REACH, ball_u), ball, centre, half),
        top,
    )
    g_centre, g_top, g_depth, g_lateral, g_half, _ = goal_bench
    if goal_bench is bench:
        # At the ball's own depth, so the job is a push along the bench rather
        # than a nudge towards the robot.
        g_spot = dr._at(bench, BALL_REACH, spot_u)
    else:
        # At L2 the spot shares a bench with the tool row, so it takes the half
        # of that bench furthest from the seam, on the strip along the far
        # edge. Drawn on its own rather than a gap from the ball: the two are
        # on different benches, and what sets the push is the bench offset.
        # The spot shares this bench with the tool row, so it is redrawn until
        # it lands clear of every tool lying there. A 0.50 m stick sweeps most
        # of the bench's depth, and a spot drawn blind had one lying across it
        # in most scenes, which left the ball nothing to come to rest in.
        for _ in range(SPOT_TRIES):
            g_spot = (
                g_centre
                + g_depth * GOAL_STRIP_DEPTH
                + g_lateral * float(rng.uniform(0.0, SPOT_LATERAL))
            )
            if dr._clear_of_tools(g_spot, TARGET_RADIUS + SPOT_CLEARANCE):
                break
    marker = ring_marker("push_spot", TARGET_RADIUS)
    # A ring is painted on the bench, not stood on it, so it keeps its centre
    # over the top and nothing more; reserving its radius as well shortened the
    # push on the seeds where it landed near an edge.
    dr._add(
        layout,
        "goal",
        marker,
        dr._clamp(g_spot, marker, g_centre, g_half, extent=np.zeros(2)),
        g_top,
    )


@TaskRegistry.register("g1_ball_move_teleop")
class G1BallMoveTeleop(ToolReasoningTask):
    uid: str = "g1_ball_move_teleop"
    label: str = "G1: move the ball to the spot"
    description: str = (
        "Push a ball to a marked spot with a stick long enough to reach it."
    )

    reasoning_axis: str = "spatial"
    phrase: str = "move the ball to the target"
    target_name: str = "ball"

    dr_cfgs: dict[str, RandomizerCfg] = {
        **ToolReasoningTask.dr_cfgs,
        "tools": ToolReasoningDRCfg(
            tools=SPATIAL_TOOLS,
            correct_tool="long_stick",
            place_objects=_place,
        ),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._start_xy: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None) -> None:
        from humanoidtoolbench.core.task import Task

        Task.reset(self, seed)
        split = self.metadata.get("split", "train")
        self.apply_scene_dr(split)

        tools_dr = self.dr.get_randomizer("tools")
        assert tools_dr is not None, "the scenario needs its tool randomizer"
        self._roles = tools_dr.apply(self.layout, split)

        self._start_xy = None
        self._reset_pick_state()
        self.reward = 0.0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    # -- scoring ----------------------------------------------------------

    def _moved(self, info: dict[str, Any]) -> tuple[float, bool]:
        """(distance travelled, inside the marked spot)."""
        layout = getattr(self, "_layout", None)
        if layout is None:
            return 0.0, False
        ball = layout.actors.get("target")
        goal = layout.actors.get("goal")
        if ball is None or goal is None:
            return 0.0, False

        pose = info.get(ball.asset.label)
        xy = (
            np.asarray(pose[:2], dtype=float)
            if pose is not None
            else np.asarray(ball.pose.position[:2], dtype=float)
        )
        if self._start_xy is None:
            self._start_xy = xy.copy()

        travelled = float(np.linalg.norm(xy - self._start_xy))
        goal_xy = np.asarray(goal.pose.position[:2], dtype=float)
        inside = bool(np.linalg.norm(xy - goal_xy) <= TARGET_RADIUS)
        return travelled, inside

    def job_done(self, info: dict[str, Any], **kwargs: Any) -> bool:
        del kwargs
        _travelled, inside = self._moved(info)
        # Inside the ring, and nothing else. The distance was a second, and
        # stricter, gate on the same event: a ball plainly in the circle could
        # be 0.08 m from where it started against the 0.18 it wanted.
        return inside

    def reach_shortfall(self) -> float:
        """How far past a short stick's reach the ball starts.

        Positive means the short stick cannot get there, which is what makes
        the scene a question rather than a preference. Reported so a scene that
        stops posing the question is visible rather than silently trivial.

        Measured from the spawn rather than from `stand_xy`: the ball is on the
        right bench at every level, so the spawn is where it is first worked,
        and at L2 `stand_xy` has already moved to the bench the spot is on.
        """
        layout = getattr(self, "_layout", None)
        ball = None if layout is None else layout.actors.get("target")
        if ball is None:
            return 0.0
        xy = np.asarray(ball.pose.position[:2], dtype=float)
        spawn = np.asarray(self.layout.robot.pose.position[:2], dtype=float)
        return float(np.linalg.norm(xy - spawn) - (ARM_REACH + SHORT_STICK_LENGTH))

    def task_info(self, info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        report = super().task_info(info, **kwargs)
        travelled, inside = self._moved(info)
        report["metrics"].update(
            {
                "push_distance": float(travelled),
                "on_spot": bool(inside),
                "reach_shortfall": self.reach_shortfall(),
            }
        )
        return report


# How far the robot can reach without a tool, shoulder to fingertip with the
# lean a stationary G1 can manage. Used only to report whether a scene is
# actually posing its question.
ARM_REACH = 0.62
