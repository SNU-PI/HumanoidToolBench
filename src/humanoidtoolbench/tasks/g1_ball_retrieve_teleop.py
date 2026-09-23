"""Retrieve an object that can only be reached from one side. Affordance.

The object sits beyond the robot's reach on the far part of the bench, and the
robot cannot walk around to its far side. That geometry is the whole scenario:
it removes pushing as an option, because anything the robot can push the object
with drives it further away. Only a tool that can take hold of the object's far
side and carry it back does the job.

The candidates in R are the same length, the same mass and the same colour,
so nothing but their shape separates them. The stick reaches the object and can
only push it. The hook's crook goes past the object and catches it.

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
    AFFORDANCE_TOOLS,
    OOD_AFFORDANCE_TOOLS,
    OOD_CORRECT_AFFORDANCE_TOOLS,
    TARGET_HALF,
    push_ball,
    ring_marker,
)
from humanoidtoolbench.dr.tool_reasoning import ToolReasoningDRCfg
from humanoidtoolbench.tasks.registry import TaskRegistry
from humanoidtoolbench.tasks.tool_reasoning import ToolReasoningTask

# How far the ball sits from the robot, along the bench's depth axis, measured
# from the robot rather than from the bench centre so the spawn jitter cannot
# change it. It has a wall either side: inside 0.62 m a hand just picks it up,
# and past the crook's 0.935 m catch no candidate can pull it, which asks
# nothing about shape.
OBJECT_REACH = 0.77
# Where the ball ends up. Near enough that the ring is a pull towards the
# robot, far enough out that it is not under its nose: the separation is 0.23
# and the ring's radius 0.12, so the ball still travels 0.11 m to cross in.
GOAL_REACH = 0.54
# At L2 the object is out on the other half, so the area moves to meet it:
# further from the robot and over towards the seam, which shortens the drag
# back from about a metre to two thirds of one.
L2_GOAL_REACH = 0.70
L2_GOAL_LATERAL = (0.12, 0.27)
# How far out on the other half the L2 object sits, from that half's centre
# away from the robot. The other scenarios stand their crossed mark on the
# strip the tool row leaves along the far edge, 0.24 out; the ball comes a
# little in off it, because out there it sat 0.99 m from where the robot ends
# up working, past the 0.935 m catch named above.
L2_OBJECT_STRIP = 0.21
# Coming in off the strip brings the ball nearer the tool row it shares that
# half with, so it is redrawn along the half until it clears every tool by its
# own radius and a little more. Without this the ball spawned resting on a tool
# on 5 seeds in 120, where it had been 1 out on the strip.
L2_OBJECT_CLEARANCE = 0.02
L2_OBJECT_TRIES = 24
# How far back towards the seam the L2 ball comes from its half's centre. Out
# at the centre it sat 0.70 m along the bench from the area it has to reach,
# which is most of the bench's width. This shortens the drag, and the walk with
# it: `stand_xy` follows the ball, so the two cannot be set apart from here.
L2_OBJECT_SEAM_SHIFT = 0.15
# The marked area, and the whole of the success condition; same radius as the
# spatial scenario's, so a ring means one thing across the benchmark.
TARGET_RADIUS = 0.12
# How far the ring sits from the ball, towards the robot. Less the ring's
# radius, this is the travel the job asks for: 0.08 m.
SEPARATION = 0.20


def _place(dr, layout, bench, goal_bench, rng) -> None:
    """The object out of reach, and the area it has to end up in.

    `goal_bench` is the same bench below L2 and the left one at L2, where the
    area crosses the seam and the pull has to bring the object over to it.
    """
    centre, top, depth, lateral, half, base = bench
    g_centre, g_top, g_depth, g_lateral, g_half, _ = goal_bench
    drift = float(rng.uniform(-0.10, 0.10))

    ball = push_ball(rng)
    if goal_bench is bench:
        ball_bench, ball_spot = bench, dr._at(bench, OBJECT_REACH, drift)
    else:
        # L2 sends the object across to the other bench, out on its far edge,
        # and leaves the area where the robot starts. Retrieving is then a walk
        # out to the object and a drag back rather than a pull where it stands.
        ball_bench = goal_bench
        ball_spot = (
            g_centre
            + g_depth * L2_OBJECT_STRIP
            + g_lateral * (drift - L2_OBJECT_SEAM_SHIFT)
        )
        for _ in range(L2_OBJECT_TRIES):
            if dr._clear_of_tools(ball_spot, TARGET_HALF + L2_OBJECT_CLEARANCE):
                break
            drift = float(rng.uniform(-0.10, 0.10))
            ball_spot = (
                g_centre
                + g_depth * L2_OBJECT_STRIP
                + g_lateral * (drift - L2_OBJECT_SEAM_SHIFT)
            )
    dr._add(
        layout,
        "target",
        ball,
        dr._clamp(ball_spot, ball, ball_bench[0], ball_bench[4]),
        ball_bench[1],
    )
    # Nearer the robot than the object, so retrieving is still a pull towards
    # it over open bench. The area stays on this bench at every level; what L2
    # moves is the object.
    marker = ring_marker("retrieve_area", TARGET_RADIUS)
    if goal_bench is bench:
        # Measured from the ball rather than from the robot. An absolute reach
        # asks for a spot the bench may not have, and the clamp then pushes the
        # area out towards the ball: the pull collapsed to 0.026 m on the seeds
        # where the ring landed on the near edge.
        ball_xy = np.asarray(layout.actors["target"].pose.position[:2], dtype=float)
        area = ball_xy - depth * SEPARATION
    else:
        area = dr._at(bench, L2_GOAL_REACH, float(rng.uniform(*L2_GOAL_LATERAL)))
    # A ring is painted on the bench, not stood on it, so it keeps its centre
    # over the top and nothing more. Reserving its radius as well cost up to
    # 0.05 m of the pull on the seeds where it landed near an edge.
    dr._add(
        layout,
        "goal",
        marker,
        dr._clamp(area, marker, centre, half, extent=np.zeros(2)),
        top,
    )


@TaskRegistry.register("g1_ball_retrieve_teleop")
class G1BallRetrieveTeleop(ToolReasoningTask):
    uid: str = "g1_ball_retrieve_teleop"
    label: str = "G1: retrieve the object"
    description: str = (
        "Bring an object that cannot be reached from behind back to the marked "
        "area, with a tool that can pull."
    )

    reasoning_axis: str = "affordance"
    phrase: str = "retrieve the object"
    target_name: str = "object"

    recording_object_slots = (
        *ToolReasoningTask.recording_object_slots,
        "target",
        "goal",
    )

    dr_cfgs: dict[str, RandomizerCfg] = {
        **ToolReasoningTask.dr_cfgs,
        "tools": ToolReasoningDRCfg(
            tools=AFFORDANCE_TOOLS,
            correct_tool="hook",
            place_objects=_place,
            extra_builders={
                "push_ball": push_ball,
                "retrieve_area": lambda: ring_marker("retrieve_area", TARGET_RADIUS),
            },
        ),
    }
    # An OOD twin's hooks and straight sticks; see `ToolReasoningTask.ood_tools`.
    ood_tool_builders = OOD_AFFORDANCE_TOOLS
    # A correct-tool twin's hook, one shown to pull the ball back.
    ood_correct_tool_builders = OOD_CORRECT_AFFORDANCE_TOOLS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._start_distance: float | None = None
        super().__init__(*args, **kwargs)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
        from humanoidtoolbench.core.task import Task

        Task.reset(self, seed, options)
        split = self.metadata.get("split", "train")
        self.apply_scene_dr(split)

        tools_dr = self.dr.get_randomizer("tools")
        assert tools_dr is not None, "the scenario needs its tool randomizer"
        self._roles = tools_dr.apply(self.layout, split)

        self._start_distance = None
        self._reset_pick_state()
        self.reward = 0.0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    # -- scoring ----------------------------------------------------------

    def _retrieved(self, info: dict[str, Any]) -> tuple[float, bool]:
        """(distance gained towards the robot, inside the marked area)."""
        layout = getattr(self, "_layout", None)
        if layout is None:
            return 0.0, False
        target = layout.actors.get("target")
        goal = layout.actors.get("goal")
        if target is None or goal is None:
            return 0.0, False

        pose = info.get(target.asset.label)
        xy = (
            np.asarray(pose[:2], dtype=float)
            if pose is not None
            else np.asarray(target.pose.position[:2], dtype=float)
        )
        distance = float(np.linalg.norm(xy - self.stand_xy))
        if self._start_distance is None:
            self._start_distance = distance

        gained = self._start_distance - distance
        goal_xy = np.asarray(goal.pose.position[:2], dtype=float)
        inside = bool(np.linalg.norm(xy - goal_xy) <= TARGET_RADIUS)
        return gained, inside

    def job_done(self, info: dict[str, Any], **kwargs: Any) -> bool:
        del kwargs
        _gained, inside = self._retrieved(info)
        # The ring is the whole condition. A distance gate as well made most
        # of the ring a losing square; the layout does that work instead.
        return inside

    def metric_spec(self) -> dict[str, str]:
        return {
            **super().metric_spec(),
            "retrieve_gain": "float32",
            "in_target_area": "bool",
        }

    def task_info(self, info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        report = super().task_info(info, **kwargs)
        gained, inside = self._retrieved(info)
        report["metrics"].update(
            {"retrieve_gain": float(gained), "in_target_area": bool(inside)}
        )
        return report
