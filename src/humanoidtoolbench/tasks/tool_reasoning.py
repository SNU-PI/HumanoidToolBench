"""The Level x Mode axis every reasoning scenario is built on.

A cell is a (level, mode) pair, and the two axes are deliberately independent:

* **Level** fixes what the robot has to do. L0 pick the tool; L1 pick it and use
  it where it stands; L2 carry it to the other bench and use it there.
* **Mode** fixes how hard picking it is. S offers the correct tool beside two
  unrelated objects; R offers it beside two plausible but wrong tools.

The differences between cells are meant to be read as costs:

    R - S    the cost of the reasoning the scene demands
    L2 - L1  the cost of the locomotion
    L1 - L0  the cost of the execution

Each scenario uses the same instruction across modes. L0 asks for a pick,
while L1 and L2 ask for a pick and execution without naming the correct tool.
`humanoidtoolbench.instructions.INSTRUCTIONS` is the shared source of wording.

L0 scores the choice alone, and it borrows its definition of a pick from
SIMPLE, which scored every one of its grasp tasks the same way: the height the
object has gained over where it started, as a fraction of `LIFT_HEIGHT`, with
no hold. SIMPLE's teleoperated pick tasks used 0.1 m and a 0.8 threshold, so
that is what a pick means here: eight centimetres of daylight.

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from humanoidtoolbench.instructions import INSTRUCTIONS, LEVELS, MODES
from humanoidtoolbench.tasks.g1_toolbench_tabletop import G1ToolbenchTabletop

# SIMPLE's `_LIFT_HEIGHT` and `success_criteria` for its teleoperated pick
# tasks (`g1_wholebody_bend_pick_teleop`), reused unchanged so a pick means the
# same thing here as it did there.
LIFT_HEIGHT = 0.1
PICK_CRITERIA = 0.8

# The walk L2 adds. The base has to leave its spawn by this much before its
# direction is judged, and come within this radius of the spot in front of
# the other bench to count as having arrived.
MOVE_TRIGGER = 0.15
STAND_RADIUS = 0.40

# How long the outcome has to stand before the episode is called. The
# environment terminates the moment `check_success` is true, so without a hold
# a single frame of the ball crossing the ring, or of the block letting go,
# would end the episode before the camera had seen the result. 50 steps is
# 1 s at the 50 Hz render rate, which also asks that the ball come to rest
# inside the area rather than roll through it.
SUCCESS_HOLD = 50

# The bench slots a hand can touch, and the role each reports as: 0 the
# correct tool, 1 the confusable one, 2 an object that is no tool at all.
TOUCH_ROLE: dict[str, int] = {
    "tool": 0,
    "confusing_tool": 1,
    "irrelevant_1": 2,
    "irrelevant_2": 2,
}

# The keyword arguments a canonical environment hands its task besides the
# cell: `gym.make` passes them through the environment, and the task forwards
# them to the robot and the base task. Anything else is refused rather than
# silently dropped.
TASK_OPTIONS = frozenset({"split", "physics_dt", "sonic_config"})


class ToolReasoningTask(G1ToolbenchTabletop):
    """Shared level and mode machinery for a reasoning scenario.

    A scenario supplies its tools and what counts as the job being done; its
    wording comes from `INSTRUCTIONS`. Everything about the axis lives here so
    the three scenarios cannot drift apart in how they are scored.
    """

    metadata: dict[str, Any] = {
        **G1ToolbenchTabletop.metadata,
        "success_criteria": PICK_CRITERIA,
    }

    def __init__(self, level: int = 1, mode: str = "S", **kwargs: Any) -> None:
        unknown = set(kwargs) - TASK_OPTIONS
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword arguments: "
                f"{', '.join(sorted(unknown))}. A task takes level, mode and "
                f"{', '.join(sorted(TASK_OPTIONS))}."
            )
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.level = int(level)
        self.mode = str(mode)

        # The tool draw depends on the cell, so the config carries it rather
        # than the randomizer guessing from the task.
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        tool_cfg = self.dr_cfgs.get("tools")
        if tool_cfg is not None:
            tool_cfg.mode = self.mode
            tool_cfg.level = self.level

        self._picked: str | None = None
        self._pick_height = 0.0
        self._init_tool_height: float | None = None
        self._first_touch_role = -1
        self._tool_touched = False
        self._wrong_touches: set[str] = set()
        self._tool_touched_target = False
        self._wrong_tool_touched_target = False
        self._first_move = -1
        self._reached_bench = False
        self._dist_to_bench = 0.0
        self._spawn_xy: np.ndarray | None = None
        # The compiled model the body tables below were built for.
        self._contact_model: Any = None
        self._hand_bodies: set[int] = set()
        self._slot_bodies: dict[int, str] = {}
        self._target_bodies: set[int] = set()
        self._pelvis_body = -1

        super().__init__(**kwargs)

    # -- identity ---------------------------------------------------------

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[(self.uid, self.level, self.mode)]

    # -- where the work is done from ---------------------------------------

    @property
    def stand_xy(self) -> np.ndarray:
        """The spot the robot works from: its spawn, or at L2 the matching
        spot in front of the bench the target is on."""
        tools_dr = self.dr.get_randomizer("tools")
        spot = getattr(tools_dr, "stand_xy", None)
        if spot is None:
            spot = self.layout.robot.pose.position[:2]
        return np.asarray(spot, dtype=float)

    # -- the pick ---------------------------------------------------------

    def _tool_actor(self):
        layout = getattr(self, "_layout", None)
        return None if layout is None else layout.actors.get("tool")

    def _lifted(self, info: dict[str, Any]) -> float:
        """How much of a pick the correct tool has had, 0 to 1.

        SIMPLE's formula, unchanged: height gained over the starting height,
        divided by `LIFT_HEIGHT`, clipped. No hold, because SIMPLE kept none.
        """
        actor = self._tool_actor()
        if actor is None:
            return 0.0
        pose = info.get(self._tool_key())
        height = float(pose[2]) if pose is not None else float(actor.pose.position[2])
        if self._init_tool_height is None:
            self._init_tool_height = height
        self._pick_height = height - self._init_tool_height
        return float(np.clip(self._pick_height / LIFT_HEIGHT, 0.0, 1.0))

    def _tool_key(self) -> str:
        actor = self._tool_actor()
        return "" if actor is None else actor.asset.label

    def picked_correct_tool(self, info: dict[str, Any]) -> bool:
        """Whether the tool that got picked up is the one that works."""
        return self._lifted(info) >= PICK_CRITERIA

    # -- the process ------------------------------------------------------
    #
    # What the hands did on the way to the outcome. The scores above only
    # read the end state, so a run that grabbed the wrong tool first and
    # switched looks exactly like one that went straight to the right one.
    # These latch the first contacts and hold them for the rest of the
    # episode, so the last frame of an episode is its verdict.

    def _reset_pick_state(self) -> None:
        """Forget the last episode's pick and contacts. Called from `reset`."""
        self._init_tool_height = None
        self._pick_height = 0.0
        self._first_touch_role = -1
        self._tool_touched = False
        self._wrong_touches = set()
        self._tool_touched_target = False
        self._wrong_tool_touched_target = False
        self._first_move = -1
        self._reached_bench = False
        self._dist_to_bench = 0.0
        self._spawn_xy = None
        self._contact_model = None
        self._held = 0

    def _target_labels(self) -> list[str]:
        """MuJoCo body names of what the tool acts on. Overridden by a
        scenario whose target is more than one body."""
        layout = getattr(self, "_layout", None)
        target = None if layout is None else layout.actors.get("target")
        return [] if target is None else [target.asset.label]

    def _index_bodies(self, model) -> None:
        """Map the compiled model's bodies to the roles the metrics report.

        Every reset compiles a new model, so this runs whenever the model
        object changes. A hand is the wrist-yaw link, which carries the palm,
        and every finger link below it.
        """
        import mujoco

        def body_id(name: str) -> int:
            return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

        self._hand_bodies = set()
        for body in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if "_hand_" in name or name.endswith("_wrist_yaw_link"):
                self._hand_bodies.add(body)

        layout = getattr(self, "_layout", None)
        actors = getattr(layout, "actors", {}) if layout is not None else {}
        self._slot_bodies = {}
        for slot in TOUCH_ROLE:
            actor = actors.get(slot)
            if actor is not None and body_id(actor.asset.label) >= 0:
                self._slot_bodies[body_id(actor.asset.label)] = slot
        self._target_bodies = {
            body for body in map(body_id, self._target_labels()) if body >= 0
        }
        self._pelvis_body = body_id("pelvis")
        self._contact_model = model

    def _note_touch(self, slot: str) -> None:
        if self._first_touch_role < 0:
            self._first_touch_role = TOUCH_ROLE[slot]
        if slot == "tool":
            self._tool_touched = True
        elif not self._tool_touched:
            self._wrong_touches.add(slot)

    def _scan_contacts(self, mujoco_env) -> None:
        """Read this step's contacts for hand-on-object and tool-on-target."""
        if mujoco_env is None:
            return
        model, data = mujoco_env.mjModel, mujoco_env.mjData
        if model is not self._contact_model:
            self._index_bodies(model)
        for i in range(data.ncon):
            contact = data.contact[i]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            for first, second in ((b1, b2), (b2, b1)):
                if first in self._hand_bodies and second in self._slot_bodies:
                    self._note_touch(self._slot_bodies[second])
                if second in self._target_bodies:
                    slot = self._slot_bodies.get(first)
                    if slot == "tool":
                        self._tool_touched_target = True
                    elif slot == "confusing_tool":
                        self._wrong_tool_touched_target = True

    # -- scoring ----------------------------------------------------------

    def job_done(self, info: dict[str, Any], **kwargs: Any) -> bool:
        """Whether the scenario's effect has been achieved. L1 and L2 only."""
        raise NotImplementedError

    def _track_walk(self, mujoco_env) -> None:
        """At L2, where the base is relative to the spot it has to reach.

        The first direction is judged once, when the base has left the spot
        it was first seen at by `MOVE_TRIGGER`: towards the other bench or
        not. Arrival latches.
        """
        if mujoco_env is None or self._pelvis_body < 0:
            return
        base = np.asarray(mujoco_env.mjData.xpos[self._pelvis_body][:2], dtype=float)
        if self._spawn_xy is None:
            self._spawn_xy = base.copy()
        stand = self.stand_xy
        self._dist_to_bench = float(np.linalg.norm(stand - base))
        if self._dist_to_bench <= STAND_RADIUS:
            self._reached_bench = True
        if self._first_move < 0:
            moved = base - self._spawn_xy
            if np.linalg.norm(moved) >= MOVE_TRIGGER:
                self._first_move = int(np.dot(moved, stand - self._spawn_xy) > 0.0)

    def check_success(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> bool:
        self._scan_contacts(kwargs.get("mujoco_env"))
        if self.level == 2:
            self._track_walk(kwargs.get("mujoco_env"))
        if self.level == 0:
            # The choice is the whole task: lifting the right tool is success,
            # and lifting a wrong one simply is not this.
            done = self.picked_correct_tool(info)
        else:
            done = self.job_done(info, **kwargs)
        # Hold it, so the recording carries the result and not just the frame
        # it happened on. An outcome that lapses starts the count again.
        self._held = self._held + 1 if done else 0
        return self._held >= SUCCESS_HOLD

    def compute_reward(self, info: dict[str, Any], *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return float(self._lifted(info))

    # -- metrics -----------------------------------------------------------

    def task_info(self, info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        picked = self.picked_correct_tool(info)
        report = {
            "metrics": {
                "level": self.level,
                # 0 for S, 1 for R: the paper writes the modes as R0 and R1.
                "reasoning_mode": MODES.index(self.mode),
                "tool_lift": float(self._pick_height),
                "picked_correct_tool": bool(picked),
                "first_touch_role": int(self._first_touch_role),
                "first_touch_correct": self._first_touch_role == 0,
                "wrong_touch_count": len(self._wrong_touches),
                "tool_touched_target": bool(self._tool_touched_target),
                "wrong_tool_touched_target": bool(self._wrong_tool_touched_target),
            }
        }
        if self.level == 2:
            report["metrics"].update(
                {
                    "dist_to_target_bench": float(self._dist_to_bench),
                    "reached_target_bench": bool(self._reached_bench),
                    # -1 has not moved yet, 0 moved away, 1 moved towards.
                    "first_move_toward_target": int(self._first_move),
                }
            )
        return report
