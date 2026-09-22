"""Lay out a reasoning scenario: three candidate tools and what they act on.

The three tools go down in a random order, which is the point: their positions
carry no information, so a policy that learns "the correct tool is the middle
one" learns nothing that transfers. What separates the two modes is only which
three tools are on the bench.

* **S (R0).** The correct tool and two objects that have nothing to do with
  the job. Picking it needs recognition, not reasoning.
* **R (R1).** The same three, plus one tool that looks like a plausible answer
  and is not. R is S with one thing added, so the difference between the two
  cells is exactly that thing: the instruction is word for word the same, the
  irrelevant objects are drawn the same way, and any drop in success is the
  cost of having to rule the extra tool out.

The scene is two benches butted together, one at each of the robot's hands,
with the seam straight ahead. The tools lie on the left bench, within reach of
a turn, at every level. What the tool acts on goes on the right bench, also at
every level. What moves with the level is the goal: at L0 and L1 it is on the
right bench beside the object, and at L2 it crosses to the left bench, so the
job is to bring the object over rather than to work it where it lies. The L2
goal shares its bench with the tool row, so it goes on the strip along that
bench's far edge, past the row. `stand_xy` is where the work is done from: the
spawn, or at L2 the spot level with the goal, a step along the pair. A scene
with only one bench puts the tools on it too.

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np
import transforms3d as t3d

from humanoidtoolbench.assets.benchmark import BenchmarkCatalog
from humanoidtoolbench.core.actor import ObjectActor
from humanoidtoolbench.core.randomizer import Randomizer, RandomizerCfg
from humanoidtoolbench.core.types import Pose

# How far the tool row sits from the robot, along the bench's depth axis. The
# spawn moves by up to 0.2 m between episodes, so an offset taken from the
# bench centre let that jitter through into how far away the tools actually
# were; measured from the robot, the row is the same reach every episode.
#
# Tools lie along the depth axis, handles towards the robot, which is how they
# fall to hand on a bench; they are spaced sideways, so all three are the same
# reach away and the slot a tool landed in says nothing about it.
# One line for every scenario and for both levels. 0.63 leaves the longest
# tool's handle 3 cm over the bench's near edge, which reads as a tool to hand
# rather than one falling off, and keeps its far end clear of the strip behind
# the row where L2 stands the goal or the second ice block. Further out and
# that strip goes; nearer and the handles jut, 15 cm of them at 0.49. The hard
# floor is 0.47, where the far spawns push a tool's centre of mass off the edge
# and the clamp pulling it back breaks the row into two lines.
TOOL_ROW_REACH = 0.63
# What the tool row keeps between a tool's centre of mass and the bench's near
# edge. Smaller than EDGE_MARGIN because only the centre has to be on the
# bench here: the handle is meant to be over the edge, towards the robot.
ROW_EDGE_MARGIN = 0.02
# Clear space between neighbours; the fixed 0.22 pitch was narrower than the
# widest tools, and pairs overlapped, one by 8 cm.
TOOL_GAP = 0.06
# How far off the row a tool may turn; 0.50 m at the old 0.12 rad ate a gap.
TOOL_JITTER = 0.06
EDGE_MARGIN = 0.06
# The bench slots the tool row fills. Everything else a scenario places is its
# own, and is what the level may send across to the other bench.
TOUCHABLE_SLOTS = ("tool", "confusing_tool", "irrelevant_1", "irrelevant_2")
# Where the L2 goal sits on the tool bench, from its centre away from the
# robot. At L2 the goal shares a bench with the tool row, and the row takes the
# near two thirds of it: a 0.50 m tool laid in the row reaches 0.03 m past the
# bench centre, so a goal at the centre had a tool lying inside it in 38 of 40
# scenes. This is the strip left along the far edge, clear of those tips by
# more than the goal's own 0.12 m radius.
GOAL_STRIP_DEPTH = 0.24


def _quat_mul(a, b):
    """Hamilton product, w first."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _segment_gap(start, end, point) -> float:
    """Distance from `point` to the segment `start`-`end`, in the bench plane."""
    span = end - start
    length = float(np.dot(span, span))
    if length <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, span) / length, 0.0, 1.0))
    return float(np.linalg.norm(start + t * span - point))


def _turn(asset, yaw: float) -> list[float]:
    """Yaw about the bench, after the tool's own roll onto its flat side."""
    q = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    roll = float(getattr(asset, "roll", 0.0) or 0.0)
    if roll:
        turn = np.zeros(4)
        turn[0] = np.cos(0.5 * roll)
        turn[1 + int(getattr(asset, "length_axis", 0))] = np.sin(0.5 * roll)
        q = _quat_mul(q, turn)
    return [float(v) for v in q]


class _Fixed:
    """An rng stand-in that always draws one particular asset.

    Replaying an episode means rebuilding the tool it drew, and the builders
    take a generator rather than an id. Handing them this keeps one code path
    for both the draw and the replay.
    """

    def __init__(self, asset_id: str, role: str) -> None:
        del role  # the builder knows the pool; the id is the same in all of them
        # `draw` reads this and returns the mesh itself. Resolving to an index
        # here cannot work: 48 meshes were added across four slots and some sit
        # in two pools at different positions, so an index found in one pool
        # picked a different tool out of the other.
        self.asset_id = asset_id
        self._index = 0

    def integers(self, _n: int) -> int:
        return self._index


class ToolReasoningDR(Randomizer):
    """Place a scenario's tool row and its object."""

    def __init__(self, cfg: ToolReasoningDRCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self._catalog: BenchmarkCatalog | None = None
        self.placed: dict[str, ObjectActor] = {}
        self.correct_slot: int = -1
        self.stand_xy: np.ndarray | None = None

    # -- geometry ---------------------------------------------------------

    @staticmethod
    def _bench(layout, key: str = "table"):
        """(centre, top, depth axis, lateral axis, half extent) or None.

        The axes are the bench's own rather than a constant: the robot stands
        along +x in one layout and +y in another, and a hard-coded axis silently
        puts the tool row out of reach. Which of the bench's two axes is the
        depth is decided by where the robot stands, but the axis itself belongs
        to the bench, so rows and objects lie square to it. Taking the heading
        straight from the robot's base instead raked everything across the bench
        whenever the robot stood off to one side, and that cost most of the
        bench: the clamp bites on each world axis separately, so a raked row
        runs out well before the bench's width says it should.
        """
        # One slab, two working halves. The bench at the robot's right hand and
        # the one at its left are the two ends of the same top, so a scene that
        # has only `table` splits it here rather than carrying a second slab,
        # which is what keeps the top seamless and the legs at its four outer
        # corners. A scene that really does have a `table2` is left alone.
        halved = layout.actors.get("table2") is None and key in ("table", "table2")
        table = layout.actors.get("table" if halved else key)
        if table is None or not hasattr(table, "size"):
            return None
        centre = np.asarray(table.pose.position[:2], dtype=float)
        top = float(table.pose.position[2] + 0.5 * table.size[2])
        half = 0.5 * np.asarray(table.size[:2], dtype=float)

        robot = getattr(layout, "robot", None)
        base = (
            np.asarray(robot.pose.position[:2], dtype=float)
            if robot is not None
            else centre - np.array([1.0, 0.0])
        )
        away = centre - base
        if float(np.linalg.norm(away)) <= 1e-6:
            away = np.array([1.0, 0.0])
        # Whichever of the bench's axes lies closer to the line the robot looks
        # along becomes the depth, pointed away from the robot.
        rotation = t3d.quaternions.quat2mat(table.pose.quaternion)
        depth = max(
            (rotation[:2, 0], rotation[:2, 1]),
            key=lambda axis: abs(float(np.dot(axis, away))),
        )
        if float(np.dot(depth, away)) < 0.0:
            depth = -depth
        depth = np.asarray(depth, dtype=float)
        depth = depth / float(np.linalg.norm(depth))
        lateral = np.array([-depth[1], depth[0]])
        if halved:
            across = int(np.argmax(np.abs(lateral)))
            half = half.copy()
            half[across] *= 0.5
            # `lateral` points to the robot's left, which is the left bench.
            centre = centre + lateral * (
                (1.0 if key == "table2" else -1.0) * half[across]
            )
        return centre, top, depth, lateral, half, base

    @staticmethod
    def _at(bench, reach: float, lateral_offset: float = 0.0):
        """A spot `reach` metres from the robot along the bench's depth axis.

        Measured from the robot rather than from the bench centre, so the spawn
        jitter moves the whole scene with the robot instead of changing how far
        away everything is. `lateral_offset` still runs from the centre line,
        which is what keeps a row centred on the bench it lies on.
        """
        centre, _, depth, lateral, _, base = bench
        standoff = float(np.dot(centre - base, depth))
        # Sideways as well as in depth: the bench pair is built around the
        # origin and the spawn box is centred on it, so the robot's own offset
        # from the origin is how far the scene has to slide to stay in front of
        # it. Without this the scene stayed put and the robot arrived up to
        # 0.1 m off to one side of it.
        drift = float(np.dot(base, lateral))
        return centre + depth * (reach - standoff) + lateral * (lateral_offset + drift)

    def _clear_of_tools(self, spot, radius: float) -> bool:
        """Whether a mark of this radius lands clear of every tool already down.

        A tool is taken as the segment its length sweeps in the bench plane,
        which is what a stick lying flat actually covers. Only the marks that
        share a bench with the tool row need this, and only after the row is
        placed, so it reads `self.placed` rather than the layout.
        """
        spot = np.asarray(spot, dtype=float)[:2]
        for name in ("tool", "confusing_tool", "irrelevant_1", "irrelevant_2"):
            actor = self.placed.get(name)
            if actor is None:
                continue
            asset = actor.asset
            axis = np.zeros(3)
            axis[int(getattr(asset, "length_axis", 0))] = 1.0
            heading = t3d.quaternions.quat2mat(actor.pose.quaternion) @ axis
            centre = np.asarray(actor.pose.position[:2], dtype=float)
            half = 0.5 * float(asset.length) * heading[:2]
            if _segment_gap(centre - half, centre + half, spot) < radius:
                return False
        return True

    @staticmethod
    def _clamp(spot, asset, centre, half, extent=None):
        """Pull a spot in until the asset lies wholly on the bench.

        `extent` is the asset's half-size along each world axis. Without it the
        clamp assumes the object is as wide as it is long, which for a 0.45 m
        tool leaves a few centimetres of a 0.9 m bench usable and stacks every
        tool on the same spot.
        """
        if extent is None:
            extent = np.full(2, 0.5 * float(asset.footprint))
        limit = np.maximum(
            np.asarray(half, dtype=float) - np.asarray(extent) - EDGE_MARGIN, 0.0
        )
        return np.clip(np.asarray(spot, dtype=float), centre - limit, centre + limit)

    @staticmethod
    def _row_reach(half, depth, lateral, half_depth, half_lateral) -> float:
        """How far along the row this tool can sit before `_clamp` bites.

        The clamp limits each world axis on its own, so a row heading that is
        not axis-aligned runs out before the bench's width says it should.
        """
        extent = np.abs(depth) * half_depth + np.abs(lateral) * half_lateral
        limit = np.maximum(np.asarray(half, dtype=float) - extent - EDGE_MARGIN, 0.0)
        along = np.abs(np.asarray(lateral, dtype=float))
        return float(
            min(limit[k] / along[k] for k in range(len(along)) if along[k] > 1e-6)
        )

    @staticmethod
    def _laid(asset, depth, rng) -> tuple[float, float, float]:
        """(yaw, half width across the row, half depth along it).

        A scanned mesh is long along whichever axis it happened to be scanned
        on, so the yaw that points a hammer down the bench is a quarter turn
        from the one that points a stick down it. That turn says nothing about
        which end leads: a quarter turn sends the +y end towards the robot and
        no turn sends the +x end away, so the handle wants to be at the high
        end in the first case and the low end in the second, and half a turn
        more when it is not. Every tool gets this, so a hammer is never laid
        head-first under the hand.

        The footprint comes back with the yaw because it depends on it: a
        0.50 m tool turned 0.12 rad reaches 6 cm further sideways.
        """
        long_axis = int(getattr(asset, "length_axis", 0))
        axis_turn = 0.5 * np.pi if long_axis == 1 else 0.0
        wants_high = long_axis == 1
        handle_at_high = getattr(asset, "handle_at_high", None)
        # None means the tool is the same at both ends, so there is nothing to
        # point and it is left as it lies.
        handle_turn = (
            np.pi
            if handle_at_high is not None and bool(handle_at_high) != wants_high
            else 0.0
        )
        jitter = float(rng.uniform(-TOOL_JITTER, TOOL_JITTER))
        yaw = float(np.arctan2(depth[1], depth[0]) + axis_turn + handle_turn + jitter)
        length = float(asset.length)
        across = float(getattr(asset, "cross_extent", asset.grip_width))
        turn, straight = abs(np.sin(jitter)), abs(np.cos(jitter))
        return (
            yaw,
            0.5 * (length * turn + across * straight),
            0.5 * (length * straight + across * turn),
        )

    def _add(self, layout, name: str, asset, xy, top: float, yaw: float = 0.0):
        layout.add_object(name, asset)
        actor = layout.actors[name]
        quaternion = _turn(asset, yaw)
        # `xy` is where the object should look like it is, which for a scanned
        # mesh is not where its origin goes: the origin sits wherever the scan
        # was taken from. Shifting by the shape's own middle is what puts a row
        # of different meshes on one line rather than merely their origins.
        offset = np.asarray(
            getattr(asset, "centre_offset", None) or [0.0, 0.0, 0.0], dtype=float
        )
        if offset[:2].any():
            offset = t3d.quaternions.rotate_vector(offset, quaternion)
        actor.pose = Pose(
            position=[
                float(xy[0]) - float(offset[0]),
                float(xy[1]) - float(offset[1]),
                top + float(asset.height),
            ],
            quaternion=quaternion,
        )
        self.placed[name] = actor
        return actor

    # -- the draw ---------------------------------------------------------

    def _irrelevant(self, rng, count: int = 2) -> list[tuple[str, Any]]:
        """Objects that are plainly not tools for any of these jobs."""
        from humanoidtoolbench.assets import distractors

        return [
            (f"irrelevant_{i}", asset)
            for i, asset in enumerate(distractors.sample(count, rng))
        ]

    def _tool_assets(self, rng) -> list[tuple[str, Any]]:
        """The bench, in a random order.

        S is the correct tool and two irrelevant objects. R is the same, with
        one confusable tool added, drawn from the scenario's own set so the
        thing that has to be ruled out varies between episodes.
        """
        cfg = self.cfg
        candidates = [("correct", cfg.tools[cfg.correct_tool](rng))]

        if cfg.mode == "R":
            confusable = [r for r in cfg.tools if r != cfg.correct_tool]
            role = confusable[int(rng.integers(len(confusable)))]
            candidates.append((role, cfg.tools[role](rng)))

        candidates += self._irrelevant(rng)
        order = rng.permutation(len(candidates))
        return [candidates[int(i)] for i in order]

    def apply(self, layout, split: str = "train") -> dict[str, ObjectActor]:
        del split
        if self._inner_state is not None:
            return self._restore(layout)

        self.placed = {}
        bench = self._bench(layout)
        if bench is None:
            return {}
        centre, top, depth, lateral, half, base = bench

        # Draw from the global stream so `DRManager.reset(seed)` reaches this
        # generator too and an episode replays with the same layout.
        rng = np.random.default_rng(int(np.random.randint(0, 2**32, dtype=np.uint32)))

        drawn = self._tool_assets(rng)
        self.correct_slot = next(
            i for i, (role, _) in enumerate(drawn) if role == "correct"
        )

        # The tool row goes on the left bench, with that bench's own axes: its
        # depth runs from the robot out along the bench, so the handles are the
        # near end of each tool, and its lateral runs square to the robot, so
        # every slot in the row is the same reach away. A scene with only one
        # bench takes the near half of it for the row and the far half for the
        # target.
        tool_bench = self._bench(layout, "table2") or bench
        t_centre, t_top, t_depth, t_lateral, t_half, _ = tool_bench
        row = self._at(tool_bench, TOOL_ROW_REACH)
        # The irrelevant objects are numbered along the row, from the seam
        # outwards. Nothing tells the two apart: both report role 2 and both
        # get a recording column, so the number is only there to give the two
        # columns a stable shape however the permutation came out. The draw
        # order they were picked in survives on `reasoning_role`.
        # Yaw and the footprint it gives, so the row packs by what it lays.
        laid = [self._laid(asset, t_depth, rng) for _, asset in drawn]
        widths = [half_lateral for _, half_lateral, _ in laid]
        # Spread the bench's slack over the gaps; a gap it cannot give makes
        # the clamp pull an end in, which pushes it into its neighbour.
        usable = 2 * min(
            self._row_reach(t_half, t_depth, t_lateral, half_depth, half_lateral)
            for _, half_lateral, half_depth in laid
        )
        slack = usable - 2 * float(np.sum(widths))

        # Never below zero: that would command the overlap, not just allow it.
        gap = (
            max(0.0, min(TOOL_GAP, slack / max(len(drawn) - 1, 1)))
            if len(drawn) > 1
            else 0.0
        )
        pitches = [widths[i] + widths[i + 1] + gap for i in range(len(drawn) - 1)]
        offsets = np.concatenate([[0.0], np.cumsum(pitches)]) if drawn else np.zeros(0)
        offsets = offsets - offsets.mean()
        # Slide the row whole until both ends fit; moving keeps the spacing.
        reaches = [
            self._row_reach(t_half, t_depth, t_lateral, half_depth, half_lateral)
            for _, half_lateral, half_depth in laid
        ]
        if len(offsets):
            low = max(-r - o for r, o in zip(reaches, offsets))
            high = min(r - o for r, o in zip(reaches, offsets))
            offsets = offsets + (
                float(np.clip(0.0, low, high)) if low <= high else 0.5 * (low + high)
            )

        irrelevant = 0
        for slot, (role, asset) in enumerate(drawn):
            yaw, half_lateral, half_depth = laid[slot]
            # Along the depth axis only the tool's centre of mass has to be on
            # the bench: a tool lies with its handle towards the robot and the
            # handle over the near edge, which is how one falls to hand. Held
            # to its whole footprint instead, a 0.50 m tool was dragged back
            # while a short one was not, and the row that should have been one
            # line spread from 0.52 to 0.73 m out. Sideways it is still clamped
            # by its full width, so nothing hangs off the ends.
            extent = np.abs(t_lateral) * half_lateral + np.abs(t_depth) * (
                ROW_EDGE_MARGIN - EDGE_MARGIN
            )
            spot = self._clamp(
                row + t_lateral * float(offsets[slot]), asset, t_centre, t_half, extent
            )
            if role == "correct":
                name = "tool"
            elif role.startswith("irrelevant_"):
                irrelevant += 1
                name = f"irrelevant_{irrelevant}"
            else:
                name = "confusing_tool"
            # Flat on the bench, one grip thickness, own rolling friction.
            actor = self._add(layout, name, asset, spot, t_top, yaw=yaw)
            actor.reasoning_role = role

        # The scenario's own objects go on the right bench, out of reach of a
        # bare hand, which is what makes a tool necessary at all. The goal they
        # have to reach is on that bench too until L2 moves it across.
        goal_bench = self._goal_bench(layout, bench)
        self.cfg.place_objects(self, layout, bench, goal_bench, rng)
        self.stand_xy = self._stand_spot(bench, goal_bench)

        state = {
            "tools": [role for role, _ in drawn],
            "correct_slot": self.correct_slot,
            "poses": {
                name: {
                    "position": list(a.pose.position),
                    "quaternion": list(a.pose.quaternion),
                    "uid": a.asset.uid,
                }
                for name, a in self.placed.items()
            },
        }
        super()._transient(state)
        return self.placed

    def _goal_bench(self, layout, bench):
        """The bench the scenario's goal goes on.

        Below L2 that is the bench the objects are already on, with its own
        axes. At L2 it is the left bench, squared to the bench pair rather than
        to the robot's line of sight, for the reason below.
        """
        centre, top, depth, lateral, half, base = bench
        if self.cfg.level != 2:
            return bench
        other = self._bench(layout, "table2")
        if other is None:
            raise ValueError("L2 puts the goal on table2, which this scene lacks")
        centre2, top2, _, _, half2, _ = other
        # The bench pair's own axes rather than the robot's line of sight to
        # this bench. The tool row is laid on that line, which runs across the
        # bench at an angle, so a goal laid on it swings into the row at one
        # end of its draw or the other. Square to the pair, the strip along the
        # far edge is clear of the row for every draw.
        along = centre2 - centre
        along = along / float(np.linalg.norm(along))
        out = np.array([along[1], -along[0]])
        if float(np.dot(out, depth)) < 0.0:
            out = -out
        return centre2, top2, out, along, half2, base

    def _stand_spot(self, bench, goal_bench):
        """Where the work is done from: the spawn, or a step along the bench.

        At L2 one of the scenario's own objects has crossed to the other bench,
        so the work finishes level with it: the spawn slid along the bench pair
        until it is square in front, which keeps the robot the same clear
        distance from the bench edge as it spawned. Which one crossed is read
        off the placement rather than assumed, because the scenarios differ:
        the spatial one sends its goal across and the affordance one its
        object. A scenario that sends neither keeps the spawn, since walking
        off would be walking away from the job.
        """
        base = np.asarray(bench[5], dtype=float)
        if goal_bench is bench:
            return base
        along = np.asarray(goal_bench[3], dtype=float)
        near, far = np.asarray(bench[0]), np.asarray(goal_bench[0])
        for name, actor in self.placed.items():
            # The scenario's own objects, whatever it calls them: the spatial
            # one sends its goal across, the affordance one its object, and the
            # physical one its second block. The tool row never crosses.
            if name in TOUCHABLE_SLOTS:
                continue
            spot = np.asarray(actor.pose.position[:2], dtype=float)
            # Which bench it landed on, by which centre it sits nearer. Taking
            # the furthest object instead picked the wrong one whenever the two
            # ended up on opposite benches.
            if np.linalg.norm(spot - far) < np.linalg.norm(spot - near):
                return base + along * float(np.dot(spot - base, along))
        return base

    def _restore(self, layout) -> dict[str, ObjectActor]:
        """Rebuild a recorded layout so a replay sees the same bench."""
        state = self._inner_state
        self.placed = {}
        self.correct_slot = int(state["correct_slot"])
        bench = self._bench(layout)
        if self._catalog is None:
            self._catalog = BenchmarkCatalog()

        for name, saved in state["poses"].items():
            asset = self.cfg.asset_for_uid(saved["uid"], self._catalog)
            layout.add_object(name, asset)
            actor = layout.actors[name]
            actor.pose = Pose(
                position=list(saved["position"]), quaternion=list(saved["quaternion"])
            )
            self.placed[name] = actor

        # After the rebuild, so the stand spot sees whether the episode had a
        # goal to cross to.
        if bench is not None:
            self.stand_xy = self._stand_spot(bench, self._goal_bench(layout, bench))
        return self.placed


@dataclass
class ToolReasoningDRCfg(RandomizerCfg):
    # The scenario's confusable set, by role. `correct_tool` names the one that
    # works; the rest are the R mode's distractors.
    tools: dict[str, Callable[[], Any]] = field(default_factory=dict)
    correct_tool: str = ""
    mode: str = "S"
    level: int = 0
    # Places whatever the tool acts on. Scenario-specific, so the scenario
    # supplies it rather than this generator guessing.
    place_objects: Callable[..., None] = lambda *_: None
    randmizer_class: type[Randomizer] = ToolReasoningDR

    def asset_for_uid(self, uid: str, catalog: BenchmarkCatalog):
        """Rebuild an asset from its uid, whichever source it came from.

        A reasoning tool's uid is `tool_<role>_<asset id prefix>`, because a
        slot holds a pool and the role alone would not say which mesh the
        episode drew. The prefix is matched back against the pools, whichever
        pool the role draws from (the two sticks share one), the OOD pools
        included so that an evaluation on unseen tools replays too. An irrelevant
        object carries its id the same way. A scenario's own objects are
        rebuilt by the builder registered under their exact uid.
        """
        from humanoidtoolbench.assets import distractors
        from humanoidtoolbench.assets import tools as tool_assets

        if uid.startswith("tool_"):
            for role, build in self.tools.items():
                if not uid.startswith(f"tool_{role}_"):
                    continue
                prefix = uid[len(f"tool_{role}_") :]
                for table in (tool_assets.ASSET_POOLS, tool_assets.OOD_ASSET_POOLS):
                    for ids in table.values():
                        for asset_id in ids:
                            if asset_id.startswith(prefix):
                                return build(_Fixed(asset_id, role))
                return build(None)
        if uid.startswith("irrelevant_"):
            return distractors.rebuild(uid)
        if uid in self.extra_builders:
            return self.extra_builders[uid]()
        for extra in self.extra_assets:
            asset = extra()
            if getattr(asset, "uid", None) == uid:
                return asset
        return catalog.load(uid)

    # Scenario objects that a replay has to be able to rebuild by uid.
    extra_assets: tuple[Callable[..., Any], ...] = ()
    # The scenario's own objects, keyed by the exact uid they are placed
    # under, so a replay can rebuild them. Called with no arguments.
    extra_builders: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def to_dict(self):
        """The recorded copy of this config, with the builders by name.

        The recorder writes this into episodes.jsonl, which a function cannot
        go into. A replay never reads the builders back from there: it uses
        the task's own config and only the drawn state from the record.
        """
        cfg = super().to_dict()
        cfg["tools"] = {
            role: _builder_name(build) for role, build in self.tools.items()
        }
        cfg["place_objects"] = _builder_name(self.place_objects)
        cfg["extra_assets"] = [_builder_name(build) for build in self.extra_assets]
        cfg["extra_builders"] = {
            uid: _builder_name(build) for uid, build in self.extra_builders.items()
        }
        return cfg


def _builder_name(build: Callable[..., Any]) -> str:
    if isinstance(build, partial):
        args = ", ".join(repr(arg) for arg in build.args)
        return f"{_builder_name(build.func)}({args})"
    return f"{build.__module__}.{build.__qualname__}"
