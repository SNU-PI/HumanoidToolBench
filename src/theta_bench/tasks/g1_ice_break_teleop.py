"""Break an ice block. Physical.

Reasoning mode adds a fly swatter, paint roller, or plunger beside the metal
hammer. These decoys reuse the former foam hammer's low mass and whole-tool
compliance. Their visible structure cues striking suitability, while the
declared physical properties are a benchmark abstraction of weak tools.

MuJoCo does not fracture bodies, so the block is one solid cube that a hard
enough blow takes out of play, replaced in place by the eight shards it
carried inside it. What counts is what arrives through the contact and how
fast the tool was closing. A compliant tool spreads its momentum over time,
but any tool that delivers a qualifying impact can break the block.

THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from theta_bench.core.randomizer import RandomizerCfg

from theta_bench.assets.tools import (
    ASSET_POOLS,
    ICE_BLOCK_HALF,
    ICE_GRID,
    ICE_HALF,
    ICE_RGBA,
    OOD_PHYSICAL_TOOLS,
    PHYSICAL_TOOLS,
    foam_hammer,
    ice_block,
    ice_offsets,
    ice_shard,
)
from theta_bench.dr.tool_reasoning import (
    EDGE_MARGIN,
    TOOL_ROW_REACH,
    ToolReasoningDRCfg,
)
from theta_bench.tasks.registry import TaskRegistry
from theta_bench.tasks.tool_reasoning import ToolReasoningTask

# Benchmark fracture score: peak qualifying normal contact force times the
# physics step, in newton seconds. This is not a full impact integral or a
# measured material constant. Keep the existing threshold when replacing the
# foam meshes; sufficiently driven compliant tools can still exceed it.
FRACTURE_IMPULSE = 0.042
# Once broken, the shards have to actually come apart, not merely be released.
SCATTER = 0.03
# How fast the tool must close on the ice to count as a blow, in m/s. A swing
# arrives at 1 m/s and a lean at a tenth of that; force cannot tell them
# apart, and without this the block broke on a hammer lowered at 5 cm/s.
STRIKE_SPEED = 0.5
# The push each shard gets outwards on breaking, in m/s: enough to part.
BURST = 0.6
# How far the blocks sit from the robot, along the bench's depth axis. Measured
# from the robot rather than from the bench centre, so the spawn jitter cannot
# change how far the robot has to swing.
BLOCK_REACH = 0.665
# Two blocks, because breaking one is a single swing and the level is meant to
# ask for more than that. Below L2 they share a bench, half this far either
# side of the row centre; at L2 one crosses, so the second is a walk away.
ICE_BLOCKS = 2
SHARDS_PER_BLOCK = ICE_GRID[0] * ICE_GRID[1] * ICE_GRID[2]
BLOCK_SPREAD = 0.20
BLOCK_JITTER = 0.08
# How far past the tool row the block that crosses at L2 stands. It shares a
# bench with the row, so it has to stand behind it rather than among it, and it
# is measured from the row rather than from the bench: the row is placed from
# the robot, so a depth fixed to the bench walked into it as the spawn moved.
# At spawn y = 0.75 the two overlapped by a centimetre and at 0.65 by eleven,
# which is a hammer's claw through the ice.
#
# Half the longest tool, the block's own half width, and a margin.
CROSSED_BLOCK_CLEARANCE = 0.21 + 0.08 + 0.04
# How close that block may sit to the bench's back edge. The usual edge margin
# keeps a 0.06 m rim clear all round, which is a rim this block cannot spare:
# it has to stand behind a tool row that reaches to within 0.19 m of the back
# edge at the nearest spawn. A centimetre is enough for a block resting on a
# top, and it buys the 0.04 m of daylight that keeps a hammer's claw out of it.
CROSSED_BLOCK_EDGE_MARGIN = 0.01
# How many places along its half the crossed block may try before it settles
# for the last one. Bounded, so a crowded bench costs a scene rather than a
# hang.
CROSSED_BLOCK_TRIES = 24


def ice_block_at(block: int):
    """The block for one slot, labelled apart so two can share a scene.

    MuJoCo names a body after the asset's label, so two blocks built from the
    same builder would collide on that name.
    """
    asset = ice_block()
    asset.uid = asset.label = f"ice_block_{block}"
    return asset


def ice_shard_at(block: int, index: int):
    """One shard of one block, labelled apart for the same reason."""
    asset = ice_shard(index)
    asset.uid = asset.label = f"ice_shard_{block}_{index}"
    return asset


def _place(dr, layout, bench, goal_bench, rng) -> None:
    """Two blocks, and the shards each holds inside.

    Below L2 both sit on the bench the robot starts at, either side of the row
    centre. At L2 the second crosses to the other bench, so reaching it is a
    walk and breaking both is what the level asks for.
    """
    same_bench = goal_bench is bench
    for block in range(ICE_BLOCKS):
        own = bench if (same_bench or block == 0) else goal_bench
        centre, top, half = own[0], own[1], own[4]
        drift = float(rng.uniform(-BLOCK_JITTER, BLOCK_JITTER))
        if same_bench:
            drift += BLOCK_SPREAD if block else -BLOCK_SPREAD

        cube = ice_block_at(block)
        reach = (
            BLOCK_REACH if own is bench else TOOL_ROW_REACH + CROSSED_BLOCK_CLEARANCE
        )
        spot = dr._at(own, reach, drift)
        if own is not bench:
            # Standing behind the row is not enough on its own: the row runs
            # the length of this half, so the block is redrawn along it until
            # it clears every tool by its own half width.
            for _ in range(CROSSED_BLOCK_TRIES):
                if dr._clear_of_tools(spot, ICE_BLOCK_HALF + CROSSED_BLOCK_EDGE_MARGIN):
                    break
                drift = float(rng.uniform(-BLOCK_SPREAD, BLOCK_SPREAD))
                spot = dr._at(own, reach, drift)
        if own is bench:
            spot = dr._clamp(spot, cube, centre, half)
        else:
            # Only the depth rim is given up; sideways it keeps the usual one.
            rim = ICE_BLOCK_HALF - (EDGE_MARGIN - CROSSED_BLOCK_EDGE_MARGIN)
            extent = np.abs(own[2]) * rim + np.abs(own[3]) * ICE_BLOCK_HALF
            spot = dr._clamp(spot, cube, centre, half, extent)
        dr._add(layout, f"ice_{block}", cube, spot, top)

        # The shards start where they will be when the block breaks, so letting
        # them go is only a matter of switching them on.
        for index, offset in enumerate(ice_offsets()):
            actor = dr._add(
                layout,
                f"shard_{block}_{index}",
                ice_shard_at(block, index),
                spot + np.asarray(offset[:2], dtype=float),
                top + ICE_BLOCK_HALF + offset[2] - ICE_HALF,
            )
            actor.ice_index = index


@TaskRegistry.register("g1_ice_break_teleop")
class G1IceBreakTeleop(ToolReasoningTask):
    uid: str = "g1_ice_break_teleop"
    label: str = "G1: break the ice block"
    description: str = (
        "Fracture a block of ice with a tool that delivers enough impact."
    )

    reasoning_axis: str = "physical"
    phrase: str = "break the ice block"
    target_name: str = "ice block"

    # The blocks and their shards, so a recording carries the whole scene and
    # can be posed back from it. Without them a kinematic replay leaves the ice
    # wherever the reset put it while everything around it moves.
    recording_object_slots = (
        *ToolReasoningTask.recording_object_slots,
        *(f"ice_{block}" for block in range(ICE_BLOCKS)),
        *(
            f"shard_{block}_{index}"
            for block in range(ICE_BLOCKS)
            for index in range(SHARDS_PER_BLOCK)
        ),
    )

    dr_cfgs: dict[str, RandomizerCfg] = {
        **ToolReasoningTask.dr_cfgs,
        "tools": ToolReasoningDRCfg(
            tools=PHYSICAL_TOOLS,
            correct_tool="metal_hammer",
            place_objects=_place,
            extra_builders={
                **{
                    f"tool_foam_hammer_{uid[:8]}": partial(foam_hammer, uid=uid)
                    for uid in ASSET_POOLS["foam_hammer"]
                },
                **{
                    f"ice_block_{block}": partial(ice_block_at, block)
                    for block in range(ICE_BLOCKS)
                },
                **{
                    f"ice_shard_{block}_{index}": partial(ice_shard_at, block, index)
                    for block in range(ICE_BLOCKS)
                    for index in range(SHARDS_PER_BLOCK)
                },
            },
        ),
    }
    # An OOD twin's hammers and decoys; see `ToolReasoningTask.ood_tools`.
    ood_tool_builders = OOD_PHYSICAL_TOOLS

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._peak_impulse = [0.0] * ICE_BLOCKS
        self._broken = [False] * ICE_BLOCKS
        self._broken_count = 0
        self._prepared = False
        self._closing: dict[tuple[int, int], float] = {}
        super().__init__(*args, **kwargs)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
        from theta_bench.core.task import Task

        Task.reset(self, seed, options)
        split = self.metadata.get("split", "train")
        self.apply_scene_dr(split)

        tools_dr = self.dr.get_randomizer("tools")
        assert tools_dr is not None, "the scenario needs its tool randomizer"
        self._roles = tools_dr.apply(self.layout, split)

        self._peak_impulse = [0.0] * ICE_BLOCKS
        self._broken = [False] * ICE_BLOCKS
        self._broken_count = 0
        self._prepared = False
        self._closing = {}
        self._reset_pick_state()
        self.reward = 0.0
        self.apply_isaac_dr(split)
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    # -- the blow ---------------------------------------------------------

    def _shard_labels(self, block: int) -> list[str]:
        layout = getattr(self, "_layout", None)
        if layout is None:
            return []
        return [
            actor.asset.label
            for name, actor in layout.actors.items()
            if name.startswith(f"shard_{block}_")
        ]

    def _block_label(self, block: int) -> str | None:
        layout = getattr(self, "_layout", None)
        cube = None if layout is None else layout.actors.get(f"ice_{block}")
        return None if cube is None else cube.asset.label

    def _block_labels(self) -> list[str]:
        """What counts as each block: itself, or its shards once it is gone."""
        labels = []
        for block in range(ICE_BLOCKS):
            cube = self._block_label(block)
            if self._broken[block] or cube is None:
                labels.append(self._shard_labels(block))
            else:
                labels.append([cube])
        return labels

    def _target_labels(self) -> list[str]:
        """Every body that is still ice, whichever block it belongs to."""
        return [label for group in self._block_labels() for label in group]

    # -- the block, whole and broken --------------------------------------

    def _free_joint(self, model, label: str) -> int:
        """Where this body's free joint starts in qpos, or -1."""
        import mujoco

        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
        if body < 0 or int(model.body_jntnum[body]) != 1:
            return -1
        joint = int(model.body_jntadr[body])
        if int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
            return -1
        return int(model.jnt_qposadr[joint])

    def _free_dof(self, model, label: str) -> int:
        """Where this body's free joint starts in qvel, or -1."""
        import mujoco

        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
        if body < 0 or int(model.body_jntnum[body]) != 1:
            return -1
        return int(model.jnt_dofadr[int(model.body_jntadr[body])])

    def _show(self, model, label: str, visible: bool) -> None:
        import mujoco

        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
        if body < 0:
            return
        adr, count = int(model.body_geomadr[body]), int(model.body_geomnum[body])
        for geom in range(adr, adr + count):
            model.geom_rgba[geom][3] = ICE_RGBA[3] if visible else 0.0
            model.geom_contype[geom] = 1 if visible else 0
            model.geom_conaffinity[geom] = 1 if visible else 0

    def on_physics_step(self, mujoco_env) -> None:
        """Carry the shards along inside the block while it is whole.

        They are switched off, so gravity would drop them through the bench;
        writing their pose from the block's is what holds them in it.
        """
        if mujoco_env is None or all(self._broken):
            return
        import mujoco

        model, data = mujoco_env.mjModel, mujoco_env.mjData
        if not self._prepared:
            for block in range(ICE_BLOCKS):
                for label in self._shard_labels(block):
                    self._show(model, label, False)
            self._prepared = True
        for block in range(ICE_BLOCKS):
            if self._broken[block]:
                continue
            cube = self._block_label(block)
            if cube is None:
                continue
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube)
            if body < 0:
                continue
            origin = np.asarray(data.xpos[body], dtype=float)
            rotation = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
            quat = np.asarray(data.xquat[body], dtype=float)
            for label, offset in zip(self._shard_labels(block), ice_offsets()):
                adr = self._free_joint(model, label)
                if adr < 0:
                    continue
                data.qpos[adr : adr + 3] = origin + rotation @ np.asarray(offset)
                data.qpos[adr + 3 : adr + 7] = quat
                dof = self._free_dof(model, label)
                if dof >= 0:
                    data.qvel[dof : dof + 6] = 0.0

    def _blow(self, mujoco_env) -> list[float]:
        """Per block, the largest impulse a struck tool-on-ice contact delivered.

        Read off the contacts rather than from the tool's velocity, because
        what fractures ice is what arrives through the contact: a foam head
        moving just as fast spreads the same momentum over a longer, softer
        push and delivers far less. But a contact force on its own cannot tell
        a blow from a lean, so a contact only counts when the tool was already
        travelling at the ice when it arrived.
        """
        if mujoco_env is None:
            return [0.0] * ICE_BLOCKS
        import mujoco

        model, data = mujoco_env.mjModel, mujoco_env.mjData
        # Every tool, not just the one the scenario calls correct: measuring
        # that slot alone decided the outcome by name rather than by physics.
        tool_bodies = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, actor.asset.label)
            for slot, actor in self.layout.actors.items()
            if slot in ("tool", "confusing_tool")
        }
        tool_bodies.discard(-1)
        if not tool_bodies:
            return [0.0] * ICE_BLOCKS
        # Which block each ice body belongs to, so a blow is credited to the
        # block it landed on rather than to the ice in general.
        block_of: dict[int, int] = {}
        for block, labels in enumerate(self._block_labels()):
            for label in labels:
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
                if body >= 0:
                    block_of[body] = block
        if not block_of:
            return [0.0] * ICE_BLOCKS

        force = np.zeros(6)
        step = float(model.opt.timestep)
        touching: set[int] = set()
        peak = [0.0] * ICE_BLOCKS
        for i in range(data.ncon):
            contact = data.contact[i]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            pair = {b1, b2}
            hitting = pair & tool_bodies
            struck = pair & set(block_of)
            if not hitting or not struck:
                continue
            touching |= hitting
            mujoco.mj_contactForce(model, data, i, force)
            blow = abs(float(force[0])) * step
            # Only what the tool was already carrying counts; a press puts
            # the same force through the contact as a swing does.
            block = block_of[min(struck)]
            if self._closing.get((min(hitting), block), 0.0) >= STRIKE_SPEED:
                peak[block] = max(peak[block], blow)

        # Read before the contact: the solver takes the speed out on landing.
        # Per block, not per tool: a swing at the far block closes on it and not
        # on the near one, and measuring against whichever block happened to
        # have the lower body id threw every blow on the far block away.
        by_block: dict[int, set[int]] = {}
        for ice, block in block_of.items():
            by_block.setdefault(block, set()).add(ice)
        for body in tool_bodies - touching:
            for block, ice in by_block.items():
                self._closing[(body, block)] = self._closing_speed(
                    model, data, body, ice
                )
        return peak

    def _closing_speed(self, model, data, body: int, ice_bodies: set[int]) -> float:
        """How fast this tool is going at one block, in metres per second."""
        ice = min(ice_bodies) if ice_bodies else -1
        if ice < 0:
            return 0.0
        gap = np.asarray(data.xpos[ice], dtype=float) - np.asarray(
            data.xpos[body], dtype=float
        )
        span = float(np.linalg.norm(gap))
        if span < 1e-6:
            return 0.0
        velocity = np.asarray(data.cvel[body][3:], dtype=float) - np.asarray(
            data.cvel[ice][3:], dtype=float
        )
        return float(np.dot(velocity, gap / span))

    def _fracture(self, mujoco_env, block: int) -> None:
        """Take the block out of play and let the shards it held go.

        They already sit in their places inside it, so this is only a switch,
        plus a push outwards so the break reads as one.
        """
        if mujoco_env is None:
            return
        import mujoco

        model, data = mujoco_env.mjModel, mujoco_env.mjData
        cube = self._block_label(block)
        if cube is None:
            return
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube)
        if body < 0:
            return
        origin = np.asarray(data.xpos[body], dtype=float)
        rotation = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
        carried = np.asarray(data.cvel[body][3:], dtype=float)

        for label, offset in zip(self._shard_labels(block), ice_offsets()):
            self._show(model, label, True)
            dof = self._free_dof(model, label)
            if dof < 0:
                continue
            outward = rotation @ np.asarray(offset, dtype=float)
            data.qvel[dof : dof + 3] = carried + BURST * outward / max(
                float(np.linalg.norm(outward)), 1e-6
            )
            data.qvel[dof + 3 : dof + 6] = 0.0

        self._show(model, cube, False)
        adr = self._free_joint(model, cube)
        if adr >= 0:
            # Out from under the bench, where nothing can meet it again.
            data.qpos[adr : adr + 3] = origin - np.array([0.0, 0.0, 10.0])
        dof = self._free_dof(model, cube)
        if dof >= 0:
            data.qvel[dof : dof + 6] = 0.0

    def check_failure(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False

    def job_done(self, info: dict[str, Any], **kwargs: Any) -> bool:
        """Every block broken and its shards apart. One of two is not enough."""
        del info
        mujoco_env = kwargs.get("mujoco_env")
        for block, blow in enumerate(self._blow(mujoco_env)):
            self._peak_impulse[block] = max(self._peak_impulse[block], blow)
            if self._broken[block] or self._peak_impulse[block] < FRACTURE_IMPULSE:
                continue
            self._fracture(mujoco_env, block)
            self._broken[block] = True
        # Cached here rather than recomputed on demand: the shard spread is
        # only readable while the environment is in hand, and the metrics are
        # written after the step has returned it.
        self._broken_count = sum(
            1
            for block in range(ICE_BLOCKS)
            if self._broken[block] and self._scatter(mujoco_env, block) >= SCATTER
        )
        return self._broken_count == ICE_BLOCKS

    def _scatter(self, mujoco_env, block: int) -> float:
        """How far one block's shards have drifted from their formation."""
        if mujoco_env is None or not self._broken[block]:
            return 0.0
        import mujoco

        model, data = mujoco_env.mjModel, mujoco_env.mjData
        spots = []
        for label in self._shard_labels(block):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
            if body >= 0:
                spots.append(np.asarray(data.xpos[body], dtype=float))
        if len(spots) < 2:
            return 0.0
        spots = np.asarray(spots)
        return float(np.max(np.linalg.norm(spots - spots.mean(axis=0), axis=1)))

    def metric_spec(self) -> dict[str, str]:
        spec = {
            **super().metric_spec(),
            "peak_impulse": "float32",
            "ice_broken": "bool",
            "ice_broken_count": "int64",
        }
        # One column per count, so a run that broke one of two is visible as a
        # partial result rather than folded into a single failed flag.
        spec.update({f"ice_broke_{n}": "bool" for n in range(1, ICE_BLOCKS + 1)})
        return spec

    def recording_object_map(self) -> dict[str, str | None]:
        """As the base, except that a hidden block or shard reports absent.

        Breaking is not a move: the block is switched off and its shards are
        switched on, in the model rather than in `qpos`. A recording that only
        carried poses would leave a replay unable to tell a whole block from a
        broken one, so presence carries it. `observation.object_present` is
        written every frame, which is what makes it the right column: it says
        which pieces of ice exist at that moment.
        """
        visible = super().recording_object_map()
        for block in range(ICE_BLOCKS):
            gone = self._broken[block]
            if gone:
                visible[f"ice_{block}"] = None
            else:
                for index in range(SHARDS_PER_BLOCK):
                    visible[f"shard_{block}_{index}"] = None
        return visible

    def show_recording_object(self, model, slot: str, visible: bool) -> None:
        """Switch a block or a shard on or off, as `_show` does during a run.

        Only the ice: `_show` paints ice's own alpha back, and the tools are
        there for the whole episode anyway.
        """
        if not (slot.startswith("ice_") or slot.startswith("shard_")):
            return
        layout = getattr(self, "_layout", None)
        actor = None if layout is None else layout.actors.get(slot)
        if actor is not None:
            self._show(model, actor.asset.label, visible)

    def task_info(self, info: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        report = super().task_info(info, **kwargs)
        report["metrics"].update(
            {
                "peak_impulse": float(max(self._peak_impulse)),
                "ice_broken": self._broken_count == ICE_BLOCKS,
                "ice_broken_count": int(self._broken_count),
                **{
                    f"ice_broke_{n}": self._broken_count >= n
                    for n in range(1, ICE_BLOCKS + 1)
                },
            }
        )
        return report
