"""The candidate tools a reasoning scenario offers the robot.

Each scenario offers a correct tool and two irrelevant objects. Reasoning
mode adds a confusing tool, with a contrast specific to the task:

* **Affordance.** Same length and same mass. Only the shape differs, so a
  policy cannot tell the hook from the stick by its size.
* **Physical.** A stiff metal hammer contrasts with a fly swatter, paint roller,
  or plunger modeled as light and compliant. Their visible structure cues
  striking suitability; contact physics determines whether the ice breaks.

The tools are scanned meshes from MolmoSpaces rather than assembled primitives,
and both controls survive that, because neither is a property of the mesh.
`mesh_scale` is chosen per asset so every tool in a set measures the same along
its longest axis, and `mass` is declared rather than derived from the geometry.

One channel cannot be closed this way. A scanned asset brings its own colour
and texture, so unlike a set of identically coloured primitives these tools can
be told apart by appearance as well as by shape, and a policy can learn "the
dark curved one" without reasoning about what a crook is for. That is the price
of using real assets, and it is worth stating rather than hiding.

Fetch the meshes with `scripts/dataset/fetch_reasoning_tools.py`.

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from humanoidtoolbench.assets.benchmark import BenchmarkAsset, PrimitiveToolAsset

# Where `msassets.py fetch` puts what it downloads.
MS_ASSETS_DIR = Path(
    os.environ.get("HUMANOIDTOOLBENCH_MS_ASSETS", "data/ms_assets")
).expanduser()
MS_OBJAVERSE = MS_ASSETS_DIR / "objects" / "objaverse"

# What the Dex3 hand can lift and hold. Above this a tool is picked at and
# never raised, which reads as a failure to choose rather than a failure to
# lift, so every set is held to it. Lowered from 0.25 once teleoperation
# showed the hand losing tools at that weight.
MAX_TOOL_MASS = 0.15

# Where along a tool the operator's hand ends up: not at the very end, and not
# at the middle. What a tool buys is therefore not its length but the part
# beyond the grip, so a tool has to be a third longer than the gap it has to
# cross.
GRIP_FRACTION = 0.25

# The gap a tool has to cross, measured from the robot's base to the ice: 0.96 m
# at the furthest the layout puts it, against ARM_REACH of 0.62. That is
# 0.34 m of reach the tool has to supply, and at GRIP_FRACTION it takes
# 0.34 / 0.75 = 0.45 to supply it, which is the floor. The ceiling is the
# bench: the placer pulls the tool row in until a tool lies wholly on it, and
# at 0.55 the far end of a tool at rest already touches the cube it is meant
# to reach. 0.50 leaves 5 cm of clearance.
MAX_TOOL_LENGTH = 0.50

# How thick a tool is where the hand goes. A scanned stick scaled to 0.45 m
# comes out 1 to 3 cm thick, which the Dex3 hand closes on but cannot hold,
# so the two axes across the length are scaled together until the grip (the
# slim quarter of the tool, see `_grip_section`) measures this across its
# thinner axis. Raised from 0.04, at which two hooks came out 4.1 cm and read
# as wire; the whole cross-section scales with it, so no grip stands out. The affordance and spatial sets hold every tool to exactly
# this, so a grip never says which tool is the answer; the hammers only have
# it as a floor, because there the size of the head is the decoy. `MAX_THICKEN`
# bounds the scale either way: a 5 mm spoon handle does not become a 4 cm
# one, it becomes a mesh that should not be in the pool.
GRIP_THICKNESS = 0.05
MAX_THICKEN = 3.0
# Thickening the grip scales the whole cross-section, and a flat blade would
# grow sideways as fast as its handle grows thick. The wider axis stops at
# the widest tool the row was laid out for; the thin axis carries on alone.
MAX_TOOL_WIDTH = 0.26


# Every tool in a set measures this along its longest axis. The affordance set
# matches the long stick: the question is which shape pulls, not which is
# longer, so where its object sat beyond reach the object moved in.
AFFORDANCE_LENGTH = MAX_TOOL_LENGTH
AFFORDANCE_MASS = MAX_TOOL_MASS

_FRICTION = [1.5, 0.1, 0.005]
# A tool lying on the bench with a round head or shaft rolls, and the rests
# that used to stop it are gone. Rolling friction stops it instead; it needs
# condim 6 and a contact priority above the bench's, or the bench's own
# parameters would apply. Measured over the pools: a mallet that rolled off
# the bench now stays within a few centimetres of where it was laid.
TOOL_ROLLING_FRICTION = 0.02
TOOL_CONTACT_PRIORITY = 11

# MolmoSpaces asset ids per slot. A slot holds a pool, not one asset: the
# scenario draws from it each episode, so a policy sees many hooks rather than
# memorising one. The controls hold across the pool because they are applied
# per draw, not per asset: whichever hook comes out is scaled to the set's
# length and given the set's mass.
#
# Curate these with `scripts/dataset/tool_candidates.py`.
ASSET_POOLS: dict[str, list[str]] = {
    "hook": [
        "36eed3c8eda149feaeb14a1d5401f392",
        "1a717ce40b2d49ec961fb6e46e32e9fa",
        "7be0661092ae444882d19814e5de2853",
        "a25a12860e834fafa182e2907ad8db1d",
        "1254687d099c4068a1bd80e40bd140cc",
        # Added 2026-09-07. Each passes the same controlled pull search the
        # five above pass, and places no worse than they do. Two of the six the
        # audit shortlisted are not here: ded805f4 reads as a copy of 36eed3c8
        # and adds no shape, and 44b7df77 was turned down on review.
        "934c3cb6ed06462e8afee848b7942d6e",
        "71359c1397da4ca080dca0d80400b880",
        "f6aef21eab0247dd820be1ed32327dc6",
        "6f36616001cf4037b104cca4967f5027",
    ],
    # 26cb919d dropped: a hockey stick, its foot 0.069 m off the shaft, more
    # hooked than two of the hooks, so the wrong answer worked.
    "straight_stick": [
        "3ebe7900b5914ba1a2b85abd3135abe3",
        "3814b5ab62f9427e9da5fbb1d01f62ec",
        "392d625c88934a64b08c628e640fafd1",
        "2b429b2d6618419d8480251d6243397c",
        "c8714f25b5d844f7a2d28b3e69f2ea36",
        # Added 2026-09-07, the same six rods the spatial slot takes. None of
        # them gives the ball a single place to catch when the tool is drawn
        # back, so the wrong answer stays wrong.
        "9712c578203d4574a8cc092ff527290d",
        "ab513e12c1c6432faf6410fb0e5088a5",
        "bfd89cebcb3f4835ae4cc64a9e1d46f3",
        "695757a99900484a82f12ae30e8bf4ec",
        "bdf07f6c648b4d74af80ea7103ac384c",
        "cb84d2732c9840498598e9838b368385",
    ],
    # Spatial: one stick, drawn at two lengths. The pool is shared because
    # the scenario is about reach, not about which stick.
    # 1f08842d (a crossbar) and 18484b0d (a hockey stick) were dropped: neither
    # is a straight rod, and both topple 9 to 12 cm as soon as they are laid.
    "stick": [
        "3ebe7900b5914ba1a2b85abd3135abe3",
        "0eedc541d1bb4a32935fc6ed7a2fea08",
        "19cb31b210f646919b56ebf897b22b33",
        # Added 2026-09-07. No initial overlap in 1200 placements, and the
        # reach split holds: long pushes the ball 61 to 66 mm, short never
        # reaches it.
        "9712c578203d4574a8cc092ff527290d",
        "ab513e12c1c6432faf6410fb0e5088a5",
        "bfd89cebcb3f4835ae4cc64a9e1d46f3",
        "695757a99900484a82f12ae30e8bf4ec",
        "bdf07f6c648b4d74af80ea7103ac384c",
        "cb84d2732c9840498598e9838b368385",
    ],
    # Physical: a heavy metal head, and a foam one that is deliberately the
    # larger of the two.
    # Dropped for reading as steel rather than rubber: 07e36824, 3018091c,
    # 4bc67992, 1f44cf28. What is left is toy hammers and rubber mallets, all
    # 1.5x to 3.5x the metal heads' thickness, so size stays the decoy.
    "metal_hammer": [
        "0133e824d5154a8fa575d9d3027ded6d",
        "1fc9ef60b96340ac8ae6a706aa068c33",
        "1fb0baea870c4acd8b3457f1bb7c15b2",
        "113a47f834b4400fb1500314d482847a",
        "200bc458e8184653a049c5449737c67a",
        # Added 2026-09-07. All fracture the block across the eight tested
        # strike conditions, and all sit at or under 6.54 cm of head, so the
        # foam decoy stays the visibly larger tool.
        "01e5b5acad9843818529d15663df5b9e",
        "11d6edfdf58142e78d5435dc4bc03460",
        "27a9c65a5fcd4f958a6e1c7e693c3cc5",
        "291f15c9c1754e4084a5655783efcbd8",
        "36b1679072d9412fa4b7a153732d3a5d",
        "37fc0ecb0a2f4693979bfdf8a0f1f19a",
        "429bd43cb7464e29842fbc748572f32c",
        "56ca74a3eae041b98e3f0c8c1f26b8f2",
        "784fd1ea80da47629f56be0e99fa2a3c",
        "79bf24b80c024455b16369d6c137052b",
        "8bcba48e80d745c5a51fc67cee5861b6",
        "9207d6a0c4874d6082daee015f53be11",
        "a775c69d36f04e4c9a738d53e6fe5115",
        "ac029344ea9c41f8900b50e9a628444d",
        "b2d45fba3c364669a624e60884b88bb2",
        "c388ab0c5fa24bd1a1e36b1a71556617",
        "c445ea18b910491c9bb057e81cee0dfa",
        "c624accefb6e489582d5bbeed5e773f8",
        "c69309ddb90845c7a62050fe8aacef23",
        "cce3dec670fc412b89a5ed963c79b7ca",
        "d1f67e80395b4b12b9e746d2e412c197",
        "d5c3f159bccb41a8b5d18314bc1d099f",
        "dd51c7a336b54cb3985fb2014a3d5405",
        "e0fd0afb8bb4468986d6d5bd729104a9",
        "e1f7278ffd7c4878a3cffbc98f05359a",
        "e2010011a9574210becf909968db4166",
        "e3710ddc972b4c3bb0a42e583b9d9f8a",
        "ebc33cd8ae204bf1bea0111f80401388",
        "f38bdd18a6e7475198228810f77dc240",
        "f9d8546f08b2426a82b2995c171c518f",
        "fb70be99961c44039145856d51251e1e",
        "fc6a748b0a3f48e5a2109e2d20554fa5",
    ],
    "physical_distractor": [
        # Fly swatter.
        "9ce080633f6f48be9ecc761bbeb7f640",
        # Paint rollers.
        "248507ee27dd4e57a75c2efac03fc636",
        "6a1a1a0bef1145e6a9ff5893dc384745",
        "9a4c9d19a1bc4bb5820e9fb5e98e9b29",
        # Plungers.
        "22b3239bd45a424a8c2e38755557accf",
        "40fff48218ce43119ffea5be7719352d",
        "bd05149f050c43ecb70716278dcf948f",
        "d0a2f0234b414e8bb2b95890c0d5d759",
        "8a796d92b21e4257a2ed79d973064318",
    ],
    # Retained for replay of episodes recorded before the decoy replacement.
    "foam_hammer": [
        "2de2cbb69ad44bc3ae2e01eacce9f174",
        "a190e92f53814532bf2ef03ec56bd16a",
        "ae3b71ad58004a6f9124869ffe02545d",
        "95447c4b8c824fb7888cc9bed57a87f7",
        "57fc7be2efcd4284a9514f05dbb51942",
    ],
}

# Meshes for an out-of-distribution test. None is in ASSET_POOLS, and none
# appears in any recording (checked 2026-09-14 against the `tool_<role>_<id>`
# uids of every episode). The `ood_*` builders below make them with the same
# factories as the pools above, so length, grip thickness and mass follow the
# same rules. Those rules do not always reach their target: thickening stops at
# MAX_THICKEN and at MAX_TOOL_WIDTH, which leaves some pool tools thinner than
# GRIP_THICKNESS. A mesh is here only if what the rules make of it lands inside
# the range its role's pool lands in: grips of 50 mm for hooks, 26.8 to 50 mm
# for sticks, 50 to 51.6 mm for metal hammers, 24.9 to 75.9 mm for decoys.
OOD_ASSET_POOLS: dict[str, list[str]] = {
    "hook": [
        # Crowbars with a crook at one end.
        "bd150d5016c443e68463583e3b3c4d0d",
        "f4cf6f8c1e8040718a68b46517531a67",
        "f342bc93ab7b4c0fb3a52d51fe179311",
        # Candy canes.
        "684c7052d0cb470da71a67a0aca148d6",
        "5128c71a17af4bac879807ace1e82bd6",
        "b5b1280f5561460688a7cb73224b7995",
    ],
    "stick": [
        "41462f7e54e24b4294f62104cb085996",
        "403a49ff877046588ab7782e021679dd",
        "b474c060ea084235a23817d97ef2b909",
        "f4910bd3c6af455f9ac988639f9368b1",
        "84bdeaad10e347fe83f5292186e1ced7",
        "43a3626e04af45dbac13b152d84d1c01",
        "4099fa96c15a4c0889953cb280f21b2e",
        "577a66fc394e4c70834cf622364a8dee",
        "27624ee05fea49899ba2a63e2f6d1e34",
        "c4a5c09cfda54a6b8ab67ef949e42536",
        "cd6467ab49094b1784e414db5613f2db",
    ],
    "metal_hammer": [
        "13f18f64c9bd4130b2f17c84832819ac",
        "237435ea473a4880a0feec50da473536",
        "bd57a5ce07914ba7a3093beb14891494",
        "1bfd4e1f28e144ad991ea8b413fab344",
    ],
    "physical_distractor": [
        # Paint roller.
        "002a355c77774228b15db50d77d0e002",
        # Fly swatters, one of them electric.
        "30e08be02db047a39882403e48694c58",
        "6655a2fa5e984c9cab631a7f94fba275",
        # Plungers.
        "81e1c64fa7834f588bb03c3084ddf450",
        "d2c6575ef88747a28d207668ffbd44f2",
    ],
}
# The affordance decoy and the spatial stick draw from one list, as the two
# benchmark pools share most of their rods.
OOD_ASSET_POOLS["straight_stick"] = OOD_ASSET_POOLS["stick"]

# The correct-tool OOD twins score the task as the cell does, so the unseen
# correct tool has to be able to do the job, and a grip in range does not show
# that. Two crowbars above stand on end, one bend is too shallow to catch the
# ball, one candy cane never pulls it, and some rods stand up, bend away or
# barely touch it. These are the unseen meshes that pass the prescribed-motion
# probes the benchmark pools pass (artifacts/s_to_r_ood_20260915/
# tool_function_audit): a hook that pulls the ball back on the fine grid, and a
# long stick that lies flat and pushes the ball as far as a pool stick does.
# f4910bd3 pushes well but failed the rod audit's finite-table support test.
# 44b7df77 is not in OOD_ASSET_POOLS; the 2026-09-07 hook audit shortlisted it
# and no prepared recording contains it.
OOD_CORRECT_ASSET_POOLS: dict[str, list[str]] = {
    "hook": [
        # Curved green hook.
        "44b7df77727143169c545e39956a77e2",
        # Candy canes.
        "684c7052d0cb470da71a67a0aca148d6",
        "b5b1280f5561460688a7cb73224b7995",
    ],
    "stick": [
        "27624ee05fea49899ba2a63e2f6d1e34",
        "43a3626e04af45dbac13b152d84d1c01",
        "577a66fc394e4c70834cf622364a8dee",
        "84bdeaad10e347fe83f5292186e1ced7",
        "b474c060ea084235a23817d97ef2b909",
        "c4a5c09cfda54a6b8ab67ef949e42536",
        "cd6467ab49094b1784e414db5613f2db",
    ],
}


def pool(role: str) -> list[str]:
    return list(ASSET_POOLS[role])


def draw(role: str, rng=None, pools: dict[str, list[str]] | None = None) -> str:
    """One asset id from a slot's pool.

    Without an rng the first entry comes back, which keeps anything that just
    wants "a hook" deterministic; the scenario passes its own generator so the
    draw replays with the episode. `pools` is the table the slot is read from:
    ASSET_POOLS, unless an OOD builder hands in OOD_ASSET_POOLS.
    """
    ids = (ASSET_POOLS if pools is None else pools)[role]
    # A replay hands in the id it recorded rather than a generator. Taking it
    # here is what makes the rebuild exact: an index cannot, because the same
    # mesh sits in more than one pool and its position differs between them.
    # The recorded ID also takes precedence across standard/OOD pools; only
    # a fresh draw is restricted to the task's selected pool.
    wanted = getattr(rng, "asset_id", None)
    if wanted is not None:
        return wanted
    if rng is None or len(ids) == 1:
        return ids[0]
    return ids[int(rng.integers(len(ids)))]


class MissingToolAssets(FileNotFoundError):
    """The meshes have not been downloaded yet."""


def _package(uid: str) -> Path:
    path = MS_OBJAVERSE / uid
    if not (path / f"{uid}_visual.obj").is_file():
        raise MissingToolAssets(
            f"{path} is missing. Run uv run --no-project scripts/setup_evaluation.py "
            "to download the reasoning tools."
        )
    return path


def _bounds(visual: Path) -> tuple[np.ndarray, np.ndarray]:
    """(low corner, high corner) of the mesh, in its own units."""
    import trimesh

    mesh = trimesh.load(visual, force="mesh")
    return np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])


# How much slimmer one end has to be before it counts as the handle. A stick is
# the same all the way along (measured ratios 1.00 to 1.27) and there is no
# handle to find; a hammer or a hook is not (1.79 to 8.11). Below this the tool
# is left as it lies, because turning it on noise is worse than not turning it.
HANDLE_RATIO = 1.5


def _grip_section(visual: Path, axis: int) -> np.ndarray | None:
    """Extents across the length of the quarter the hand holds, mesh units.

    The grip is the slimmer end quarter of the tool: a hammer's or paddle's
    handle, a hook's shaft, either end of a stick. Measured as the bounding
    extents of that quarter's vertices across the two axes that are not the
    length, so the head of a hammer or the crook of a hook does not count.
    """
    import trimesh

    mesh = trimesh.load(visual, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    along = vertices[:, axis]
    low, high = float(along.min()), float(along.max())
    quarter = GRIP_FRACTION * (high - low)
    others = [a for a in (0, 1, 2) if a != axis]

    def section(mask) -> np.ndarray | None:
        if int(mask.sum()) < 8:
            return None
        part = vertices[mask][:, others]
        return part.max(axis=0) - part.min(axis=0)

    ends = [
        s
        for s in (section(along <= low + quarter), section(along >= high - quarter))
        if s is not None
    ]
    if not ends:
        return None
    return min(ends, key=lambda s: float(np.min(s)))


def _handle_end(visual: Path, axis: int) -> bool | None:
    """Is the slim end at the +axis end once scaled? None if there is no slim end.

    Compare the two end quarters, the same ones `_grip_section` measures, by
    their widest cross-section. Halves miss a short head: a toy hammer whose
    head is a tenth of its length sits in a half that is mostly handle and
    reads as symmetric, which lays it head towards the robot. `mesh_scale`
    mirrors x, so a tool that is long on x has its two ends exchanged in the
    frame the placer sees.
    """
    import trimesh

    mesh = trimesh.load(visual, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    along = vertices[:, axis]
    low_end, high_end = float(along.min()), float(along.max())
    quarter = GRIP_FRACTION * (high_end - low_end)
    others = [a for a in (0, 1, 2) if a != axis]

    def girth(mask) -> float:
        if int(mask.sum()) < 8:
            return float("inf")
        part = vertices[mask][:, others]
        return float(np.max(part.max(axis=0) - part.min(axis=0)))

    low = girth(along <= low_end + quarter)
    high = girth(along >= high_end - quarter)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if max(low, high) < HANDLE_RATIO * max(min(low, high), 1e-9):
        return None  # symmetric: either end will do

    handle_low = low < high
    if axis == 0:
        handle_low = not handle_low  # the x mirror swaps the ends
    return not handle_low


def _rolled(low, high, scales, axis: int, roll: float):
    """(extent, lift) once the mesh is turned `roll` about its length axis.

    Both come from the turned bounding box rather than from the mesh's own
    axes, so every consumer keeps reading a world-aligned box.
    """
    sin, cos = np.sin(roll), np.cos(roll)
    turn = np.eye(3)
    others = [i for i in (0, 1, 2) if i != axis]
    a, b = others
    turn[a, a] = turn[b, b] = cos
    turn[a, b], turn[b, a] = -sin, sin
    corners = np.array(np.meshgrid(*zip(low, high))).reshape(3, -1).T
    box = corners * np.asarray(scales, dtype=float)
    turned = box @ turn.T
    return turned.max(axis=0) - turned.min(axis=0), float(-turned[:, 2].min())


def mesh_asset(
    role: str,
    *,
    length: float | None = None,
    long_length: float | None = None,
    mass: float = 0.2,
    uid: str | None = None,
    rng=None,
    min_thickness: float | None = None,
    max_thickness: float | None = None,
    pools: dict[str, list[str]] | None = None,
) -> BenchmarkAsset:
    """One fetched MolmoSpaces object, scaled and weighed to order.

    `length` scales the mesh so its longest axis measures that, which is what
    holds a set's tools to one size. `mass` is declared rather than derived,
    which is what holds them to one weight.

    `long_length` stretches the long axis alone to that measurement and leaves
    the cross-section at whatever `length` gave it: a longer handle, not a
    bigger tool. Use it where the reach has to grow but the head must not.

    `min_thickness` and `max_thickness` bound how thick the grip is (see
    `_grip_section`), scaling the two axes across the length together; give
    both the same value to fix it.

    `pools` is only where the id is drawn from (see `draw`). Nothing about the
    scaling reads it, which is what keeps an OOD tool to a pool tool's size.
    """
    asset_uid = uid or draw(role, rng, pools)
    package = _package(asset_uid)
    visual = package / f"{asset_uid}_visual.obj"
    from humanoidtoolbench.assets.evaluation import canonical_colliders

    # Canonical release assets have one fixed collision representation.
    # Local decomposition files must not silently change benchmark physics.
    colliders = canonical_colliders(package, asset_uid)
    if colliders is None:
        colliders = sorted(
            package.glob(f"{asset_uid}_acd*.obj"),
            key=lambda path: int(path.stem.rsplit("_acd", 1)[1]),
        ) or sorted(package.glob(f"{asset_uid}_collider*.obj"))
    texture = package / f"{asset_uid}_visual_0.png"

    low, high = _bounds(visual)
    natural = high - low
    scale = 1.0 if length is None else float(length) / float(np.max(natural))
    axis = int(np.argmax(natural))
    scales = np.full(3, scale, dtype=float)
    if long_length is not None:
        scales[axis] = float(long_length) / float(natural[axis])
    if min_thickness is not None or max_thickness is not None:
        # Scale the cross-section, both axes by one factor so the shape keeps
        # its proportions, until the grip's thinner axis lands in the band.
        across = [i for i in range(3) if i != axis]
        grip = _grip_section(visual, axis)
        section = (natural[across] if grip is None else grip) * scales[across]
        thin_axis = across[int(np.argmin(section))]
        wide_axis = across[1 - int(np.argmin(section))]
        thinnest = float(np.min(section))
        factor = 1.0
        if min_thickness is not None and thinnest < min_thickness:
            factor = min_thickness / thinnest
        elif max_thickness is not None and thinnest > max_thickness:
            factor = max_thickness / thinnest
        factor = float(np.clip(factor, 1.0 / MAX_THICKEN, MAX_THICKEN))
        # Both cross axes follow, so a round grip stays round, but each stops
        # at the width cap: a blade gets thick without getting any wider.
        # The cap has to read each axis' own span. A thin handle on a fat head
        # is thin across the tool's wide axis, so capping only the axis the
        # grip calls wide let one toy hammer's head out to 0.41 m.
        for cross_axis in (thin_axis, wide_axis):
            span = float(natural[cross_axis] * scales[cross_axis])
            capped = factor
            if capped > 1.0 and span * capped > MAX_TOOL_WIDTH:
                capped = max(1.0, MAX_TOOL_WIDTH / span)
            scales[cross_axis] *= capped
    extent = natural * scales
    # A mesh's origin is wherever it was scanned from, so a tool is placed by
    # how far its lowest point sits below that origin, and by how far its middle
    # sits to one side of it. Without the second, a row laid by origin put the
    # hammers 0.18 m apart down the bench while the numbers said one line.
    lift = float(-low[2] * scales[2])
    centre = 0.5 * (low + high) * np.array([-scales[0], scales[1], scales[2]])
    # Lay the tool on its flat side. Two of the hammers are scanned with the
    # head across z, and the placer only turns about z, so they stood on the
    # head a quarter of a metre tall with the striking face at the ceiling.
    roll = 0.0
    if min_thickness is not None or max_thickness is not None:
        across = [i for i in (0, 1, 2) if i != axis]
        flat = next((i for i in across if i != 2), None)
        if flat is not None and extent[2] > extent[flat]:
            roll = 0.5 * np.pi
            extent, lift = _rolled(
                low, high, [-scales[0], scales[1], scales[2]], axis, roll
            )

    asset = BenchmarkAsset(
        {
            # The id is in the uid so a recorded episode replays with the
            # same mesh, not merely with the same kind of tool.
            "uid": f"tool_{role}_{asset_uid[:8]}",
            "label": f"tool_{role}_{asset_uid[:8]}",
            "name": role.replace("_", " "),
            "role": role,
            "kind": "mesh",
            "group": "reasoning_tool",
            "group_label": "Reasoning tool",
            "source": "molmospaces",
            "license": "objaverse",
            "collision_meshes_mujoco": [str(p) for p in colliders],
            "visual_mesh": str(visual),
            "texture": str(texture) if texture.is_file() else None,
            # The packages are authored mirrored on x; keeping that is what
            # makes the visual mesh and its colliders agree.
            "mesh_scale": [-scales[0], scales[1], scales[2]],
            "mass": float(mass),
            "friction": _FRICTION,
            "up_axis": "z",
            "extent": [float(v) for v in extent],
            "height": lift,
            "footprint": float(np.max(extent[:2])),
            "rest_offset": lift,
            "n_convex_parts": len(colliders),
            "length": float(np.max(extent)),
            "grip_width": float(np.min(extent)),
            "colour": None,
            "holds": None,
            "interior": None,
        }
    )
    # Which way the mesh is long, so the placer can point every tool the same
    # way however the asset happened to be scanned.
    asset.length_axis = axis
    # How wide the tool is across that axis, on the bench. `grip_width` is the
    # smallest extent, which for a hammer is the handle's thickness and not the
    # head's span, so a placer that reserved room by it would hang the head
    # over the edge.
    asset.cross_extent = float(extent[1 - asset.length_axis])
    if min_thickness is not None or max_thickness is not None:
        # Only the reasoning tools (the ones with a grip to hold) get the
        # rolling friction; an irrelevant object keeps the bench's contact.
        asset.friction = [_FRICTION[0], _FRICTION[1], TOOL_ROLLING_FRICTION]
        asset.condim = 6
        asset.contact_priority = TOOL_CONTACT_PRIORITY
    # Which end the hand goes on. A tool is thin where it is held and bulky
    # where it works, so the slimmer half is the handle; a hammer laid head
    # towards the robot is a hammer the robot cannot pick up by the handle.
    asset.handle_at_high = _handle_end(visual, asset.length_axis)
    # How far to turn the tool about its own length before laying it down.
    asset.roll = float(roll)
    # Where the shape's middle sits relative to the origin the scan left it,
    # so a row can line up what is seen rather than what is stored.
    asset.centre_offset = [float(v) for v in centre]
    return asset


# -- affordance: retrieve an object that cannot be reached from behind --------


def hook(rng=None, *, pools=None) -> BenchmarkAsset:
    """The crook goes past the object and catches its far side."""
    return mesh_asset(
        "hook",
        length=AFFORDANCE_LENGTH,
        mass=AFFORDANCE_MASS,
        rng=rng,
        pools=pools,
        min_thickness=GRIP_THICKNESS,
        max_thickness=GRIP_THICKNESS,
    )


def straight_stick(rng=None, *, pools=None) -> BenchmarkAsset:
    """Reaches the object and can only push it away."""
    return mesh_asset(
        "straight_stick",
        length=AFFORDANCE_LENGTH,
        mass=AFFORDANCE_MASS,
        rng=rng,
        pools=pools,
        min_thickness=GRIP_THICKNESS,
        max_thickness=GRIP_THICKNESS,
    )


# The paddle slot is gone: a flat blade can scoop a ball and carry it back, so
# it was not reliably the wrong answer this axis needs.
AFFORDANCE_TOOLS = {
    "hook": hook,
    "straight_stick": straight_stick,
}


# -- spatial: reach an object that is further away than it looks -------------
#
# One tool at two lengths. The §5 rule for this axis is that the candidates be
# the same tool differing in a single spatial parameter, so both are the same
# mesh: nothing but reach separates them, and a policy cannot answer from
# appearance.
SHORT_STICK_LENGTH = 0.30
LONG_STICK_LENGTH = MAX_TOOL_LENGTH
STICK_MASS = MAX_TOOL_MASS


def short_stick(rng=None, *, pools=None) -> BenchmarkAsset:
    """Reaches part of the way. Everything else about it is the long one.

    Scaled as the long stick is and then shortened along its length alone, so
    the two have the same cross-section and differ in nothing but reach.
    """
    asset = mesh_asset(
        "stick",
        length=LONG_STICK_LENGTH,
        long_length=SHORT_STICK_LENGTH,
        mass=STICK_MASS,
        rng=rng,
        pools=pools,
        min_thickness=GRIP_THICKNESS,
        max_thickness=GRIP_THICKNESS,
    )
    # Keep the mesh id, so a replay rebuilds the stick that was drawn.
    asset.uid = asset.label = f"tool_short_stick_{asset.uid.rsplit('_', 1)[1]}"
    return asset


def long_stick(rng=None, *, pools=None) -> BenchmarkAsset:
    """The same stick, long enough to reach."""
    asset = mesh_asset(
        "stick",
        length=LONG_STICK_LENGTH,
        mass=STICK_MASS,
        rng=rng,
        pools=pools,
        min_thickness=GRIP_THICKNESS,
        max_thickness=GRIP_THICKNESS,
    )
    asset.uid = asset.label = f"tool_long_stick_{asset.uid.rsplit('_', 1)[1]}"
    return asset


SPATIAL_TOOLS = {
    "long_stick": long_stick,
    "short_stick": short_stick,
}


# -- physical: break an ice block with a concentrated impact ----------------
#
# New decoys reuse the former foam hammer's size, mass, and whole-asset
# compliance. These are normalized benchmark properties, not measurements of
# the source objects or separate models of their handles and working surfaces.
METAL_HAMMER_HEAD = 0.34
METAL_HAMMER_LENGTH = 0.42
FOAM_HAMMER_HEAD = 0.42
FOAM_HAMMER_LENGTH = 0.42
METAL_HAMMER_MASS = MAX_TOOL_MASS
FOAM_HAMMER_MASS = 0.04
# Contact compliance reduces peak impact in the tested free strikes. Sustained
# external drive can still exceed the fracture threshold for compliant tools.
METAL_CONTACT_SOLREF = [0.004, 1.0]
FOAM_CONTACT_SOLREF = [0.10, 1.5]


def metal_hammer(rng=None, *, pools=None) -> BenchmarkAsset:
    """The intended ice-breaking tool, heavier and stiffer than the decoys."""
    asset = mesh_asset(
        "metal_hammer",
        length=METAL_HAMMER_HEAD,
        long_length=METAL_HAMMER_LENGTH,
        mass=METAL_HAMMER_MASS,
        rng=rng,
        pools=pools,
        min_thickness=GRIP_THICKNESS,
    )
    asset.contact_solref = list(METAL_CONTACT_SOLREF)
    return asset


def foam_hammer(rng=None, *, uid: str | None = None) -> BenchmarkAsset:
    """Legacy compliant hammer, retained for recorded episode replay."""
    asset = mesh_asset(
        "foam_hammer",
        length=FOAM_HAMMER_HEAD,
        long_length=FOAM_HAMMER_LENGTH,
        mass=FOAM_HAMMER_MASS,
        uid=uid,
        rng=rng,
        min_thickness=GRIP_THICKNESS,
    )
    asset.contact_solref = list(FOAM_CONTACT_SOLREF)
    return asset


def physical_distractor(rng=None, *, pools=None) -> BenchmarkAsset:
    """One swatter, roller, or plunger with the former foam tool's physics."""
    asset = mesh_asset(
        "physical_distractor",
        length=FOAM_HAMMER_HEAD,
        long_length=FOAM_HAMMER_LENGTH,
        mass=FOAM_HAMMER_MASS,
        rng=rng,
        pools=pools,
        min_thickness=GRIP_THICKNESS,
    )
    if Path(asset.visual_mesh).stem == "9ce080633f6f48be9ecc761bbeb7f640_visual":
        import trimesh

        # The swatter's grid is thinner than its handle, so the generic grip
        # estimate picks the grid. Size depth from the actual handle quarter
        # while keeping the paddle width and the tool length unchanged.
        vertices = np.asarray(trimesh.load(asset.visual_mesh, force="mesh").vertices)
        along = vertices[:, asset.length_axis]
        handle = vertices[along <= along.min() + GRIP_FRACTION * float(np.ptp(along))]
        depth = float(np.ptp(handle[:, 2])) * asset.mesh_scale[2]
        factor = GRIP_THICKNESS / depth
        asset.mesh_scale[2] *= factor
        asset.extent[2] *= factor
        asset.height *= factor
        asset._rest_offset *= factor
        asset.centre_offset[2] *= factor
        asset.grip_width = float(np.min(asset.extent))
    asset.contact_solref = list(FOAM_CONTACT_SOLREF)
    return asset


PHYSICAL_TOOLS = {
    "metal_hammer": metal_hammer,
    "physical_distractor": physical_distractor,
}


# -- out of distribution: the same builders, drawing from OOD_ASSET_POOLS -----
#
# Each wraps its benchmark builder instead of repeating the arguments, so an
# OOD tool can differ from a pool tool only by its mesh. The role in the uid
# stays the benchmark's (`tool_hook_<id>`), so metrics and replays read an OOD
# hook as they read any hook.


def ood_hook(rng=None) -> BenchmarkAsset:
    """A hook no recording has seen, scaled as `hook` scales its own."""
    return hook(rng, pools=OOD_ASSET_POOLS)


def ood_straight_stick(rng=None) -> BenchmarkAsset:
    """A straight stick no recording has seen, scaled as `straight_stick`."""
    return straight_stick(rng, pools=OOD_ASSET_POOLS)


def ood_short_stick(rng=None) -> BenchmarkAsset:
    """A short stick no recording has seen, scaled as `short_stick`."""
    return short_stick(rng, pools=OOD_ASSET_POOLS)


def ood_long_stick(rng=None) -> BenchmarkAsset:
    """A long stick no recording has seen, scaled as `long_stick`."""
    return long_stick(rng, pools=OOD_ASSET_POOLS)


def ood_metal_hammer(rng=None) -> BenchmarkAsset:
    """A metal hammer no recording has seen, scaled as `metal_hammer`."""
    return metal_hammer(rng, pools=OOD_ASSET_POOLS)


def ood_physical_distractor(rng=None) -> BenchmarkAsset:
    """A decoy no recording has seen, with `physical_distractor`'s physics."""
    return physical_distractor(rng, pools=OOD_ASSET_POOLS)


OOD_AFFORDANCE_TOOLS = {
    "hook": ood_hook,
    "straight_stick": ood_straight_stick,
}
OOD_SPATIAL_TOOLS = {
    "long_stick": ood_long_stick,
    "short_stick": ood_short_stick,
}
OOD_PHYSICAL_TOOLS = {
    "metal_hammer": ood_metal_hammer,
    "physical_distractor": ood_physical_distractor,
}


def ood_correct_hook(rng=None) -> BenchmarkAsset:
    """An unseen hook that pulls the ball back, scaled as `hook`."""
    return hook(rng, pools=OOD_CORRECT_ASSET_POOLS)


def ood_correct_long_stick(rng=None) -> BenchmarkAsset:
    """An unseen long stick that pushes the ball, scaled as `long_stick`."""
    return long_stick(rng, pools=OOD_CORRECT_ASSET_POOLS)


# What a correct-tool OOD twin swaps in. The hammers have no function audit,
# so IceBreak has no correct-tool twin.
OOD_CORRECT_AFFORDANCE_TOOLS = {"hook": ood_correct_hook}
OOD_CORRECT_SPATIAL_TOOLS = {"long_stick": ood_correct_long_stick}


# -- the objects the tools act on --------------------------------------------


# What the hook pulls back and the stick pushes across: one ball for both, so
# neither scenario varies the thing the tool acts on, and it shows the same
# face from every side the robot can come at it from.
TARGET_HALF = 0.03
TARGET_MASS = 0.05
TARGET_RGBA = [0.82, 0.42, 0.24, 1.0]


# A ball keeps rolling on the rolling friction the bench carries, and a light
# push would take it off the far edge. Rolling friction needs condim 6, and
# the bench's contact priority would otherwise override it, so the ball
# outranks the bench and brings its own contact parameters. Measured with the
# robot held still: a push at 0.5 m/s coasts 6 cm at this value, at 1 m/s 24 cm.
BALL_ROLLING_FRICTION = 0.0005
BALL_CONTACT_PRIORITY = 11


def push_ball(rng=None) -> PrimitiveToolAsset:
    """What the stick pushes and the hook pulls back.

    A ball rolls where a cube drags, so a push has to be measured, not shoved.
    """
    del rng  # one ball, so there is nothing to draw
    asset = _primitive(
        "push_ball",
        "target",
        [
            {
                "type": "sphere",
                "pos": [0.0, 0.0, 0.0],
                "size": [TARGET_HALF],
                "rgba": TARGET_RGBA,
                "mass": TARGET_MASS,
            }
        ],
        length=2 * TARGET_HALF,
        mass=TARGET_MASS,
        grip_width=2 * TARGET_HALF,
    )
    asset.height = TARGET_HALF
    asset.friction = [_FRICTION[0], _FRICTION[1], BALL_ROLLING_FRICTION]
    asset.condim = 6
    asset.contact_priority = BALL_CONTACT_PRIORITY
    # A stiff contact makes a sphere chatter on the bench; this one settles.
    asset.contact_solref = [0.02, 1.0]
    return asset


def _primitive(
    uid: str,
    role: str,
    geoms: list[dict],
    *,
    length: float,
    mass: float,
    grip_width: float,
    static: bool = False,
) -> PrimitiveToolAsset:
    """A procedural asset, for the things a scanned mesh cannot express.

    The target cube, the ice shards and the goal marker are geometry with a
    job rather than objects with a look: the cube has to present one face and
    one edge whatever the episode, the shards have to tile into a block, and
    the marker has to be a ring thin enough never to touch anything.
    """
    return PrimitiveToolAsset(
        {
            "uid": uid,
            "label": uid,
            "name": uid.replace("_", " "),
            "role": role,
            "kind": "primitive",
            "geoms": geoms,
            "static": static,
            "mass": mass,
            "friction": _FRICTION,
            "length": length,
            "footprint": length,
            "height": grip_width,
            "extent": [length, grip_width, grip_width],
            "grip_width": grip_width,
            "rest_offset": 0.0,
            "n_convex_parts": len(geoms),
            "mesh_scale": [1.0, 1.0, 1.0],
            "visual_mesh": None,
            "texture": None,
            "up_axis": "z",
            "source": "primitive",
            "license": "procedural",
            "group": "reasoning_tool",
            "group_label": "Reasoning tool",
            "colour": None,
            "holds": None,
            "interior": None,
        }
    )


ICE_HALF = 0.04
ICE_GRID = (2, 2, 2)
ICE_BLOCK_HALF = 2 * ICE_HALF
ICE_SHARD_MASS = 0.35
ICE_BLOCK_MASS = ICE_SHARD_MASS * ICE_GRID[0] * ICE_GRID[1] * ICE_GRID[2]
ICE_RGBA = [0.68, 0.85, 0.92, 0.65]


def ice_block() -> PrimitiveToolAsset:
    """The block, whole: the one cube the robot is asked to break.

    MuJoCo cannot fracture a body, so a break is this being taken out of play
    and the shards below being let in. Until then it is the only ice there is.
    """
    return _primitive(
        "ice_block",
        "ice",
        [
            {
                "type": "box",
                "pos": [0.0, 0.0, 0.0],
                "size": [ICE_BLOCK_HALF, ICE_BLOCK_HALF, ICE_BLOCK_HALF],
                "rgba": ICE_RGBA,
                "mass": ICE_BLOCK_MASS,
            }
        ],
        length=2 * ICE_BLOCK_HALF,
        mass=ICE_BLOCK_MASS,
        grip_width=ICE_BLOCK_HALF,
    )


def ice_shard(index: int) -> PrimitiveToolAsset:
    """One eighth of the block, for after it breaks.

    They exist from the start because MuJoCo cannot add a body to a running
    model, but stay inert and invisible inside the block until it breaks.
    """
    return _primitive(
        f"ice_shard_{index}",
        "ice",
        [
            {
                "type": "box",
                "pos": [0.0, 0.0, 0.0],
                "size": [ICE_HALF, ICE_HALF, ICE_HALF],
                "rgba": ICE_RGBA,
                "mass": ICE_SHARD_MASS,
            }
        ],
        length=2 * ICE_HALF,
        mass=ICE_SHARD_MASS,
        grip_width=ICE_HALF,
    )


def ice_offsets() -> list[list[float]]:
    """Where each shard sits inside the block, from its centre.

    Along the world axes, which is each shard's own frame too: a grid laid
    along anything else tiles square boxes on tilted rows and leaves gaps.
    """
    nx, ny, nz = ICE_GRID
    step = 2 * ICE_HALF
    return [
        [
            step * (ix - 0.5 * (nx - 1)),
            step * (iy - 0.5 * (ny - 1)),
            step * (iz - 0.5 * (nz - 1)),
        ]
        for ix in range(nx)
        for iy in range(ny)
        for iz in range(nz)
    ]


def marker_asset(uid: str, radius: float, rgba=(0.16, 0.62, 0.36, 0.55)):
    """The goal spot: a flat disc that is drawn but never collides.

    It has to be visible, because the instruction speaks of a marked spot and a
    policy that cannot see the mark cannot aim at it. It must also not touch
    anything, or an object would ride up onto it and the success radius would
    be measuring a collision instead of a position.
    """
    return _primitive(
        uid,
        "goal",
        [
            {
                "type": "cylinder",
                "size": [radius, 0.0015],
                "pos": [0.0, 0.0, 0.0015],
                "rgba": list(rgba),
                "mass": 0.0,
                "visual_only": True,
            }
        ],
        length=2 * radius,
        mass=0.0,
        grip_width=0.003,
        static=True,
    )


RING_LINE = 0.005
RING_SEGMENTS = 32


def ring_marker(uid: str, radius: float, rgba=(0.95, 0.95, 0.95, 1.0)):
    """The goal spot as a chalk-like circle on the bench: an outline, not a disc.

    Drawn as short capsules laid end to end around the circle, none of which
    collide, so the mark is visible from the head camera but nothing rides up
    onto it. `radius` is the success radius, so the line shows exactly where
    the ball has to end up.
    """
    geoms = []
    for i in range(RING_SEGMENTS):
        a0 = 2 * np.pi * i / RING_SEGMENTS
        a1 = 2 * np.pi * (i + 1) / RING_SEGMENTS
        geoms.append(
            {
                "type": "capsule",
                "size": [RING_LINE],
                "fromto": [
                    radius * np.cos(a0),
                    radius * np.sin(a0),
                    RING_LINE,
                    radius * np.cos(a1),
                    radius * np.sin(a1),
                    RING_LINE,
                ],
                "rgba": list(rgba),
                "mass": 0.0,
                "visual_only": True,
            }
        )
    return _primitive(
        uid,
        "goal",
        geoms,
        length=2 * radius,
        mass=0.0,
        grip_width=0.001,
        static=True,
    )


def available() -> bool:
    """Whether every mesh a pool names has been downloaded.

    The irrelevant objects count: a scene is short two objects without them,
    which is a different scene rather than a duller one.
    """
    from humanoidtoolbench.assets.distractors import DISTRACTOR_POOL

    try:
        for ids in (*ASSET_POOLS.values(), DISTRACTOR_POOL):
            for asset_id in ids:
                _package(asset_id)
    except MissingToolAssets:
        return False
    return True
