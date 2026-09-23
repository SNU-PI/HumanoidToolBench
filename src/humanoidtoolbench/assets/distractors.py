"""The irrelevant objects that share the bench with the tools.

Every scene carries two of these, in both modes. They are what makes the tool
row a choice rather than a single object waiting to be picked up, and they are
deliberately not tools: an apple or a mug is no answer to any of these jobs, so
ruling them out takes recognition and nothing more. That is the point. The
reasoning the benchmark prices lives in the confusable tool that mode R adds,
and it can only be read as such if the rest of the bench is constant between
the two modes.

They are drawn from a pool per episode rather than fixed, so a policy meets a
different bench each time and cannot learn the scene by heart. They are also
small: an object the size of a tool would compete with it for the grasp, and
one the size of the block would be mistaken for the target.

The pool below is uncurated on purpose. Unlike the tools there is nothing
to choose between here: any of these is equally not an answer, and a wider
pool only makes the bench vary more, so it holds every candidate the
MolmoSpaces catalogue returned for four household-object queries (food,
tableware, containers, odds and ends).

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from humanoidtoolbench.assets.tools import (
    MissingToolAssets,
    _bounds,
    _package,
    mesh_asset,
)

# Small household things, none of which is a tool for pushing, pulling or
# striking. Sized to a common bulk so which one is drawn changes the look of
# the bench and nothing about the difficulty of clearing it.
#
# Bulk, not length. Holding the longest axis to one figure made a candle
# 0.140 x 0.019 x 0.016 and an apple 0.140 x 0.135 x 0.135, sixty four times
# the volume, so beside a 0.36 m hammer half the pool could not be made out at
# all and the other half crowded the row.
DISTRACTOR_BULK = 0.08
# A candle at that bulk runs to 0.29 m, longer than the ice block and close to
# a tool, so the long axis is capped and the thinnest things stay short.
DISTRACTOR_MAX_LENGTH = 0.18
DISTRACTOR_MASS = 0.18


def build(asset_id: str) -> Any:
    """One irrelevant object, sized for the bench."""
    return mesh_asset(
        "irrelevant",
        length=_length(asset_id),
        mass=DISTRACTOR_MASS,
        uid=asset_id,
    )


@lru_cache(maxsize=None)
def _length(asset_id: str) -> float:
    """The long axis that gives this mesh the pool's common bulk."""
    low, high = _bounds(_package(asset_id) / f"{asset_id}_visual.obj")
    natural = np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
    bulk = float(np.cbrt(np.prod(np.maximum(natural, 1e-6))))
    return min(DISTRACTOR_MAX_LENGTH, float(np.max(natural)) * DISTRACTOR_BULK / bulk)


# Grouped by the catalogue query that returned them; the draw ignores the
# grouping.
DISTRACTOR_POOL: list[str] = [
    # food
    "005a246f8c304e77b27cf11cd53ff4ed",  # apple
    "0098150095bb4b0f80a7dce8764ed392",  # apple
    "0231b3f6d9164c38b072d691bc1ea95b",  # apple
    "02a86f92fd824f5fa9da13ff1075c17f",  # apple
    "033fd9c58c1944a7b742b421ad0a8ef9",  # apple
    "04171af27fa541dead4476307a5559df",  # apple
    "0238adfcca3e4727b9c36eba3a9eb7cb",  # banana
    "03b7e92ec16948959169b42e436ebe81",  # banana
    "05df5a56978e4477aebc7bf155e94236",  # banana
    "2b2d1ec0c18d455798d46cd02c8afa86",  # banana
    "31de74dc44724c67934d1375a7fc07e0",  # banana
    "3ff314a82d41478382e322798739dde7",  # banana
    "05ff58cabc054c45a7df8442d59d0e13",  # lemon
    "0701a4213dad42c1a3c2947a0cf92abf",  # lemon
    "1fad12825a2d483182f86869d2ad111b",  # lemon
    "2a9e10ffd5b0405eaabfe496d61de0bd",  # lemon
    "2e2d067b8a2845acb6f585c1c5eac840",  # lemon
    "4ee454c08f5c45a7ac12ffd484c6ec14",  # lemon
    "0d6f1b3b07174a169d518ad12f6e2c0b",  # egg
    "0e40664775db477eb91abb56e61b84c4",  # decorative egg
    "12cbd6dd23af4dd2a510339a8bf0d281",  # decorative egg
    "16fc37d3e9474c20ab52491e2a84f6af",  # decorative egg
    "1a4eb7ec11bf44358f5cfe2577a0fa49",  # egg
    "2125c39434ed4b3696bdff0a0bdb6fb2",  # decorative egg
    # tableware
    "026122925fab446494e815c4fccf0960",  # mug
    "02d6c5052e3e4a659ab64c2525f4a822",  # mug
    "0423a2de1d4942e6a2778353994fad42",  # mug
    "047462f5fdc244339511f87f4df49446",  # mug
    "04d0553202d34b299bc0bf43025b6ef8",  # mug
    "0592fecb8a124d598befd935159bd561",  # mug
    "022b877f1a3e42aa89cc508bc16c3e00",  # cup
    "027f9a512d2e49a582290e8a47ae68ed",  # teacup
    "06405732c5eb4e7c9ed27536a91e9209",  # coffee cup
    "09c0c98f282d43c2a138c0b114d9ae95",  # cup
    "0beab25d297644c6a0ab7508e45ee553",  # cup
    "10899ecd56d74abf91674254b0435ef3",  # coffee cup
    "005f5630a54b442291aeb0a5d487353b",  # bowl
    "01395fa3a8e54abc8c32a016b2d8c5f5",  # bowl
    "01ec7658fef64e18b69d68cf8b06548b",  # bowl
    "036abe36af224e588d3f573f44c24ab3",  # bowl
    "03f1185ff6d4428fa81f6fc9db589338",  # bowl
    "03fa6768245446f4bab866db37a6caec",  # bowl
    "031d94f071594cae98565833cf686ec3",  # nameplate
    "05204d13ca344f069a5a4ac3c6e796bf",  # nameplate
    "0762cb8db9c24337b6e946c3e03b6bcd",  # nameplate
    "1669f0e8e23d4160ae064c39867f6218",  # ceramic plate
    "1c809c3aa8564665beaab1d1950cb3c7",  # nameplate
    "1e264fb3307d4cecaf903a2e8d55cb23",  # decorative plate
    # containers
    "013b0fff25ab49c08ba1195ca7d7df46",  # beverage can
    "015cc2427bc74ba79231e3d1354b6e9b",  # beverage can
    "03222789235742bdb9bccd4d6859624d",  # can
    "0364ab96f338493c972248102b462aa4",  # beverage can
    "03dad84f1f424aae8521cf8eda940936",  # beverage can
    "08129621ded8409c951bd1ae9a260127",  # beverage can
    "017b70bb9ff2459cbf522460dd3da182",  # bottle
    "05ef0e44c661413a86ccdb7332a456c6",  # bottle
    "0937863cbbf34247a9d1200db980b414",  # bottle
    "0ddbd8be52794fa3a9f58e26e9d0d082",  # bottle
    "12fdf6eefdfe425c881048271c663a22",  # bottle
    "17cd007c38824321b184c763b1eec4e2",  # bottle
    "0200f72df1234e98b83820d2ab4adc12",  # box
    "05624f21a8684474b641ab7865d5c3b0",  # jewelry box
    "062a5843f2214b5ebf9c90273f803e16",  # medicine box
    "08e7c58bae834ee1a53759b6719c079b",  # decorative box
    "0f43887bf9da4ba6b6bc1e6ff1e911a7",  # decorative box
    "0fde9522f9104eac932ec90dd44b333b",  # game box
    "411652ce228c4689afc013a702f4e705",  # jar
    "4911919955ff4f788b60950e64c0a61f",  # jar
    "4e43eb29aa144bdbada1ec500d218a81",  # jar
    "50f676e0b0dc4931b2f95ea7fde295e4",  # jar
    "73e266e435b24c0dbee40b274de71082",  # jar
    "a2f9fcfa1b414daa858616bad5bc5198",  # jar
    # odds and ends
    "0056bf73a1474aaf9d60227776a3e167",  # book
    "06bf1edc7a7b4a3486dae576df1e9121",  # book
    "091e139f26554be4844712ab50f8ad2a",  # book
    "0b1124e7d0ff4bfcb397fd50bf2ce7cf",  # book
    "0a074433f30d4e819c19fa7fc01f3211",  # candle
    "1d03fb33f15b4f778eec5a8617ff70c4",  # candle
    "1dd1c02247754af297948be324b14479",  # candle
    "24948fe838294d53b15e05bed1c2e5bb",  # candle
    "00325ac6b516409a80e122997e1914b6",  # soap dish
    "21346853e28c44fba72f83d1b681bfba",  # soap dish
    "4a04be27a4c94a52a7939303db225197",  # soap
    "539c979c75b34576b75f3383e96eadb5",  # soap
    "0849a8eb5c974781b94d05a35c5125ce",  # digital alarm clock
    "0e09a2d7fdf34f4487e2646b435b3b78",  # tabletop clock
    "1361433bf5d9419aa24f9d0036c6eb1a",  # alarm clock
    "15c626a3aab443998123f10dc0d38a65",  # alarm clock
    "03772ec493194563a4141a91a9fb609c",  # remote control
    "08a82d744b39481dbb40289faea1ba8f",  # remote control
    "2094c2498e9846ad849113a98f4b8eb9",  # remote control
    "24b0ef19fb344e8aa1cc4e79e5a81564",  # remote control
    "2cb7026ffacf4889bb265099b8460c7f",  # sponge
    "46a3949089bb43d8a4c1f5fd93ae8c4f",  # cleaning sponge
    "5527fc83613a4653b35a3c1a75bee9a5",  # sponge
    "77b6d6c657c143c7a64d93bf362bcf43",  # natural sponge
]


def sample(count: int, rng: Any) -> list[Any]:
    """`count` distinct irrelevant objects, drawn without replacement.

    A pool smaller than `count` repeats rather than failing: an under-curated
    pool should make the bench monotonous, not stop the episode.
    """
    if not DISTRACTOR_POOL:
        return []

    order = rng.permutation(len(DISTRACTOR_POOL))
    picked: list[Any] = []
    index = 0
    while len(picked) < count:
        asset_id = DISTRACTOR_POOL[int(order[index % len(order)])]
        index += 1
        try:
            asset = build(asset_id)
        except MissingToolAssets:
            # A pool entry that was never downloaded should cost one object,
            # not the whole episode.
            if index > 4 * max(count, len(DISTRACTOR_POOL)):
                break
            continue
        asset.uid = asset.label = f"irrelevant_{len(picked)}_{asset_id[:8]}"
        picked.append(asset)
    return picked
