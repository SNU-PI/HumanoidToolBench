"""Procedural textures for the tool-use room.

The upstream scene binds NVIDIA's `Walnut_Planks.mdl`; MuJoCo cannot read MDL
and the table asset only names the material, so there is no image to extract.
These are generated instead: deterministic (a fixed seed: a texture is an
asset, not a sample), no download, and tuned to roughly the tone the MDL
renders so the room keeps its intended look.

Generation is lazy: `ensure(name)` writes the PNG on first use and returns the
path thereafter. Nothing needs to be built ahead of time or committed.

    from humanoidtoolbench.assets.textures import ensure
    path = ensure("wood")          # -> ~/.cache/humanoidtoolbench/textures/wood.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
TEXTURE_DIR = Path(
    os.environ.get("HUMANOIDTOOLBENCH_TEXTURE_DIR", _CACHE_ROOT / "humanoidtoolbench" / "textures")
).expanduser()

SIZE = 1024


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _smooth_noise(rng, w, h, cells):
    """Low-frequency noise, bicubic-resampled up to (h, w)."""
    from PIL import Image

    small = rng.normal(0.0, 1.0, (cells, cells)).astype(np.float32)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC))


def wood(w=SIZE, h=SIZE, base=(0.52, 0.33, 0.19), plank_rows=6) -> np.ndarray:
    """Plank grain: rings along x, a seam every plank, per-plank tone variation."""
    rng = _rng()
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / w, y / h

    # Rings: a gently warped sine along the plank's length. Turn the warp up
    # and it stops reading as sawn boards and starts reading as burl plywood,
    # which is what a first pass at this looked like.
    warp = _smooth_noise(rng, w, h, 10)
    grain = np.sin((u * 24.0 + warp * 0.6) * np.pi)
    grain = 0.5 + 0.5 * np.sign(grain) * np.abs(grain) ** 0.6

    # Planks: a dark seam at each boundary, and each plank a shade different.
    # The per-plank tone does more for realism than the grain contrast does.
    row = np.floor(v * plank_rows).astype(int)
    seam = np.abs((v * plank_rows) % 1.0 - 0.5) * 2.0
    seam = np.clip((seam - 0.965) / 0.035, 0.0, 1.0)
    tone = (rng.random(plank_rows)[row] - 0.5) * 0.20

    shade = 0.90 + 0.10 * grain + tone - 0.45 * seam
    shade += rng.normal(0, 0.010, (h, w))  # fibre speckle
    rgb = np.clip(np.asarray(base)[None, None, :] * shade[..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


def wood_edge(w=SIZE // 4, h=SIZE, base=(0.46, 0.29, 0.17)) -> np.ndarray:
    """End grain for the slab's sides: same wood, no planks, tighter rings."""
    rng = _rng()
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = _smooth_noise(rng, w, h, 12)
    grain = np.sin((y / h * 90.0 + warp * 2.2) * np.pi)
    shade = 0.84 + 0.16 * (0.5 + 0.5 * grain) + rng.normal(0, 0.01, (h, w))
    rgb = np.clip(np.asarray(base)[None, None, :] * shade[..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


def plaster(w=512, h=512, base=(0.90, 0.90, 0.89)) -> np.ndarray:
    """Wall: almost flat, just enough noise that it does not band."""
    rng = _rng()
    fine = rng.normal(0, 0.010, (h, w)).astype(np.float32)
    broad = _smooth_noise(rng, w, h, 8) * 0.012  # faint trowel unevenness
    rgb = np.clip(
        np.asarray(base)[None, None, :] * (1.0 + fine + broad)[..., None], 0, 1
    )
    return (rgb * 255).astype(np.uint8)


def concrete(w=512, h=512, base=(0.74, 0.74, 0.72)) -> np.ndarray:
    """Floor: polished screed, broad mottling plus a sparse aggregate speckle."""
    rng = _rng()
    mottle = _smooth_noise(rng, w, h, 10) * 0.05
    fine = rng.normal(0, 0.018, (h, w)).astype(np.float32)
    aggregate = (rng.random((h, w)) > 0.9975).astype(np.float32) * rng.uniform(
        -0.22, -0.10
    )
    shade = np.clip(1.0 + mottle + fine + aggregate, 0.55, 1.25)
    rgb = np.clip(np.asarray(base)[None, None, :] * shade[..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


def steel(w=256, h=256, base=(0.33, 0.34, 0.36)) -> np.ndarray:
    """Table legs: brushed dark steel, streaked along one axis."""
    rng = _rng()
    streak = rng.normal(0, 1.0, (1, w)).astype(np.float32)
    streak = np.repeat(streak, h, axis=0) * 0.035
    shade = np.clip(1.0 + streak + rng.normal(0, 0.008, (h, w)), 0.7, 1.3)
    rgb = np.clip(np.asarray(base)[None, None, :] * shade[..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


def tiles(w=512, h=512, base=(0.72, 0.70, 0.67), rows=6, grout=0.06) -> np.ndarray:
    """Ceramic / masonry: a grid of slabs with grout lines and per-tile tone."""
    rng = _rng()
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / w * rows, y / h * rows
    line = np.minimum(np.abs(u % 1.0 - 0.5), np.abs(v % 1.0 - 0.5)) * 2.0
    seam = np.clip((line - (1.0 - grout)) / grout, 0.0, 1.0)
    tone = (
        rng.random((rows, rows))[
            np.floor(v).astype(int) % rows, np.floor(u).astype(int) % rows
        ]
        - 0.5
    ) * 0.14
    shade = np.clip(1.0 + tone - 0.35 * seam + rng.normal(0, 0.012, (h, w)), 0.5, 1.3)
    rgb = np.clip(np.asarray(base)[None, None, :] * shade[..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


def weave(w=512, h=512, base=(0.55, 0.52, 0.50), pitch=7.0) -> np.ndarray:
    """Fabric / carpet: a plain over-under weave with fibre noise."""
    rng = _rng()
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = np.sin(x / pitch * np.pi) * np.sin(y / pitch * np.pi)
    shade = np.clip(1.0 + 0.13 * warp + rng.normal(0, 0.03, (h, w)), 0.6, 1.35)
    rgb = np.clip(np.asarray(base)[None, None, :] * shade[..., None], 0, 1)
    return (rgb * 255).astype(np.uint8)


GENERATORS = {
    "wood": wood,
    "wood_edge": wood_edge,
    "plaster": plaster,
    "concrete": concrete,
    "steel": steel,
    "tiles": tiles,
    "weave": weave,
}

# ------------------------------------------------------- vMaterials -> MuJoCo
# RoomDR picks NVIDIA vMaterials, which are MDL: a shader graph that
# only an MDL-capable renderer can instantiate. There is no image anywhere on disk for MuJoCo to
# use (`resources/vMaterials_2/` holds only the index), and no MDL renderer in
# a mujoco-only checkout. So a material is reproduced rather than loaded: its
# path gives the family, its name seeds the colour and scale. Same material
# name always yields the same texture, different ones look different, which is
# what randomising materials is for.

_FAMILY = {
    "Wood": ("wood", (0.52, 0.33, 0.19)),
    "Concrete": ("concrete", (0.74, 0.74, 0.72)),
    "Ground": ("concrete", (0.62, 0.58, 0.52)),
    "Stone": ("tiles", (0.66, 0.64, 0.61)),
    "Masonry": ("tiles", (0.60, 0.36, 0.30)),
    "Ceramic": ("tiles", (0.86, 0.85, 0.83)),
    "Plaster": ("plaster", (0.90, 0.90, 0.89)),
    "Paint": ("plaster", (0.80, 0.80, 0.82)),
    "Paper": ("plaster", (0.92, 0.90, 0.86)),
    "Metal": ("steel", (0.55, 0.56, 0.58)),
    "Carpaint": ("steel", (0.45, 0.48, 0.55)),
    "Fabric": ("weave", (0.55, 0.52, 0.50)),
    "Carpet": ("weave", (0.45, 0.40, 0.36)),
    "Leather": ("weave", (0.40, 0.28, 0.22)),
    "Mesh": ("weave", (0.50, 0.50, 0.52)),
}
_FAMILY_DEFAULT = ("plaster", (0.78, 0.78, 0.78))


def _material_key(material: dict) -> tuple[str, str]:
    """(family, name) for a recorded vMaterials entry."""
    path = str(material.get("path", ""))
    family = Path(path).parent.name if path else ""
    return family, str(material.get("name") or Path(path).stem or "unnamed")


def material_texture(
    material: dict, out_dir: Path | None = None
) -> tuple[str, list[float]]:
    """Render a vMaterials entry as a MuJoCo-usable texture.

    Returns (png path, rgba tint). The tint carries the family's base colour so
    a caller that only wants an albedo can use it without the image.
    """
    import hashlib

    from PIL import Image

    family, name = _material_key(material)
    generator, base = _FAMILY.get(family, _FAMILY_DEFAULT)

    # Deterministic per material name: same name -> same look, every run.
    #
    # Mostly brightness, only a little hue. An equal swing on all three
    # channels reads as a lighter or darker version of the same material; an
    # independent swing per channel re-colours it, which is how concrete came
    # out pink and plaster came out green. Wood should still look like wood.
    digest = hashlib.sha1(f"{family}/{name}".encode()).digest()
    luma = 1.0 + 0.22 * (digest[0] / 255.0 - 0.5) * 2.0  # +/- 22% overall
    hue = (np.array([digest[i + 1] / 255.0 for i in range(3)]) - 0.5) * 2.0 * 0.05
    tint = np.clip(np.asarray(base) * luma * (1.0 + hue), 0.05, 1.0)

    out_dir = Path(out_dir or TEXTURE_DIR)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{family}_{name}")[
        :80
    ]
    path = out_dir / f"mat_{safe}.png"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        img = GENERATORS[generator](base=tuple(float(v) for v in tint))
        tmp = path.with_name(f".{safe}.{os.getpid()}.png")
        Image.fromarray(img).save(tmp)
        tmp.replace(path)
    return str(path), [float(tint[0]), float(tint[1]), float(tint[2]), 1.0]


def ensure(name: str, out_dir: Path | None = None, force: bool = False) -> str:
    """Path to the texture, generating it the first time it is asked for."""
    if name not in GENERATORS:
        raise KeyError(f"unknown texture {name!r}; have {sorted(GENERATORS)}")
    out_dir = Path(out_dir or TEXTURE_DIR)
    path = out_dir / f"{name}.png"
    if force or not path.exists():
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{name}.{os.getpid()}.png")
        Image.fromarray(GENERATORS[name]()).save(tmp)
        tmp.replace(path)  # atomic: two processes may race on first use
    return str(path)


def ensure_all(out_dir: Path | None = None, force: bool = False) -> dict[str, str]:
    return {name: ensure(name, out_dir, force) for name in GENERATORS}


__all__ = ["ensure", "ensure_all", "material_texture", "GENERATORS", "TEXTURE_DIR"]


if __name__ == "__main__":
    for name, path in ensure_all(force=True).items():
        print(f"{name:12s} {path}")
