"""Randomize the Isaac renderer's materials.

This does not draw the room's materials. `RoomDR` already picks a vMaterials
entry per surface and records it under the surface's `vmaterial` key, so this
randomizer copies that identity as a portable reference. The Isaac scene
builder resolves the `.mdl` file only when RTX rendering is actually used.

The two engines then draw the same material to different fidelities, which is
the point. MuJoCo cannot instantiate an MDL shader graph at all, so
`humanoidtoolbench.assets.textures` generates a procedural stand-in from the entry's
family and a hash of its name: a "Wood" entry becomes a generated wood pattern
tinted by that hash, and the `.mdl` is never opened. Isaac loads the real
material. Drawing a second, independent material here would not sharpen that
difference, it would replace it with a contradiction, one bench being wood in
MuJoCo and paper in Isaac. Reusing the entry keeps a single material identity
per episode, recorded once in the state dict.

What is left to randomize is what has no MuJoCo counterpart at all: the PBR
constants on the robot and the objects, which is what SIMPLE varied.

The `.mdl` files are a multi-gigabyte download separate from the index, so a
missing library degrades to no MDL rather than failing the episode.

HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from humanoidtoolbench.core.layout import Layout

from humanoidtoolbench.core.actor import ObjectActor, RobotActor
from humanoidtoolbench.core.randomizer import Randomizer, RandomizerCfg


class IsaacMaterialDR(Randomizer):
    """Carry room material identities and draw PBR constants for the rest."""

    @staticmethod
    def _reference(vmaterial: Any) -> dict[str, str] | None:
        """Keep RoomDR's vMaterials identity portable until Isaac builds."""
        if not isinstance(vmaterial, dict) or "path" not in vmaterial:
            return None
        relative = str(vmaterial["path"])
        # The index stores paths as they sit under `data/`.
        relative = (
            relative[len("data/") :] if relative.startswith("data/") else relative
        )
        return {"path": relative, "name": vmaterial.get("name", "")}

    def _shaders(self, randomize: bool) -> dict[str, float]:
        if not randomize:
            return {
                "reflection_roughness_constant": 0.5,
                "metallic_constant": 0.0,
                "specular_level": 0.0,
            }

        cfg = self.cfg
        return {
            "reflection_roughness_constant": float(
                np.random.uniform(*cfg.roughness_range)
            ),
            "metallic_constant": float(np.random.uniform(*cfg.metallic_range)),
            "specular_level": float(np.random.uniform(*cfg.specular_range)),
        }

    def _apply(self, layout: Layout, state: dict[str, Any]) -> None:
        for surface in layout.actors.values():
            if hasattr(surface, "set_isaac_material"):
                vmaterial = getattr(surface, "material", {}).get("vmaterial")
                surface.set_isaac_material(self._reference(vmaterial))

        robot = layout.actors.get("robot")
        if isinstance(robot, RobotActor):
            robot.set_shaders(state["robot_shader_params"])

        objects = [a for a in layout.actors.values() if isinstance(a, ObjectActor)]
        shaders = state["object_shader_params"]
        if len(objects) != len(shaders):
            raise ValueError(
                "Isaac material state has "
                f"{len(shaders)} object shaders for {len(objects)} objects"
            )
        for actor, shader in zip(objects, shaders):
            actor.set_isaac_shaders(shader)

    def apply(self, split: str, layout: Layout) -> dict[str, Any]:
        """The name every other HumanoidToolBench randomizer is driven by."""
        return self(split, layout)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner_state = state_dict or None

    def __call__(self, split: str, layout: Layout, **kwargs: Any) -> dict[str, Any]:
        del split, kwargs
        if self._inner_state is not None:
            self._apply(layout, self._inner_state)
            return self._inner_state

        mode = self.cfg.material_mode
        if mode not in {"fixed", "rand_all", "rand_objects"}:
            raise ValueError(f"Invalid Isaac material mode {mode!r}")
        objects = [a for a in layout.actors.values() if isinstance(a, ObjectActor)]
        state = {
            "robot_shader_params": self._shaders(mode == "rand_all"),
            "object_shader_params": [
                self._shaders(mode in {"rand_all", "rand_objects"}) for _ in objects
            ],
        }
        self._apply(layout, state)
        return super()._transient(state)


@dataclass
class IsaacMaterialDRCfg(RandomizerCfg):
    # fixed, rand_all, rand_objects
    material_mode: str = "fixed"

    # SIMPLE drew all three across the full 0 to 1 range. Under RTX that is
    # wide enough to turn a wooden block into a chrome mirror, which reflects
    # the room back at the camera instead of showing the object.
    #
    # Narrower still than that, and towards matte, because the objects these
    # land on are photogrammetry scans: their textures already have the light
    # of the room they were scanned in baked into them. A specular highlight
    # from RTX is therefore a second highlight on top of one that is already
    # painted on, and the surface reads as wet. Matte lets the baked lighting
    # be the lighting, which is also what makes the two renderers agree, since
    # MuJoCo has no PBR to speak of.
    roughness_range: tuple[float, float] = (0.65, 0.95)
    metallic_range: tuple[float, float] = (0.0, 0.05)
    specular_range: tuple[float, float] = (0.0, 0.15)

    randmizer_class: type[Randomizer] = IsaacMaterialDR
