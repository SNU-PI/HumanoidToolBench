"""Build the Isaac scene, the way SIMPLE builds it where the assets allow.

Three different sources feed one stage:

* **The robot is its USD**, `robots/g1/g1_29dof_wholebody_dex3.usd`, the same
  asset SIMPLE referenced. Its links are flat siblings under the articulation
  root and carry the vendor's own materials, so the G1 looks exactly as it does
  in SIMPLE rather than merely similar.
* **The room and the bench are USD geometry with MDL materials**, the NVIDIA
  vMaterials that `RoomDR` already drew for the episode. RTX loads the material
  itself instead of the flat texture MuJoCo has to settle for.
* **The task's objects come from the compiled model.** Catalog objects use OBJ
  meshes while canonical tasks also create procedural primitives. They have no
  authored USD to reference, so they keep their MuJoCo geometry and textures
  and take their PBR constants from the material randomizer.

Lighting is `layout.lights`, the RTX rig drawn by `IsaacLightingDR`; MuJoCo's
own light rig is a separate thing and is not read here.

Nothing in this module carries physics. Links and bodies are plain `Xform`s
whose transforms are overwritten from `mjData` each frame, so Isaac needs no
articulation, no PhysX scene and no joint-name mapping: MuJoCo has already
solved the kinematics and this just reads the answer.

Only what a MuJoCo camera would draw is built: render groups 0-2 with a
non-zero alpha. That is what excludes the invisible convex collision hulls the
object builder parks under every textured visual mesh.

THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from theta_bench.utils import resolve_data_path

# MuJoCo's default `mjvOption` draws geom groups 0, 1 and 2 and hides the rest.
VISIBLE_GEOM_GROUPS = frozenset({0, 1, 2})

# An infinite MuJoCo plane has to become a finite USD quad.
INFINITE_PLANE_HALF_EXTENT = 50.0


def _float_array(values: Any) -> np.ndarray:
    """A contiguous float32 copy, which is what `Vt.*Array.FromNumpy` needs."""
    return np.ascontiguousarray(values, dtype=np.float32)


def _name(model: Any, objtype: Any, index: int, fallback: str) -> str:
    raw = mujoco.mj_id2name(model, objtype, index)
    return Tf.MakeValidIdentifier(raw if raw else f"{fallback}_{index}")


def _transform(position: Any, quaternion: Any, scale: Any | None = None) -> Gf.Matrix4d:
    """Build a USD transform from a MuJoCo position, wxyz quaternion and scale."""
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotateOnly(
        Gf.Quatd(
            float(quaternion[0]),
            Gf.Vec3d(float(quaternion[1]), float(quaternion[2]), float(quaternion[3])),
        )
    )
    matrix.SetTranslateOnly(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    if scale is None:
        return matrix
    # USD composes row vectors, so the scale has to be applied before the
    # rotation and translation that follow it.
    scale_matrix = Gf.Matrix4d(1.0)
    scale_matrix.SetScale(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
    return scale_matrix * matrix


def _quat(attribute: Any, w: float, x: float, y: float, z: float) -> Any:
    """A quaternion of the precision the existing xformOp was authored with."""
    single = attribute.GetTypeName() == Sdf.ValueTypeNames.Quatf
    make = Gf.Quatf if single else Gf.Quatd
    vector = Gf.Vec3f if single else Gf.Vec3d
    return make(w, vector(x, y, z))


def _set_transform(prim: Any) -> Any:
    """Give a prim exactly one matrix xformOp and return that op."""
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    return xformable.AddTransformOp()


class IsaacSceneBuilder:
    """One episode's Isaac stage, refreshed from `mjData` each frame."""

    def __init__(
        self,
        stage: Any,
        texture_dir: str | os.PathLike[str],
        root_path: str = "/World/theta_bench",
        camera_root_path: str = "/World/cameras",
    ) -> None:
        self.stage = stage
        self.texture_dir = Path(texture_dir)
        self.root_path = Sdf.Path(root_path)
        self.camera_root_path = Sdf.Path(camera_root_path)

        self._body_ops: dict[int, Any] = {}
        self._camera_ops: dict[int, Any] = {}
        self._camera_paths: dict[str, str] = {}
        self._materials: dict[Any, Any] = {}
        self._textures: dict[int, str] = {}
        self._geom_opacities: dict[int, Any] = {}
        # MuJoCo body id -> the USD link Xform standing in for it.
        self._robot_link_ops: dict[int, tuple[Any, Any]] = {}
        self._robot_body_ids: set[int] = set()
        # Links with no MuJoCo body, carried by a driven ancestor:
        # (xform ops, anchor body id, rest offset to the anchor).
        self._robot_attached: list[tuple[tuple[Any, Any], int, Any]] = []
        self._layout: Any = None
        self._model: Any = None
        self._last_body_poses: tuple[np.ndarray, np.ndarray] | None = None
        self._last_camera_poses: tuple[np.ndarray, np.ndarray] | None = None
        self._last_opacities: np.ndarray | None = None
        self._mesh_cache: dict[str, Any] = {}
        self._previous_mesh_cache: dict[str, Any] = {}
        self._mesh_keys: dict[int, str] = {}

    # -- lifecycle ---------------------------------------------------------

    def build(
        self,
        model: Any,
        camera_resolutions: dict[str, tuple[int, int]],
        layout: Any = None,
        robot_usd: str | None = None,
    ) -> None:
        """Author the whole stage for `model`, replacing anything already there.

        `camera_resolutions` maps each MuJoCo camera name to the (width, height)
        it will be rendered at. It has to be passed in: MuJoCo derives a
        camera's horizontal field of view from the viewport it is rendered
        into, and `mjModel.cam_resolution` is left at its 1x1 default unless the
        task declares explicit intrinsics, so the model alone cannot say what
        the aspect ratio is.
        """
        previous_mesh_cache = self._mesh_cache
        self.clear()
        self._previous_mesh_cache = previous_mesh_cache

        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        self._layout = layout
        self._model = model
        UsdGeom.Xform.Define(self.stage, self.root_path)
        UsdGeom.Scope.Define(self.stage, self.root_path.AppendChild("Looks"))
        # The robot claims its bodies first so the prop builder skips them and
        # the MJCF's own robot meshes are never drawn over the USD.
        try:
            self._build_robot(model, robot_usd)
            self._build_bodies(model)
            self._build_lights(model, layout)
            self._build_cameras(model, camera_resolutions)
        finally:
            # Retain only geometry used by this build, including after a
            # failed reset. Memory cannot accumulate across domain variants.
            self._previous_mesh_cache = {}

    def clear(self) -> None:
        """Drop the mirrored geometry so the next reset can rebuild it.

        Cameras deliberately survive: their prim paths back Replicator render
        products, which would be invalidated by a delete.
        """
        self._body_ops.clear()
        self._materials.clear()
        self._textures.clear()
        self._geom_opacities.clear()
        self._robot_link_ops.clear()
        self._robot_body_ids.clear()
        self._robot_attached.clear()
        self._model = None
        self._last_body_poses = None
        self._last_camera_poses = None
        self._last_opacities = None
        self._mesh_cache = {}
        self._previous_mesh_cache = {}
        self._mesh_keys.clear()
        if self.stage.GetPrimAtPath(self.root_path):
            self.stage.RemovePrim(self.root_path)

    def camera_path(self, camera_name: str) -> str | None:
        return self._camera_paths.get(camera_name)

    # -- per-frame ---------------------------------------------------------

    def sync(self, data: Any) -> None:
        """Copy changed poses and opacity out of `mjData`, without tolerances."""
        xpos = data.xpos
        xquat = data.xquat
        body_changed = np.ones(len(xpos), dtype=bool)
        if self._last_body_poses is not None:
            previous_pos, previous_quat = self._last_body_poses
            body_changed = np.any(xpos != previous_pos, axis=1) | np.any(
                xquat != previous_quat, axis=1
            )
        camera_changed = np.ones(len(data.cam_xpos), dtype=bool)
        if self._last_camera_poses is not None:
            previous_pos, previous_mat = self._last_camera_poses
            camera_changed = np.any(data.cam_xpos != previous_pos, axis=1) | np.any(
                data.cam_xmat != previous_mat, axis=1
            )
        opacities = None
        if self._model is not None:
            # Use float64 just as _opacity does after converting to Python
            # floats. Evaluate all alphas together instead of clipping each
            # unchanged geom separately at every observation.
            opacities = self._model.geom_rgba[:, 3].astype(np.float64)
            material_ids = self._model.geom_matid
            has_material = material_ids >= 0
            opacities[has_material] *= self._model.mat_rgba[
                material_ids[has_material], 3
            ]
            np.clip(opacities, 0.0, 1.0, out=opacities)
        # One change block per frame: USD notifies Hydra once instead of once
        # per attribute, which is the difference between a usable and an
        # unusable frame rate at teleop cadence.
        with Sdf.ChangeBlock():
            for body_id, op in self._body_ops.items():
                if body_changed[body_id]:
                    op.Set(_transform(xpos[body_id], xquat[body_id]))

            if opacities is not None:
                for geom_id, opacity in self._geom_opacities.items():
                    if (
                        self._last_opacities is None
                        or opacities[geom_id] != self._last_opacities[geom_id]
                    ):
                        opacity.Set(float(opacities[geom_id]))

            # The USD robot's links are siblings under one root held at the
            # origin, so each link's local transform is its world pose and
            # MuJoCo's forward kinematics places the whole robot directly.
            for body_id, (translate, orient) in self._robot_link_ops.items():
                if not body_changed[body_id]:
                    continue
                position, quaternion = xpos[body_id], xquat[body_id]
                translate.Set(
                    Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
                )
                orient.Set(
                    _quat(
                        orient,
                        float(quaternion[0]),
                        float(quaternion[1]),
                        float(quaternion[2]),
                        float(quaternion[3]),
                    )
                )

            for (translate, orient), anchor, offset in self._robot_attached:
                if not body_changed[anchor]:
                    continue
                matrix = offset * _transform(xpos[anchor], xquat[anchor])
                rotation = matrix.ExtractRotationQuat()
                imaginary = rotation.GetImaginary()
                translate.Set(matrix.ExtractTranslation())
                orient.Set(
                    _quat(
                        orient,
                        float(rotation.GetReal()),
                        float(imaginary[0]),
                        float(imaginary[1]),
                        float(imaginary[2]),
                    )
                )

            quaternion = np.empty(4)
            for camera_id, op in self._camera_ops.items():
                if not camera_changed[camera_id]:
                    continue
                mujoco.mju_mat2Quat(quaternion, data.cam_xmat[camera_id])
                op.Set(_transform(data.cam_xpos[camera_id], quaternion))

        # mjData buffers are mutable views. Own the snapshots so in-place
        # physics updates cannot hide a changed pose on the next sync.
        self._last_body_poses = (xpos.copy(), xquat.copy())
        self._last_camera_poses = (data.cam_xpos.copy(), data.cam_xmat.copy())
        self._last_opacities = opacities

    # -- geometry ----------------------------------------------------------

    def _build_bodies(self, model: Any) -> None:
        geoms_by_body: dict[int, list[int]] = {}
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) in self._robot_body_ids:
                # Drawn from the robot's own USD instead.
                continue
            if int(model.geom_group[geom_id]) not in VISIBLE_GEOM_GROUPS:
                continue
            if float(model.geom_rgba[geom_id][3]) <= 0.0:
                continue
            geoms_by_body.setdefault(int(model.geom_bodyid[geom_id]), []).append(
                geom_id
            )

        for body_id, geom_ids in geoms_by_body.items():
            body_path = self.root_path.AppendChild(
                _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body")
            )
            body_prim = UsdGeom.Xform.Define(self.stage, body_path)
            self._body_ops[body_id] = _set_transform(body_prim)
            for geom_id in geom_ids:
                self._build_geom(model, body_path, geom_id)

    def _build_robot(self, model: Any, robot_usd: str | None) -> None:
        """Reference the robot's USD and bind its links to MuJoCo bodies.

        Most links share a name with a MuJoCo body and are driven from it
        directly. The rest, the head, the palms, the sensor and trim frames,
        exist only in the USD because the MJCF folds them into their parent.
        They are attached to that parent through the asset's own
        `PhysicsFixedJoint`s, which is the same relationship SIMPLE got for
        free by loading the robot as a PhysX articulation. Left unattached they
        keep the pose the asset shipped with while the rest of the robot moves,
        which tears the robot apart and loses its head.
        """
        if not robot_usd:
            return

        robot_path = self.root_path.AppendChild("Robot")
        robot_prim = self.stage.DefinePrim(robot_path, "Xform")
        robot_prim.GetReferences().AddReference(str(robot_usd))
        # The reference arrives with the asset's own root transform; the links
        # below carry world poses, so the root has to be the identity.
        _set_transform(robot_prim).Set(Gf.Matrix4d(1.0))

        links: dict[str, Any] = {}
        rest: dict[str, Gf.Matrix4d] = {}
        for link in Usd.PrimRange(robot_prim):
            name = link.GetName()
            if name == "collisions":
                # Collision proxies sit on top of the visual meshes. `guide`
                # is the purpose a renderer skips.
                UsdGeom.Imageable(link).CreatePurposeAttr(UsdGeom.Tokens.guide)
                continue
            if not link.IsA(UsdGeom.Xform) or name in links:
                continue
            ops = {
                op.GetOpName(): op
                for op in UsdGeom.Xformable(link).GetOrderedXformOps()
            }
            if "xformOp:translate" in ops and "xformOp:orient" in ops:
                links[name] = ops
                rest[name] = UsdGeom.Xformable(link).GetLocalTransformation()

        self._apply_robot_shaders(robot_prim)
        parent_of = self._fixed_joint_parents(robot_prim)

        for name, ops in links.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            pair = (ops["xformOp:translate"], ops["xformOp:orient"])
            if body_id >= 0:
                self._robot_body_ids.add(body_id)
                self._robot_link_ops[body_id] = pair
                continue

            anchor, offset = self._anchor(model, name, parent_of, rest)
            if anchor is not None:
                self._robot_attached.append((pair, anchor, offset))

    def _apply_robot_shaders(self, robot_prim: Any) -> None:
        """Push the randomizer's PBR constants into the robot's own shaders.

        The G1's USD is shaded with OmniPBR and OmniSurface, which take these
        three by name; this is the one place the robot's appearance varies, so
        without it the material domain does nothing for the robot at all.
        """
        shaders = getattr(getattr(self._layout, "robot", None), "shaders", None)
        if not shaders:
            return
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(prim)
            for name, value in shaders.items():
                shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(value))

    @staticmethod
    def _fixed_joint_parents(robot_prim: Any) -> dict[str, str]:
        """child link -> parent link, for every joint the asset declares."""
        parents: dict[str, str] = {}
        for prim in Usd.PrimRange(robot_prim):
            if "Joint" not in prim.GetTypeName():
                continue
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if body0 and body1:
                parents[body1[0].name] = body0[0].name
        return parents

    def _anchor(
        self,
        model: Any,
        name: str,
        parent_of: dict[str, str],
        rest: dict[str, Gf.Matrix4d],
    ) -> tuple[int | None, Gf.Matrix4d]:
        """Find the nearest driven ancestor, and the rest offset up to it.

        Rest transforms are authored against the robot's root, so the offset
        from a link to its anchor is `rest[link] * inv(rest[anchor])`; applying
        it to the anchor's world pose puts the link where the anchor carries
        it.
        """
        seen = {name}
        current = name
        while True:
            parent = parent_of.get(current)
            if parent is None or parent in seen:
                return None, Gf.Matrix4d(1.0)
            seen.add(parent)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent)
            if body_id >= 0 and parent in rest and name in rest:
                anchor_rest = Gf.Matrix4d(rest[parent])
                # The offset has to be rigid; a scale left in it would be
                # applied twice once the anchor's own transform lands.
                anchor_rest = anchor_rest.RemoveScaleShear()
                link_rest = Gf.Matrix4d(rest[name]).RemoveScaleShear()
                return body_id, link_rest * anchor_rest.GetInverse()
            current = parent

    def _actor_for_geom(self, model: Any, geom_id: int) -> Any:
        """The layout actor a geom belongs to, or None.

        Room primitives are keyed in the layout by the same name their MuJoCo
        body carries. Objects are keyed by role ("target", "stick") while their
        body carries the asset label, so both routes are tried.
        """
        if self._layout is None:
            return None
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        actors = getattr(self._layout, "actors", {})
        actor = actors.get(body)
        if actor is not None:
            return actor
        for candidate in actors.values():
            asset = getattr(candidate, "asset", None)
            if asset is not None and getattr(asset, "label", None) == body:
                return candidate
        return None

    def _isaac_material(self, model: Any, geom_id: int) -> Any:
        """The MDL material the room randomizer chose for this geom's surface.

        The MuJoCo body and the layout actor share a name, which is what lets a
        geom find the surface it belongs to.
        """
        actor = self._actor_for_geom(model, geom_id)
        material = getattr(actor, "isaac_material", None)
        if not material:
            return None

        source = str(material["path"])
        try:
            material_path = (
                source
                if Path(source).is_absolute() and Path(source).is_file()
                else resolve_data_path(source, auto_download=True)
            )
        except Exception:
            # The multi-gigabyte MDL library is optional. Only the Isaac
            # renderer resolves it, and a missing material falls back to the
            # surface's plain colour without affecting MuJoCo-only runs.
            return None

        key = ("mdl", material_path, material["name"])
        cached = self._materials.get(key)
        if cached is not None:
            return cached

        path = self.root_path.AppendChild("Looks").AppendChild(
            f"mdl_{len(self._materials)}"
        )
        usd_material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path.AppendChild("Shader"))
        # Authored the plain USD way rather than through Kit's material
        # library, so the stage can be built and tested without Kit.
        shader.SetSourceAsset(Sdf.AssetPath(material_path), "mdl")
        shader.SetSourceAssetSubIdentifier(material["name"], "mdl")
        usd_material.CreateSurfaceOutput("mdl").ConnectToSource(
            shader.ConnectableAPI(), "out"
        )
        self._materials[key] = usd_material
        return usd_material

    def _build_geom(self, model: Any, body_path: Any, geom_id: int) -> None:
        geom_path = body_path.AppendChild(
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "geom")
        )
        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        scale = None

        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            prim = self._define_plane(geom_path, size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            prim = UsdGeom.Sphere.Define(self.stage, geom_path)
            prim.GetRadiusAttr().Set(float(size[0]))
            self._set_extent(prim, [size[0]] * 3)
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            prim = UsdGeom.Sphere.Define(self.stage, geom_path)
            prim.GetRadiusAttr().Set(1.0)
            self._set_extent(prim, [1.0, 1.0, 1.0])
            scale = size[:3]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            prim = UsdGeom.Capsule.Define(self.stage, geom_path)
            prim.GetRadiusAttr().Set(float(size[0]))
            # MuJoCo sizes a capsule by the half-length of its cylindrical part.
            prim.GetHeightAttr().Set(float(2.0 * size[1]))
            prim.GetAxisAttr().Set(UsdGeom.Tokens.z)
            self._set_extent(prim, [size[0], size[0], size[1] + size[0]])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            prim = UsdGeom.Cylinder.Define(self.stage, geom_path)
            prim.GetRadiusAttr().Set(float(size[0]))
            prim.GetHeightAttr().Set(float(2.0 * size[1]))
            prim.GetAxisAttr().Set(UsdGeom.Tokens.z)
            self._set_extent(prim, [size[0], size[0], size[1]])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            # Not a `UsdGeom.Cube`: that is an implicit surface carrying no `st`
            # primvar, so a texture bound to it samples a single texel and the
            # box renders flat. The room, the benches and the floor are all
            # textured boxes, so they need real UVs.
            prim = self._define_box(geom_path, size, self._texuniform(model, geom_id))
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            prim = self._define_mesh(model, geom_path, int(model.geom_dataid[geom_id]))
        else:
            # Height fields, SDFs and the decorative geom types are not used by
            # any THETA(θ)-Bench task; drawing nothing is better than drawing a
            # wrong stand-in.
            return

        _set_transform(prim.GetPrim()).Set(
            _transform(model.geom_pos[geom_id], model.geom_quat[geom_id], scale)
        )
        opacity = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "thetaOpacity", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant
        )
        opacity.Set(self._opacity(model, geom_id))
        self._geom_opacities[geom_id] = opacity
        material = self._isaac_material(model, geom_id) or self._material(
            model, geom_id
        )
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)

    @staticmethod
    def _set_extent(prim: Any, half_extent: Any) -> None:
        extent = [
            Gf.Vec3f(*(-float(value) for value in half_extent)),
            Gf.Vec3f(*(float(value) for value in half_extent)),
        ]
        prim.GetExtentAttr().Set(extent)

    @staticmethod
    def _texuniform(model: Any, geom_id: int) -> bool:
        material_id = int(model.geom_matid[geom_id])
        return material_id >= 0 and bool(model.mat_texuniform[material_id])

    def _define_box(self, geom_path: Any, size: Any, texuniform: bool) -> Any:
        """A MuJoCo box as an explicit mesh, with per-face texture coordinates.

        MuJoCo textures a box face by face. Under `texuniform` it reads
        `texrepeat` as repeats per metre, so the coordinates are the face's own
        extent in metres; otherwise the texture is fitted once to each face.
        Either way the material applies `texrepeat` on top, so this only has to
        get the parameterisation right.
        """
        half = np.asarray(size[:3], dtype=np.float64)
        points, normals, texcoords, indices = [], [], [], []

        for axis in range(3):
            for sign in (1.0, -1.0):
                # Pick the two in-plane axes so that their cross product is the
                # outward normal, which keeps every face wound the same way.
                u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
                if sign < 0:
                    u_axis, v_axis = v_axis, u_axis

                normal = np.zeros(3)
                normal[axis] = sign
                u_vec, v_vec = np.zeros(3), np.zeros(3)
                u_vec[u_axis], v_vec[v_axis] = 1.0, 1.0

                base = len(points)
                for u_sign, v_sign in (
                    (-1.0, -1.0),
                    (1.0, -1.0),
                    (1.0, 1.0),
                    (-1.0, 1.0),
                ):
                    u = u_sign * half[u_axis]
                    v = v_sign * half[v_axis]
                    points.append(normal * half[axis] + u * u_vec + v * v_vec)
                    normals.append(normal)
                    texcoords.append(
                        (u, v)
                        if texuniform
                        else (0.5 * (u_sign + 1), 0.5 * (v_sign + 1))
                    )
                indices.extend([base, base + 1, base + 2, base + 3])

        mesh = UsdGeom.Mesh.Define(self.stage, geom_path)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(_float_array(points)))
        mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(6, 4, dtype=np.int32))
        )
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.asarray(indices, dtype=np.int32))
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(_float_array(normals)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec2fArray.FromNumpy(_float_array(texcoords)))
        self._set_extent(mesh, half)
        return mesh

    def _define_plane(self, geom_path: Any, size: Any) -> Any:
        """A MuJoCo plane as a USD quad; a 0 half-extent means infinite."""
        half_x = float(size[0]) or INFINITE_PLANE_HALF_EXTENT
        half_y = float(size[1]) or INFINITE_PLANE_HALF_EXTENT
        mesh = UsdGeom.Mesh.Define(self.stage, geom_path)
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(-half_x, -half_y, 0.0),
                Gf.Vec3f(half_x, -half_y, 0.0),
                Gf.Vec3f(half_x, half_y, 0.0),
                Gf.Vec3f(-half_x, half_y, 0.0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetNormalsAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st = primvars.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        # One texture tile per metre, which is how MuJoCo draws a `texuniform`
        # ground plane.
        st.Set(
            [
                Gf.Vec2f(-half_x, -half_y),
                Gf.Vec2f(half_x, -half_y),
                Gf.Vec2f(half_x, half_y),
                Gf.Vec2f(-half_x, half_y),
            ]
        )
        self._set_extent(mesh, [half_x, half_y, 0.0])
        return mesh

    def _mesh_key(self, model: Any, mesh_id: int) -> str:
        """Hash exact compiled geometry once per mesh in each scene build."""
        if mesh_id in self._mesh_keys:
            return self._mesh_keys[mesh_id]
        digest = hashlib.sha256()
        arrays = [model.mesh_scale[mesh_id]]
        for name in ("vert", "face"):
            start = int(getattr(model, f"mesh_{name}adr")[mesh_id])
            count = int(getattr(model, f"mesh_{name}num")[mesh_id])
            arrays.append(getattr(model, f"mesh_{name}")[start : start + count])
        # Hash the same indexed values the conversion reads, including any
        # missing-corner sentinels. Mesh-local slices alone can miss them.
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        for name in ("normal", "texcoord"):
            start = int(getattr(model, f"mesh_{name}adr")[mesh_id])
            digest.update(str(start >= 0).encode())
            if start >= 0:
                indices = getattr(model, f"mesh_face{name}")[
                    face_start : face_start + face_count
                ]
                arrays.extend(
                    (
                        indices,
                        getattr(model, f"mesh_{name}")[start + indices.reshape(-1)],
                    )
                )
        for array in arrays:
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(np.ascontiguousarray(array))
        key = digest.hexdigest()
        self._mesh_keys[mesh_id] = key
        return key

    def _define_mesh(self, model: Any, geom_path: Any, mesh_id: int) -> Any:
        key = self._mesh_key(model, mesh_id)
        target = self.stage.GetEditTarget()
        spec_path = target.MapToSpecPath(geom_path)
        cached = self._mesh_cache.get(key) or self._previous_mesh_cache.get(key)
        if cached is not None:
            Sdf.CopySpec(cached, "/Mesh", target.GetLayer(), spec_path)
            self._mesh_cache[key] = cached
            return UsdGeom.Mesh(self.stage.GetPrimAtPath(geom_path))

        mesh = UsdGeom.Mesh.Define(self.stage, geom_path)

        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])

        # The Vt array conversions matter: the G1 alone contributes hundreds of
        # thousands of vertices, and building one Python Gf.Vec3f per vertex is
        # slow enough to be felt on every reset.
        points = _float_array(
            model.mesh_vert[vertex_start : vertex_start + vertex_count]
        )
        faces = np.ascontiguousarray(
            model.mesh_face[face_start : face_start + face_count], dtype=np.int32
        )

        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(face_count, 3, dtype=np.int32))
        )
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        # Without this the renderer subdivides, which both softens the silhouette
        # and discards the explicit normals below.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        # The MolmoSpaces packages are authored mirrored on x, so their meshes
        # arrive with a negative scale and every triangle wound the other way.
        # USD assumes right-handed, so say otherwise rather than leave the
        # surface declared inside out. Nothing visible turned on this, which
        # is worth knowing: it is correctness, not a fix.
        scale = np.asarray(model.mesh_scale[mesh_id], dtype=np.float64)
        if float(np.prod(scale)) < 0.0:
            mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
        self._set_extent(
            mesh, np.max(np.abs(points), axis=0) if vertex_count else [0.0, 0.0, 0.0]
        )

        normal_start = int(model.mesh_normaladr[mesh_id])
        face_normals = np.asarray(
            model.mesh_facenormal[face_start : face_start + face_count], dtype=np.int32
        )
        normals = _float_array(
            np.asarray(model.mesh_normal, dtype=np.float32)[
                normal_start + face_normals.reshape(-1)
            ]
        )
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

        texcoord_start = int(model.mesh_texcoordadr[mesh_id])
        if texcoord_start >= 0:
            face_texcoords = np.asarray(
                model.mesh_facetexcoord[face_start : face_start + face_count],
                dtype=np.int32,
            )
            # MuJoCo's V runs the other way from USD's: it measures down from
            # the top row of the image where USD measures up from the bottom.
            # A tool with one texture all over survives the confusion, which is
            # why this went unnoticed, but one with regions, a black head on a
            # wooden handle, comes out with the two swapped: wood on the head
            # and a black band across the handle.
            texcoords = _float_array(
                np.asarray(model.mesh_texcoord, dtype=np.float32)[
                    texcoord_start + face_texcoords.reshape(-1)
                ]
            ).copy()
            texcoords[:, 1] = 1.0 - texcoords[:, 1]
            primvars = UsdGeom.PrimvarsAPI(mesh)
            st = primvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            st.Set(Vt.Vec2fArray.FromNumpy(texcoords))

        # Capture only immutable geometry. Geom transforms, opacity and
        # material bindings are authored by _build_geom after this returns,
        # so every episode still receives its current poses and appearance.
        cached = Sdf.Layer.CreateAnonymous("theta-mesh.usda")
        Sdf.CopySpec(target.GetLayer(), spec_path, cached, "/Mesh")
        self._mesh_cache[key] = cached
        return mesh

    # -- appearance --------------------------------------------------------

    @staticmethod
    def _opacity(model: Any, geom_id: int) -> float:
        opacity = float(model.geom_rgba[geom_id][3])
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            opacity *= float(model.mat_rgba[material_id][3])
        return float(np.clip(opacity, 0.0, 1.0))

    def _material(self, model: Any, geom_id: int) -> Any:
        """The USD material for a geom, shared by every geom that matches it."""
        material_id = int(model.geom_matid[geom_id])
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64)
        shaders = getattr(self._actor_for_geom(model, geom_id), "isaac_shaders", None)

        # The finish belongs in the key: without it the first object to build a
        # material would lend its roughness to every object drawn after it.
        finish = tuple(sorted((shaders or {}).items()))
        key: Any = (
            ("rgba", tuple(np.round(rgba, 4)), finish)
            if material_id < 0
            else ("mat", material_id, finish)
        )

        cached = self._materials.get(key)
        if cached is not None:
            return cached

        material_path = self.root_path.AppendChild("Looks").AppendChild(
            f"mat_{len(self._materials)}"
        )
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(
            self.stage, material_path.AppendChild("Surface")
        )
        shader.CreateIdAttr("UsdPreviewSurface")

        if material_id < 0:
            colour, roughness, metallic = rgba[:3], 0.6, 0.0
        else:
            material_rgba = np.asarray(model.mat_rgba[material_id], dtype=np.float64)
            colour = material_rgba[:3] * rgba[:3]
            shininess = float(model.mat_shininess[material_id])
            roughness = float(np.clip(1.0 - shininess, 0.05, 1.0))
            metallic = float(np.clip(model.mat_reflectance[material_id], 0.0, 1.0))
            self._bind_texture(model, material_id, material, shader)

        if not shader.GetInput("diffuseColor"):
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*(float(channel) for channel in colour))
            )
        if shaders:
            # What the material randomizer drew for this object. MuJoCo has no
            # equivalent knob, so this domain exists only under RTX.
            roughness = float(shaders.get("reflection_roughness_constant", roughness))
            metallic = float(shaders.get("metallic_constant", metallic))
        opacity_reader = UsdShade.Shader.Define(
            self.stage, material_path.AppendChild("opacityReader")
        )
        opacity_reader.CreateIdAttr("UsdPrimvarReader_float")
        opacity_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(
            "thetaOpacity"
        )
        opacity_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float).Set(1.0)
        opacity_reader.CreateOutput("result", Sdf.ValueTypeNames.Float)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            opacity_reader.ConnectableAPI(), "result"
        )
        # Without this UsdPreviewSurface reads opacity as a cutout mask: above
        # the threshold the surface is drawn solid, below it not at all. The
        # ice block is 0.65 and would come out opaque. Zero asks for the
        # blending MuJoCo does, where 0.65 means two thirds of the light.
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        # Roughness spreads a highlight but never removes it: UsdPreviewSurface
        # keeps a dielectric Fresnel lobe at its default index of refraction,
        # so even a fully rough surface catches the ceiling lights. These
        # objects are photogrammetry scans whose textures already carry the
        # light of the room they were scanned in, and a second highlight on
        # top of a painted one is what makes an apple look like a bauble. An
        # index of one is no refraction, which leaves the baked lighting to do
        # the work. The room's own surfaces are MDL and go nowhere near this.
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        self._materials[key] = material
        return material

    def _bind_texture(
        self, model: Any, material_id: int, material: Any, shader: Any
    ) -> None:
        texture_id = self._rgb_texture_id(model, material_id)
        if texture_id < 0:
            return
        texture_file = self._texture_png(model, texture_id)
        if texture_file is None:
            return

        material_path = material.GetPath()
        reader = UsdShade.Shader.Define(
            self.stage, material_path.AppendChild("stReader")
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        # The geometry's own texture coordinates already carry the `texuniform`
        # distinction (metres across the face, or a single fitted tile), so
        # `texrepeat` applies unchanged on top of them.
        repeat = np.asarray(model.mat_texrepeat[material_id], dtype=np.float64)
        transform = UsdShade.Shader.Define(
            self.stage, material_path.AppendChild("stTransform")
        )
        transform.CreateIdAttr("UsdTransform2d")
        transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(repeat[0]) or 1.0, float(repeat[1]) or 1.0)
        )

        texture = UsdShade.Shader.Define(
            self.stage, material_path.AppendChild("diffuseTexture")
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        # Say it rather than leave it to be guessed. These are 8-bit PNGs
        # written out of the model, which MuJoCo reads as sRGB and paints
        # straight on; UsdUVTexture defaults to "auto" and lets the renderer
        # decide, and a renderer that decides "raw" applies no gamma, which
        # comes out as a different colour rather than a different finish.
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            transform.ConnectableAPI(), "result"
        )
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )

    @staticmethod
    def _rgb_texture_id(model: Any, material_id: int) -> int:
        texture_ids = model.mat_texid
        if np.ndim(texture_ids) == 2:
            # MuJoCo 3 stores one texture per PBR role.
            return int(texture_ids[material_id][mujoco.mjtTextureRole.mjTEXROLE_RGB])
        return int(texture_ids[material_id])

    def _texture_png(self, model: Any, texture_id: int) -> str | None:
        """Write one of the model's textures out so USD can reference it."""
        cached = self._textures.get(texture_id)
        if cached is not None:
            return cached
        if int(model.tex_type[texture_id]) != mujoco.mjtTexture.mjTEXTURE_2D:
            # Skyboxes and cube maps have no single-image USD equivalent here.
            return None

        try:
            from PIL import Image  # noqa: PLC0415 - optional, ships with Isaac Sim
        except ImportError:
            return None

        height = int(model.tex_height[texture_id])
        width = int(model.tex_width[texture_id])
        channels = int(np.atleast_1d(model.tex_nchannel)[texture_id])
        start = int(model.tex_adr[texture_id])
        pixels = np.asarray(
            model.tex_data[start : start + height * width * channels], dtype=np.uint8
        ).reshape(height, width, channels)

        self.texture_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{height}x{width}x{channels}:".encode() + pixels.tobytes()
        ).hexdigest()
        # Hydra can still read the previous scene's textures during a rebuild.
        path = self.texture_dir / f"tex_{digest}.png"
        if not path.is_file():
            Image.fromarray(pixels).save(path)
        self._textures[texture_id] = str(path)
        return str(path)

    # -- lights ------------------------------------------------------------

    def _build_lights(self, model: Any, layout: Any) -> None:
        """Build the RTX rig `IsaacLightingDR` drew for this episode.

        MuJoCo's own lights are not converted. Its model has no emitter size
        and no colour temperature, so a rig that reads well under RTX cannot be
        expressed in it; the two engines carry one rig each and this reads the
        Isaac one.
        """
        del model
        lights_path = self.root_path.AppendChild("Lights")
        UsdGeom.Xform.Define(self.stage, lights_path)

        rig = list(getattr(layout, "lights", []) or [])
        if not rig:
            return

        # Every light is positioned relative to the rig's centre, so the whole
        # ceiling grid moves and tilts as one.
        centre = _transform(
            rig[0].center_light_position, rig[0].center_light_orientation
        )
        _set_transform(self.stage.GetPrimAtPath(lights_path)).Set(centre)

        for light in rig:
            path = lights_path.AppendChild(Tf.MakeValidIdentifier(light.uid))
            if light.type == "CylinderLight":
                prim = UsdLux.CylinderLight.Define(self.stage, path)
                prim.CreateRadiusAttr(float(light.light_radius))
                prim.CreateLengthAttr(float(light.light_length))
            elif light.type == "SphereLight":
                prim = UsdLux.SphereLight.Define(self.stage, path)
                prim.CreateRadiusAttr(float(light.light_radius))
            else:
                prim = UsdLux.DistantLight.Define(self.stage, path)

            prim.CreateIntensityAttr(float(light.light_intensity))
            prim.CreateEnableColorTemperatureAttr(True)
            prim.CreateColorTemperatureAttr(float(light.light_color_temperature))
            UsdLux.ShadowAPI.Apply(prim.GetPrim()).CreateShadowEnableAttr(True)
            _set_transform(prim.GetPrim()).Set(
                _transform(light.pose.position, [1.0, 0.0, 0.0, 0.0])
            )

    # -- cameras -----------------------------------------------------------

    def _build_cameras(
        self, model: Any, camera_resolutions: dict[str, tuple[int, int]]
    ) -> None:
        """Author one USD camera per MuJoCo camera, matching its intrinsics.

        The cameras sit outside the mirrored body tree and are driven from
        `mjData.cam_xpos`/`cam_xmat`, so a reset can rebuild the scene without
        invalidating the render products attached to them. MuJoCo and USD share
        the same camera convention (-Z forward, +Y up), so the pose copies
        across unchanged.
        """
        UsdGeom.Xform.Define(self.stage, self.camera_root_path)
        near = float(model.vis.map.znear) * float(model.stat.extent)
        far = float(model.vis.map.zfar) * float(model.stat.extent)

        self._camera_ops.clear()
        self._camera_paths.clear()
        for camera_id in range(model.ncam):
            name = _name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id, "camera")
            camera_path = self.camera_root_path.AppendChild(name)
            camera = UsdGeom.Camera.Define(self.stage, camera_path)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
            self._set_intrinsics(model, camera, camera_id, camera_resolutions.get(name))

            self._camera_ops[camera_id] = _set_transform(camera.GetPrim())
            self._camera_paths[name] = str(camera_path)

    @staticmethod
    def _set_intrinsics(
        model: Any,
        camera: Any,
        camera_id: int,
        resolution: tuple[int, int] | None,
    ) -> None:
        """Copy a MuJoCo camera's intrinsics onto a USD camera.

        USD derives the field of view from the aperture/focal-length ratio, so
        both may be carried in MuJoCo's length units rather than millimetres.
        """
        sensor_size = np.asarray(model.cam_sensorsize[camera_id], dtype=np.float64)
        if float(sensor_size[0]) > 0.0 and float(sensor_size[1]) > 0.0:
            focal_x, focal_y, offset_x, offset_y = (
                float(value) for value in model.cam_intrinsic[camera_id]
            )
            del focal_y  # USD has one focal length; the apertures carry the rest.
            camera.GetFocalLengthAttr().Set(focal_x)
            camera.GetHorizontalApertureAttr().Set(float(sensor_size[0]))
            camera.GetVerticalApertureAttr().Set(float(sensor_size[1]))
            camera.GetHorizontalApertureOffsetAttr().Set(offset_x)
            camera.GetVerticalApertureOffsetAttr().Set(offset_y)
            return

        # `cam_fovy` is the vertical field of view; MuJoCo widens it into the
        # horizontal one using the aspect ratio of the viewport it renders
        # into. That viewport is the render product, so its resolution is what
        # decides the horizontal aperture. Reading `mjModel.cam_resolution`
        # here instead would silently yield a square 1x1 frustum and crop
        # everything outside the middle of the frame.
        if resolution is None:
            raise ValueError(
                f"camera {camera_id} has no render resolution, and "
                "mjModel.cam_resolution cannot supply one"
            )
        width, height = float(resolution[0]), float(resolution[1])
        focal_length = 24.0  # Any value works; only the ratio is observable.
        vertical_aperture = (
            2.0
            * focal_length
            * np.tan(np.deg2rad(float(model.cam_fovy[camera_id])) / 2)
        )
        camera.GetFocalLengthAttr().Set(focal_length)
        camera.GetVerticalApertureAttr().Set(float(vertical_aperture))
        camera.GetHorizontalApertureAttr().Set(
            float(vertical_aperture * width / height)
        )
