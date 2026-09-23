"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from humanoidtoolbench.core.task import Task
    from humanoidtoolbench.assets.primitive import Primitive
    from humanoidtoolbench.core.actor import CameraEntity, ObjectActor, RobotActor

import mujoco
import numpy as np
import transforms3d as t3d

from humanoidtoolbench.core.object import SemanticAnnotated
from humanoidtoolbench.core.simulator import Simulator
from humanoidtoolbench.robots.protocols import Controllable


def apply_hand_contact_overrides(robot_spec: mujoco.MjSpec) -> None:
    """Stiffen every hand and gripper geom of a robot spec, in place.

    The solimp and solref values come from the Shadow Hand xml. They generate
    larger force with smaller penetration, so the fingers read as rigid rather
    than soft.
    """
    for g in robot_spec.geoms:
        if "hand" in (g.name or g.meshname) or "gripper" in (g.name or g.meshname):
            g.solimp[:3] = [0.9, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]


# How far a wrist camera looks down from the wrist's own forward
# axis, in degrees. At 0 the hand sits along the bottom edge and
# whatever it holds is half out of frame.
# Where `{side}_hand_camera_base_link` sits in the wrist frame, read out of
# Unitree's own G1 dex3 USD. The left hand is written here and the right
# mirrors in y. It is the palm link's origin.
WRIST_CAM_BASE = (0.0415, 0.00299, 0.0)
# The camera's orientation on that link, in ROS camera convention, straight
# from Unitree's camera_configs.py. Both hands share it there.
WRIST_CAM_ROS_QUAT = (0.00539, 0.86024, 0.0424, 0.50809)


class MujocoSimulator(Simulator):
    def __init__(
        self,
        task: Task,
        render_hz: int = 30,
        physics_dt: float = 0.002,
        headless=True,
    ) -> None:
        # Keep the constructor keywords compatible; cadence belongs to the task
        # and the Sonic environment owns its on-screen viewer.
        del render_hz, headless
        self.task = task
        self.physics_dt = (
            task.metadata["physics_dt"]
            if "physics_dt" in self.task.metadata
            else physics_dt
        )

        self.mj_worldbody = None
        self.render_option = None
        self.renderers = {}
        self.render_step = 0

        self.need_gravity = self.task.metadata.get("need_gravity", False)

    def update_layout(self, **kwargs) -> None:
        del kwargs
        self._setup_scene()

    def step(self, render=True, **kwargs) -> dict[str, np.ndarray] | None:
        del kwargs
        mujoco.mj_step(self.mjModel, self.mjData, nstep=1)
        physics_step = getattr(self.task, "on_physics_step", None)
        if physics_step is not None:
            physics_step(self)

        self.render_step += 1

        # Update layout poses after the initial settling steps.
        if self.render_step > 5:
            for objtype, mj_obj in self.mj_objects.items():
                self.task.layout.actors[objtype].pose.position = list(mj_obj.xpos)
                self.task.layout.actors[objtype].pose.quaternion = list(mj_obj.xquat)
            self.task.layout.actors["robot"].pose.position = list(
                np.round(self.mjData.qpos[:3], 3)
            )
            self.task.layout.actors["robot"].pose.quaternion = list(
                np.round(self.mjData.qpos[3:7], 3)
            )

        if render:
            return self.render()

    def _setup_scene(self):
        from humanoidtoolbench.assets.primitive import Primitive
        from humanoidtoolbench.core.actor import ObjectActor, RobotActor

        # https://mujoco.readthedocs.io/en/stable/computation/index.html
        # https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip
        mjSpec = mujoco.MjSpec()
        # Texture names are per-spec, and every reset builds a new spec. Keeping
        # the cache across resets makes the second episode reference a texture
        # that no longer exists ("texture 'tex_0' not found in material 0").
        self._texture_cache = {}
        mjSpec.option.timestep = self.physics_dt
        mjSpec.option.impratio = 10
        mjSpec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        mjSpec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        mjSpec.option.noslip_iterations = 2

        mj_worldbody = mjSpec.worldbody

        # A task can declare its own lighting rig as `add_light` keyword maps.
        # One overhead lamp is the fallback.
        task_lights = getattr(self.task, "mujoco_lights", None)
        if task_lights:
            for light_kwargs in task_lights:
                mj_worldbody.add_light(**light_kwargs)
        else:
            # One lamp is all a task gets unless it declares `mujoco_lights`.
            # Directional and brighter than the old point lamp at 1.5 m: that
            # one left anything more than a metre away, and any dark-textured
            # object, unreadable in a render.
            mj_worldbody.add_light(
                pos=[0, 0, 2.6],
                dir=[0, 0, -1],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                castshadow=False,
                diffuse=[0.65, 0.65, 0.65],
                specular=[0.15, 0.15, 0.15],
                ambient=[0.35, 0.35, 0.35],
            )

        for objtype, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                self._build_object(mjSpec, mj_worldbody, actor)
            elif isinstance(actor, RobotActor):
                self._build_robot(mjSpec, mj_worldbody, actor)
            elif isinstance(actor, Primitive):
                self._build_primitive(mjSpec, mj_worldbody, actor, table_name=objtype)
            else:
                raise TypeError(f"Unsupported actor type: {type(actor)}")

        self.mj_worldbody = mj_worldbody

        for cname, camera in self.task.layout.cameras.items():
            self._build_camera(cname, camera)

        # Add the ground plane at the Sonic world origin.
        ground = mj_worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,  # type: ignore
            name="ground",
            size=[0, 0, 1],
            pos=[0, 0, 0],
            rgba=[0.25, 0.25, 0.25, 1.0],
        )
        # Add friction for contact stability.
        ground.friction = [1.0, 0.005, 0.0001]  # [sliding, torsional, rolling]

        # MuJoCo applies gravity to every dynamic body, so a second force through
        # ``xfrc_applied`` would make light objects fall faster than the robot.
        self.mjModel = mjSpec.compile()
        self.mjData = mujoco.MjData(self.mjModel)
        self.mjSpec = mjSpec
        self.mjModel.opt.gravity = (0, 0, -9.81 if self.need_gravity else 0.0)

        mj_objects = {}
        obj_names = []
        for objtype, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                label = actor.asset.uid
                if isinstance(actor.asset, SemanticAnnotated):
                    label = actor.asset.label
                mj_obj = self.mjData.body(f"{label}")
                mj_objects[objtype] = mj_obj
                obj_names.append(label)

        self.mj_objects = mj_objects
        self.obj_names = obj_names

        # Bind the G1 controller to the compiled model and data.
        assert isinstance(self.task.robot, Controllable), "Task robot is None."
        self.joints, self.actuators = self.task.robot.setup_control(
            self.mjData, self.mjModel, mjSpec=self.mjSpec
        )

        mujoco.mj_forward(self.mjModel, self.mjData)

        self.render_option = mujoco.MjvOption()  # type: ignore
        mujoco.mjv_defaultOption(self.render_option)  # type: ignore

        # in case of forgetting to close the env before reset
        if self.renderers:
            self.close()

        self.renderers = {}
        for cname, camera in self.task.layout.cameras.items():
            self.renderers[cname] = mujoco.Renderer(
                self.mjModel,
                height=camera.resolution[1],
                width=camera.resolution[0],
            )  # type: ignore

        self.render_step = 0

    _PRIM_GEOM = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    }

    def _build_primitive_object(self, mjWorld, actor: ObjectActor):
        """An object whose asset declares `geoms` instead of collider meshes.

        The benchmark's ball, ice block and shards, and goal ring are
        procedural: their geometry has to be exact, and a primitive is also
        the only way to vary colour independently of shape. Everything else
        about the object (pose, free joint, friction) works as it does for a
        mesh, so tasks never branch on which kind they were handed.

        An asset marked `static` gets no free joint: that is how a goal marker
        stays where it was put.
        """
        asset = actor.asset
        label = getattr(asset, "label", None) or asset.uid
        friction = getattr(asset, "friction", None) or [0.9, 0.05, 0.005]
        contact_solref = getattr(asset, "contact_solref", [0.005, 2.0])
        # The benches carry priority 10, so by default their friction and
        # softness win every contact. An asset that has to roll on its own
        # terms (the ball) outranks them and brings its own condim.
        condim = int(getattr(asset, "condim", 4))
        priority = int(getattr(asset, "contact_priority", 0))
        body = mjWorld.add_body(
            name=label, pos=actor.pose.position, quat=actor.pose.quaternion
        )
        for i, g in enumerate(asset.geoms):
            where = (
                {"fromto": g["fromto"]}
                if "fromto" in g
                else {"pos": g.get("pos", [0, 0, 0])}
            )
            size = list(g["size"]) + [0.0] * (3 - len(g["size"]))
            visual_only = bool(g.get("visual_only"))
            body.add_geom(
                name=f"{label}_g{i}",
                type=self._PRIM_GEOM[g["type"]],
                size=size,
                rgba=g["rgba"],
                mass=float(g.get("mass", 0.05)),
                friction=friction,
                condim=condim,
                priority=priority,
                contype=0 if visual_only else 1,
                conaffinity=0 if visual_only else 1,
                solref=contact_solref,
                **where,
            )
        if not getattr(asset, "static", False):
            body.add_freejoint(name=f"{label}_joint")

    def _build_object(self, mjSpec, mjWorld, actor: ObjectActor):
        if getattr(actor.asset, "geoms", None):
            self._build_primitive_object(mjWorld, actor)
            return

        collision_meshes = actor.asset.collision_meshes_mujoco
        num_convex = len(collision_meshes)

        label = actor.asset.uid
        if isinstance(actor.asset, SemanticAnnotated):
            label = actor.asset.label

        # Assets may carry their own physics and appearance; anything they do
        # not declare keeps the historical defaults.
        asset = actor.asset
        mesh_scale = getattr(asset, "mesh_scale", None)
        total_mass = getattr(asset, "mass", None) or 0.1
        friction = getattr(asset, "friction", None) or [0.8, 0.05, 0.005]
        # As for primitives: an asset can outrank the benches' priority 10 and
        # bring its own condim, which is how a tool gets rolling friction.
        condim = int(getattr(asset, "condim", 4))
        priority = int(getattr(asset, "contact_priority", 0))
        solref = list(getattr(asset, "contact_solref", [0.005, 2]))
        collision_solrefs = getattr(asset, "collision_solrefs", None)
        if collision_solrefs is None:
            collision_solrefs = [solref] * num_convex
        elif len(collision_solrefs) != num_convex:
            raise ValueError(
                f"{asset.uid}: collision_solrefs must match collision_meshes_mujoco "
                f"({len(collision_solrefs)} != {num_convex})"
            )
        visual_mesh = getattr(asset, "visual_mesh", None)
        texture = getattr(asset, "texture", None)

        mesh_kwargs = {"scale": mesh_scale} if mesh_scale is not None else {}
        for i in range(num_convex):
            mjSpec.add_mesh(
                name=f"{label}_mesh_convex{i}",
                file=collision_meshes[i],
                **mesh_kwargs,
            )

        material_name = None
        if visual_mesh is not None:
            mjSpec.add_mesh(
                name=f"{label}_mesh_visual", file=visual_mesh, **mesh_kwargs
            )
            if texture is not None:
                mjSpec.add_texture(
                    name=f"{label}_tex",
                    type=mujoco.mjtTexture.mjTEXTURE_2D,
                    file=texture,
                )
                material = mjSpec.add_material(name=f"{label}_mat")
                material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{label}_tex"
                material_name = f"{label}_mat"

        mj_obj = mjWorld.add_body(
            name=label, pos=actor.pose.position, quat=actor.pose.quaternion
        )

        if visual_mesh is not None:
            # The textured mesh is what a camera sees; the convex parts below it
            # are collision only, so they must not be drawn on top of it.
            mj_obj.add_geom(
                name=f"{label}_visual",
                meshname=f"{label}_mesh_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                contype=0,
                conaffinity=0,
                group=1,
                mass=1e-8,
                **(
                    {"material": material_name}
                    if material_name
                    else {"rgba": [0.75, 0.75, 0.78, 1]}
                ),
            )

        num_convex = len(collision_meshes)
        for i in range(num_convex):
            mj_obj.add_geom(
                name=f"{label}_convex_{i}",
                meshname=f"{label}_mesh_convex{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                # opposing slip in the tangent plane, rotation around the contact normal
                # and rotation around the two axes of the tangent plane
                condim=condim,
                priority=priority,
                # spread the object's mass evenly over its convex parts
                mass=total_mass / num_convex,
                # rubber on rough ground: large static, sliding and torisonal friction
                friction=friction,
                group=3 if visual_mesh is not None else 0,
                rgba=[1, 1, 1, 0] if visual_mesh is not None else [1, 1, 1, 1],
                solref=collision_solrefs[i],
            )
        mj_obj.add_freejoint(name=f"{label}_joint")

    def _build_robot(self, mjSpec, mjWorld, actor: RobotActor):
        """Build the robot in the Mujoco simulator."""
        mjcf_path = actor.robot.mjcf_path
        if not os.path.isfile(mjcf_path):
            raise FileNotFoundError(
                f"Robot model not found: {mjcf_path}. It ships with the gear_sonic "
                "package; rerun `uv run --no-project scripts/setup_evaluation.py`."
            )
        robot_mjcf = mujoco.MjSpec.from_file(mjcf_path)
        apply_hand_contact_overrides(robot_mjcf)

        frame = mjWorld.add_frame(pos=actor.pose.position, quat=actor.pose.quaternion)
        # MuJoCo 3.11 prefixes attached names with "/" unless told otherwise;
        # the controller and the head camera look the robot up by its MJCF names.
        mjSpec.attach(robot_mjcf, prefix="", frame=frame)

    def _primitive_appearance(self, mjSpec, name: str, actor) -> dict:
        """Turn a primitive's `set_material` dict into add_geom kwargs.

        Understands `rgba` and an image `texture` (with `texrepeat`,
        `specular`, `shininess`). Textures are shared between primitives that
        name the same file, so a room of walls costs one texture, not four.
        """
        material = getattr(actor, "material", None) or {}
        if not material:
            return {}

        texture = material.get("texture")
        if not texture:
            return {"rgba": material["rgba"]} if "rgba" in material else {}

        tex_name = self._texture_cache.get(texture)
        if tex_name is None:
            tex_name = f"tex_{len(self._texture_cache)}"
            mjSpec.add_texture(
                name=tex_name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture)
            )
            self._texture_cache[texture] = tex_name

        mj_material = mjSpec.add_material(name=f"{name}_mat")
        mj_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex_name
        mj_material.texrepeat = material.get("texrepeat", [1.0, 1.0])
        # texuniform makes texrepeat mean "repeats per metre" on a box, which is
        # what lets one wood image tile correctly across differently sized parts.
        mj_material.texuniform = bool(material.get("texuniform", True))
        mj_material.specular = float(material.get("specular", 0.1))
        mj_material.shininess = float(material.get("shininess", 0.2))
        if "rgba" in material:
            mj_material.rgba = material["rgba"]
        return {"material": f"{name}_mat"}

    def _build_primitive(
        self, mjSpec, mjWorld, actor: Primitive, table_name: str = "table"
    ):
        from humanoidtoolbench.assets.primitive import Box

        if isinstance(actor, Box):
            table_size = 0.5 * np.array(actor.size)
            table_position = actor.pose.position.copy()

            table = mjWorld.add_body(
                name=table_name, pos=table_position, quat=actor.pose.quaternion
            )
            # `set_material` is the only way a primitive can ask for an
            # appearance; without one it keeps MuJoCo's default.
            appearance = self._primitive_appearance(mjSpec, table_name, actor)
            # Use table_name as geom name to avoid conflicts when multiple tables exist
            table.add_geom(
                name=f"{table_name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=table_size,
                condim=6,
                # Sliding and torsional as before; rolling raised from 0.0005
                # so a mallet or a hook laid on its side stays where it was
                # put now that the tool rests are gone. The bench outranks the
                # tools, so this is what every tool-on-bench contact uses.
                friction=[2, 0.04, 0.02],
                priority=10,
                **appearance,
            )
        else:
            raise TypeError(f"Unsupported primitive type: {type(actor)}")

    def _mujoco_camera_intrinsic_kwargs(self, camera: CameraEntity) -> dict[str, Any]:
        if camera.cam_cfg.intrinsics is None:
            W, H = camera.resolution
            fovy = 2 * np.arctan(H / (2 * camera.fy)) * 180 / np.pi
            return {"fovy": fovy}

        W, H = camera.resolution
        return {
            "fovy": 0.0,
            "resolution": [int(W), int(H)],
            "sensor_size": [float(W), float(H)],
            "focal_pixel": [float(camera.fx), float(camera.fy)],
            "principal_pixel": [
                float(camera.cx - 0.5 * W),
                float(camera.cy - 0.5 * H),
            ],
        }

    @staticmethod
    def _set_mujoco_camera_attr(mj_camera, names: tuple[str, ...], value) -> bool:
        for name in names:
            if hasattr(mj_camera, name):
                setattr(mj_camera, name, value)
                return True
        return False

    def _apply_mujoco_camera_intrinsics(self, mj_camera, camera: CameraEntity) -> None:
        if camera.cam_cfg.intrinsics is None:
            return

        W, H = camera.resolution
        self._set_mujoco_camera_attr(mj_camera, ("fovy",), 0.0)
        self._set_mujoco_camera_attr(mj_camera, ("resolution",), [int(W), int(H)])
        self._set_mujoco_camera_attr(
            mj_camera,
            ("sensor_size", "sensorsize"),
            [float(W), float(H)],
        )
        self._set_mujoco_camera_attr(
            mj_camera,
            ("focal_pixel", "focalpixel"),
            [float(camera.fx), float(camera.fy)],
        )
        self._set_mujoco_camera_attr(
            mj_camera,
            ("principal_pixel", "principalpixel"),
            [
                float(camera.cx - 0.5 * W),
                float(camera.cy - 0.5 * H),
            ],
        )

    def _add_mujoco_camera(self, parent, camera: CameraEntity, **kwargs) -> None:
        intrinsic_kwargs = self._mujoco_camera_intrinsic_kwargs(camera)
        try:
            mj_camera = parent.add_camera(**kwargs, **intrinsic_kwargs)
        except TypeError:
            W, H = camera.resolution
            fovy = 2 * np.arctan(H / (2 * camera.fy)) * 180 / np.pi
            mj_camera = parent.add_camera(**kwargs, fovy=fovy)
            self._apply_mujoco_camera_intrinsics(mj_camera, camera)
        else:
            self._apply_mujoco_camera_intrinsics(mj_camera, camera)

    def _build_camera(self, cname: str, camera: CameraEntity):
        # Camera poses are authored in the USD/OpenGL camera convention (-Z
        # forward, +Y up); this rotation maps them onto MuJoCo's camera frame.
        q_cam_convention = t3d.quaternions.mat2quat(
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        )
        if camera.mount == "eye_in_head":
            torso_body = None
            for body in self.mj_worldbody.find_all("body"):
                if body.name == "torso_link":
                    torso_body = body
                    break
            assert torso_body is not None
            # Calibrated offsets align the head camera with torso_link in the
            # G1 MuJoCo model.
            DEFAULT_HEAD_CAM_POSITION = np.array(
                [0.05366004 + 0.0039635, 0.01752999 + 0, 0.4738702 + -0.044],
                dtype=np.float32,
            )
            DEFAULT_HEAD_CAM_ORIENTATION = np.array(
                [0.91496, 0.0, 0.40355, 0.0], dtype=np.float32
            )
            q = np.asarray(camera.pose.quaternion, dtype=np.float32)
            is_identity_quat = np.allclose(q[1:], 0.0, atol=1e-6) and np.isclose(
                abs(float(q[0])), 1.0, atol=1e-6
            )
            assert is_identity_quat, (
                "Expected eye_in_head camera quaternion to be identity (wxyz)"
            )

            self._add_mujoco_camera(
                torso_body,
                camera,
                name=cname,
                pos=DEFAULT_HEAD_CAM_POSITION + camera.pose.position,  # FIXME
                quat=t3d.quaternions.qmult(
                    DEFAULT_HEAD_CAM_ORIENTATION, q_cam_convention
                ),
            )
        elif camera.mount == "eye_in_hand":
            # Which hand: the camera's own name says so.
            side = "left" if "left" in cname else "right"
            link = self.task.robot.wrist_cam_link(side)
            wrist_body = None
            for body in self.mj_worldbody.find_all("body"):
                if body.name == link:
                    wrist_body = body
                    break
            assert wrist_body is not None, f"no body named {link!r} to mount on"
            # The wrist frame, measured off the model rather than assumed:
            # +x runs down the fingers (0.165 m to the knuckles on both hands),
            # +z is the index-to-middle spread, and the palm faces -y on the
            # left hand and +y on the right, because that is the way each thumb
            # folds across it. So the back of the hand is +y on the left and -y
            # on the right, and the two mounts mirror.
            #
            # Reading +z as the back of the hand instead put the camera out on
            # the index-finger side looking across the palm, which filled the
            # frame with the hand and hid whatever it was holding.
            # Unitree's own mount, copied rather than invented, structure and
            # all. Their Isaac Lab sim hangs the camera off a link named
            # `{side}_hand_camera_base_link`, so that link is built here too
            # rather than folded into the camera's own offset.
            #
            # Where it sits comes from the USD they ship with
            # unitree_sim_isaaclab; the offset on top of it is what their
            # camera_configs.py gives relative to it. Their two hands share one
            # rotation and mirror only the position, so this does the same.
            mirror = 1.0 if side == "left" else -1.0
            base = wrist_body.add_body(
                name=f"{side}_hand_camera_base_link",
                pos=[
                    WRIST_CAM_BASE[0],
                    mirror * WRIST_CAM_BASE[1],
                    WRIST_CAM_BASE[2],
                ],
                quat=[1.0, 0.0, 0.0, 0.0],
                mass=0.0,
            )
            offset = camera.pose.position
            rotation = t3d.quaternions.quat2mat(WRIST_CAM_ROS_QUAT)
            # ROS camera convention: +z is the line of sight, +x right, +y down.
            # MuJoCo wants the right and up axes, so the heading is implied.
            across = rotation @ np.array([1.0, 0.0, 0.0])
            above = rotation @ np.array([0.0, -1.0, 0.0])
            self._add_mujoco_camera(
                base,
                camera,
                name=cname,
                pos=[
                    float(offset[0]),
                    mirror * float(offset[1]),
                    float(offset[2]),
                ],
                xyaxes=[*(float(v) for v in across), *(float(v) for v in above)],
            )
        else:
            raise ValueError(
                f"Toolbench cameras must use eye_in_head or eye_in_hand, "
                f"got {camera.mount!r}"
            )

    def get_robot_qpos(self) -> dict[str, float]:
        from humanoidtoolbench.robots.protocols import Controllable

        if not isinstance(self.task.robot, Controllable):
            raise TypeError("The task robot is not a Robot instance.")
        return self.task.robot.get_robot_qpos()

    def apply_action(self, action_cmd) -> None:
        self.task.robot.apply_action(action_cmd)

    def render(self) -> dict[str, np.ndarray]:
        image_observations: dict[str, np.ndarray] = {}
        for camera in self.mj_worldbody.find_all("camera"):
            renderer = self.renderers.get(camera.name)
            if renderer is None:
                continue
            renderer.update_scene(
                self.mjData,
                scene_option=self.render_option,
                camera=camera.name,
            )
            render_product = renderer.render()
            image_observations[camera.name] = render_product[..., :3].astype(
                np.uint8, copy=False
            )
        return image_observations

    def close(self):
        if hasattr(self, "renderers") and self.renderers:
            for renderer in self.renderers.values():
                renderer.close()
            self.renderers = {}
