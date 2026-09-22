"""Asset data required by the MuJoCo toolbench builder."""


class Asset:
    def __init__(self, uid: str, collision_meshes_mujoco: list[str]) -> None:
        self.uid = uid
        self.collision_meshes_mujoco = collision_meshes_mujoco
