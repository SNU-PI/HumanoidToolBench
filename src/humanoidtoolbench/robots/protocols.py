"""Small robot interfaces used by the active MuJoCo toolbench path."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from humanoidtoolbench.core.action import ActionCmd
from humanoidtoolbench.core.controller import Controller, ControllerCfg


@runtime_checkable
class HeadCamMountable(Protocol):
    head_camera_orientation: list[float]

    @property
    def head_cam_link(self) -> str: ...


@runtime_checkable
class WristCamMountable(Protocol):
    def wrist_cam_link(self, side: str) -> str:
        """The body a wrist camera hangs off, for "left" or "right"."""
        ...


class Controllable:
    _controller: Controller | None
    controller_cfg: ControllerCfg
    joint_names: list[str]

    @property
    def controller(self) -> Controller:
        if self._controller is None:
            self._controller = self.controller_cfg.clazz(self.controller_cfg)
        return self._controller

    def setup_control(
        self, mjData: Any, mjModel: Any, **kwargs: Any
    ) -> tuple[dict, dict]:
        raise NotImplementedError

    def apply_action(self, action_cmd: ActionCmd) -> None:
        raise NotImplementedError
