"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from humanoidtoolbench.core.asset import Asset


class Scene:
    uid: str
    data_dir: str
    name: str

    def to_dict(self):
        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif hasattr(obj, "__dict__"):
                return {k: _convert(v) for k, v in vars(obj).items()}
            elif isinstance(obj, list):
                return [_convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_convert(v) for v in obj)
            else:
                return obj

        return _convert(self)


class TabletopScene(Scene):
    def __init__(
        self,
        uid: str = "toolbench",
        name: str = "toolbench",
        data_dir: str = "",
    ) -> None:
        self.uid = uid
        self.name = name
        self.data_dir = data_dir
        self.table: Asset
        self.table2: Asset | None = None
        # The small bench at the robot's right that the tools are laid on.
        self.tool_table: Asset | None = None

    def set_table(self, table: Asset) -> None:
        self.table = table

    def set_table2(self, table: Asset) -> None:
        self.table2 = table

    def set_tool_table(self, table: Asset) -> None:
        self.tool_table = table
