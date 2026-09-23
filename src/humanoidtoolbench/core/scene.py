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

    def set_table(self, table: Asset) -> None:
        self.table = table
