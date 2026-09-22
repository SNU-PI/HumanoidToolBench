"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from gymnasium.envs.registration import register  # noqa: E402

from humanoidtoolbench.instructions import SCENARIO_UIDS

from .base_dual_env import BaseDualSim  # noqa: E402, F401

# The benchmark is a grid: three scenarios, one per reasoning axis, each run at
# three levels in two modes. A cell is a (level, mode) pair. Instructions are
# identical across modes: L0 asks for a pick, while L1 and L2 ask for a pick
# and execution with the same wording.
#
#   R - S    the cost of the reasoning the scene demands
#   L2 - L1  the cost of the locomotion
#   L1 - L0  the cost of the execution
#
# The paper writes the modes R0 and R1; S and R are the same two.
# Public scenario names and task UIDs describe the target and action.
SCENARIOS = SCENARIO_UIDS

# Every cell gets the same budget: 3000 steps is one minute at 50 Hz.
LEVEL_STEPS = {0: 3000, 1: 3000, 2: 3000}

for _name, _task in SCENARIOS.items():
    for _level, _steps in LEVEL_STEPS.items():
        for _mode in ("S", "R"):
            register(
                id=f"humanoidtoolbench/{_name}-L{_level}-{_mode}",
                entry_point="humanoidtoolbench.envs.sonic_loco_manip:SonicLocoManipEnv",
                kwargs={"task": _task, "level": _level, "mode": _mode},
                max_episode_steps=_steps,
            )
