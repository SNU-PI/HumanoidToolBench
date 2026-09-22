"""The public task runtime preserves canonical scoring and rejects variants."""

from types import SimpleNamespace

import numpy as np
import pytest

from humanoidtoolbench.tasks.g1_ball_move_teleop import G1BallMoveTeleop
from humanoidtoolbench.tasks.g1_ball_retrieve_teleop import G1BallRetrieveTeleop
from humanoidtoolbench.tasks.g1_ice_break_teleop import (
    FRACTURE_IMPULSE,
    SCATTER,
    G1IceBreakTeleop,
)
from humanoidtoolbench.tasks.g1_toolbench_tabletop import G1ToolbenchTabletop

TASKS = (G1BallMoveTeleop, G1BallRetrieveTeleop, G1IceBreakTeleop)
REMOVED_OPTIONS = (
    "gap_scale",
    "sticky_grasp",
    "hand_reach",
    "no_tools",
    "fail_on_tool_lift",
    "touch_breaks",
    "near_blocks",
)


@pytest.mark.parametrize("task_type", TASKS)
@pytest.mark.parametrize("option", REMOVED_OPTIONS)
@pytest.mark.parametrize("enabled", [False, True])
def test_removed_task_options_fail_before_initializing_the_robot(
    task_type, option, enabled, monkeypatch
):
    def unexpected_initialization(*args, **kwargs):
        pytest.fail("unsupported options must fail before robot initialization")

    monkeypatch.setattr(G1ToolbenchTabletop, "__init__", unexpected_initialization)
    value = (0.5 if enabled else 1.0) if option == "gap_scale" else enabled
    with pytest.raises(TypeError, match=f"Unsupported task options: {option}"):
        task_type(level=1, mode="R", **{option: value})


@pytest.fixture(params=TASKS)
def task_type(request, monkeypatch):
    monkeypatch.setattr(G1ToolbenchTabletop, "__init__", lambda *a, **k: None)
    return request.param


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("mode", ["S", "R"])
def test_canonical_cells_keep_their_success_hold(task_type, level, mode):
    task = task_type(level=level, mode=mode)
    task._reset_pick_state()
    task.picked_correct_tool = lambda info: True
    task.job_done = lambda info, **kwargs: False

    for option in REMOVED_OPTIONS:
        assert not hasattr(task, option)
    assert not hasattr(task, "edit_spec")
    assert not hasattr(task, "_apply_easy")
    assert task.metadata["max_episode_steps"] == 3000
    assert task.metadata["render_hz"] == 50

    # Picking the correct tool completes L0 only. Execution levels also need
    # the scenario's geometric outcome, held for fifty consecutive controls.
    for _ in range(49):
        assert not task.check_success({})
    assert task.check_success({}) is (level == 0)
    assert not task.check_failure({})

    task.picked_correct_tool = lambda info: False
    assert not task.check_success({})
    task.picked_correct_tool = lambda info: True
    task.job_done = lambda info, **kwargs: True
    for _ in range(49):
        assert not task.check_success({})
    assert task.check_success({})


def test_l0_still_requires_eight_centimetres_of_tool_lift(task_type):
    task = task_type(level=0, mode="R")
    task._layout = SimpleNamespace(
        actors={
            "tool": SimpleNamespace(
                asset=SimpleNamespace(label="tool"),
                pose=SimpleNamespace(position=[0.0, 0.0, 0.0]),
            )
        }
    )
    task._reset_pick_state()
    assert not task.picked_correct_tool({"tool": np.array([0.0, 0.0, 0.0])})
    assert not task.picked_correct_tool({"tool": np.array([0.0, 0.0, 0.079])})
    assert task.picked_correct_tool({"tool": np.array([0.0, 0.0, 0.081])})


def test_ice_requires_qualifying_impacts_and_both_blocks_to_scatter(monkeypatch):
    monkeypatch.setattr(G1ToolbenchTabletop, "__init__", lambda *a, **k: None)
    task = G1IceBreakTeleop(level=1, mode="R")
    impulses = [0.0, 0.0]
    spread = [0.0, 0.0]
    fractured = []
    task._blow = lambda env: impulses
    task._fracture = lambda env, block: fractured.append(block)
    task._scatter = lambda env, block: spread[block]

    assert not task.job_done({})
    impulses[:] = [FRACTURE_IMPULSE - 1e-6] * 2
    assert not task.job_done({})
    assert not fractured

    impulses[0] = FRACTURE_IMPULSE
    spread[0] = SCATTER
    assert not task.job_done({})
    assert fractured == [0]

    impulses[1] = FRACTURE_IMPULSE
    assert not task.job_done({})
    assert fractured == [0, 1]
    spread[1] = SCATTER
    assert task.job_done({})
    assert task._broken_count == 2
