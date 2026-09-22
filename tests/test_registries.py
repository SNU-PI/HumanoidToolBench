"""Regression tests for task and robot registry lifecycle behavior."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
from gymnasium import spaces

from theta_bench.core.registry import RegistryMixin
from theta_bench.envs.base_dual_env import BaseDualSim
from theta_bench.robots.registry import RobotRegistry
from theta_bench.tasks.registry import TaskRegistry


def test_registry_tables_are_isolated_between_subclasses() -> None:
    class FirstRegistry(RegistryMixin[object]):
        pass

    class SecondRegistry(RegistryMixin[object]):
        pass

    @FirstRegistry.register("shared")
    class FirstValue:
        pass

    assert FirstRegistry.make("shared").__class__ is FirstValue
    try:
        SecondRegistry.make("shared")
    except ValueError as exc:
        assert "No class registered" in str(exc)
    else:
        raise AssertionError("registry entries leaked between subclasses")


def test_task_registry_returns_a_fresh_instance(monkeypatch) -> None:
    monkeypatch.setattr(TaskRegistry, "_registry", {})

    @TaskRegistry.register("test_task")
    class TestTask:
        pass

    first = TaskRegistry.make("test_task")
    second = TaskRegistry.make("test_task")

    assert first is not second


def test_robot_registry_bootstraps_g1_sonic_and_returns_fresh_instances(
    monkeypatch,
) -> None:
    monkeypatch.setattr(RobotRegistry, "_registry", {})
    imported: list[str] = []

    def import_robot_module(module_name: str):
        imported.append(module_name)

        @RobotRegistry.register("g1_sonic")
        class TestRobot:
            def __init__(self, marker: object) -> None:
                self.marker = marker

        return SimpleNamespace(G1Sonic=TestRobot)

    monkeypatch.setattr(
        "theta_bench.robots.registry.import_module", import_robot_module
    )
    marker = object()

    first = RobotRegistry.make("g1_sonic", marker=marker)
    second = RobotRegistry.make("g1_sonic", marker=marker)

    assert imported == ["theta_bench.robots.g1_sonic"]
    assert first is not second
    assert first.marker is marker


def test_base_dual_sim_creates_task_once(monkeypatch) -> None:
    task = SimpleNamespace(
        action_space=spaces.Discrete(2),
        observation_space=spaces.Discrete(3),
        metadata={},
    )
    calls: list[tuple[str, dict]] = []

    def make_task(uid: str, *args, **kwargs):
        del args
        calls.append((uid, kwargs))
        return task

    class FakeMujocoSimulator:
        def __init__(self, received_task, headless: bool, make_renderers=True) -> None:
            self.task = received_task
            self.headless = headless

        def close(self) -> None:
            pass

    monkeypatch.setattr(TaskRegistry, "make", make_task)
    monkeypatch.setattr("theta_bench.engines.MujocoSimulator", FakeMujocoSimulator)

    env = BaseDualSim("test_task", headless=False, marker="value")

    assert calls == [("test_task", {"marker": "value"})]
    assert env.task is task
    assert env.mujoco.task is task
    assert not env.mujoco.headless


def test_cell_config_is_instance_local(monkeypatch) -> None:
    """Two cells of one scenario must not share a randomizer config.

    The mode decides which three tools go on the bench, and it is written onto
    the tool randomizer's config at construction. Were that config the class
    attribute, building an R cell would silently turn every S cell of the same
    scenario into an R one.
    """
    from theta_bench.tasks.g1_ball_retrieve_teleop import G1BallRetrieveTeleop
    from theta_bench.tasks.g1_toolbench_tabletop import G1ToolbenchTabletop

    monkeypatch.setattr(G1ToolbenchTabletop, "__init__", lambda self, *a, **k: None)
    class_cfg = G1BallRetrieveTeleop.dr_cfgs["tools"]
    original_mode = class_cfg.mode

    standard = G1BallRetrieveTeleop(level=1, mode="S")
    reasoning = G1BallRetrieveTeleop(level=1, mode="R")

    assert standard.dr_cfgs is not reasoning.dr_cfgs
    assert standard.dr_cfgs["tools"] is not reasoning.dr_cfgs["tools"]
    assert standard.dr_cfgs["tools"].mode == "S"
    assert reasoning.dr_cfgs["tools"].mode == "R"
    assert class_cfg.mode == original_mode


def test_the_grid_is_three_scenarios_by_three_levels_by_two_modes() -> None:
    """Every cell exists, and each carries the arguments that define it."""
    from theta_bench.envs import LEVEL_STEPS, SCENARIOS

    assert set(SCENARIOS) == {"G1BallMove", "G1BallRetrieve", "G1IceBreak"}
    seen = set()
    for name, task in SCENARIOS.items():
        for level in LEVEL_STEPS:
            for mode in ("S", "R"):
                env_id = f"theta_bench/{name}-L{level}-{mode}"
                spec = gym.spec(env_id)
                assert spec.version is None
                kwargs = spec.kwargs
                assert kwargs["task"] == task
                assert kwargs["level"] == level
                assert kwargs["mode"] == mode
                seen.add(env_id)

    assert len(seen) == 18
    registered = {i for i in gym.registry if i.startswith("theta_bench/")}
    assert registered == seen, "the reasoning grid contains exactly these cells"


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("mode", ["S", "R"])
@pytest.mark.parametrize(
    "canonical, legacy",
    [("G1BallMove", "G1StickMove"), ("G1BallRetrieve", "G1HookRetrieve")],
)
def test_legacy_environment_names_are_not_registered(canonical, legacy, level, mode):
    canonical_spec = gym.spec(f"theta_bench/{canonical}-L{level}-{mode}")
    assert canonical_spec.max_episode_steps == 3000
    with pytest.raises(gym.error.NameNotFound):
        gym.spec(f"theta_bench/{legacy}-L{level}-{mode}")


def test_every_cell_has_an_instruction() -> None:
    from theta_bench.envs import SCENARIOS
    from theta_bench.tasks.tool_reasoning import INSTRUCTIONS, LEVELS, MODE_ALIAS, MODES

    assert MODE_ALIAS == {"S": "R0", "R": "R1"}
    expected = {
        (uid, level, mode)
        for uid in SCENARIOS.values()
        for level in LEVELS
        for mode in MODES
    }
    assert set(INSTRUCTIONS) == expected, "all eighteen cells, and nothing else"
    for text in INSTRUCTIONS.values():
        assert text.endswith(".")
        assert "{" not in text, "written out, not a template"


def test_scenario_instructions_match_each_level_and_are_identical_across_modes() -> (
    None
):
    """All eighteen cells use the requested scenario wording without tool hints."""
    from theta_bench.tasks.tool_reasoning import INSTRUCTIONS, LEVELS, MODES

    scenarios = {
        "g1_ball_move_teleop": (
            "Pick the tool for moving the ball to the target.",
            "Pick the tool and move the ball to the target.",
        ),
        "g1_ball_retrieve_teleop": (
            "Pick the tool for retrieving the ball to the target.",
            "Pick the tool and retrieve the ball to the target.",
        ),
        "g1_ice_break_teleop": (
            "Pick the tool for breaking the ice blocks.",
            "Pick the tool and break the ice blocks.",
        ),
    }
    for uid, (pick, execute) in scenarios.items():
        for level in LEVELS:
            for mode in MODES:
                assert INSTRUCTIONS[(uid, level, mode)] == (
                    pick if level == 0 else execute
                )
