from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from humanoidtoolbench.policies.reasoner import ReasonerConfig
from humanoidtoolbench.scenario_names import canonicalize_env_id


@dataclass(frozen=True)
class EvalConfig:
    env_id: str
    policy: str
    split: str = "train"
    host: str = "172.17.0.1"
    port: int = 21000
    data_format: str = "seeds"
    sim_mode: str = "mujoco"
    headless: bool = False
    eval_dir: str = "data/evals"  # Each run writes to a unique child directory.
    max_episode_steps: int = 15000
    num_episodes: int = 100
    episode_start: int = 0
    data_dir: str = "data/datagen"
    success_criteria: float | None = None
    save_video: bool = True
    num_workers: int = 1
    controller: str = "decoupled_wbc"
    gear_sonic_artifact_root: str | None = None
    gear_sonic_initial_token: str | None = None
    # A BallMove RL run (legacy L0 or SONIC L0/L1/L2) scored in-process
    # instead of a policy server; ``policy`` is then only the run's label.
    rl_checkpoint: str | None = None
    reasoner: ReasonerConfig | None = None
    record_trajectory: bool = False
    trajectory_policy_id: str | None = None
    instruction_override: str | None = None
    pair_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_id", canonicalize_env_id(self.env_id))
        if self.record_trajectory and not self.trajectory_policy_id:
            raise ValueError("Trajectory recording requires --trajectory-policy-id")
        if self.pair_reference and not self.record_trajectory:
            raise ValueError("--pair-reference requires --record-trajectory")
        if (
            self.instruction_override is not None
            and not self.instruction_override.strip()
        ):
            raise ValueError("Instruction override must be nonempty")
        if self.reasoner is not None and (
            self.record_trajectory or self.instruction_override
        ):
            raise ValueError(
                "Paired instruction recording requires the fixed execution policy"
            )


@dataclass
class EvalResult:
    env_id: str
    policy: str
    split: str
    stats: dict[str, bool]
    success_rate: float
    log_path: str
    eval_dir: str


class EvalRunner:
    def __init__(
        self,
        config: EvalConfig,
        *,
        show_progress: bool = True,
    ):
        self.config = config
        self.show_progress = show_progress

    def run(self, policy: Any | None = None) -> EvalResult:
        config = self.config

        if policy is not None:
            if isinstance(policy, str):
                config = replace(config, policy=policy)
            else:
                raise TypeError("policy must be a registered policy name")

        from humanoidtoolbench.cli.eval_decoupled_wbc import run_eval as _run_eval

        return _run_eval(
            config,
            show_progress=self.show_progress,
        )
