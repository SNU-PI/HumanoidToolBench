from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """One evaluation run; the defaults are the canonical public protocol."""

    env_id: str
    policy: str = "http"
    split: str = "test"
    host: str = "127.0.0.1"
    port: int = 21000
    # Episode i is whatever reset(seed=i) builds; no other episode source exists.
    data_format: str = "seeds"
    sim_mode: str = "mujoco"
    headless: bool = True
    eval_dir: str = "data/evals"  # Each run writes to a unique child directory.
    max_episode_steps: int = 3000
    num_episodes: int = 100
    episode_start: int = 10000
    save_video: bool = True
    controller: str = "decoupled_wbc"

    def __post_init__(self) -> None:
        if self.data_format != "seeds":
            raise ValueError(
                f"Unsupported data format {self.data_format!r}; only 'seeds' is supported."
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
