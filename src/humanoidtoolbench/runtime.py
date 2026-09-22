"""Simulator mode validation shared by HumanoidToolBench entry points."""

from __future__ import annotations

VALID_SIM_MODES = frozenset({"mujoco"})


def validate_sim_mode(sim_mode: str) -> str:
    """Validate a mode before simulator imports and return its canonical name."""

    if sim_mode not in VALID_SIM_MODES:
        choices = ", ".join(sorted(VALID_SIM_MODES))
        raise ValueError(f"Invalid sim_mode {sim_mode!r}; choose one of: {choices}")
    return sim_mode
