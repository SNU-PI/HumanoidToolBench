"""
HumanoidToolBench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

__version__ = "0.1.0"
__all__ = ["__version__", "EvalConfig", "EvalResult", "EvalRunner"]


def __getattr__(name: str):
    if name in {"EvalConfig", "EvalResult", "EvalRunner"}:
        from theta_bench.evals import EvalConfig, EvalResult, EvalRunner

        exports = {
            "EvalConfig": EvalConfig,
            "EvalResult": EvalResult,
            "EvalRunner": EvalRunner,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
