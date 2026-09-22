"""
THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Importing ``theta_bench.robots.protocols`` executes this module first, so keep
# the G1 implementation lazy and let lightweight protocol imports stay lightweight.
_EXPORTS = {
    "G1Sonic": (".g1_sonic", "G1Sonic"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
