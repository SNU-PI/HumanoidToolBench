"""Lazy public exports for benchmark assets."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BenchmarkAsset": (".benchmark", "BenchmarkAsset"),
    "Box": (".primitive", "Box"),
    "Primitive": (".primitive", "Primitive"),
    "PrimitiveToolAsset": (".benchmark", "PrimitiveToolAsset"),
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
