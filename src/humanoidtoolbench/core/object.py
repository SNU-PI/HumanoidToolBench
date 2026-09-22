"""Structural metadata required by the MuJoCo object builder."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SemanticAnnotated(Protocol):
    uid: str
    label: str
    name: str
    description: str | None
