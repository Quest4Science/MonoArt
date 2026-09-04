"""Part-aware semantic reasoning from TRELLIS point features."""

from __future__ import annotations

from typing import Any

__all__ = ["PartReasoner", "infer_part_features"]


def __getattr__(name: str) -> Any:
    """Import the inference facade lazily to keep module execution warning-free."""
    if name in __all__:
        from .inference import PartReasoner, infer_part_features

        return {"PartReasoner": PartReasoner, "infer_part_features": infer_part_features}[name]
    raise AttributeError(name)
