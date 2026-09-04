"""Small configuration helpers used to reconstruct trained modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigNode:
    """Expose a nested mapping through both attributes and mapping methods."""

    def __init__(self, values: Mapping[str, Any]):
        for key, value in values.items():
            if isinstance(value, Mapping) and key != "feature_root_paths":
                value = ConfigNode(value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
