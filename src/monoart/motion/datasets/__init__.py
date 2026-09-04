# Lazy imports to avoid circular dependencies and missing modules during development
__all__ = [
    "ArticulatedDataset",
    "MotionParser",
    "MOTION_TYPE_MAP_3CLASS",
    "CATEGORY_MAP",
    "articulated_collate_fn",
]


def __getattr__(name):
    if name == "MotionParser":
        from .motion_parser import MotionParser

        return MotionParser
    elif name == "MOTION_TYPE_MAP_3CLASS":
        from .motion_parser import MOTION_TYPE_MAP_3CLASS

        return MOTION_TYPE_MAP_3CLASS
    elif name == "CATEGORY_MAP":
        from .motion_parser import CATEGORY_MAP

        return CATEGORY_MAP
    elif name == "ArticulatedDataset":
        from .articulated_dataset import ArticulatedDataset

        return ArticulatedDataset
    elif name == "articulated_collate_fn":
        from .collate_fn import articulated_collate_fn

        return articulated_collate_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
