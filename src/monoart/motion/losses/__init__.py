# Losses module
# Lazy imports to avoid circular dependencies


def __getattr__(name):
    if name == "MotionPredictionLoss":
        from .motion_loss import MotionPredictionLoss

        return MotionPredictionLoss
    elif name == "CategoryLoss":
        from .motion_loss import CategoryLoss

        return CategoryLoss
    elif name == "CombinedLoss":
        from .combined_loss import CombinedLoss

        return CombinedLoss
    elif name == "HungarianMatcher":
        from .combined_loss import HungarianMatcher

        return HungarianMatcher
    elif name == "MaskLoss":
        from .combined_loss import MaskLoss

        return MaskLoss
    elif name == "ScoreLoss":
        from .combined_loss import ScoreLoss

        return ScoreLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MotionPredictionLoss",
    "CategoryLoss",
    "CombinedLoss",
    "HungarianMatcher",
    "MaskLoss",
    "ScoreLoss",
]
