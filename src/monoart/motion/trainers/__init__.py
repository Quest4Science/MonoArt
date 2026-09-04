# Trainers module
# Lazy imports to avoid circular dependencies


def __getattr__(name):
    if name == "BaseTrainer":
        from .base_trainer import BaseTrainer

        return BaseTrainer
    elif name == "AverageMeter":
        from .base_trainer import AverageMeter

        return AverageMeter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseTrainer", "AverageMeter"]
