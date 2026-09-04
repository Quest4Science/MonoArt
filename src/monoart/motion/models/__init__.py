# Models module
# Lazy imports to avoid circular dependencies


def __getattr__(name):
    if name == "ArticulatedMAFT":
        from .articulated_maft import ArticulatedMAFT

        return ArticulatedMAFT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ArticulatedMAFT"]
