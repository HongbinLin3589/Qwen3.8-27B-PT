"""SFT preparation, batching, training, and progress auditing."""

__all__ = ["main", "run_sft"]


def __getattr__(name: str):
    """Load existing entrypoint exports without coupling data tools to Hydra."""
    if name in __all__:
        from . import entrypoint

        return getattr(entrypoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
