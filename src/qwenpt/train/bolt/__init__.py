"""Offline BOLT rollout, grouped data, training, and verification."""

__all__ = ["main", "seed_training"]


def __getattr__(name: str):
    """Keep entrypoint exports compatible without loading Hydra for rollouts."""
    if name in __all__:
        from . import entrypoint

        return getattr(entrypoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
