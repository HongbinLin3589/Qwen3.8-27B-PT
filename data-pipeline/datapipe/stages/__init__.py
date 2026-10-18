"""Pipeline stages. Each module registers its Stage with @register_stage;
the default order lives in datapipe/defaults.yaml (`order`)."""
from .base import STAGES, Context, Stage, register_stage
from . import dedup, env_check, verifier_review, leak_check, difficulty, trajectory  # noqa: F401  (registers)

__all__ = ["Context", "Stage", "STAGES", "register_stage"]
