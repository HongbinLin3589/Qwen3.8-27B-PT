"""Backend factory. A config entry is {"type": <name or dotted.path:Class>, ...kwargs}.
A dotted path lets you plug in your own class without editing this file."""
from __future__ import annotations

import importlib
from typing import Any, Optional

from .base import Container, Embedder, LogprobScorer, Policy, ReviewAgent, Sandbox, ShortcutAgent

REGISTRY = {
    # sandboxes
    "local_dir": "datapipe.backends.sandboxes:LocalDirSandbox",
    "docker": "datapipe.backends.sandboxes:DockerSandbox",
    # embedders
    "hashing": "datapipe.backends.embedders:HashingEmbedder",
    "sentence_transformers": "datapipe.backends.embedders:SentenceTransformerEmbedder",
    "openai_embeddings": "datapipe.backends.embedders:OpenAICompatEmbedder",
    # models
    "mock_policy": "datapipe.backends.mock:MockPolicy",
    "openai_policy": "datapipe.backends.openai_compat:OpenAICompatPolicy",
    "mock_scorer": "datapipe.backends.mock:MockScorer",
    "vllm_scorer": "datapipe.backends.openai_compat:VLLMLogprobScorer",
    # agents
    "mock_review": "datapipe.backends.mock:MockReviewAgent",
    "grep_shortcut": "datapipe.backends.mock:GrepShortcutAgent",
}


def register_backend(name: str):
    """Decorator: make a class usable as {type: name} in the `backends` config."""
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def build(spec: Optional[dict]) -> Any:
    if not spec:
        return None
    spec = dict(spec)
    target = REGISTRY.get(spec["type"], spec["type"])
    spec.pop("type")
    if not isinstance(target, str):
        return target(**spec)
    mod, _, cls = target.partition(":")
    return getattr(importlib.import_module(mod), cls)(**spec)


__all__ = ["build", "register_backend", "Container", "Embedder", "LogprobScorer", "Policy", "ReviewAgent", "Sandbox", "ShortcutAgent"]
