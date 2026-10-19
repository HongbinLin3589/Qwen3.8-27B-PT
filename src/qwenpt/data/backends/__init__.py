"""Backend factory. A config entry is {"type": <name or dotted.path:Class>, ...kwargs}.
A dotted path lets you plug in your own class without editing this file."""
from __future__ import annotations

import importlib
from typing import Any, Optional

from .base import Container, Embedder, LogprobScorer, Policy, ReviewAgent, Sandbox, ShortcutAgent

REGISTRY = {
    # sandboxes
    "local_dir": "qwenpt.data.backends.sandboxes:LocalDirSandbox",
    "docker": "qwenpt.data.backends.sandboxes:DockerSandbox",
    # embedders
    "hashing": "qwenpt.data.backends.embedders:HashingEmbedder",
    "sentence_transformers": "qwenpt.data.backends.embedders:SentenceTransformerEmbedder",
    "openai_embeddings": "qwenpt.data.backends.embedders:OpenAICompatEmbedder",
    # models
    "mock_policy": "qwenpt.data.backends.mock:MockPolicy",
    "openai_policy": "qwenpt.data.backends.openai_compat:OpenAICompatPolicy",
    "mock_scorer": "qwenpt.data.backends.mock:MockScorer",
    "vllm_scorer": "qwenpt.data.backends.openai_compat:VLLMLogprobScorer",
    # agents
    "mock_review": "qwenpt.data.backends.mock:MockReviewAgent",
    "grep_shortcut": "qwenpt.data.backends.mock:GrepShortcutAgent",
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
