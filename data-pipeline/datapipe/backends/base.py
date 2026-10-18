"""Backend interfaces. Everything that needs real infrastructure sits behind one
of these, so the pipeline logic can run against mocks here and against your
cluster in production.

  Sandbox        fresh container per run (SWE / terminal environments)
  Embedder       text -> vector, for near-dup and decontamination
  Policy         samples rollouts (Qwen3.8-27B student, or the teacher)
  LogprobScorer  per-token log-probs of a fixed token sequence under a model
  ReviewAgent    judges whether a solution really meets the task statement
  ShortcutAgent  tries to get the answer without doing the work
"""
from __future__ import annotations

import abc
from typing import Optional, Sequence

from ..schema import Rollout, Solution, Task


class Container(abc.ABC):
    """One fresh, isolated copy of a task environment."""

    @abc.abstractmethod
    def apply(self, solution: Optional[Solution]) -> None:
        """Write solution file edits into the workdir. None = no change."""

    @abc.abstractmethod
    def run(self, cmd: str, timeout: float) -> tuple[int, str, str]:
        """Run a shell command in the workdir. Returns (exit_code, stdout, stderr).
        Exit code 124 means timeout."""

    @abc.abstractmethod
    def read_file(self, path: str) -> Optional[str]: ...

    @abc.abstractmethod
    def list_files(self) -> list[str]:
        """Paths relative to the workdir."""

    @abc.abstractmethod
    def destroy(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


class Sandbox(abc.ABC):
    name: str = "sandbox"

    @abc.abstractmethod
    def create(self, task: Task) -> Container:
        """Build a brand-new container for `task` (never reused)."""


class Embedder(abc.ABC):
    name: str = "embedder"

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> "list[list[float]]":
        """Return L2-normalized vectors."""


class Policy(abc.ABC):
    """A model plus the harness that drives it on a task."""
    model: str = "policy"
    harness: str = "single_turn"

    @abc.abstractmethod
    def generate(self, task: Task, n: int, sampling: dict, sandbox: Optional[Sandbox] = None) -> list[Rollout]: ...


class LogprobScorer(abc.ABC):
    model: str = "scorer"

    @abc.abstractmethod
    def score(self, task: Task, rollout: Rollout) -> Optional[list[float]]:
        """Per-token log-probs of rollout.tokens under this model, conditioned on
        the same prompt. Returns None if the tokenizers are incompatible."""


class ReviewAgent(abc.ABC):
    name: str = "review_agent"

    @abc.abstractmethod
    def judge(self, task: Task, solution: Solution) -> dict:
        """Read the statement and the solution, decide whether the solution is
        correct *by the statement*, ignoring the tests.
        Returns {"correct": bool, "confidence": float, "rationale": str}."""

    def review_tests(self, task: Task) -> list[str]:
        """Optional: flag tests that check things the statement does not ask
        for (exact wording, ordering, internals). Returns a list of issues."""
        return []


class ShortcutAgent(abc.ABC):
    name: str = "shortcut_agent"

    @abc.abstractmethod
    def attempt(self, task: Task, container: Container) -> Optional[Solution]:
        """Try to find the answer in the environment without doing the
        development work (leftover artifacts, git history, installed copies,
        network). Return the solution it would submit, or None."""
