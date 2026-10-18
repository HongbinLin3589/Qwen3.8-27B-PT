from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import os
import threading
from typing import Any, Callable, Iterable, Optional

from ..schema import Task


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._f = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, obj: Any):
        line = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self):
        self._f.close()


class Context:
    """Shared state for one pipeline run: config, backends, output files, stats."""

    def __init__(self, cfg: dict, out_dir: str, backends: dict, run_id: str):
        self.cfg, self.out_dir, self.backends, self.run_id = cfg, out_dir, backends, run_id
        self._writers: dict[str, JsonlWriter] = {}
        self.stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self._lock = threading.Lock()

    def writer(self, name: str) -> JsonlWriter:
        with self._lock:
            if name not in self._writers:
                self._writers[name] = JsonlWriter(os.path.join(self.out_dir, name))
            return self._writers[name]

    def count(self, stage: str, key: str, n: int = 1):
        with self._lock:
            self.stats[stage][key] += n

    def drop(self, task: Task, stage: str, reason: str, **info):
        task.note(stage, decision="drop", reason=reason, **info)
        self.count(stage, f"drop:{reason}")
        self.writer("dropped.jsonl").write({"id": task.id, "source": task.source, "stage": stage,
                                            "reason": reason, "info": info})

    def divert(self, task: Task, stage: str, file: str, reason: str, **info):
        """Remove from the training pool but keep the task in another file."""
        task.note(stage, decision="divert", to=file, reason=reason, **info)
        self.count(stage, f"divert:{reason}")
        self.writer(file).write(task.to_dict())

    def close(self):
        for w in self._writers.values():
            w.close()


STAGES: dict = {}


def register_stage(name: str):
    """Register a Stage subclass under `name`; the config section with the
    same name holds its options and its `enabled` switch."""
    def deco(cls):
        cls.name = name
        STAGES[name] = cls
        return cls
    return deco


class Stage:
    name = "stage"

    def __init__(self, cfg: dict, ctx: Context):
        self.cfg, self.ctx = cfg, ctx

    def run(self, tasks: list[Task]) -> list[Task]:
        raise NotImplementedError

    def b(self, key: str, required: bool = True):
        v = self.ctx.backends.get(key)
        if v is None and required:
            raise RuntimeError(f"stage {self.name} needs backend '{key}'")
        return v

    def pmap(self, fn: Callable[[Task], Any], tasks: Iterable[Task]) -> list:
        tasks = list(tasks)
        workers = int(self.cfg.get("workers", self.ctx.cfg.get("workers", 4)))
        if workers <= 1:
            return [fn(t) for t in tasks]
        with cf.ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, tasks))
