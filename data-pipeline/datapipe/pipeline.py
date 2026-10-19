from __future__ import annotations

import importlib
import json
import os
import shutil
import time

from . import backends as B
from .schema import Task
from .stages import STAGES, Context

OUTPUT_FILES = ["tasks.jsonl", "dropped.jsonl", "eval_holdout.jsonl", "compose_pool.jsonl",
                "rollouts.jsonl", "trajectories.jsonl", "report.json", "report.md", "audit"]


def load_tasks(path: str, fmt: str = "task") -> list[Task]:
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if fmt == "math2code":
                from .adapters import math2code_to_task
                d = math2code_to_task(d)
            t = Task.from_dict(d)
            if t.env.repo_dir and not os.path.isabs(t.env.repo_dir):
                t.env.repo_dir = os.path.join(base, t.env.repo_dir)
            out.append(t)
    return out


def _stage_class(name: str):
    if name in STAGES:
        return STAGES[name]
    if ":" not in name:
        raise ValueError(f"unknown stage {name!r}; known: {sorted(STAGES)} (or use pkg.module:Class)")
    mod, _, cls = name.partition(":")
    return getattr(importlib.import_module(mod), cls)


def run_pipeline(tasks: list[Task], cfg: dict, out_dir: str, overwrite: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    for f in OUTPUT_FILES:
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            if not overwrite:
                raise FileExistsError(f"{p} exists; pass overwrite=True / --overwrite")
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    run_id = f"{cfg.get('run_name', 'run')}-{time.strftime('%Y%m%d-%H%M%S')}"
    backends = {k: B.build(v) for k, v in (cfg.get("backends") or {}).items()}
    ctx = Context(cfg, out_dir, backends, run_id)
    flow = [{"stage": "input", "n": len(tasks)}]
    try:
        for name in cfg.get("order", list(STAGES)):
            section = name.split(":")[-1] if ":" in name else name
            scfg = cfg.get(section, {}) or {}
            if not scfg.get("enabled", True):
                flow.append({"stage": section, "skipped": True, "n": len(tasks)})
                continue
            t0 = time.time()
            tasks = _stage_class(name)(scfg, ctx).run(tasks)
            flow.append({"stage": section, "n": len(tasks), "secs": round(time.time() - t0, 2)})
        w = ctx.writer("tasks.jsonl")
        for t in tasks:
            w.write(t.to_dict())
    finally:
        ctx.close()
    report = {"run_id": run_id, "flow": flow, "stats": {k: dict(v) for k, v in ctx.stats.items()},
              "backends": {k: (v or {}).get("type") for k, v in (cfg.get("backends") or {}).items()}}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(report))
    return report


def render_report(r: dict) -> str:
    lines = [f"# Pipeline run {r['run_id']}", "", "| 阶段 | 剩余任务数 | 用时 (s) |", "| --- | --- | --- |"]
    for s in r["flow"]:
        lines.append(f"| {s['stage']} | {s['n']} | {'跳过' if s.get('skipped') else s.get('secs', '')} |")
    lines += ["", "## 各阶段计数", ""]
    for stage, st in r["stats"].items():
        lines.append(f"**{stage}**: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
        lines.append("")
    lines += ["## 后端", ""] + [f"- {k}: `{v}`" for k, v in r["backends"].items()]
    return "\n".join(lines) + "\n"
