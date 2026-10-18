"""Download raw datasets and convert them into pipeline inputs.

A sources config lists datasets; each has a loader (where rows come from) and
a mapper (how a row becomes a Task). Train sources go to tasks.jsonl, eval
sources to eval/<name>.jsonl (used by decontamination).

    sources:
      - name: gsm8k
        role: train                       # train | eval
        loader: {type: hf, repo: openai/gsm8k, config: main, split: train}
        mapper: {type: fields, prompt: question, answer: answer, domain: math}
        where: [{field: level, in: [4, 5]}]   # optional row filter
"""
from __future__ import annotations

import json
import os
import time
import traceback

from .loaders import LOADERS, load_rows, register_loader
from .mappers import MAPPERS, register_mapper


def _keep(row: dict, where: list[dict]) -> bool:
    """where: [{field: task_id, min: 11, max: 510}, {field: level, in: [4, 5]}, {field: x, equals: y},
    {field: question, regex: "..."}, {field: question, not_regex: "..."}]"""
    import re
    from .mappers import get_path
    for w in where:
        v = get_path(row, w["field"])
        if "equals" in w and v != w["equals"]:
            return False
        if "in" in w and v not in w["in"]:
            return False
        if "min" in w and (v is None or v < w["min"]):
            return False
        if "max" in w and (v is None or v > w["max"]):
            return False
        if "regex" in w and not re.search(w["regex"], str(v or "")):
            return False
        if "not_regex" in w and re.search(w["not_regex"], str(v or "")):
            return False
    return True


def map_rows(rows, mapper: dict, source: str, stats: dict | None = None, where: list[dict] | None = None):
    mapper = dict(mapper)
    kind = mapper.pop("type", "fields")
    if kind not in MAPPERS:
        raise ValueError(f"unknown mapper {kind!r}; known: {sorted(MAPPERS)}")
    fn = MAPPERS[kind]
    stats = stats if stats is not None else {}
    stats.update(rows=0, skipped=0)
    for i, row in enumerate(rows):
        stats["rows"] += 1
        if where and not _keep(row, where):
            stats["skipped"] += 1
            continue
        t = fn(row, i, source=source, **mapper)
        if t is None:
            stats["skipped"] += 1
            continue
        yield t


def download_sources(cfg: dict, out_dir: str, only: list[str] | None = None) -> dict:
    """Run every source in cfg["sources"]. Writes:
         out_dir/tasks.jsonl            all train sources, concatenated
         out_dir/train/<name>.jsonl     per train source
         out_dir/eval/<name>.jsonl      per eval source ({"id", "prompt", "repo"})
         out_dir/manifest.json          what was downloaded, counts, errors
    """
    cache = cfg.get("cache_dir") or os.path.join(out_dir, "cache")
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": []}
    for src in cfg.get("sources", []):
        name, role = src["name"], src.get("role", "train")
        if only and name not in only:
            continue
        if src.get("enabled", True) is False:
            continue
        entry = {"name": name, "role": role, "loader": src["loader"], "license": src.get("license")}
        path = os.path.join(out_dir, role if role == "eval" else "train", f"{name}.jsonl")
        t0 = time.time()
        try:
            n, st = 0, {}
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                for t in map_rows(load_rows(src["loader"], cache), src.get("mapper", {}), name, st,
                                  src.get("where")):
                    if role == "eval":
                        t = {"id": t["id"], "prompt": t["prompt"], "repo": t.get("repo")}
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
                    n += 1
            os.replace(path + ".tmp", path)
            entry.update(status="ok", rows=st["rows"], written=n, skipped=st["skipped"],
                         path=os.path.relpath(path, out_dir))
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
        entry["secs"] = round(time.time() - t0, 1)
        manifest["sources"].append(entry)
    # concatenate train sources that exist (including ones from earlier runs)
    with open(os.path.join(out_dir, "tasks.jsonl"), "w", encoding="utf-8") as out:
        for src in cfg.get("sources", []):
            p = os.path.join(out_dir, "train", f"{src['name']}.jsonl")
            if src.get("role", "train") == "train" and src.get("enabled", True) is not False and os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    out.writelines(f)
    mpath = os.path.join(out_dir, "manifest.json")
    if only and os.path.exists(mpath):  # keep entries of sources not re-run this time
        with open(mpath, encoding="utf-8") as f:
            done = {e["name"] for e in manifest["sources"]}
            manifest["sources"] = [e for e in json.load(f).get("sources", []) if e["name"] not in done] + \
                manifest["sources"]
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


__all__ = ["download_sources", "load_rows", "map_rows", "register_loader", "register_mapper", "LOADERS", "MAPPERS"]
