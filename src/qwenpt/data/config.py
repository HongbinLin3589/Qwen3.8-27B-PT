from __future__ import annotations

import copy
import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.yaml")


def _load_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value):
    """`--set difficulty.enabled=true` style overrides. Value is parsed as YAML."""
    import yaml
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(value)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = _load_file(DEFAULT_CONFIG_PATH)
    if path:
        cfg = deep_merge(cfg, _load_file(path))
    for o in overrides or []:
        k, _, v = o.partition("=")
        set_dotted(cfg, k.strip(), v.strip())
    base = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
    # Relative eval-set paths resolve against the config file.
    ev = cfg.get("decontam_split", {}).get("eval_sets", [])
    cfg.setdefault("decontam_split", {})["eval_sets"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in ev]
    return cfg
