"""Turn a pipeline run directory into training files.

Exporters are registered by name (`export.format` in the config picks one):

    @register_exporter("my_format")
    def export_my_format(run_dir, out_dir, cfg) -> dict: ...

Only `verl` is built in. Its options live under export.sft / export.rl.
"""
from __future__ import annotations

import json
import os
import hashlib

from ..schema import Task
from .verl import (apply_sft_format, sft_row, drop_nulls, read_parquet, rl_row, roundtrip_report,
                   sft_from_task_reference, sft_from_trajectory, validate_rl_row, validate_sft_row, verl_view,
                   write_parquet)

EXPORTERS: dict = {}


def register_exporter(name: str):
    def deco(fn):
        EXPORTERS[name] = fn
        return fn
    return deco


def export(run_dir: str | None, out_dir: str, cfg: dict) -> dict:
    name = cfg.get("format", "verl")
    if name not in EXPORTERS:
        raise ValueError(f"unknown export format {name!r}; known: {sorted(EXPORTERS)}")
    return EXPORTERS[name](run_dir, out_dir, cfg)


def export_sft_files(paths: list[str], out_dir: str, cfg: dict | None = None) -> dict:
    """Import already prepared SFT rows without requiring internal Task records."""
    cfg = dict(cfg or {})
    cfg["rl"] = {"enabled": False}
    cfg["sft"] = {**cfg.get("sft", {}), "enabled": True, "sources": [], "extra_files": paths}
    return export(None, out_dir, cfg)


def _sft_file(path: str) -> list:
    if path.endswith(".parquet"):
        return [drop_nulls(r) for r in read_parquet(path)]
    if path.endswith(".jsonl"):
        # An explicitly requested input must exist, unlike optional run outputs.
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if not path.endswith(".json"):
        raise ValueError(f"unsupported SFT input {path!r}; expected .json, .jsonl or .parquet")
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else [obj]


def _canonical_sft(row):
    if not isinstance(row, dict):
        return row
    # Keep invalid explicit values (including null) for the validator to report.
    return {**row, **sft_row(row.get("messages"), row.get("tools"), row.get("images"), row.get("videos"),
                            row.get("enable_thinking"))}


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@register_exporter("verl")
def export_verl(run_dir: str | None, out_dir: str, cfg: dict) -> dict:
    """Writes (file names are fixed; content follows export.sft / export.rl):
         out_dir/rl/train.parquet                tasks.jsonl -> RL rows
         out_dir/sft/<set>.parquet               text-only SFT rows (trajectories / reference / extra files)
         out_dir/sft/<set>.multimodal.parquet    rows with images / videos
         out_dir/sft/invalid.jsonl               SFT rows that failed validate_sft_row, with reasons
         out_dir/export_report.json              counts + what verl will see after reading back
    """
    systems = cfg.get("system_prompts", {})  # domain -> system prompt
    sources_map = cfg.get("data_source_map", {})
    json_cols = tuple(cfg.get("json_columns", []))
    sft_cfg, rl_cfg = cfg.get("sft", {}), cfg.get("rl", {})
    report = {}
    written = []  # (rows, path) for the read-back check
    tasks = [Task.from_dict(d) for d in _jsonl(os.path.join(run_dir, "tasks.jsonl"))] if run_dir else []

    if rl_cfg.get("enabled", True):
        from ..verifiers import VERIFIERS
        types = set(rl_cfg.get("verifier_types") or VERIFIERS)  # default: every registered verifier
        suffix = rl_cfg.get("prompt_suffix", {})                # verifier type -> appended text
        rl = [rl_row(t, i, rl_cfg.get("split", "train"), systems.get(t.domain), sources_map.get(t.source),
                     suffix.get(t.verifier.type), rl_cfg)
              for i, t in enumerate(x for x in tasks if x.verifier.type in types)]
        good, n_bad = [], 0
        os.makedirs(os.path.join(out_dir, "rl"), exist_ok=True)
        with open(os.path.join(out_dir, "rl", "invalid.jsonl"), "w", encoding="utf-8") as bad:
            for r in rl:
                errs = validate_rl_row(r, rl_cfg.get("keys"))
                if errs:
                    n_bad += 1
                    bad.write(json.dumps({"errors": errs, "row": r}, ensure_ascii=False) + "\n")
                    if rl_cfg.get("on_invalid", "drop") == "error":
                        raise ValueError(f"invalid RL row: {errs}")
                else:
                    good.append(r)
        rl = good
        report["rl/invalid.jsonl"] = n_bad
        if rl:
            p = os.path.join(out_dir, "rl", f"{rl_cfg.get('split', 'train')}.parquet")
            report[os.path.relpath(p, out_dir)] = write_parquet(rl, p, json_cols)
            written.append((rl, p))

    if sft_cfg.get("enabled", True):
        protected = {(sft_cfg.get("keys") or {}).get(k, k) for k in
                     ("messages", "tools", "images", "videos", "enable_thinking")}
        if protected.intersection(json_cols):
            raise ValueError("SFT columns must remain structured values, not json_columns strings")
        sets = {}
        wanted = sft_cfg.get("sources", ["trajectories"])
        if "trajectories" in wanted and run_dir:
            sets["trajectories"] = [sft_from_trajectory(tr, systems.get(tr.get("domain")))
                                    for tr in _jsonl(os.path.join(run_dir, "trajectories.jsonl"))]
        if "reference" in wanted:
            sets["reference"] = [sft_from_task_reference(t, systems.get(t.domain), sft_cfg.get("reference_strip"))
                                 for t in tasks]
        for name in sets:
            # These None entries mean a generated task has no usable SFT target.
            sets[name] = [row for row in sets[name] if row is not None]
        for path in sft_cfg.get("extra_files", []):  # SFT rows prepared elsewhere (canonical key names)
            name = os.path.splitext(os.path.basename(path))[0]
            sets.setdefault(name, []).extend(_canonical_sft(r) for r in _sft_file(path))
        os.makedirs(os.path.join(out_dir, "sft"), exist_ok=True)
        keys = sft_cfg.get("keys") or {}
        n_bad = 0
        with open(os.path.join(out_dir, "sft", "invalid.jsonl"), "w", encoding="utf-8") as bad:
            for name, rows in sets.items():
                groups: dict = {}
                for r in rows:
                    try:
                        # Validate before formatting so bad types cannot crash media
                        # resolution or be concealed by a configured boolean override.
                        errs = validate_sft_row(r)
                        if not errs:
                            r = apply_sft_format(r, sft_cfg)
                            errs = validate_sft_row(r, keys)
                    except OSError as e:  # missing media file
                        errs = [f"media: {e}"]
                    if errs:
                        n_bad += 1
                        bad.write(json.dumps({"set": name, "errors": errs, "row": r}, ensure_ascii=False,
                                             default=lambda b: f"<{len(b)} bytes>") + "\n")
                        if sft_cfg.get("on_invalid", "drop") == "error":
                            raise ValueError(f"invalid SFT row in {name}: {errs}")
                        continue
                    groups.setdefault(_modality(r, keys, sft_cfg), []).append(r)
                for mod, good in groups.items():
                    _fill_media_columns(good, keys)
                    # Different function schemas otherwise acquire null properties
                    # from each other through Arrow's nested struct unification.
                    tool_key = keys.get("tools", "tools")
                    signatures = {json.dumps(r.get(tool_key) or [], sort_keys=True) for r in good if r.get(tool_key)}
                    partitions = {}
                    for r in good:
                        suffix = ""
                        if len(signatures) > 1:
                            sig = json.dumps(r.get(tool_key) or [], sort_keys=True)
                            suffix = ".tools-" + hashlib.sha256(sig.encode()).hexdigest()[:12]
                        partitions.setdefault(suffix, []).append(r)
                    for suffix, rows in partitions.items():
                        p = os.path.join(out_dir, "sft", f"{name}{'.' + mod if mod else ''}{suffix}.parquet")
                        report[os.path.relpath(p, out_dir)] = write_parquet(rows, p, json_cols)
                        # Validate actual stored rows, including schema widening.
                        for stored in read_parquet(p):
                            errors = validate_sft_row(drop_nulls(stored), keys)
                            if errors:
                                os.remove(p)
                                raise ValueError(f"invalid SFT after Parquet write: {errors}")
                        written.append((rows, p))
        report["sft/invalid.jsonl"] = n_bad

    if cfg.get("check_roundtrip", True):
        rt = {os.path.relpath(p, out_dir): roundtrip_report(rows, p) for rows, p in written}
        with open(os.path.join(out_dir, "export_report.json"), "w", encoding="utf-8") as f:
            json.dump({"counts": report, "verl_read_back": rt}, f, ensure_ascii=False, indent=2)
        for path, r in rt.items():
            if r["null_keys_added"]:
                report[f"{path} null keys after read-back"] = ", ".join(r["null_keys_added"])
    return report


def _fill_media_columns(rows: list[dict], keys: dict):
    """In one file, a media column a row lacks is read back as null and verl
    fails on len(None); give such rows an empty list instead."""
    for k in ({"images": "images", "videos": "videos", **keys}[x] for x in ("images", "videos")):
        if any(k in r for r in rows):
            for r in rows:
                if r.get(k) is None:
                    r[k] = []


def _modality(row: dict, keys: dict, cfg: dict) -> str:
    """verl's SFT dataset requires a processor as soon as an images/videos
    column exists, and a text-only row in a file with that column fails, so by
    default text and multimodal rows go to separate files."""
    if not cfg.get("split_by_modality", True):
        return ""
    k = {"images": "images", "videos": "videos", **keys}
    return "multimodal" if (row.get(k["images"]) or row.get(k["videos"])) else ""


def validate_file(path: str, kind: str = "auto", keys: dict | None = None, *,
                  check_media_exists: bool = True, media_root: str | None = None) -> dict:
    """Check a .parquet / .jsonl / .json file of SFT or RL rows against the
    format (and, for Parquet, what verl reads back). kind: auto | sft | rl."""
    if path.endswith(".parquet"):
        rows = verl_view(path)
    elif path.endswith(".jsonl"):
        rows = _jsonl(path)
    else:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        rows = obj if isinstance(obj, list) else [obj]
    keys = keys or {}
    if kind == "auto":
        kind = "sft" if rows and isinstance(rows[0], dict) and keys.get("messages", "messages") in rows[0] else "rl"
    check = validate_sft_row if kind == "sft" else validate_rl_row
    bad = []
    for i, r in enumerate(rows):
        r = drop_nulls(r) if path.endswith(".parquet") else r
        if not isinstance(r, dict):
            errs = ["sample is not an object"]
        else:
            errs = check(r, keys, check_media_exists=check_media_exists, media_root=media_root) if kind == "sft" else check(r, keys)
        if errs:
            bad.append({"row": i, "errors": errs})
    out = {"file": path, "kind": kind, "rows": len(rows), "invalid": len(bad), "examples": bad[:20]}
    if path.endswith(".parquet"):
        cols = set(rows[0]) if rows else set()
        k = {"images": "images", "videos": "videos", **keys}
        media_cols = [c for c in (k["images"], k["videos"]) if c in cols]
        nulls = [i for i, r in enumerate(rows) for c in media_cols if r.get(c) is None]
        if nulls:
            out["media_column_nulls"] = f"{len(nulls)} rows have null {media_cols}; verl fails on these (len(None))"
        if media_cols:
            out["note"] = f"file has {media_cols} columns: verl will require a processor (multimodal model)"
    return out


__all__ = ["export", "export_sft_files", "validate_file", "export_verl", "register_exporter", "EXPORTERS", "read_parquet", "drop_nulls",
           "validate_sft_row", "verl_view", "roundtrip_report"]
