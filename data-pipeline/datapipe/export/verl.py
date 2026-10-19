"""verl Parquet: row builders, format checks, media handling, read / write.

Every choice that varies between projects is a config value (export.sft /
export.rl in datapipe/defaults.yaml); this module only reads them.

SFT row (verl MultiTurnSFTDataset; key names follow data.messages_key etc.):
    {"messages": [{"role": "system"|"user"|"assistant"|"tool", "content": str,
                   "tool_calls"?: [{"id", "type": "function",
                                    "function": {"name", "arguments": <JSON string>}}],
                   "tool_call_id"?: str}, ...],
     "tools"?: [OpenAI function schema], "images"?: [{"image": path} | {"bytes": b"..."}],
     "videos"?: [{"video": path}], "enable_thinking"?: bool}
  verl computes loss on assistant messages; the last message must be the
  assistant target. Each <image>/<video> placeholder maps, in order, to
  images/videos.

RL row (verl RLHFDataset, its standard layout):
    {"data_source": str, "prompt": [messages], "ability": str,
     "reward_model": {"style": "rule", "ground_truth": str},
     "extra_info": {...}, "images"?: [...]}
  `ground_truth` is the gold answer for answer_match tasks, otherwise a JSON
  verifier spec that datapipe.export.verl_reward.compute_score can grade.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import Iterable, Optional

from ..schema import Task

ROLES = {"system", "user", "assistant", "tool"}
SFT_KEYS = {"messages": "messages", "tools": "tools", "images": "images", "videos": "videos",
            "enable_thinking": "enable_thinking"}
RL_KEYS = {"prompt": "prompt", "data_source": "data_source", "ability": "ability",
           "reward_model": "reward_model", "extra_info": "extra_info", "images": "images"}


# ---------------------------------------------------------------- SFT rows
def sft_row(messages: list[dict], tools: Optional[list] = None, images: Optional[list] = None,
            videos: Optional[list] = None, enable_thinking: Optional[bool] = None) -> dict:
    """Canonical SFT row (internal key names; renamed by `apply_sft_format`)."""
    row = {"messages": [_clean_msg(m) for m in messages] if isinstance(messages, list) else messages}
    if tools is not None:
        row["tools"] = tools
    if images is not None:
        row["images"] = [_media(x, "image") for x in images] if isinstance(images, list) else images
    if videos is not None:
        row["videos"] = [_media(x, "video") for x in videos] if isinstance(videos, list) else videos
    if enable_thinking is not None:
        row["enable_thinking"] = enable_thinking
    return row


def _media(x, kind: str) -> dict:
    return {kind: x} if isinstance(x, str) else x


def _clean_msg(m: dict) -> dict:
    if not isinstance(m, dict):
        return m  # Keep malformed values so validation can report them.
    out = dict(m)
    if m.get("role") == "assistant" and m.get("tool_calls") and m.get("content") is None:
        out["content"] = ""
    if isinstance(m.get("tool_calls"), list):
        calls = []
        for c in m["tool_calls"]:
            if not isinstance(c, dict) or not isinstance(c.get("function"), dict):
                calls.append(c)
                continue
            fn = dict(c["function"])
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
            calls.append({**c, "function": fn})
        out["tool_calls"] = calls
    return out


def validate_sft_row(row: dict, keys: Optional[dict] = None, *,
                     check_media_exists: bool = False, media_root: Optional[str] = None) -> list[str]:
    """Checks from the data spec: roles, last message is the assistant target,
    tool_call ids pair with tool_call_id, placeholder counts match media,
    tool calls name declared tools, media entries have exactly one source."""
    from jsonschema import Draft202012Validator
    from jsonschema.validators import validator_for
    from referencing import Registry

    if not isinstance(row, dict):
        return ["sample is not an object"]
    k = {**SFT_KEYS, **(keys or {})}
    errs = []
    msgs = row.get(k["messages"])
    if not isinstance(msgs, list) or not msgs:
        return ["messages is not a non-empty list"]
    if k["enable_thinking"] in row and type(row[k["enable_thinking"]]) is not bool:
        errs.append("enable_thinking must be true or false (boolean)")

    validators = {}
    tools = row.get(k["tools"], [])
    if not isinstance(tools, list):
        errs.append("tools is not a list")
        tools = []
    for j, tool in enumerate(tools):
        where = f"tools[{j}]"
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            errs.append(f"{where}: expected an OpenAI function tool")
            continue
        fn = tool["function"]
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            errs.append(f"{where}: missing function name")
            continue
        if name in validators:
            errs.append(f"{where}: duplicate tool name {name!r}")
        if "description" in fn and not isinstance(fn["description"], str):
            errs.append(f"{where}: description is not a string")
        validators[name] = None
        schema = fn.get("parameters", {"type": "object"})
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            errs.append(f"{where}: parameters must be an object schema")
            continue
        try:
            cls = validator_for(schema, default=Draft202012Validator)
            cls.check_schema(schema)
            # Local $refs work; external references never trigger network access.
            validators[name] = cls(schema, registry=Registry())
        except Exception as e:
            errs.append(f"{where}: invalid parameter schema ({type(e).__name__})")

    pending: set = set()
    seen_ids: set = set()
    final_ids: set = set()
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            errs.append(f"messages[{i}]: message is not an object")
            continue
        r = m.get("role")
        if not isinstance(r, str) or r not in ROLES:
            errs.append(f"messages[{i}]: bad role {r!r}")
        if r == "system" and i != 0:
            errs.append(f"messages[{i}]: system message not first")
        if not isinstance(m.get("content"), str):
            errs.append(f"messages[{i}]: content is not a string")
        if r in ("assistant", "user") and pending:
            errs.append(f"messages[{i}]: tool_calls never answered before the next {r} message: {sorted(pending)}")
        calls = m.get("tool_calls", [])
        if not isinstance(calls, list):
            errs.append(f"messages[{i}]: tool_calls is not a list")
            calls = []
        for c in calls:
            if r != "assistant":
                errs.append(f"messages[{i}]: tool_calls on a {r} message")
            if not isinstance(c, dict):
                errs.append(f"messages[{i}]: tool_call is not an object")
                continue
            cid = c.get("id")
            if not isinstance(cid, str) or not cid.strip():
                errs.append(f"messages[{i}]: tool_call id must be a non-empty string")
            else:
                if cid in seen_ids:
                    errs.append(f"messages[{i}]: duplicate tool_call id {cid}")
                seen_ids.add(cid)
                pending.add(cid)
                if i == len(msgs) - 1 and r == "assistant":
                    final_ids.add(cid)
            if c.get("type") != "function" or not isinstance(c.get("function"), dict):
                errs.append(f"messages[{i}]: expected a function tool_call")
                continue
            fn = c["function"]
            name = fn.get("name")
            if not isinstance(name, str) or name not in validators:
                errs.append(f"messages[{i}]: tool_call to undeclared tool {name!r}")
            try:
                if not isinstance(fn.get("arguments"), str):
                    raise TypeError("arguments is not a string")
                arguments = json.loads(fn["arguments"], parse_constant=_reject_json_constant)
            except (ValueError, TypeError, KeyError):
                errs.append(f"messages[{i}]: tool_call {c.get('id')} arguments is not a JSON string")
                continue
            if not isinstance(arguments, dict):
                errs.append(f"messages[{i}]: tool_call arguments must decode to an object")
                continue
            if isinstance(name, str) and validators.get(name) is not None:
                try:
                    for error in validators[name].iter_errors(arguments):
                        errs.append(f"messages[{i}]: {name} arguments: {error.message}")
                except Exception as e:
                    errs.append(f"messages[{i}]: cannot validate {name} arguments ({type(e).__name__})")
            if name == "computer_use" and "coordinate" in arguments:
                xy = arguments["coordinate"]
                if (not isinstance(xy, list) or len(xy) != 2 or
                    any(type(v) not in (int, float) or not 0 <= v <= 999 or not math.isfinite(v) for v in xy)):
                    errs.append(f"messages[{i}]: computer_use coordinate must contain two numbers in 0-999")
        if r == "tool":
            tid = m.get("tool_call_id")
            if not isinstance(tid, str) or tid not in pending:
                errs.append(f"messages[{i}]: tool_call_id {tid!r} has no earlier matching tool_call")
            else:
                pending.remove(tid)
        elif "tool_call_id" in m:
            errs.append(f"messages[{i}]: tool_call_id on a non-tool message")
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        errs.append("last message is not assistant (no supervision target)")
    elif not last.get("content") and not last.get("tool_calls"):
        errs.append("last assistant has neither content nor tool_calls (empty supervision target)")
    unanswered = pending - final_ids
    if unanswered:
        errs.append(f"tool_calls never answered by a tool message: {sorted(unanswered)}")
    text = "".join(m.get("content", "") for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), str))
    for tag, key, kind in (("<image>", k["images"], "image"), ("<video>", k["videos"], "video")):
        media = row.get(key, [])
        if not isinstance(media, list):
            errs.append(f"{key} is not a list")
            continue
        if text.count(tag) != len(media):
            errs.append(f"{text.count(tag)} {tag} placeholders but {len(media)} {key}")
        for j, x in enumerate(media):
            if not isinstance(x, dict) or ((kind in x) == ("bytes" in x)):
                errs.append(f"{key}[{j}]: needs exactly one of {kind!r} or 'bytes'")
            elif "bytes" in x:
                if not isinstance(x["bytes"], bytes) or not x["bytes"]:
                    errs.append(f"{key}[{j}]: bytes must be non-empty binary data")
            elif not isinstance(x[kind], str) or not x[kind].strip():
                errs.append(f"{key}[{j}]: {kind} must be a non-empty path or URL")
            elif check_media_exists:
                path = x[kind]
                if "://" not in path and not os.path.isfile(os.path.join(media_root or "", path)):
                    errs.append(f"{key}[{j}]: media file not found: {path}")
    return errs


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def sft_from_task_reference(task: Task, system: Optional[str] = None,
                            strip: Optional[str] = None) -> Optional[dict]:
    """A one-turn SFT sample from the task's own worked solution (GSM8K's
    solution text, LeetCode's reference code). Answer-only references are skipped."""
    ref = task.reference
    if ref is None or ref.meta.get("answer_only"):
        return None
    if ref.text:
        answer = ref.text
    elif ref.code:
        answer = f"```python\n{ref.code.strip()}\n```"
    else:
        return None
    if strip:
        answer = re.sub(strip, "", answer)
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": task.prompt}, {"role": "assistant", "content": answer}]
    return sft_row(msgs, images=task.meta.get("images"))


def sft_from_trajectory(tr: dict, system: Optional[str] = None) -> Optional[dict]:
    """trajectories.jsonl record -> SFT row. The harness stores OpenAI-format
    messages (tool_calls / role=tool) in `messages`, media paths in extra.images."""
    msgs = list(tr.get("messages") or [])
    if not msgs:
        return None
    if system and msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": system}] + msgs
    ex = tr.get("extra") or {}
    level = tr.get("reasoning_level")
    return sft_row(msgs, tools=tr.get("tools") or ex.get("tools"), images=ex.get("images"),
                   videos=ex.get("videos"),
                   enable_thinking=None if level is None else level not in ("off", "none"))


def apply_sft_format(row: dict, fmt: dict) -> dict:
    """Apply export.sft settings to a canonical row: enable_thinking policy,
    media as paths or embedded bytes, optional empty media columns, key names."""
    row = dict(row)
    et = fmt.get("enable_thinking", "keep")
    if type(et) is bool:
        row["enable_thinking"] = et
    elif et == "omit":
        row.pop("enable_thinking", None)
    elif et != "keep":
        raise ValueError("export.sft.enable_thinking must be keep, omit, true or false")
    has_media = bool(row.get("images") or row.get("videos"))
    for key, kind in (("images", "image"), ("videos", "video")):
        if key in row:
            row[key] = [_resolve_media(x, kind, fmt) for x in row[key]]
        if has_media and fmt.get("empty_media_columns", False):
            row.setdefault(key, [])
        if not row.get(key) and not has_media:
            row.pop(key, None)  # text-only rows carry no media columns (see split_by_modality)
    names = {**SFT_KEYS, **(fmt.get("keys") or {})}
    return {names.get(k, k): v for k, v in row.items()}


def _resolve_media(x: dict, kind: str, fmt: dict) -> dict:
    root = fmt.get("media_root")
    path = x.get(kind)
    if isinstance(path, str) and root and not os.path.isabs(path) and "://" not in path:
        full = os.path.join(root, path)
    else:
        full = path
    mode = fmt.get("media", "path")
    if mode == "bytes" and kind == "image" and isinstance(full, str):
        with open(full, "rb") as f:  # verl: {"bytes": ...} and {"image": ...} are mutually exclusive
            return {"bytes": f.read()}
    if fmt.get("check_media_exists", True) and isinstance(full, str) and "://" not in full and not os.path.isfile(full):
        raise FileNotFoundError(full)
    if mode == "absolute_path" and isinstance(full, str):
        return {**x, kind: os.path.abspath(full)}
    return x


# ---------------------------------------------------------------- RL rows
def ground_truth(task: Task) -> str:
    v = task.verifier
    if v.type == "answer_match":
        return str(v.answer)
    spec = {"type": v.type, "function": v.function, "tests": v.tests, "time_limit_sec": v.time_limit_sec,
            "test_setup_code": task.meta.get("test_setup_code"), "code_prelude": task.meta.get("code_prelude")}
    if v.type == "command":
        spec.update(image=task.env.image, test_cmd=task.env.test_cmd, repo=task.repo)
    return json.dumps(spec, ensure_ascii=False)


def rl_row(task: Task, index: int, split: str = "train", system: Optional[str] = None,
           data_source: Optional[str] = None, prompt_suffix: Optional[str] = None,
           fmt: Optional[dict] = None) -> dict:
    fmt = fmt or {}
    # the suffix may name the function the tests call: "... named `{function}`"
    content = task.prompt + (prompt_suffix or "").replace("{function}", task.verifier.function or "solve")
    prompt = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    info_all = {"index": index, "split": split, "task_id": task.id, "source": task.source,
                "verifier": task.verifier.type, "verifier_version": task.verifier.version,
                "difficulty": json.dumps(task.difficulty, ensure_ascii=False) if task.difficulty else None}
    fields = fmt.get("extra_info_fields") or list(info_all)
    row = {
        "data_source": data_source or task.source,
        "prompt": prompt,
        "ability": (fmt.get("ability_map") or {}).get(task.domain, task.domain),
        "reward_model": {"style": fmt.get("reward_style", "rule"), "ground_truth": ground_truth(task)},
        "extra_info": {f: info_all.get(f) for f in fields},
    }
    if task.meta.get("images"):
        row["images"] = [_media(x, "image") for x in task.meta["images"]]
    names = {**RL_KEYS, **(fmt.get("keys") or {})}
    return {names.get(k, k): v for k, v in row.items()}


def validate_rl_row(row: dict, keys: Optional[dict] = None) -> list[str]:
    k = {**RL_KEYS, **(keys or {})}
    errs = []
    prompt = row.get(k["prompt"])
    if not isinstance(prompt, list) or not prompt:
        errs.append("prompt is not a non-empty list of messages")
    else:
        for i, m in enumerate(prompt):
            if m.get("role") not in ROLES or not isinstance(m.get("content"), str):
                errs.append(f"prompt[{i}]: needs role and string content")
        if prompt[-1].get("role") != "user":
            errs.append("last prompt message is not user")
    if not isinstance(row.get(k["data_source"]), str):
        errs.append("data_source is not a string")
    rm = row.get(k["reward_model"]) or {}
    if not isinstance(rm.get("ground_truth"), str):
        errs.append("reward_model.ground_truth is not a string")
    text = "".join(m.get("content") or "" for m in prompt or [])
    n_img = len(row.get(k["images"]) or [])
    if text.count("<image>") != n_img:
        errs.append(f"{text.count('<image>')} <image> placeholders but {n_img} images")
    return errs


# ---------------------------------------------------------------- write / read
def write_parquet(rows: Iterable[dict], path: str, json_columns: tuple = ()) -> int:
    """json_columns: columns stored as JSON strings instead of nested structs.
    Note verl does not parse JSON-string columns; use this only for columns
    your own loader decodes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = list(rows)
    if json_columns:
        rows = [{k: (json.dumps(v, ensure_ascii=False) if k in json_columns and v is not None else v)
                 for k, v in r.items()} for r in rows]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # from_pylist infers top-level columns from the FIRST row only. Build
    # columns from the entire batch so later tools/enable_thinking/etc survive.
    columns = dict.fromkeys(k for row in rows for k in row)
    table = pa.table({k: [row.get(k) for row in rows] for k in columns})
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, temporary)
        restored = pq.read_table(temporary).to_pylist()
        if len(restored) != len(rows) or any(drop_nulls(a) != drop_nulls(b) for a, b in zip(rows, restored)):
            raise ValueError("Parquet round-trip changed sample data; output was not published")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return len(rows)


def read_parquet(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def drop_nulls(obj):
    """Undo Parquet's struct widening: remove keys whose value is None."""
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [drop_nulls(x) for x in obj]
    return obj


def verl_view(path: str, limit: Optional[int] = None) -> list[dict]:
    """Reconstruct nested row values with PyArrow, preserving added null keys.
    This checks storage layout; it does not execute verl's tokenizer/processor.
    """
    import pyarrow.parquet as pq
    import numpy as np

    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        return x
    t = pq.read_table(path)
    if limit:
        t = t.slice(0, limit)
    return [conv(r) for r in t.to_pylist()]


def roundtrip_report(rows: list[dict], path: str, limit: int = 2000) -> dict:
    """Compare written rows with what verl will load. Parquet stores a list of
    dicts as one struct type, so a key present in any message appears in all
    (as null). Reports those added keys by path, and anything else that changed."""
    added, other = {}, 0
    for a, b in zip(rows[:limit], verl_view(path, limit)):
        _diff(a, b, "", added)
        if drop_nulls(b) != drop_nulls(a):
            other += 1
    return {"rows_checked": min(limit, len(rows)), "null_keys_added": dict(sorted(added.items())),
            "rows_changed_beyond_nulls": other}


def _diff(a, b, path, added):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in b:
            p = f"{path}.{k}" if path else k
            if k not in a:
                if b[k] is None:
                    added[p] = added.get(p, 0) + 1
            else:
                _diff(a[k], b[k], p, added)
    elif isinstance(a, list) and isinstance(b, list):
        for x, y in zip(a, b):
            _diff(x, y, path + "[]", added)
