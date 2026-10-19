"""Backends for a model served behind an OpenAI-compatible API (vLLM / SGLang).

OpenAICompatPolicy   single-turn sampling for function tasks (math x code).
                     Repo tasks (SWE / terminal) need a multi-turn agent
                     harness; plug yours in by subclassing `Policy`.
VLLMLogprobScorer    per-token log-probs of a given token sequence under a
                     model (teacher log-probs on student rollouts, or student
                     log-probs on teacher rollouts). Needs vLLM's /tokenize
                     endpoint and `prompt_logprobs`, and the two models must
                     share a tokenizer (Qwen family to Qwen family).

Not exercised against a live server in this repo; the request shapes follow
vLLM's OpenAI server.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Optional

from ..schema import Rollout, Solution, Task
from .base import LogprobScorer, Policy, Sandbox

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _post(url: str, body: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build_messages(task: Task, system_prompt: Optional[str] = None) -> list[dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    user = task.prompt
    if task.verifier.type in ("python_function", "python_asserts"):
        user += f"\n\n请用 Python 实现函数 `{task.verifier.function or 'solve'}`，最后给出一个完整的 ```python 代码块。"
    elif task.verifier.type == "answer_match":
        user += "\n\n请把最终答案写在 \\boxed{} 里。"
    msgs.append({"role": "user", "content": user})
    return msgs


class OpenAICompatPolicy(Policy):
    harness = "single_turn_chat"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", system_prompt: Optional[str] = None,
                 reasoning_level: Optional[str] = None, extra_body: Optional[dict] = None, timeout: float = 1800):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.system_prompt, self.reasoning_level = system_prompt, reasoning_level
        self.extra_body, self.timeout = extra_body or {}, timeout

    def generate(self, task: Task, n: int, sampling: dict, sandbox: Optional[Sandbox] = None) -> list[Rollout]:
        if task.verifier.type not in ("python_function", "python_asserts", "answer_match"):
            raise NotImplementedError("repo tasks need an agent harness; subclass Policy")
        msgs = build_messages(task, self.system_prompt)
        body = {"model": self.model, "messages": msgs, "n": n, "logprobs": True,
                "return_tokens_as_token_ids": True, **sampling, **self.extra_body}
        resp = _post(f"{self.base_url}/chat/completions", body, self.api_key, self.timeout)
        out = []
        for ch in resp["choices"]:
            text = ch["message"]["content"] or ""
            lp = (ch.get("logprobs") or {}).get("content") or []
            toks = [int(x["token"].split(":", 1)[1]) if str(x["token"]).startswith("token_id:") else x["token"] for x in lp]
            codes = _CODE.findall(text)
            out.append(Rollout(
                text=text, solution=Solution(code=codes[-1] if codes else None, text=text),
                tokens=toks, logprobs=[x["logprob"] for x in lp],
                messages=msgs + [{"role": "assistant", "content": text}],
                truncated=ch.get("finish_reason") == "length", finish_reason=ch.get("finish_reason"),
                model=self.model, harness=self.harness, sampling=dict(sampling, n=n),
                reasoning_level=self.reasoning_level))
        return out


class VLLMLogprobScorer(LogprobScorer):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", system_prompt: Optional[str] = None,
                 timeout: float = 600):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.system_prompt, self.timeout = system_prompt, timeout
        self._root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def score(self, task: Task, rollout: Rollout) -> Optional[list[float]]:
        if not rollout.tokens or not all(isinstance(t, int) for t in rollout.tokens):
            return None  # need token ids from a shared tokenizer
        prompt_msgs = [m for m in rollout.messages if m["role"] != "assistant"] or build_messages(task, self.system_prompt)
        tok = _post(f"{self._root}/tokenize", {"model": self.model, "messages": prompt_msgs,
                                               "add_generation_prompt": True}, self.api_key, self.timeout)
        prompt_ids = tok["tokens"]
        ids = prompt_ids + list(rollout.tokens)
        resp = _post(f"{self.base_url}/completions", {"model": self.model, "prompt": ids, "max_tokens": 1,
                                                      "prompt_logprobs": 0}, self.api_key, self.timeout)
        plp = resp["choices"][0]["prompt_logprobs"]
        out = []
        for tid, entry in zip(rollout.tokens, plp[len(prompt_ids):]):
            e = (entry or {}).get(str(tid))
            out.append(e["logprob"] if e else float("nan"))
        return out
