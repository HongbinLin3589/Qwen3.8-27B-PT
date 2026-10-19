"""Step 2: dedup, decontamination, time split, repo split."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Optional

from ..schema import Task
from ..textutil import MinHasher, exact_key, lsh_buckets, ngrams, tokens, word_tokens
from .base import Stage, register_stage


def task_text(t: Task, ignore: tuple = ()) -> str:
    return strip_patterns(t.prompt, ignore)


def strip_patterns(text: str, ignore: tuple = ()) -> str:
    """Remove boilerplate shared by many prompts (config `ignore_patterns`, a
    list of regexes) so it cannot create false n-gram / embedding matches."""
    for p in ignore:
        text = re.sub(p, " ", text)
    return text


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)  # smaller index (= preferred) is the root


def _cosine_pairs(vecs: list[list[float]], thr: float, block: int = 2048):
    """All pairs i<j with cosine >= thr. numpy, blockwise. For millions of
    items swap this for a FAISS / ANN index."""
    import numpy as np
    if not vecs:
        return []
    X = np.asarray(vecs, dtype=np.float32)
    out = []
    for s in range(0, len(X), block):
        S = X[s:s + block] @ X.T
        ii, jj = np.nonzero(S >= thr)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if s + i < j:
                out.append((s + i, j, float(S[i, j])))
    return out


def _answers_differ(a: Task, b: Task) -> bool:
    """True if both tasks have tests on a shared input with different expected
    outputs. Text near-dup alone over-merges templated math problems (2xn vs
    3xn tilings, sum of floor(n/k) vs its square), which differ in one token.
    answer_match tasks differ when their normalized gold answers differ."""
    if a.verifier.type == b.verifier.type == "answer_match":
        from ..verifiers import normalize_answer
        return normalize_answer(str(a.verifier.answer)) != normalize_answer(str(b.verifier.answer))
    if a.verifier.type != b.verifier.type or a.verifier.type not in ("python_function", "stdio"):
        return False
    ta = {json.dumps(t["input"]): t["output"] for t in a.verifier.tests}
    tb = {json.dumps(t["input"]): t["output"] for t in b.verifier.tests}
    return any(ta[k] != tb[k] for k in ta.keys() & tb.keys())


@register_stage("dedup")
class Dedup(Stage):
    """Exact dedup, then MinHash n-gram near-dup, then embedding near-dup.
    Within a duplicate cluster the task from the highest-priority source wins
    (config `prefer_sources`), ties go to input order."""
    name = "dedup"

    def run(self, tasks):
        prefer = {s: i for i, s in enumerate(self.cfg.get("prefer_sources", []))}
        ign = tuple(self.cfg.get("ignore_patterns", []))
        tasks = sorted(tasks, key=lambda t: prefer.get(t.source, len(prefer)))  # stable
        # 1. exact
        seen, kept = {}, []
        for t in tasks:
            k = exact_key(task_text(t, ign))
            if k in seen:
                self.ctx.drop(t, self.name, "exact_dup", dup_of=seen[k])
            else:
                seen[k] = t.id
                kept.append(t)
        # 2. near-dup
        uf = _UF(len(kept))
        evidence = {}
        guard = self.cfg.get("answer_guard", True)

        def merge(i, j, ev):
            if guard and _answers_differ(kept[i], kept[j]):
                self.ctx.count(self.name, "near_dup_kept_by_answer_guard")
                return
            uf.union(i, j)
            evidence.setdefault((i, j), ev)

        mh_cfg = self.cfg.get("minhash", {})
        if mh_cfg.get("enabled", True):
            mh = MinHasher(mh_cfg.get("num_perm", 128), mh_cfg.get("ngram", 5), mh_cfg.get("seed", 1))
            sigs = [mh.signature(task_text(t, ign)) for t in kept]
            buckets = defaultdict(list)
            for i, s in enumerate(sigs):
                for b in lsh_buckets(s, mh_cfg.get("bands", 32)):
                    buckets[b].append(i)
            thr = mh_cfg.get("threshold", 0.7)
            checked = set()
            for idx in buckets.values():
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        i, j = idx[a], idx[b]
                        if (i, j) in checked:
                            continue
                        checked.add((i, j))
                        jac = MinHasher.jaccard(sigs[i], sigs[j])
                        if jac >= thr:
                            merge(i, j, ("minhash", round(jac, 3)))
        emb_cfg = self.cfg.get("embedding", {})
        embedder = self.b("embedder", required=False)
        if emb_cfg.get("enabled", True) and embedder is not None:
            vecs = embedder.embed([task_text(t, ign) for t in kept])
            for i, j, c in _cosine_pairs(vecs, emb_cfg.get("threshold", 0.92)):
                merge(i, j, ("embedding", round(c, 3)))
        out = []
        for i, t in enumerate(kept):
            r = uf.find(i)
            if r == i:
                out.append(t)
            else:
                ev = evidence.get((min(i, r), max(i, r)))
                self.ctx.drop(t, self.name, "near_dup", dup_of=kept[r].id, evidence=ev)
        self.ctx.count(self.name, "kept", len(out))
        return out


def _load_eval(paths: list[str]) -> list[dict]:
    import glob
    items = []
    expanded = [m for p in paths for m in (sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])]
    for p in expanded:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    d.setdefault("_file", os.path.basename(p))
                    items.append(d)
    return items


def _eval_text(d: dict) -> str:
    for k in ("prompt", "problem", "question", "statement", "text", "problem_statement"):
        if d.get(k):
            return d[k]
    return ""


def _parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


@register_stage("decontam_split")
class DecontamSplit(Stage):
    """Remove train tasks that overlap any eval set (n-gram + embedding), then
    hold out tasks dated on/after the cutoff and tasks from eval repos."""
    name = "decontam_split"

    def run(self, tasks):
        evals = _load_eval(self.cfg.get("eval_sets", []))
        ign = tuple(self.cfg.get("ignore_patterns", []))
        ng = self.cfg.get("ngram", {})
        n, min_shared = ng.get("n", 13), ng.get("min_shared", 1)
        # word_tokens: n-grams over words only. With punctuation and LaTeX symbols
        # counted as tokens, a 13-gram can be just "$ a $ , $ b $ , and $ c $ ,".
        tok = word_tokens if ng.get("word_tokens", False) else tokens
        eval_grams = [ngrams(tok(strip_patterns(_eval_text(e), ign)), n) for e in evals]
        index = defaultdict(set)
        for k, gs in enumerate(eval_grams):
            for g in gs:
                index[g].add(k)
        # Template n-grams ("time limit: {'seconds': 2 ...", "the first line contains an
        # integer n") recur across many unrelated problems. An n-gram found in more
        # than max_eval_df eval items, or more than max_train_df train tasks, is
        # treated as boilerplate and ignored. None = keep every n-gram.
        max_eval_df, max_train_df = ng.get("max_eval_df"), ng.get("max_train_df")
        if max_eval_df:
            index = defaultdict(set, {g: ks for g, ks in index.items() if len(ks) <= max_eval_df})
        task_sizes, task_grams = [], []
        for t in tasks:
            gs = ngrams(tok(task_text(t, ign)), n)
            task_sizes.append(len(gs))
            task_grams.append([g for g in gs if g in index])
        if max_train_df:
            df = defaultdict(int)
            for gs in task_grams:
                for g in gs:
                    df[g] += 1
            common = {g for g, c in df.items() if c > max_train_df}
            task_grams = [[g for g in gs if g not in common] for gs in task_grams]
            self.ctx.count(self.name, "boilerplate_ngrams_ignored", len(common))
        emb_cfg = self.cfg.get("embedding", {})
        embedder = self.b("embedder", required=False)
        use_emb = emb_cfg.get("enabled", True) and embedder is not None and evals
        if use_emb:
            import numpy as np
            E = np.asarray(embedder.embed([strip_patterns(_eval_text(e), ign) for e in evals]), dtype=np.float32)
            T = np.asarray(embedder.embed([task_text(t, ign) for t in tasks]), dtype=np.float32) if tasks else None
        clean = []
        for ti, t in enumerate(tasks):
            hits = defaultdict(int)
            for g in task_grams[ti]:
                for k in index.get(g, ()):
                    hits[k] += 1
            best = max(hits.items(), key=lambda kv: kv[1], default=None)
            # min_overlap: shared n-grams / n-grams of the shorter of the two texts.
            # Unrelated problems written in the same house style share a few
            # n-grams; real copies share a large fraction.
            overlap = best[1] / max(1, min(len(eval_grams[best[0]]), task_sizes[ti])) if best else 0.0
            if best and best[1] >= min_shared and overlap >= ng.get("min_overlap", 0.0):
                e = evals[best[0]]
                example = next(g for g in task_grams[ti] if best[0] in index[g])
                self.ctx.drop(t, self.name, "contaminated_ngram", eval_file=e["_file"],
                              eval_id=e.get("id"), shared_ngrams=best[1], overlap=round(overlap, 3),
                              example_ngram=" ".join(example))
                continue
            if use_emb:
                sims = E @ T[ti]
                k = int(sims.argmax())
                if sims[k] >= emb_cfg.get("threshold", 0.9):
                    self.ctx.drop(t, self.name, "contaminated_embedding", eval_file=evals[k]["_file"],
                                  eval_id=evals[k].get("id"), cosine=round(float(sims[k]), 3))
                    continue
            clean.append(t)
        self.ctx.count(self.name, "eval_items_loaded", len(evals))

        # time split
        ts = self.cfg.get("time_split", {})
        cutoff = _parse_date(ts.get("cutoff")) if ts.get("enabled", True) else None
        missing = ts.get("missing_date", "keep")  # keep | drop | holdout
        # repo split
        rs = self.cfg.get("repo_split", {})
        eval_repos = set(rs.get("eval_repos", [])) | {e["repo"] for e in evals if e.get("repo")}
        frac = rs.get("holdout_fraction", 0.0)
        domains = set(rs.get("domains", ["swe"]))

        def repo_held(repo: str) -> bool:
            if repo in eval_repos:
                return True
            return frac > 0 and int(hashlib.sha256(repo.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac

        out = []
        for t in clean:
            if cutoff:
                d = _parse_date(t.created_at)
                if d is None and missing != "keep":
                    if missing == "drop":
                        self.ctx.drop(t, self.name, "missing_date")
                    else:
                        self.ctx.divert(t, self.name, "eval_holdout.jsonl", "missing_date")
                    continue
                if d is not None and d >= cutoff:
                    self.ctx.divert(t, self.name, "eval_holdout.jsonl", "after_cutoff", created_at=t.created_at)
                    continue
            if rs.get("enabled", True) and t.domain in domains and t.repo and repo_held(t.repo):
                self.ctx.divert(t, self.name, "eval_holdout.jsonl", "eval_repo", repo=t.repo)
                continue
            out.append(t)
        self.ctx.count(self.name, "kept", len(out))
        return out
