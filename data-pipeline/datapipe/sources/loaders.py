"""Loaders: where raw rows come from. Each loader yields plain dicts.

  hf     a Hugging Face dataset (datasets library -> huggingface_hub files ->
         datasets-server rows API, whichever is available). Honors HF_ENDPOINT
         (mirrors) and HF_TOKEN.
  url    a .jsonl / .json / .parquet file (optionally .gz) at an http(s) URL
  file   the same formats from a local path
  git    files inside a git repository (any host `git clone` can reach), at a
         branch / tag / commit; `path` may be a glob. Only the matched files
         are fetched (partial clone).

Add a loader with @register_loader("name"); it receives the loader config
(minus `type`) plus `cache_dir`, and must yield dicts.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import os
import subprocess
import urllib.parse
import urllib.request
from typing import Iterator, Optional

LOADERS: dict = {}


def register_loader(name: str):
    def deco(fn):
        LOADERS[name] = fn
        return fn
    return deco


def load_rows(spec: dict, cache_dir: str) -> Iterator[dict]:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in LOADERS:
        raise ValueError(f"unknown loader {kind!r}; known: {sorted(LOADERS)}")
    limit = spec.pop("limit", None)
    for i, row in enumerate(LOADERS[kind](cache_dir=cache_dir, **spec)):
        if limit is not None and i >= limit:
            break
        yield row


# ---------------------------------------------------------------- file formats
def _read_bytes(data: bytes, name: str) -> Iterator[dict]:
    if name.endswith(".gz"):
        data, name = gzip.decompress(data), name[:-3]
    if name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("reading parquet needs `pip install pyarrow`") from e
        yield from pq.read_table(io.BytesIO(data)).to_pylist()
    elif name.endswith(".jsonl"):
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
    elif name.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict):  # {"data": [...]} or {"train": [...]}
            obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
        yield from obj
    else:
        raise ValueError(f"don't know how to read {name} (want .jsonl/.json/.parquet[.gz])")


@register_loader("file")
def load_file(path: str, cache_dir: str = "", **_) -> Iterator[dict]:
    with open(path, "rb") as f:
        yield from _read_bytes(f.read(), path)


def _fetch(url: str, cache_dir: str, headers: Optional[dict] = None) -> bytes:
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, hashlib.sha256(url.encode()).hexdigest()[:16] + "-" +
                     os.path.basename(urllib.parse.urlparse(url).path))
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=300) as r, open(p + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(p + ".tmp", p)
    with open(p, "rb") as f:
        return f.read()


@register_loader("url")
def load_url(url: str, cache_dir: str, format: Optional[str] = None, **_) -> Iterator[dict]:
    name = format or urllib.parse.urlparse(url).path
    yield from _read_bytes(_fetch(url, cache_dir), name)


# ---------------------------------------------------------------- Hugging Face
@register_loader("hf")
def load_hf(repo: str, cache_dir: str, split: str = "train", config: Optional[str] = None,
            revision: Optional[str] = None, backend: str = "auto", token: Optional[str] = None,
            data_files: Optional[str] = None, **_) -> Iterator[dict]:
    """backend: auto | datasets | hub | rows_api."""
    token = token or os.environ.get("HF_TOKEN")
    order = ["datasets", "hub", "rows_api"] if backend == "auto" else [backend]
    errors = []
    gens = {
        "datasets": lambda: _hf_datasets(repo, split, config, revision, token, cache_dir, data_files),
        "hub": lambda: _hf_hub(repo, split, config, revision, token, cache_dir, data_files),
        "rows_api": lambda: _hf_rows_api(repo, split, config, token),
    }
    for b in order:
        n = 0
        try:
            for row in gens[b]():
                n += 1
                yield row
            return
        except ImportError as e:
            errors.append(f"{b}: {e}")
        except Exception as e:  # network / layout problem: try the next backend
            if n:  # already yielded rows: retrying elsewhere would duplicate them
                raise
            errors.append(f"{b}: {type(e).__name__}: {str(e)[:300]}")
            if backend != "auto":
                break
    raise RuntimeError(f"could not load {repo} ({config}/{split}):\n  " + "\n  ".join(errors))


def _hf_datasets(repo, split, config, revision, token, cache_dir, data_files):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split=split, revision=revision, token=token,
                      cache_dir=os.path.join(cache_dir, "hf_datasets"), data_files=data_files)
    for row in ds:
        yield dict(row)


def _hf_hub(repo, split, config, revision, token, cache_dir, data_files):
    """Download the dataset's own data files (or the auto-converted parquet
    branch) and read them, without the `datasets` library."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    tried = []
    for rev in ([revision] if revision else [None, "refs/convert/parquet"]):
        files = api.list_repo_files(repo, repo_type="dataset", revision=rev)
        if data_files:
            pats = [data_files] if isinstance(data_files, str) else list(data_files)
        else:
            c = config or "*"
            pats = [f"{c}/{split}-*", f"{c}/{split}/*", f"data/{split}-*", f"data/{split}/*", f"{split}-*",
                    f"{split}/*", f"*/{split}-*", f"*{split}*.jsonl*", f"*{split}*.json*", f"*{split}*.parquet"]
        exts = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz")
        for pat in pats:
            hits = sorted(f for f in files if fnmatch.fnmatch(f, pat) and f.endswith(exts))
            if hits:
                for f in hits:
                    p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, token=token,
                                        cache_dir=os.path.join(cache_dir, "hf_hub"))
                    yield from load_file(p)
                return
        tried.append(f"rev={rev or 'main'}: {len(files)} files, none matched {pats[:3]}...")
    raise FileNotFoundError("; ".join(tried))


def _hf_rows_api(repo, split, config, token, page: int = 100):
    """datasets-server JSON API. Slow for big datasets, needs no extra packages."""
    base = os.environ.get("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(path, **q):
        url = f"{base}/{path}?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
            return json.load(r)

    if config is None:
        splits = get("splits", dataset=repo)["splits"]
        config = next(s["config"] for s in splits if s["split"] == split)
    off = 0
    while True:
        rows = get("rows", dataset=repo, config=config, split=split, offset=off, length=page)["rows"]
        if not rows:
            return
        for r in rows:
            yield r["row"]
        off += len(rows)


# ---------------------------------------------------------------- git
@register_loader("git")
def load_git(repo: str, path, cache_dir: str, ref: str = "HEAD", format: Optional[str] = None,
             **_) -> Iterator[dict]:
    """repo: "owner/name" (GitHub) or a full clone URL. path: a file path or a
    glob (or a list of them) inside the repo. ref: branch, tag or commit."""
    url = repo if "://" in repo or repo.startswith("git@") else f"https://github.com/{repo}"
    d = os.path.join(cache_dir, "git", hashlib.sha256(url.encode()).hexdigest()[:16])
    if not os.path.isdir(d):
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, d + ".tmp"],
                       check=True, capture_output=True, text=True, timeout=600)
        os.replace(d + ".tmp", d)
    rev = ref if ref == "HEAD" else _git_resolve(d, ref)
    names = subprocess.run(["git", "-C", d, "ls-tree", "-r", "--name-only", rev], check=True,
                           capture_output=True, text=True).stdout.splitlines()
    pats = [path] if isinstance(path, str) else list(path)
    hits = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    if not hits:
        raise FileNotFoundError(f"{pats} not found in {url}@{ref}")
    for n in sorted(hits):
        data = subprocess.run(["git", "-C", d, "show", f"{rev}:{n}"], check=True, capture_output=True,
                              timeout=600).stdout
        yield from _read_bytes(data, format or n)


def _git_resolve(d: str, ref: str) -> str:
    for cand in (f"origin/{ref}", ref):
        r = subprocess.run(["git", "-C", d, "rev-parse", "--verify", "-q", cand + "^{commit}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    subprocess.run(["git", "-C", d, "fetch", "-q", "--filter=blob:none", "origin", ref], check=True,
                   capture_output=True, text=True, timeout=600)
    return "FETCH_HEAD"
