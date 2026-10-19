"""Embedders for near-dup and decontamination.

HashingEmbedder           char n-gram feature hashing. Runs anywhere, catches
                          light rewording, misses real paraphrases. Default for
                          local runs only.
SentenceTransformerEmbedder  any sentence-transformers model (e.g. bge-m3).
OpenAICompatEmbedder      an /v1/embeddings endpoint (vLLM, TEI, ...).
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from typing import Sequence

from ..textutil import normalize
from .base import Embedder


def _l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 1024, ngram: int = 3):
        self.dim, self.ngram = dim, ngram

    def embed(self, texts: Sequence[str]):
        out = []
        for t in texts:
            s = normalize(t)
            v = [0.0] * self.dim
            for i in range(max(1, len(s) - self.ngram + 1)):
                g = s[i:i + self.ngram]
                h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
                v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            out.append(_l2(v))
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "BAAI/bge-m3", batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.name = f"st:{model}"
        self.m = SentenceTransformer(model, device=device)
        self.batch_size = batch_size

    def embed(self, texts):
        return self.m.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True).tolist()


class OpenAICompatEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", batch_size: int = 64, timeout: float = 120):
        self.name = f"api:{model}"
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.batch_size, self.timeout = batch_size, timeout

    def embed(self, texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            body = json.dumps({"model": self.model, "input": texts[i:i + self.batch_size]}).encode()
            req = urllib.request.Request(f"{self.base_url}/embeddings", data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)["data"]
            out.extend(_l2(d["embedding"]) for d in sorted(data, key=lambda d: d["index"]))
        return out
