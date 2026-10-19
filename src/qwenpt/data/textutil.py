"""Text normalization, n-grams and MinHash for dedup / decontamination."""
from __future__ import annotations

import hashlib
import re
import struct
import unicodedata

_WS = re.compile(r"\s+")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_TOKEN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _WS.sub(" ", text).strip()


def exact_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def tokens(text: str) -> list[str]:
    """Word tokens for latin text, one token per CJK character."""
    return _TOKEN.findall(normalize(text))


_WORD = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]|[A-Za-z0-9_]+")


def word_tokens(text: str) -> list[str]:
    """Like tokens() but without punctuation / symbols."""
    return _WORD.findall(normalize(text))


def ngrams(toks: list[str], n: int) -> set[tuple]:
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _h64(s: str, seed: int) -> int:
    return struct.unpack("<Q", hashlib.blake2b(s.encode(), digest_size=8, salt=seed.to_bytes(8, "little")).digest())[0]


class MinHasher:
    def __init__(self, num_perm: int = 128, ngram: int = 5, seed: int = 1):
        self.num_perm, self.ngram, self.seed = num_perm, ngram, seed
        # Use one hash and num_perm cheap affine permutations of it, mod a
        # 31-bit prime so a*h+b fits in uint64 and numpy can do all at once.
        import numpy as np
        rnd = np.random.default_rng(seed)
        self._p = np.uint64((1 << 31) - 1)
        self._a = rnd.integers(1, int(self._p), num_perm, dtype=np.uint64)
        self._b = rnd.integers(0, int(self._p), num_perm, dtype=np.uint64)

    def signature(self, text: str) -> list[int]:
        import numpy as np
        grams = ngrams(tokens(text), self.ngram)
        if not grams:
            return [0] * self.num_perm
        hs = np.fromiter((_h64(" ".join(g), 0) for g in grams), dtype=np.uint64, count=len(grams)) % self._p
        return ((np.outer(hs, self._a) + self._b) % self._p).min(axis=0).tolist()

    @staticmethod
    def jaccard(s1: list[int], s2: list[int]) -> float:
        return sum(x == y for x, y in zip(s1, s2)) / len(s1)


def lsh_buckets(sig: list[int], bands: int) -> list[tuple]:
    rows = len(sig) // bands
    return [(b, tuple(sig[b * rows:(b + 1) * rows])) for b in range(bands)]
