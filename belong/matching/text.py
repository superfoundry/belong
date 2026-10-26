"""Tiny TF-IDF text embedding, so the prototype needs no model download or API key.

Vectors are sparse dicts {term: weight}, L2-normalised; cosine is a dot product.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_STOP = set("""a an and the in on at for of to with i i'm i'd me my is are be or from who
that this it as by around want looking need someone wanted ideally hoping find like""".split())


_TOKENS: dict = {}  # text -> its tokens: the board is read again every run, and most of it hasn't changed


def tokenize(text: str) -> tuple:
    out = _TOKENS.get(text)
    if out is not None:
        return out
    words = []
    for w in re.findall(r"[a-zà-ÿ]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        for suffix in ("ing", "ers", "er", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        words.append(w)
    if len(_TOKENS) > 50_000:
        _TOKENS.clear()
    out = _TOKENS[text] = tuple(words)
    return out


class TfIdf:
    def __init__(self, docs: dict[str, str]):
        tokenized = {k: tokenize(v) for k, v in docs.items()}
        df: Counter = Counter()
        for toks in tokenized.values():
            df.update(set(toks))
        n = max(1, len(docs))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = {k: self.embed_tokens(toks) for k, toks in tokenized.items()}

    def embed_tokens(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        vec = {t: c * self._idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in vec.values())) or 1.0
        return {t: x / norm for t, x in vec.items()}

    def embed(self, text: str) -> dict[str, float]:
        return self.embed_tokens(tokenize(text))


def cosine(u: dict[str, float], v: dict[str, float]) -> float:
    if len(u) > len(v):
        u, v = v, u
    return sum(x * v.get(t, 0.0) for t, x in u.items())
