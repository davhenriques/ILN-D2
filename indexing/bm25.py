"""Lexical index (BM25 Okapi) built in parallel to the vector index."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
will with what which who whom how did does do during than then there these those their our we you your
i me my he she they them his her not no""".split())


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)   # 94,930 -> 94930 (match numbers written with or without commas)
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], tickers: list[str]):
        self.ids = ids
        self.tickers = np.array(tickers)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int, tickers: list[str] | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        if tickers:
            scores = np.where(np.isin(self.tickers, tickers), scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)
