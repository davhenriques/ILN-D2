"""Retrievers used by the RAG variants.

  DenseRetriever   top-k by cosine similarity of embeddings (ChromaDB)
  BM25Retriever    top-k by BM25 lexical score
  HybridRetriever  dense + BM25 fused with Reciprocal Rank Fusion (RRF)
  RerankRetriever  hybrid candidates re-scored by a cross-encoder

All retrievers return a list of Hit ordered by relevance and accept an
optional list of tickers used as a metadata filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.config import resolve_device
from indexing.bm25 import BM25Index
from indexing.embeddings import Embedder
from indexing.store import IndexSpec, chroma_client, load_chunks


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float
    scores: dict = field(default_factory=dict)   # per-component scores (debugging / analysis)


class IndexResources:
    """Loads once everything an index configuration needs (chunks, BM25, Chroma, embedder)."""

    def __init__(self, cfg: dict, spec: IndexSpec):
        self.spec = spec
        self.chunks = {c["id"]: c for c in load_chunks(cfg, spec)}
        self.bm25 = BM25Index.load(spec.dir(cfg) / "bm25.pkl")
        self.collection = chroma_client(cfg).get_collection(spec.name)
        self.embedder = Embedder.get(spec.embedding_model(cfg), cfg["models"]["device"], spec.max_seq_length())

    def hit(self, chunk_id: str, score: float, **scores) -> Hit:
        c = self.chunks[chunk_id]
        return Hit(chunk_id, c["text"], c["metadata"], score, scores)


class DenseRetriever:
    def __init__(self, res: IndexResources):
        self.res = res

    def retrieve(self, query: str, k: int, tickers: list[str] | None = None) -> list[Hit]:
        where = {"ticker": {"$in": tickers}} if tickers else None
        r = self.res.collection.query(query_embeddings=[self.res.embedder.embed_query(query).tolist()],
                                      n_results=k, where=where, include=["distances"])
        return [self.res.hit(i, 1 - d, dense=1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]


class BM25Retriever:
    def __init__(self, res: IndexResources):
        self.res = res

    def retrieve(self, query: str, k: int, tickers: list[str] | None = None) -> list[Hit]:
        return [self.res.hit(i, s, bm25=s) for i, s in self.res.bm25.search(query, k, tickers)]


class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (rrf_k + rank_r(d))."""

    def __init__(self, res: IndexResources, candidates_k: int = 30, rrf_k: int = 60):
        self.res, self.candidates_k, self.rrf_k = res, candidates_k, rrf_k
        self.dense, self.bm25 = DenseRetriever(res), BM25Retriever(res)

    def retrieve(self, query: str, k: int, tickers: list[str] | None = None) -> list[Hit]:
        n = max(k, self.candidates_k)
        fused: dict[str, float] = {}
        for name, hits in (("dense", self.dense.retrieve(query, n, tickers)),
                           ("bm25", self.bm25.retrieve(query, n, tickers))):
            for rank, h in enumerate(hits, start=1):
                fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [self.res.hit(i, s, rrf=s) for i, s in ranked]


class RerankRetriever:
    def __init__(self, res: IndexResources, reranker_model: str, device: str = "auto",
                 candidates_k: int = 30, rrf_k: int = 60):
        from sentence_transformers import CrossEncoder
        self.hybrid = HybridRetriever(res, candidates_k, rrf_k)
        self.candidates_k = candidates_k
        self.model = CrossEncoder(reranker_model, device=resolve_device(device), max_length=512)

    def retrieve(self, query: str, k: int, tickers: list[str] | None = None) -> list[Hit]:
        cands = self.hybrid.retrieve(query, max(k, self.candidates_k), tickers)
        if not cands:
            return []
        scores = self.model.predict([(query, h.text) for h in cands], batch_size=16, show_progress_bar=False)
        for h, s in zip(cands, scores):
            h.scores["rerank"] = float(s)
            h.score = float(s)
        return sorted(cands, key=lambda h: -h.score)[:k]
