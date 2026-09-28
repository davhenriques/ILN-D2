"""Retrieval-only evaluation (no LLM): every retriever x every index configuration.

    python -m evaluation.retrieval_eval

Cheap enough to run on all questions with reference sources, so the chunk-size
experiment is measured here on the full set; results/retrieval_by_index.csv
holds one row per (index, retriever, question).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from agent.entities import detect_tickers
from agent.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, IndexResources, RerankRetriever
from common.config import load_config, setup_logging
from evaluation import metrics as M
from indexing.store import IndexSpec

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = load_config()
    r = cfg["retrieval"]
    questions = [q for q in (json.loads(l) for l in open(cfg["paths"]["questions"])) if q["reference_sources"]]
    specs = [IndexSpec(e["chunk_size"], e["chunk_overlap"], e["embedding"]) for e in cfg["evaluation"]["chunk_experiments"]]
    rows = []
    for spec in specs:
        if not (spec.dir(cfg) / "info.json").exists():
            log.warning("Index %s not built; skipping (python -m indexing.build_index --all-experiments)", spec.name)
            continue
        res = IndexResources(cfg, spec)
        chunks = list(res.chunks.values())
        retrievers = {"bm25": BM25Retriever(res), "dense": DenseRetriever(res),
                      "hybrid": HybridRetriever(res, r["candidates_k"], r["rrf_k"]),
                      "hybrid_rerank": RerankRetriever(res, cfg["models"]["reranker"], cfg["models"]["device"],
                                                       r["candidates_k"], r["rrf_k"])}
        for q in tqdm(questions, desc=spec.name):
            rel = M.relevant_ids(chunks, q["reference_sources"])
            if not rel:
                continue
            tickers = detect_tickers(q["question"], cfg["companies"]["tickers"]) or None
            for name, ret in retrievers.items():
                ranked = [h.id for h in ret.retrieve(q["question"], 10, tickers)]
                rows.append({"index": spec.name, "chunk_size": spec.chunk_size, "embedding": spec.embedding_model(cfg),
                             "retriever": name, "id": q["id"], "type": q["type"], "n_relevant": len(rel),
                             **M.retrieval_metrics(ranked, rel, cfg["evaluation"]["retrieval_ks"])})
    out = Path(cfg["paths"]["results"]) / "retrieval_by_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info("\n%s", df.groupby(["index", "retriever"])[["R@5", "MRR"]].mean().round(3))


if __name__ == "__main__":
    main()
