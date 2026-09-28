"""Phase 2: build (or rebuild) the dense (ChromaDB) and lexical (BM25) indexes.

    python -m indexing.build_index                              # parameters from config.yaml
    python -m indexing.build_index --chunk-size 256 --overlap 32
    python -m indexing.build_index --chunk-size 1024 --overlap 128 --embedding embedding_long
    python -m indexing.build_index --all-experiments            # every entry in evaluation.chunk_experiments

Idempotent: if an index with the same parameters and the same corpus already
exists it is left untouched (use --force to rebuild).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging

from common.config import ROOT, load_config, setup_logging
from indexing.bm25 import BM25Index
from indexing.chunking import chunk_document, load_processed_docs
from indexing.embeddings import Embedder
from indexing.store import IndexSpec, chroma_client

log = logging.getLogger(__name__)


def corpus_fingerprint(cfg: dict, tickers: list[str] | None) -> str:
    manifest = json.loads(open(cfg["paths"]["manifest"]).read())
    accs = sorted(e["accession"] for e in manifest if not tickers or e["ticker"] in tickers)
    return hashlib.md5("|".join(accs).encode()).hexdigest()[:12]


def build_index(cfg: dict, spec: IndexSpec, force: bool = False, tickers: list[str] | None = None) -> None:
    out_dir = spec.dir(cfg)
    info = {**spec.info(cfg), "corpus": corpus_fingerprint(cfg, tickers)}
    info_path = out_dir / "info.json"
    client = chroma_client(cfg)
    if not force and info_path.exists() and json.loads(info_path.read_text()) == info:
        try:
            if client.get_collection(spec.name).count() > 0:
                log.info("Index %s is up to date; skipping (use --force to rebuild)", spec.name)
                return
        except Exception:
            pass

    embedder = Embedder.get(spec.embedding_model(cfg), cfg["models"]["device"], spec.max_seq_length())
    docs = load_processed_docs(cfg["paths"]["manifest"], ROOT, tickers)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, embedder.tokenizer, spec.chunk_size, spec.chunk_overlap,
                                     cfg["indexing"]["min_chunk_tokens"]))
    log.info("%s: %d documents -> %d chunks", spec.name, len(docs), len(chunks))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    texts = [c["text"] for c in chunks]
    BM25Index([c["id"] for c in chunks], texts, [c["metadata"]["ticker"] for c in chunks]).save(out_dir / "bm25.pkl")

    embeddings = embedder.embed_documents(texts)
    try:
        client.delete_collection(spec.name)
    except Exception:
        pass
    col = client.create_collection(spec.name, metadata={"hnsw:space": "cosine"})
    batch = 2000
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(ids=[c["id"] for c in part], documents=[c["text"] for c in part],
                metadatas=[c["metadata"] for c in part], embeddings=embeddings[i:i + batch].tolist())
    info_path.write_text(json.dumps(info, indent=2))
    log.info("Index %s built: %d chunks in ChromaDB + BM25", spec.name, col.count())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--embedding", default="embedding", help="key under `models` in config.yaml")
    ap.add_argument("--all-experiments", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    if args.all_experiments:
        specs = [IndexSpec(e["chunk_size"], e["chunk_overlap"], e["embedding"])
                 for e in cfg["evaluation"]["chunk_experiments"]]
    else:
        specs = [IndexSpec(args.chunk_size or cfg["indexing"]["chunk_size"],
                           args.overlap if args.overlap is not None else cfg["indexing"]["chunk_overlap"],
                           args.embedding)]
    for spec in specs:
        build_index(cfg, spec, force=args.force, tickers=args.tickers)


if __name__ == "__main__":
    main()
