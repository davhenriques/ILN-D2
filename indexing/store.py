"""Where an index configuration lives on disk and how to open it.

An index configuration = (chunk_size, chunk_overlap, embedding model key). Each
configuration gets its own folder and its own ChromaDB collection, so several
configurations can coexist and be compared:

  data/index/chroma/                 ChromaDB persistent client (all collections)
  data/index/<name>/chunks.jsonl     chunk texts + metadata
  data/index/<name>/bm25.pkl         BM25 index
  data/index/<name>/info.json        parameters used to build it
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from indexing.chunking import HEADER_BUDGET


@dataclass(frozen=True)
class IndexSpec:
    chunk_size: int
    chunk_overlap: int
    embedding_key: str = "embedding"   # key in config.yaml `models`

    @property
    def name(self) -> str:
        return f"cs{self.chunk_size}_ov{self.chunk_overlap}_{self.embedding_key}"

    @classmethod
    def from_config(cls, cfg: dict) -> "IndexSpec":
        return cls(cfg["indexing"]["chunk_size"], cfg["indexing"]["chunk_overlap"], "embedding")

    def embedding_model(self, cfg: dict) -> str:
        return cfg["models"][self.embedding_key]

    def max_seq_length(self) -> int:
        return self.chunk_size + HEADER_BUDGET // 2

    def dir(self, cfg: dict) -> Path:
        return Path(cfg["paths"]["index"]) / self.name

    def info(self, cfg: dict) -> dict:
        return {**asdict(self), "embedding_model": self.embedding_model(cfg),
                "min_chunk_tokens": cfg["indexing"]["min_chunk_tokens"]}


def chroma_client(cfg: dict):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(Path(cfg["paths"]["index"]) / "chroma"),
                                     settings=Settings(anonymized_telemetry=False))


def load_chunks(cfg: dict, spec: IndexSpec) -> list[dict]:
    with open(spec.dir(cfg) / "chunks.jsonl") as f:
        return [json.loads(line) for line in f]
