"""Sentence-transformers wrapper that applies the query/document prefixes each model expects."""
from __future__ import annotations

import numpy as np

from common.config import resolve_device

# Asymmetric retrieval models are trained with instruction prefixes.
PREFIXES = {
    "BAAI/bge-m3": ("", ""),          # no instruction prefix
    "BAAI/bge": ("Represent this sentence for searching relevant passages: ", ""),
    "nomic-ai/nomic-embed": ("search_query: ", "search_document: "),
    "intfloat/e5": ("query: ", "passage: "),
}

_CACHE: dict[str, "Embedder"] = {}


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", max_seq_length: int | None = None):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=resolve_device(device),
                                         trust_remote_code=model_name.startswith("nomic-ai/"))
        if max_seq_length:  # never exceed the encoder's position limit (512 for BERT-style models)
            self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length)
        self.tokenizer = self.model.tokenizer
        self.q_prefix, self.d_prefix = next((v for k, v in PREFIXES.items() if model_name.startswith(k)), ("", ""))

    @classmethod
    def get(cls, model_name: str, device: str = "auto", max_seq_length: int | None = None) -> "Embedder":
        key = f"{model_name}|{max_seq_length}"
        if key not in _CACHE:
            _CACHE[key] = cls(model_name, device, max_seq_length)
        return _CACHE[key]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode([self.d_prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.q_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]
