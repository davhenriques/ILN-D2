"""The agent: guardrail -> (optional) retrieval -> LLM generation.

Variants (same LLM, same system prompt):
  no_rag              LLM only (baseline)
  rag_dense           dense retrieval
  rag_hybrid          BM25 + dense with Reciprocal Rank Fusion
  rag_hybrid_rerank   hybrid + cross-encoder reranking
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from agent import guardrail
from agent.entities import detect_tickers
from agent.llm import OllamaLLM
from agent.prompts import build_messages
from agent.retrievers import (DenseRetriever, Hit, HybridRetriever, IndexResources,
                              RerankRetriever)
from indexing.store import IndexSpec

VARIANTS = ["no_rag", "rag_dense", "rag_hybrid", "rag_hybrid_rerank"]
N_RANKED = 10   # ranking depth kept for retrieval metrics (max k in {1,3,5,10})

_RESOURCES: dict[str, IndexResources] = {}
_RERANKERS: dict[str, RerankRetriever] = {}


@dataclass
class AgentResponse:
    question: str
    variant: str
    answer: str
    refused: bool = False
    tickers_filter: list[str] = field(default_factory=list)
    context: list[Hit] = field(default_factory=list)       # chunks given to the LLM
    ranked_ids: list[str] = field(default_factory=list)    # top-N_RANKED ranking (for retrieval metrics)
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0

    @property
    def latency(self) -> float:
        return self.retrieval_latency + self.generation_latency

    @property
    def cited(self) -> list[int]:
        return sorted({int(n) for n in re.findall(r"\[(\d+)\]", self.answer) if 0 < int(n) <= len(self.context)})

    def sources(self) -> list[str]:
        cited = set(self.cited)
        out = []
        for i, h in enumerate(self.context, start=1):
            m = h.metadata
            mark = "*" if i in cited else " "
            out.append(f"{mark}[{i}] {m['ticker']} {m['form']} (period {m['report_date']}, filed {m['filing_date']}) "
                       f"- {m['section']} {('- ' + m['section_title']) if m.get('section_title') else ''}".rstrip())
        return out


class RAGAgent:
    def __init__(self, cfg: dict, variant: str | None = None, spec: IndexSpec | None = None,
                 top_k: int | None = None, llm: OllamaLLM | None = None):
        self.cfg = cfg
        self.variant = variant or cfg["agent"]["default_variant"]
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant}; choose one of {VARIANTS}")
        self.top_k = top_k or cfg["retrieval"]["top_k"]
        self.llm = llm or OllamaLLM.from_config(cfg)
        self.guard_llm = self.llm if cfg["agent"].get("guardrail_use_llm") else None
        self.tickers = cfg["companies"]["tickers"]
        self.retriever = None
        if self.variant != "no_rag":
            spec = spec or IndexSpec.from_config(cfg)
            if spec.name not in _RESOURCES:
                _RESOURCES[spec.name] = IndexResources(cfg, spec)
            res = _RESOURCES[spec.name]
            r = cfg["retrieval"]
            if self.variant == "rag_dense":
                self.retriever = DenseRetriever(res)
            elif self.variant == "rag_hybrid":
                self.retriever = HybridRetriever(res, r["candidates_k"], r["rrf_k"])
            else:
                if spec.name not in _RERANKERS:   # the cross-encoder is expensive to load: share it
                    _RERANKERS[spec.name] = RerankRetriever(res, cfg["models"]["reranker"], cfg["models"]["device"],
                                                            r["candidates_k"], r["rrf_k"])
                self.retriever = _RERANKERS[spec.name]

    def retrieve(self, question: str) -> tuple[list[Hit], list[str]]:
        tickers = detect_tickers(question, self.tickers) if self.cfg["retrieval"]["ticker_filter"] else []
        hits = self.retriever.retrieve(question, max(self.top_k, N_RANKED), tickers or None)
        return hits, tickers

    def answer(self, question: str) -> AgentResponse:
        is_advice, _ = guardrail.check(question, self.guard_llm)
        if is_advice:
            return AgentResponse(question, self.variant, guardrail.refusal_message(question), refused=True)
        resp = AgentResponse(question, self.variant, "")
        hits = None
        if self.retriever is not None:
            t0 = time.perf_counter()
            ranked, resp.tickers_filter = self.retrieve(question)
            resp.retrieval_latency = time.perf_counter() - t0
            resp.ranked_ids = [h.id for h in ranked[:N_RANKED]]
            hits = resp.context = ranked[: self.top_k]
        resp.answer, resp.generation_latency = self.llm.chat(build_messages(question, hits))
        return resp
