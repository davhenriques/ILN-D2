"""Command-line interface.

    python -m agent.cli                                 # interactive, default variant
    python -m agent.cli --variant no_rag
    python -m agent.cli -q "What was NVIDIA's revenue for the quarter ended July 27, 2025?"
"""
from __future__ import annotations

import argparse
import logging

from agent.rag_agent import VARIANTS, RAGAgent
from common.config import load_config, setup_logging

BANNER = """FinDocs - questions about 10-K / 10-Q filings of {tickers}
Variant: {variant}   (type 'exit' to quit)
Note: this agent does NOT provide financial or investment advice."""


def show(resp, verbose: bool = False) -> None:
    print(f"\n{resp.answer}\n")
    if resp.context:
        print("Sources (* = cited):")
        for s in resp.sources():
            print("  " + s)
        if resp.tickers_filter:
            print(f"  (filter by ticker: {', '.join(resp.tickers_filter)})")
    if verbose:
        for i, h in enumerate(resp.context, 1):
            print(f"\n--- [{i}] score={h.score:.3f}\n{h.text[:600]}")
    print(f"\n[{resp.variant}] retrieval {resp.retrieval_latency:.2f}s | generation {resp.generation_latency:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--k", type=int, help="number of chunks passed to the LLM")
    ap.add_argument("-q", "--question", help="ask a single question and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="print retrieved chunk texts")
    args = ap.parse_args()
    setup_logging(logging.WARNING)
    cfg = load_config()
    agent = RAGAgent(cfg, args.variant, top_k=args.k)
    if args.question:
        show(agent.answer(args.question), args.verbose)
        return
    print(BANNER.format(tickers=", ".join(cfg["companies"]["tickers"]), variant=agent.variant))
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "sair"}:
            break
        if q:
            show(agent.answer(q), args.verbose)


if __name__ == "__main__":
    main()
