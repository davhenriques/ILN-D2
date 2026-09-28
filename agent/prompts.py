"""Prompts. The SAME system prompt is used by every variant (no_rag and RAG) so
that the comparison isolates the effect of retrieval; the variants differ only
in whether a CONTEXT block is present in the user message.
"""
from __future__ import annotations

NOT_FOUND = "I could not find this information in the documents."

SYSTEM_PROMPT = f"""You are FinDocs, an assistant specialised in the SEC filings (Forms 10-K and 10-Q) of US-listed companies.
Rules:
1. Answer only factual or analytical questions about the content of company filings (results, revenue, margins, segments, risk factors, guidance, employees, etc.).
2. Never give investment advice or recommendations (buy/sell/hold, price targets, whether a stock is a good investment).
3. Never invent facts or numbers. If you do not know the answer, reply exactly: "{NOT_FOUND}"
4. Be concise (1-4 sentences). When giving an amount, state the value with its unit (e.g. "$94,930 million") and the period it refers to.
5. Answer in the same language as the question."""

RAG_INSTRUCTIONS = f"""Using ONLY the information in the CONTEXT above, answer the QUESTION.
- Cite the excerpts you used as [n] right after each claim, e.g. "Revenue was $94,930 million [2]."
- Pay attention to the company, the period and the units (tables are usually "in millions").
- If the CONTEXT does not contain the answer, reply exactly: "{NOT_FOUND}" """

NO_RAG_INSTRUCTIONS = f"""Answer the QUESTION using your own knowledge.
If you do not know the answer, reply exactly: "{NOT_FOUND}" """


def format_context(hits) -> str:
    return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))


def build_messages(question: str, hits=None) -> list[dict]:
    if hits is None:
        user = f"{NO_RAG_INSTRUCTIONS}\n\nQUESTION: {question}"
    else:
        user = f"CONTEXT (excerpts from SEC filings):\n\n{format_context(hits)}\n\n---\n{RAG_INSTRUCTIONS}\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
