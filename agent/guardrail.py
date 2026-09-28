"""Guardrail: detect requests for investment advice and refuse them.

First line: hand-written regular expressions (English + Portuguese), which are
transparent and fast. Optional second line (agent.guardrail_use_llm): a
zero-shot LLM classifier for phrasings the rules miss.

Questions *about* the documents that merely mention stocks (e.g. "How many
shares did Apple repurchase?") must not be blocked.
"""
from __future__ import annotations

import re

REFUSAL = ("I can only answer questions about the content of the companies' SEC filings "
           "(10-K and 10-Q), such as results, revenue, margins, segments or risk factors. "
           "I do not provide investment advice or buy/sell recommendations.")
REFUSAL_PT = ("Só respondo a perguntas sobre o conteúdo dos documentos financeiros das empresas "
              "(10-K e 10-Q), como resultados, receitas, margens, segmentos ou fatores de risco. "
              "Não dou aconselhamento de investimento nem recomendações de compra/venda.")

ADVICE_PATTERNS = [
    # English
    r"\bshould (i|we|one)\b.{0,40}\b(buy|sell|invest|hold|short|purchase|dump)\b",
    r"\b(is|are)\b.{0,40}\b(good|bad|great|smart|safe|wise)\b.{0,20}\b(investment|buy|stock to (buy|own))\b",
    r"\bworth (investing|buying|a buy|the investment)\b",
    r"\b(recommend|recommendation|advise|advice)\b.{0,40}\b(stock|shares|invest|buy|sell|portfolio)\b",
    r"\bprice target\b",
    r"\b(will|going to)\b.{0,40}\b(stock|share price|shares)\b.{0,20}\b(go up|rise|increase|fall|drop|go down|crash)\b",
    r"\b(buy|sell) or (buy|sell|hold)\b",
    r"\bwhich (stock|company|share)s? should\b",
    r"\b(best|better) (stock|investment)\b",
    r"\bundervalued|overvalued\b",
    # Portuguese
    r"\bdevo (comprar|vender|investir|manter)\b",
    r"\bvale a pena (investir|comprar)\b",
    r"\b(recomendas|recomenda|aconselhas|aconselha)\b.{0,40}\b(a[cç][oõ]es|investir|comprar|vender)\b",
    r"\bbom investimento\b",
    r"\b(comprar|vender) (a[cç][oõ]es|t[ií]tulos)\b",
]
_ADVICE_RE = [re.compile(p, re.I) for p in ADVICE_PATTERNS]

CLASSIFIER_PROMPT = """Classify the user message. Reply with one word only:
ADVICE - it asks for investment advice, a recommendation to buy/sell/hold, a price prediction or whether something is a good investment.
FACTUAL - it asks about facts or analysis contained in company reports (financial results, risks, segments...).

Message: {question}
Answer:"""


def is_portuguese(text: str) -> bool:
    return bool(re.search(r"\b(qual|quais|quanto|devo|vale|como|empresa|receita|lucro|ações|é|são)\b", text, re.I))


def check(question: str, llm=None) -> tuple[bool, str]:
    """Return (is_advice_request, reason)."""
    for rx in _ADVICE_RE:
        if rx.search(question):
            return True, f"rule: {rx.pattern}"
    if llm is not None:
        out, _ = llm.chat([{"role": "user", "content": CLASSIFIER_PROMPT.format(question=question)}])
        if out.strip().upper().startswith("ADVICE"):
            return True, "llm classifier"
    return False, ""


def refusal_message(question: str) -> str:
    return REFUSAL_PT if is_portuguese(question) else REFUSAL
