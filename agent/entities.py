"""Detect which companies a question is about (used as a metadata filter)."""
from __future__ import annotations

import re

ALIASES = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "TSLA": ["tesla"],
    "JPM": ["jpmorgan", "jp morgan", "j.p. morgan", "jpmorgan chase", "chase"],
    "KO": ["coca-cola", "coca cola", "coke", "the coca-cola company"],
    "GOOGL": ["alphabet", "google"],
    "AMZN": ["amazon"],
    "META": ["meta platforms", "facebook"],
}


def detect_tickers(question: str, known_tickers: list[str]) -> list[str]:
    found = []
    for t in known_tickers:
        # Tickers must be written in upper case ("KO", not "ko"); names are case-insensitive.
        if re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", question):
            found.append(t)
            continue
        if any(re.search(rf"\b{re.escape(a)}\b", question, re.I) for a in ALIASES.get(t, [])):
            found.append(t)
    return found
