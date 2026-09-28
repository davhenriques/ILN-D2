"""Section-aware chunking with overlap.

Chunks never cross section boundaries. Windows are measured in tokens of the
embedding model's tokenizer (so a 512-token chunk fits a 512-token encoder),
and the text of each chunk is an exact substring of the section, recovered
through the tokenizer's character offsets.

Each chunk is prefixed with a short header (ticker, form, period, section).
This "contextual header" lets both BM25 and the dense encoder match questions
such as "... Apple's 10-Q for the quarter ended June 28, 2025", which the
body text alone often does not mention.
"""
from __future__ import annotations

import json
from pathlib import Path

HEADER_BUDGET = 40  # tokens reserved for the header inside chunk_size


def chunk_header(doc: dict, section: dict) -> str:
    title = f" ({section['title']})" if section.get("title") and not section["section"].startswith("Report:") else ""
    return (f"[{doc['ticker']} | {doc['company']} | Form {doc['form']} for the period ended "
            f"{doc['report_date']} | {section['section']}{title}]")


def chunk_text(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + size >= len(offsets):
            break
    return chunks


def chunk_document(doc: dict, tokenizer, chunk_size: int, overlap: int, min_tokens: int) -> list[dict]:
    body_size = chunk_size - HEADER_BUDGET
    out = []
    for section in doc["sections"]:
        header = chunk_header(doc, section)
        for i, body in enumerate(chunk_text(section["text"], tokenizer, body_size, overlap)):
            if len(tokenizer.tokenize(body)) < min_tokens:
                continue
            sec_slug = section["section"].replace(" ", "")
            out.append({
                "id": f"{doc['accession']}::{sec_slug}::{i}",
                "text": f"{header}\n{body}",
                "metadata": {
                    "ticker": doc["ticker"], "company": doc["company"], "form": doc["form"],
                    "filing_date": doc["filing_date"], "report_date": doc["report_date"],
                    "accession": doc["accession"], "section": section["section"],
                    "section_title": section.get("title") or "", "chunk_index": i,
                },
            })
    return out


def load_processed_docs(manifest_path: str | Path, root: Path, tickers: list[str] | None = None) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    docs = []
    for entry in manifest:
        if tickers and entry["ticker"] not in tickers:
            continue
        doc = json.loads((root / entry["processed_path"]).read_text())
        docs.append(doc)
    return docs
