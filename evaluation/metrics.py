"""Evaluation metrics: retrieval (P/R/F@k, MRR) and answer quality (numeric match,
token F1 / exact match, abstention)."""
from __future__ import annotations

import re
import string
from collections import Counter

# ------------------------------------------------------------------ retrieval
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).lower()


def is_relevant(chunk: dict, sources: list[dict]) -> bool:
    """A chunk is relevant if it comes from a reference filing (and section, when given)
    and contains at least one of the reference evidence strings."""
    md, text = chunk["metadata"], None
    for s in sources:
        if s.get("accession") and md["accession"] != s["accession"]:
            continue
        if s.get("ticker") and md["ticker"] != s["ticker"]:
            continue
        if s.get("section") and md["section"] != s["section"]:
            continue
        text = text or _norm(chunk["text"])
        if any(_norm(e) in text for e in s.get("evidence", [])):
            return True
    return False


def relevant_ids(chunks: list[dict], sources: list[dict]) -> set[str]:
    accs = {s.get("accession") for s in sources}
    return {c["id"] for c in chunks if (None in accs or c["metadata"]["accession"] in accs)
            and is_relevant(c, sources)}


def retrieval_metrics(ranked: list[str], relevant: set[str], ks=(1, 3, 5, 10)) -> dict:
    out = {}
    for k in ks:
        hits = sum(1 for i in ranked[:k] if i in relevant)
        p = hits / k
        r = hits / len(relevant) if relevant else 0.0
        out[f"P@{k}"] = p
        out[f"R@{k}"] = r
        out[f"F@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
        out[f"Hit@{k}"] = float(hits > 0)
    rr = next((1 / (i + 1) for i, d in enumerate(ranked[:max(ks)]) if d in relevant), 0.0)
    out["MRR"] = rr
    return out


# ------------------------------------------------------------------ numbers
SCALE = {"trillion": 1e12, "tn": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6,
         "m": 1e6, "mm": 1e6, "thousand": 1e3, "k": 1e3,
         # Portuguese
         "mil milhões": 1e9, "milhões": 1e6, "milhão": 1e6, "mil": 1e3, "biliões": 1e12}
NUM_RE = re.compile(
    r"(?P<neg>[-−(])?\s*\$?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?"      # 94,036 / 1,234.5
    r"|\d+,\d{1,2}(?!\d)"                               # Portuguese decimal comma: 94,0
    r"|\d+(?:\.\d+)?)\s*"
    r"(?P<pct>%|percent|por cento)?\s*"
    r"(?P<scale>mil milhões|trillion|billion|million|thousand|milhões|milhão|biliões|mil|tn|bn|mn|mm|[bmk])?(?![a-zà-ú])",
    re.I)


def extract_numbers(text: str) -> list[tuple[float, bool, bool]]:
    """Return [(value, is_percent, has_explicit_scale)] for every number in the text."""
    out = []
    text = re.sub(r"\[\d+\]", " ", text)            # drop citation markers like [2]
    for m in NUM_RE.finditer(text):
        raw = m.group("num")
        raw = raw.replace(",", ".") if re.fullmatch(r"\d+,\d{1,2}", raw) else raw.replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        scale = (m.group("scale") or "").lower()
        if scale:
            v *= SCALE.get(scale, 1)
        out.append((v, bool(m.group("pct")), bool(scale)))
    return out


def numeric_match(answer: str, reference: float, tol: float = 0.01) -> bool:
    """True if some number in the answer is within `tol` relative error of the reference.

    Unit normalisation: numbers with an explicit scale ("94.0 billion") are compared
    directly; bare numbers are also tried in millions and thousands, because
    filings' tables are "in millions" and models often copy the table value.
    Signs are ignored (a "net loss of $500 million" matches -500e6).
    """
    ref = abs(reference)
    if ref == 0:
        return any(v == 0 for v, _, _ in extract_numbers(answer))
    for v, is_pct, has_scale in extract_numbers(answer):
        if is_pct:
            continue
        candidates = [v]
        # Implicit "in millions/thousands" only for table-like numbers: >= 100 and not a year.
        if not has_scale and v >= 100 and not (1900 <= v <= 2100 and v == int(v)):
            candidates += [v * 1e6, v * 1e3]
        if any(abs(c - ref) / ref <= tol for c in candidates):
            return True
    return False


def percent_match(answer: str, reference_pct: float, abs_tol: float = 0.5) -> bool:
    return any(is_pct and abs(abs(v) - abs(reference_pct)) <= abs_tol for v, is_pct, _ in extract_numbers(answer))


# ------------------------------------------------------------------ abstention
ABSTAIN_RE = re.compile(
    r"could not find|couldn't find|cannot find|can't find|did not find|unable to find|not (?:be )?found|"
    r"no information|not (?:available|provided|mentioned|disclosed|included|specified|contain)|"
    r"does not (?:contain|provide|mention|include|specify)|do not (?:have|know)|don't (?:have|know)|"
    r"i am not able|i'm not able|cannot (?:answer|determine|provide)|can't (?:answer|determine)|"
    r"no (?:data|mention)|n[ãa]o (?:encontrei|foi poss[ií]vel|tenho|sei|cont[eé]m|est[aá] dispon)",
    re.I)


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer))


# ------------------------------------------------------------------ text overlap (SQuAD-style)
def _tokens(s: str) -> list[str]:
    s = re.sub(r"\[\d+\]", " ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"."})
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return [t.strip(".") for t in s.split() if t.strip(".")]


def token_f1(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if not p or not r or same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, ref: str) -> float:
    return float(_tokens(pred) == _tokens(ref))
