"""Build evaluation/questions.jsonl.

  numeric_fact  generated from XBRL (stratified sample over companies and metrics)
  comparison    generated from XBRL (YoY, QoQ, cross-company)
  text_fact     hand-written from the filings  (evaluation/manual_questions.jsonl)
  unanswerable  hand-written, answer not in the corpus (evaluation/manual_questions.jsonl)

    python -m evaluation.generate_questions [--n-numeric 50 --n-yoy 8 --n-qoq 6 --n-cross 6]
    python -m evaluation.generate_questions --tickers AAPL --n-numeric 3 ... (smoke test)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from common.config import ROOT, load_config, setup_logging
from evaluation.xbrl_questions import comparison_candidates, numeric_candidates

log = logging.getLogger(__name__)


def stratified_sample(cands: list[dict], n: int, rng: random.Random, keys=("companies", "metric")) -> list[dict]:
    """Round-robin over companies, and inside each company over metrics, so the sample is balanced."""
    by_company = defaultdict(lambda: defaultdict(list))
    for c in cands:
        by_company[c["companies"][0]][c["metric"]].append(c)
    for metrics in by_company.values():
        for lst in metrics.values():
            rng.shuffle(lst)
    order = {t: rng.sample(sorted(m), len(m)) for t, m in by_company.items()}
    out, pos = [], defaultdict(int)
    while len(out) < n and any(any(by_company[t][m] for m in by_company[t]) for t in by_company):
        for t in sorted(by_company):
            metrics = [m for m in order[t] if by_company[t][m]]
            if not metrics or len(out) >= n:
                continue
            m = metrics[pos[t] % len(metrics)]
            pos[t] += 1
            out.append(by_company[t][m].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-numeric", type=int, default=50)
    ap.add_argument("--n-yoy", type=int, default=8)
    ap.add_argument("--n-qoq", type=int, default=6)
    ap.add_argument("--n-cross", type=int, default=6)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--max-manual", type=int, default=None, help="limit manual questions (smoke test)")
    ap.add_argument("--out")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    rng = random.Random(cfg["evaluation"]["seed"])
    manifest = json.loads(Path(cfg["paths"]["manifest"]).read_text())
    if args.tickers:
        manifest = [m for m in manifest if m["ticker"] in args.tickers]

    numeric = stratified_sample(numeric_candidates(manifest, cfg["paths"]["xbrl"]), args.n_numeric, rng)
    yoy, qoq, cross = comparison_candidates(manifest, cfg["paths"]["xbrl"], rng)
    comparisons = (stratified_sample(yoy, args.n_yoy, rng) + stratified_sample(qoq, args.n_qoq, rng)
                   + rng.sample(cross, min(args.n_cross, len(cross))))

    manual = [json.loads(l) for l in open(ROOT / "evaluation" / "manual_questions.jsonl")]
    if args.tickers:
        manual = [q for q in manual if not q["companies"] or set(q["companies"]) & set(args.tickers)]
    if args.max_manual is not None:
        by_type = defaultdict(list)
        for q in manual:
            by_type[q["type"]].append(q)
        manual = [q for lst in by_type.values() for q in lst[: args.max_manual]]

    questions = []
    for prefix, group in (("num", numeric), ("cmp", comparisons)):
        for i, q in enumerate(group, 1):
            questions.append({"id": f"{prefix}_{i:03d}", **q})
    questions += manual

    out = Path(args.out or cfg["paths"]["questions"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    counts = defaultdict(int)
    for q in questions:
        counts[q["type"]] += 1
    log.info("%d questions -> %s  %s", len(questions), out, dict(counts))


if __name__ == "__main__":
    main()
