"""Export a random sample of the automatically generated questions to CSV for manual validation.

    python -m evaluation.export_validation_sample [--n 30]

Open results/validation_sample.csv, check each question/answer against the
filing (the accession number points to the source) and fill in the columns
`valid` (1/0) and `notes`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from common.config import load_config


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=cfg["evaluation"]["validation_sample_size"])
    args = ap.parse_args()
    qs = [json.loads(l) for l in open(cfg["paths"]["questions"])]
    auto = [q for q in qs if q["type"] in ("numeric_fact", "comparison")]
    sample = random.Random(cfg["evaluation"]["seed"]).sample(auto, min(args.n, len(auto)))
    manifest = {m["accession"]: m for m in json.load(open(cfg["paths"]["manifest"]))}
    rows = []
    for q in sample:
        srcs = q["reference_sources"]
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "reference_answer": q["reference_answer"],
            "xbrl_tag": q.get("xbrl_tag", ""), "accessions": "; ".join(s["accession"] for s in srcs),
            "filing_urls": "; ".join(manifest[s["accession"]]["url"] for s in srcs if s["accession"] in manifest),
            "evidence": "; ".join(e for s in srcs for e in s["evidence"]), "valid": "", "notes": "",
        })
    out = Path(cfg["paths"]["results"]) / "validation_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
