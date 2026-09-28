"""Smoke test: full pipeline with 1 company (AAPL) and 5 questions.

    python -m tests.smoke_test          (or: make smoke)

Uses a temporary config that shares the download cache and processed filings
with the main run, but writes its manifest, index, questions and results to
data/smoke/ and results/smoke/ so the main artefacts are never touched.
Requires Ollama running with the generator and judge models pulled.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def run(env, *args):
    print("\n$ python -m", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["companies"]["tickers"] = ["AAPL"]
    cfg["paths"].update({"index": "data/smoke/index", "manifest": "data/smoke/manifest.json",
                         "questions": "results/smoke/questions.jsonl", "results": "results/smoke"})
    cfg_path = ROOT / "data" / "smoke" / "config_smoke.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "D2_CONFIG": str(cfg_path)}

    run(env, "data_collection.collect")
    run(env, "indexing.build_index")
    run(env, "evaluation.generate_questions", "--n-numeric", "2", "--n-yoy", "1", "--n-qoq", "0",
        "--n-cross", "0", "--max-manual", "1")
    run(env, "evaluation.run_all", "--variants", "no_rag", "rag_hybrid_rerank")
    run(env, "evaluation.report")

    df = pd.read_csv(ROOT / "results" / "smoke" / "runs" / "main" / "results.csv")
    assert df["id"].nunique() == 5, f"expected 5 questions, got {df['id'].nunique()}"
    assert set(df["variant"]) == {"no_rag", "rag_hybrid_rerank"}
    assert (ROOT / "results" / "smoke" / "summary.md").exists()
    print("\nSMOKE TEST OK\n", df.groupby("variant")["correct"].mean())


if __name__ == "__main__":
    main()
