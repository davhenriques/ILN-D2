"""Run agent variants over the question set and score every answer.

    python -m evaluation.run_all                                   # main run: 4 variants, all questions
    python -m evaluation.run_all --variants rag_hybrid_rerank --k 3 --subset 40 --run-name k3
    python -m evaluation.run_all --variants rag_hybrid_rerank --chunk-size 256 --overlap 32 --subset 40 --run-name cs256

Outputs (results/runs/<run-name>/):
  answers_<variant>.jsonl   raw answers, retrieved ids, context, latencies (resumable cache)
  results.csv               one row per (question, variant) with every metric
The main run is also copied to results/results.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from agent.rag_agent import VARIANTS, RAGAgent
from common.config import load_config, setup_logging
from evaluation import metrics as M
from evaluation.judge import Judge
from indexing.store import IndexSpec, load_chunks

log = logging.getLogger(__name__)


def load_questions(cfg: dict, path: str | None = None) -> list[dict]:
    return [json.loads(l) for l in open(path or cfg["paths"]["questions"])]


def stratified_subset(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Keep the type proportions when evaluating on a subset (used by the extra experiments)."""
    rng = random.Random(seed)
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(q)
    out = []
    for t, lst in sorted(by_type.items()):
        k = max(1, round(n * len(lst) / len(questions)))
        out += rng.sample(lst, min(k, len(lst)))
    return out


# ------------------------------------------------------------------ generation
def generate(cfg: dict, questions: list[dict], variant: str, spec: IndexSpec, k: int, out_path: Path) -> list[dict]:
    done = {}
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [q for q in questions if q["id"] not in done]
    if todo:
        agent = RAGAgent(cfg, variant, spec=spec, top_k=k)
        agent.llm.chat([{"role": "user", "content": "Hi"}])        # warm-up: load model before timing
        with open(out_path, "a") as f:
            for q in tqdm(todo, desc=variant):
                r = agent.answer(q["question"])
                row = {"id": q["id"], "variant": variant, "answer": r.answer, "refused": r.refused,
                       "tickers_filter": r.tickers_filter, "ranked_ids": r.ranked_ids,
                       "context_ids": [h.id for h in r.context], "context": [h.text for h in r.context],
                       "cited": r.cited, "retrieval_latency": r.retrieval_latency,
                       "generation_latency": r.generation_latency, "latency": r.latency}
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[q["id"]] = row
    return [done[q["id"]] for q in questions]


# ------------------------------------------------------------------ scoring
def score_row(q: dict, a: dict, tol: float) -> dict:
    ans = a["answer"]
    s = {"abstained": M.is_abstention(ans), "numeric_correct": None, "token_f1": None, "exact_match": None}
    if q.get("reference_value") is not None:
        ok = M.numeric_match(ans, q["reference_value"], tol)
        if q["type"] == "comparison" and q.get("reference_pct") is not None:
            ok = ok or M.percent_match(ans, q["reference_pct"])
        s["numeric_correct"] = ok
    if q["type"] in ("text_fact", "comparison"):
        s["token_f1"] = M.token_f1(ans, q["reference_answer"])
        s["exact_match"] = M.exact_match(ans, q["reference_answer"])
    return s


def final_correct(q: dict, s: dict) -> float | None:
    """One binary 'correct' per question, used for accuracy tables and the bootstrap."""
    if q["type"] == "unanswerable":
        return float(s["abstained"])
    if s["numeric_correct"] is not None:
        return float(s["numeric_correct"])
    jc = s.get("judge_correctness")
    return None if jc is None else float(jc >= 4)


def score(cfg: dict, questions: list[dict], answers: dict[str, list[dict]], spec: IndexSpec,
          use_judge: bool = True, cache_path: Path | None = None) -> pd.DataFrame:
    tol = cfg["evaluation"]["numeric_tolerance"]
    ks = cfg["evaluation"]["retrieval_ks"]
    chunks = load_chunks(cfg, spec)
    relevant = {q["id"]: M.relevant_ids(chunks, q["reference_sources"]) for q in questions if q["reference_sources"]}
    judge = Judge(cfg) if use_judge else None
    # Judge verdicts are cached per (question, variant, answer) so re-scoring is cheap.
    cache = {}
    if cache_path and cache_path.exists():
        cache = {r["key"]: r["value"] for r in map(json.loads, open(cache_path))}
    cache_f = open(cache_path, "a") if cache_path else None
    rows = []
    for variant, ans_list in answers.items():
        for q, a in tqdm(list(zip(questions, ans_list)), desc=f"score {variant}"):
            row = {"id": q["id"], "variant": variant, "type": q["type"], "subtype": q.get("subtype", ""),
                   "companies": ",".join(q["companies"]), "question": q["question"],
                   "reference_answer": q["reference_answer"], "answer": a["answer"], "refused": a["refused"],
                   "latency": a["latency"], "retrieval_latency": a["retrieval_latency"],
                   "generation_latency": a["generation_latency"], "n_cited": len(a["cited"])}
            row.update(score_row(q, a, tol))
            rel = relevant.get(q["id"])
            row["n_relevant"] = len(rel) if rel is not None else None
            if variant != "no_rag" and rel:
                row.update(M.retrieval_metrics(a["ranked_ids"], rel, ks))
            if judge and q["type"] != "unanswerable":
                key = f"{q['id']}|{variant}|{hashlib.md5(a['answer'].encode()).hexdigest()}"
                if key not in cache:
                    ctx = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(a["context"], 1)) if a["context"] else None
                    cache[key] = judge.score(q["question"], q["reference_answer"], a["answer"], ctx)
                    if cache_f:
                        cache_f.write(json.dumps({"key": key, "value": cache[key]}) + "\n")
                        cache_f.flush()
                row.update(cache[key])
            row["correct"] = final_correct(q, row)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--embedding", default="embedding")
    ap.add_argument("--k", type=int)
    ap.add_argument("--subset", type=int, help="stratified subset of N questions")
    ap.add_argument("--questions", help="alternative questions file")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    spec = IndexSpec(args.chunk_size or cfg["indexing"]["chunk_size"],
                     args.overlap if args.overlap is not None else cfg["indexing"]["chunk_overlap"], args.embedding)
    k = args.k or cfg["retrieval"]["top_k"]
    questions = load_questions(cfg, args.questions)
    if args.subset:
        questions = stratified_subset(questions, args.subset, cfg["evaluation"]["seed"])
    run_dir = Path(cfg["paths"]["results"]) / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(
        {"index": spec.name, "k": k, "variants": args.variants, "n_questions": len(questions),
         "generator": cfg["models"]["generator"], "judge": cfg["models"]["judge"]}, indent=2))

    answers = {v: generate(cfg, questions, v, spec, k, run_dir / f"answers_{v}.jsonl") for v in args.variants}
    df = score(cfg, questions, answers, spec, use_judge=not args.no_judge, cache_path=run_dir / "judge_cache.jsonl")
    df.to_csv(run_dir / "results.csv", index=False)
    if args.run_name == "main":
        shutil.copy(run_dir / "results.csv", Path(cfg["paths"]["results"]) / "results.csv")
    log.info("Accuracy by variant:\n%s", df.groupby(["variant", "type"])["correct"].mean().unstack().round(3))


if __name__ == "__main__":
    main()
