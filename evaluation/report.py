"""Build results/summary.md and figures from the evaluation outputs.

    python -m evaluation.report

Reads
  results/runs/main/results.csv          main comparison (4 variants x all questions)
  results/retrieval_by_index.csv         retrieval-only chunk-size experiment (optional)
  results/runs/{cs*,k*}/results.csv      answer-level chunk-size / k experiments (optional)
Writes
  results/summary.md, results/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent import guardrail  # noqa: E402
from common.config import ROOT, load_config  # noqa: E402

VARIANTS = ["no_rag", "rag_dense", "rag_hybrid", "rag_hybrid_rerank"]
TYPES = ["numeric_fact", "comparison", "text_fact", "unanswerable"]
# Fixed categorical order (reference palette, light mode): colour follows the variant, never its rank.
COLORS = {"no_rag": "#2a78d6", "rag_dense": "#eb6834", "rag_hybrid": "#1baf7a", "rag_hybrid_rerank": "#eda100",
          "bm25": "#e87ba4", "dense": "#eb6834", "hybrid": "#1baf7a", "hybrid_rerank": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


# ------------------------------------------------------------------ statistics
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int, seed: int) -> tuple[float, float, float, float]:
    """Difference of means b - a over the same questions. Returns (diff, lo95, hi95, p_one_sided)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), \
        float((diffs <= 0).mean())


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> float:
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


# ------------------------------------------------------------------ sections
def accuracy_tables(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    acc = df.pivot_table(index="variant", columns="type", values="correct", aggfunc="mean").reindex(VARIANTS)
    acc = acc[[t for t in TYPES if t in acc.columns]]
    acc["overall"] = df.groupby("variant")["correct"].mean()
    n = df[df.variant == VARIANTS[0]].groupby("type").size()
    out = ["## 1. Answer accuracy by variant and question type\n",
           "`correct` = numeric match within tolerance (numeric facts, numeric comparisons, numeric text facts), "
           "LLM-judge correctness >= 4 (other text facts, 'which company' comparisons), "
           "correct abstention (unanswerable).\n",
           f"Questions per type: {', '.join(f'{t}={n.get(t, 0)}' for t in TYPES)}.\n", md(acc), ""]

    rows = []
    for v in VARIANTS:
        d = df[df.variant == v]
        ans = d[d.type != "unanswerable"]
        un = d[d.type == "unanswerable"]
        txt = d[d.type.isin(["text_fact", "comparison"])]
        rows.append({"variant": v,
                     "numeric acc.": d[d.type == "numeric_fact"]["numeric_correct"].astype(float).mean(),
                     "text token-F1": txt["token_f1"].mean(),
                     "judge correctness (1-5)": ans["judge_correctness"].mean(),
                     "judge faithfulness (1-5)": ans["judge_faithfulness"].mean(),
                     "unanswerable: abstention": un["abstained"].mean(),
                     "unanswerable: hallucination": 1 - un["abstained"].mean(),
                     "answerable: false abstention": ans["abstained"].mean()})
    out += ["### Detailed answer metrics\n", md(pd.DataFrame(rows).set_index("variant")), ""]
    return "\n".join(out), acc


def retrieval_table(df: pd.DataFrame, ks) -> tuple[str, pd.DataFrame]:
    d = df[(df.variant != "no_rag") & df["MRR"].notna()]
    cols = [f"{m}@{k}" for k in ks for m in ("P", "R", "F")] + ["MRR"]
    t = d.groupby("variant")[cols].mean().reindex([v for v in VARIANTS if v != "no_rag"])
    n = d[d.variant == "rag_dense"].shape[0]
    txt = ["## 2. Retrieval metrics (RAG variants)\n",
           f"Computed on the {n} questions with reference sources (chunk is relevant if it belongs to the "
           "reference filing and contains the reference evidence, e.g. the XBRL value as printed in the filing).\n",
           md(t), ""]
    return "\n".join(txt), t


def latency_table(df: pd.DataFrame) -> str:
    t = df.groupby("variant")[["retrieval_latency", "generation_latency", "latency"]].agg(["mean", "median"])
    t.columns = [f"{a} ({b})" for a, b in t.columns]
    return "## 3. Latency (seconds per question)\n\n" + md(t.reindex(VARIANTS), ".2f") + "\n"


def significance(df: pd.DataFrame, cfg: dict) -> str:
    n, seed = cfg["evaluation"]["bootstrap_samples"], cfg["evaluation"]["seed"]
    piv = df.pivot_table(index="id", columns="variant", values="correct", aggfunc="first")
    types = df.drop_duplicates("id").set_index("id")["type"]
    rows = []
    for base, other in [("no_rag", v) for v in VARIANTS[1:]] + [("rag_dense", "rag_hybrid"),
                                                                ("rag_dense", "rag_hybrid_rerank"),
                                                                ("rag_hybrid", "rag_hybrid_rerank")]:
        for scope in ["all"] + TYPES:
            ids = piv.index if scope == "all" else piv.index[types.reindex(piv.index) == scope]
            sub = piv.loc[ids, [base, other]].dropna()
            if len(sub) < 5:
                continue
            diff, lo, hi, p = paired_bootstrap(sub[base].values, sub[other].values, n, seed)
            rows.append({"comparison": f"{other} - {base}", "questions": scope, "n": len(sub), "diff": diff,
                         "CI95 low": lo, "CI95 high": hi, "P(diff<=0)": p,
                         "significant": "yes" if lo > 0 or hi < 0 else "no"})
    return ("## 4. Significance: paired bootstrap of accuracy differences\n\n"
            f"{n} resamples of questions (paired: same questions for both variants). "
            "A difference is significant at 95% when the CI excludes 0.\n\n"
            + md(pd.DataFrame(rows).set_index("comparison")) + "\n")


def judge_validation(df: pd.DataFrame) -> str:
    d = df[df.numeric_correct.notna() & df.judge_correctness.notna()]
    if d.empty:
        return ""
    x = d["numeric_correct"].astype(bool).astype(int).values
    y = (d["judge_correctness"] >= 4).astype(int).values
    return ("## 5. Judge validation\n\n"
            "On questions that also have a deterministic numeric check, agreement between the LLM judge "
            f"(correctness >= 4) and the numeric match: n={len(d)}, agreement={(x == y).mean():.3f}, "
            f"Cohen's kappa={cohen_kappa(x, y):.3f}. Judge said correct but numeric check failed: "
            f"{int(((y == 1) & (x == 0)).sum())}; judge said wrong but numeric check passed: "
            f"{int(((y == 0) & (x == 1)).sum())}.\n")


def guardrail_section() -> str:
    tests = [json.loads(l) for l in open(ROOT / "evaluation" / "guardrail_tests.jsonl")]
    pred = np.array([guardrail.check(t["question"])[0] for t in tests])
    gold = np.array([t["is_advice"] for t in tests])
    tp, fp, fn = int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return ("## 6. Investment-advice guardrail (rules only)\n\n"
            f"{len(tests)} test prompts (EN/PT; {int(gold.sum())} advice requests, {int((~gold).sum())} factual "
            f"questions that mention stocks/shares): precision={prec:.2f}, recall={rec:.2f} "
            f"(false positives={fp}, false negatives={fn}). The test set was written together with the "
            "rules, so this is a sanity check, not an unbiased estimate.\n")


def chunk_experiment(cfg: dict, figdir: Path) -> str:
    out = []
    p = Path(cfg["paths"]["results"]) / "retrieval_by_index.csv"
    if p.exists():
        r = pd.read_csv(p)
        t = r.groupby(["index", "retriever"])[["R@1", "R@5", "R@10", "Hit@5", "MRR"]].mean().round(3)
        out += ["## 7. Additional experiment: chunk size and embedding model (retrieval only)\n",
                f"All {r['id'].nunique()} questions with reference sources, top-10 rankings.\n", md(t), ""]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        piv = r.groupby(["index", "retriever"])["MRR"].mean().unstack()
        order = ["bm25", "dense", "hybrid", "hybrid_rerank"]
        x = np.arange(len(piv.index))
        w = 0.8 / len(order)
        for i, ret in enumerate(order):
            if ret in piv:
                ax.bar(x + i * w - 0.4 + w / 2, piv[ret].values, w - 0.02, color=COLORS[ret], label=ret)
        ax.set_xticks(x, [s.replace("_embedding_long", "\nbge-m3").replace("_embedding", "\nbge-base")
                          for s in piv.index], color=INK2)
        ax.set_ylabel("MRR", color=INK2)
        ax.set_title("MRR by index configuration and retriever", color=INK, loc="left")
        ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.22))
        _style(ax)
        fig.tight_layout()
        fig.savefig(figdir / "chunk_experiment_mrr.png", dpi=160)
        plt.close(fig)
        out.append("![chunk experiment](figures/chunk_experiment_mrr.png)\n")

    runs = sorted((Path(cfg["paths"]["results"]) / "runs").glob("*/results.csv"))
    rows = []
    for rp in runs:
        name = rp.parent.name
        if name in ("main", "dev") or name.startswith("smoke"):
            continue
        info = json.loads((rp.parent / "run_info.json").read_text())
        d = pd.read_csv(rp)
        row = {"run": name, "index": info["index"], "k": info["k"], "n": d["id"].nunique(),
               "accuracy": d["correct"].mean(), "latency": d["latency"].mean()}
        row.update(d.groupby("type")["correct"].mean().to_dict())
        rows.append(row)
    if rows:
        out += ["## 8. Additional experiment: answer accuracy vs chunk size and k (rag_hybrid_rerank, subset)\n",
                "Same stratified subset of questions for every run.\n",
                md(pd.DataFrame(rows).set_index("run")), ""]
    return "\n".join(out)


def examples(df: pd.DataFrame, n: int = 4) -> str:
    piv = df.pivot_table(index="id", columns="variant", values="correct", aggfunc="first")
    good = piv[(piv.get("no_rag") == 0) & (piv.get("rag_hybrid_rerank") == 1)].index[:n]
    bad = piv[(piv.get("rag_hybrid_rerank") == 0)].index[:n]
    out = ["## 9. Examples\n"]
    for title, ids in (("RAG fixes the baseline", good), ("Failures of the best RAG variant", bad)):
        out.append(f"### {title}\n")
        for i in ids:
            d = df[df.id == i].set_index("variant")
            out.append(f"**{i}** ({d.iloc[0]['type']}) - {d.iloc[0]['question']}  \n"
                       f"*Reference:* {d.iloc[0]['reference_answer']}  \n"
                       f"*no_rag:* {str(d.loc['no_rag', 'answer'])[:300]}  \n"
                       f"*rag_hybrid_rerank:* {str(d.loc['rag_hybrid_rerank', 'answer'])[:400]}\n")
    return "\n".join(out)


# ------------------------------------------------------------------ figures
def plot_accuracy(acc: pd.DataFrame, path: Path) -> None:
    cols = [c for c in TYPES + ["overall"] if c in acc.columns]
    x = np.arange(len(cols))
    w = 0.8 / len(VARIANTS)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, v in enumerate(VARIANTS):
        vals = acc.loc[v, cols].values.astype(float)
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w - 0.02, color=COLORS[v], label=v)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, color=INK2, padding=1)
    ax.set_xticks(x, [c.replace("_", " ") for c in cols], color=INK2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy", color=INK2)
    ax.set_title("Answer accuracy by variant and question type", color=INK, loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_recall(ret: pd.DataFrame, ks, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for v in ret.index:
        ys = [ret.loc[v, f"R@{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", markersize=5, linewidth=2, color=COLORS[v], label=v)
        ax.annotate(v, (ks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set_xticks(ks)
    ax.set_xlabel("k", color=INK2)
    ax.set_ylabel("Recall@k", color=INK2)
    ax.set_xlim(ks[0] - 0.5, ks[-1] + 3.5)
    ax.set_title("Recall@k by retrieval variant", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main", help="run folder under results/runs used for sections 1-6")
    args = ap.parse_args()
    cfg = load_config()
    res_dir = Path(cfg["paths"]["results"])
    figdir = res_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res_dir / "runs" / args.run / "results.csv")
    info = json.loads((res_dir / "runs" / args.run / "run_info.json").read_text())
    for col in ("judge_correctness", "judge_faithfulness", "numeric_correct", "token_f1"):
        if col not in df:
            df[col] = np.nan      # e.g. a run made with --no-judge
    ks = cfg["evaluation"]["retrieval_ks"]

    acc_md, acc = accuracy_tables(df)
    ret_md, ret = retrieval_table(df, ks)
    plot_accuracy(acc, figdir / "accuracy_by_variant_type.png")
    plot_recall(ret, ks, figdir / "recall_at_k.png")

    parts = [
        "# Evaluation summary: RAG vs no-RAG on SEC filings\n",
        f"Generator `{info['generator']}`, judge `{info['judge']}`, index `{info['index']}`, top-k={info['k']}, "
        f"{info['n_questions']} questions, companies {', '.join(cfg['companies']['tickers'])}.\n",
        acc_md, "![accuracy](figures/accuracy_by_variant_type.png)\n",
        ret_md, "![recall](figures/recall_at_k.png)\n",
        latency_table(df), significance(df, cfg), judge_validation(df), guardrail_section(),
        chunk_experiment(cfg, figdir), examples(df),
    ]
    (res_dir / "summary.md").write_text("\n".join(p for p in parts if p))
    print(f"Wrote {res_dir / 'summary.md'} and figures in {figdir}")


if __name__ == "__main__":
    main()
