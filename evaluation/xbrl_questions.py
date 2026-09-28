"""Automatic question generation from XBRL company facts (numeric ground truth).

For each filing in the corpus we look for facts reported *in that filing*
(`accn` == accession) for its own reporting period (`end` == period of report):

  * duration facts (income statement, cash flow): the period length is derived
    from start/end dates -> "three / six / nine months ended" or "fiscal year ended";
  * instant facts (balance sheet): "as of <date>".

Each metric has a list of alternative US-GAAP tags tried in order (e.g. revenue
is reported as Revenues, RevenueFromContractWithCustomerExcludingAssessedTax or,
for banks, RevenuesNetOfInterestExpense). If no tag exists for a filing, no
question is generated.

Question types produced here:
  numeric_fact   "What was Apple's net income for the three months ended June 28, 2025?"
  comparison     year-over-year change inside one 10-Q, quarter-over-quarter across two
                 10-Qs, and cross-company ("which company reported higher ...").
"""
from __future__ import annotations

import random
from datetime import date

from data_collection.xbrl import load_companyfacts

# metric -> (alternative tags in priority order, label, kind, unit)
METRICS = {
    "revenue": (["RevenuesNetOfInterestExpense", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "SalesRevenueNet"], "total revenue", "duration", "USD"),
    "net_income": (["NetIncomeLoss"], "net income", "duration", "USD"),
    "operating_income": (["OperatingIncomeLoss"], "operating income", "duration", "USD"),
    "gross_profit": (["GrossProfit"], "gross margin (gross profit)", "duration", "USD"),
    "eps_diluted": (["EarningsPerShareDiluted"], "diluted earnings per share", "duration", "USD/shares"),
    "rd_expense": (["ResearchAndDevelopmentExpense"], "research and development expense", "duration", "USD"),
    "income_tax": (["IncomeTaxExpenseBenefit"], "provision for income taxes", "duration", "USD"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "net cash provided by operating activities",
                            "duration", "USD"),
    "total_assets": (["Assets"], "total assets", "instant", "USD"),
    "total_liabilities": (["Liabilities"], "total liabilities", "instant", "USD"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], "cash and cash equivalents", "instant", "USD"),
    "equity": (["StockholdersEquity"], "total shareholders' equity", "instant", "USD"),
}

# How each company names total revenue in its statements (helps the question read naturally).
REVENUE_LABEL = {"AAPL": "total net sales", "KO": "net operating revenues", "JPM": "total net revenue",
                 "TSLA": "total revenues", "NVDA": "revenue", "MSFT": "total revenue"}

SHORT_NAME = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla", "JPM": "JPMorgan Chase",
              "KO": "Coca-Cola"}


def _d(s: str) -> date:
    return date.fromisoformat(s)


def fmt_date(s: str) -> str:
    d = _d(s)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def period_phrase(fact: dict, form: str) -> str | None:
    if "start" not in fact:
        return f"as of {fmt_date(fact['end'])}"
    months = round((_d(fact["end"]) - _d(fact["start"])).days / 30.44)
    if months == 12:
        return f"for the fiscal year ended {fmt_date(fact['end'])}"
    words = {3: "three", 6: "six", 9: "nine"}
    if months not in words:
        return None
    return f"for the {words[months]} months ended {fmt_date(fact['end'])}"


def fmt_value(value: float, unit: str) -> str:
    if unit == "USD/shares":
        return f"${value:,.2f} per diluted share"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e6:,.0f} million"


def evidence_strings(value: float, unit: str) -> list[str]:
    """How the value is likely written in the filing (tables are in millions)."""
    if unit == "USD/shares":
        return [f"{abs(value):.2f}"]
    return [f"{abs(value) / 1e6:,.0f}"]


def filing_facts(facts: dict, filing: dict) -> dict[str, list[dict]]:
    """metric -> facts reported in this filing for its own period (all durations)."""
    gaap = facts["facts"].get("us-gaap", {})
    out = {}
    for metric, (tags, _, kind, unit) in METRICS.items():
        for tag in tags:
            rows = [x for x in gaap.get(tag, {}).get("units", {}).get(unit, [])
                    if x["accn"] == filing["accession"]]
            if rows:
                out[metric] = [dict(r, tag=tag) for r in rows]
                break
    return out


def label_for(metric: str, ticker: str) -> str:
    return REVENUE_LABEL.get(ticker, "total revenue") if metric == "revenue" else METRICS[metric][1]


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r.get("start"), r["end"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def numeric_candidates(manifest: list[dict], xbrl_dir: str) -> list[dict]:
    cands = []
    for filing in manifest:
        facts = load_companyfacts(xbrl_dir, filing["ticker"])
        for metric, rows in filing_facts(facts, filing).items():
            tags, _, kind, unit = METRICS[metric]
            for r in _dedupe(rows):
                if r["end"] != filing["report_date"]:
                    continue       # comparative (prior-period) values are used for comparisons only
                phrase = period_phrase(r, filing["form"])
                if phrase is None:
                    continue
                if kind == "duration" and filing["form"] == "10-Q" and "three months" not in phrase \
                        and metric != "operating_cash_flow":
                    continue       # keep quarterly figures; cash flow is only reported year-to-date
                name = SHORT_NAME.get(filing["ticker"], filing["company"])
                label = label_for(metric, filing["ticker"])
                verb = "were" if metric in ("total_assets", "total_liabilities", "cash") else "was"
                q = f"What {verb} {name}'s {label} {phrase}?"
                cands.append({
                    "question": q, "type": "numeric_fact", "companies": [filing["ticker"]],
                    "reference_answer": f"{name}'s {label} {phrase} {verb} {fmt_value(r['val'], unit)}.",
                    "reference_value": r["val"], "unit": unit, "metric": metric, "xbrl_tag": r["tag"],
                    "period": {"start": r.get("start"), "end": r["end"]},
                    "reference_sources": [{"ticker": filing["ticker"], "accession": filing["accession"],
                                           "evidence": evidence_strings(r["val"], unit)}],
                })
    return cands


def _prior_year(rows: list[dict], cur: dict) -> dict | None:
    """Same-length period ending ~1 year before `cur` (comparative column in the same filing)."""
    for r in rows:
        if "start" in cur and "start" in r and abs((_d(cur["end"]) - _d(r["end"])).days - 364) <= 10 \
                and abs(((_d(r["end"]) - _d(r["start"])) - (_d(cur["end"]) - _d(cur["start"]))).days) <= 10:
            return r
    return None


def _change_answer(name, label, cur_phrase, prev_phrase, cur, prev, unit):
    delta = cur["val"] - prev["val"]
    pct = 100 * delta / abs(prev["val"]) if prev["val"] else None
    direction = "increased" if delta >= 0 else "decreased"
    ans = (f"{name}'s {label} {direction} from {fmt_value(prev['val'], unit)} ({prev_phrase}) to "
           f"{fmt_value(cur['val'], unit)} ({cur_phrase}), a change of {fmt_value(abs(delta), unit)}")
    ans += f" ({pct:+.1f}%)." if pct is not None else "."
    return ans, delta, pct


def comparison_candidates(manifest: list[dict], xbrl_dir: str, rng: random.Random) -> list[dict]:
    yoy, qoq, cross = [], [], []
    quarterly: dict[str, list[tuple[dict, dict]]] = {}   # ticker -> [(filing, {metric: fact})]
    for filing in manifest:
        if filing["form"] != "10-Q":
            continue
        facts = load_companyfacts(xbrl_dir, filing["ticker"])
        ff = filing_facts(facts, filing)
        name = SHORT_NAME.get(filing["ticker"], filing["company"])
        cur_by_metric = {}
        for metric in ("revenue", "net_income", "operating_income"):
            rows = _dedupe(ff.get(metric, []))
            cur = next((r for r in rows if r["end"] == filing["report_date"] and "start" in r
                        and round((_d(r["end"]) - _d(r["start"])).days / 30.44) == 3), None)
            if cur is None:
                continue
            cur_by_metric[metric] = cur
            prev = _prior_year(rows, cur)
            if prev is None:
                continue
            label, unit = label_for(metric, filing["ticker"]), "USD"
            cp, pp = period_phrase(cur, "10-Q"), period_phrase(prev, "10-Q")
            ans, delta, pct = _change_answer(name, label, cp, pp, cur, prev, unit)
            yoy.append({
                "question": f"How did {name}'s {label} {cp} compare with the same quarter of the previous year?",
                "type": "comparison", "subtype": "change", "companies": [filing["ticker"]],
                "reference_answer": ans, "reference_value": delta, "reference_pct": pct,
                "reference_values": [cur["val"], prev["val"]], "unit": unit, "metric": metric,
                "reference_sources": [{"ticker": filing["ticker"], "accession": filing["accession"],
                                       "evidence": evidence_strings(cur["val"], unit)}],
            })
        quarterly.setdefault(filing["ticker"], []).append((filing, cur_by_metric))

    # quarter-over-quarter across two consecutive 10-Qs of the same company
    for ticker, items in quarterly.items():
        items.sort(key=lambda x: x[0]["report_date"])
        name = SHORT_NAME.get(ticker, ticker)
        for (f0, m0), (f1, m1) in zip(items, items[1:]):
            days = (_d(f1["report_date"]) - _d(f0["report_date"])).days
            if not 80 <= days <= 100:
                continue   # not consecutive quarters (a fiscal Q4 lives in the 10-K)
            for metric in set(m0) & set(m1):
                label, unit = label_for(metric, ticker), "USD"
                cp, pp = period_phrase(m1[metric], "10-Q"), period_phrase(m0[metric], "10-Q")
                ans, delta, pct = _change_answer(name, label, cp, pp, m1[metric], m0[metric], unit)
                qoq.append({
                    "question": f"How did {name}'s {label} {cp} compare with its {label} "
                                f"{pp}?",
                    "type": "comparison", "subtype": "change", "companies": [ticker],
                    "reference_answer": ans, "reference_value": delta, "reference_pct": pct,
                    "reference_values": [m1[metric]["val"], m0[metric]["val"]], "unit": unit, "metric": metric,
                    "reference_sources": [
                        {"ticker": ticker, "accession": f1["accession"], "evidence": evidence_strings(m1[metric]["val"], unit)},
                        {"ticker": ticker, "accession": f0["accession"], "evidence": evidence_strings(m0[metric]["val"], unit)}],
                })

    # cross-company: most recent 10-Q of two companies, same metric
    latest = {t: sorted(v, key=lambda x: x[0]["report_date"])[-1] for t, v in quarterly.items()}
    tickers = sorted(latest)
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            for metric in ("revenue", "net_income"):
                fa, ma = latest[a]
                fb, mb = latest[b]
                if metric not in ma or metric not in mb:
                    continue
                va, vb = ma[metric]["val"], mb[metric]["val"]
                na, nb = SHORT_NAME[a], SHORT_NAME[b]
                label = "total revenue" if metric == "revenue" else "net income"
                pa, pb = period_phrase(ma[metric], "10-Q"), period_phrase(mb[metric], "10-Q")
                winner = na if va > vb else nb
                cross.append({
                    "question": f"Which company reported higher {label}: {na} {pa} or {nb} {pb}?",
                    "type": "comparison", "subtype": "which", "companies": [a, b],
                    "reference_answer": (f"{winner}. {na}: {fmt_value(va, 'USD')} ({pa}); "
                                         f"{nb}: {fmt_value(vb, 'USD')} ({pb})."),
                    "reference_value": None, "reference_winner": winner,
                    "reference_values": [va, vb], "unit": "USD", "metric": metric,
                    "reference_sources": [
                        {"ticker": a, "accession": fa["accession"], "evidence": evidence_strings(va, "USD")},
                        {"ticker": b, "accession": fb["accession"], "evidence": evidence_strings(vb, "USD")}],
                })
    return yoy, qoq, cross
