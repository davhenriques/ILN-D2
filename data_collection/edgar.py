"""Download 10-K / 10-Q filings from SEC EDGAR and build data/manifest.json.

Steps
  1. ticker -> CIK via https://www.sec.gov/files/company_tickers.json
  2. filing list via https://data.sec.gov/submissions/CIK##########.json
  3. keep the most recent 10-K(s) and 10-Q(s) with filing date >= min_filing_date
  4. download the primary document (HTML) of each filing into data/raw/filings/

Every HTTP response is cached by SECClient, so re-running never re-downloads.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from common.config import ROOT, load_config, setup_logging
from common.sec_client import SECClient

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


def resolve_ciks(client: SECClient, tickers: list[str]) -> dict[str, dict]:
    """Return {ticker: {"cik": int, "name": str}}."""
    data = client.get_json(TICKERS_URL)
    by_ticker = {row["ticker"].upper(): row for row in data.values()}
    out = {}
    for t in tickers:
        row = by_ticker.get(t.upper())
        if row is None:
            raise KeyError(f"Ticker {t} not found in SEC company_tickers.json")
        out[t.upper()] = {"cik": int(row["cik_str"]), "name": row["title"]}
    return out


def list_filings(client: SECClient, cik: int, forms: list[str], min_date: str) -> list[dict]:
    """Recent filings of the requested forms (amendments such as 10-K/A are excluded)."""
    sub = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    pages = [sub["filings"]["recent"]]
    # "recent" holds only the last ~1000 filings. Heavy filers (e.g. JPM issues
    # prospectuses daily) need the older pages to reach min_date.
    for extra in sub["filings"].get("files", []):
        if extra["filingTo"] >= min_date:
            pages.append(client.get_json(f"https://data.sec.gov/submissions/{extra['name']}"))
    rows = []
    for page in pages:
        for i, form in enumerate(page["form"]):
            if form not in forms or page["filingDate"][i] < min_date:
                continue
            rows.append({
                "form": form,
                "accession": page["accessionNumber"][i],
                "filing_date": page["filingDate"][i],
                "report_date": page["reportDate"][i],   # period of report (end of quarter / year)
                "primary_document": page["primaryDocument"][i],
            })
    rows.sort(key=lambda r: r["filing_date"], reverse=True)
    return rows


def collect_filings(cfg: dict, tickers: list[str] | None = None) -> list[dict]:
    client = SECClient.from_config(cfg)
    tickers = tickers or cfg["companies"]["tickers"]
    fcfg = cfg["filings"]
    companies = resolve_ciks(client, tickers)
    raw_dir = Path(cfg["paths"]["raw"]) / "filings"
    manifest = []
    for ticker, info in companies.items():
        filings = list_filings(client, info["cik"], fcfg["forms"], fcfg["min_filing_date"])
        selected = ([f for f in filings if f["form"] == "10-K"][: fcfg["max_10k_per_company"]]
                    + [f for f in filings if f["form"] == "10-Q"][: fcfg["max_10q_per_company"]])
        for f in selected:
            url = ARCHIVE_URL.format(cik=info["cik"], acc_nodash=f["accession"].replace("-", ""),
                                     doc=f["primary_document"])
            html = client.get_bytes(url)
            path = raw_dir / ticker / f"{f['form']}_{f['report_date']}_{f['accession']}.htm"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(html)
            manifest.append({
                "company": info["name"], "ticker": ticker, "cik": info["cik"], **f,
                "url": url, "raw_path": str(path.relative_to(ROOT)),
            })
            log.info("%s %s filed %s (period %s) -> %s", ticker, f["form"], f["filing_date"],
                     f["report_date"], path.name)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="*", help="override the tickers in config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    manifest = collect_filings(cfg, args.tickers)
    Path(cfg["paths"]["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["paths"]["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest with %d filings written to %s", len(manifest), cfg["paths"]["manifest"])


if __name__ == "__main__":
    main()
