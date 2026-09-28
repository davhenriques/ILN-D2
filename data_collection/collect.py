"""Phase 1 entry point: download filings, extract sections, download XBRL facts.

    python -m data_collection.collect [--tickers AAPL MSFT]

Outputs
  data/raw/filings/<TICKER>/*.htm        original primary documents
  data/raw/xbrl/<TICKER>.json            XBRL company facts (ground truth, not indexed)
  data/processed/<TICKER>/<accession>.json   sections as structured JSON
  data/manifest.json                     one entry per filing (company, ticker, CIK, form, dates, accession, paths)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from common.config import ROOT, load_config, setup_logging
from common.sec_client import SECClient
from data_collection.edgar import collect_filings
from data_collection.extract_sections import extract_sections
from data_collection.xbrl import download_companyfacts

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", help="override the tickers in config.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    paths = cfg["paths"]
    client = SECClient.from_config(cfg)

    manifest = collect_filings(cfg, args.tickers)

    for entry in manifest:
        out = Path(paths["processed"]) / entry["ticker"] / f"{entry['accession']}.json"
        if not out.exists():
            html = (ROOT / entry["raw_path"]).read_bytes()
            sections, method = extract_sections(html, entry["form"])
            doc = {k: entry[k] for k in ("company", "ticker", "cik", "form", "accession",
                                          "filing_date", "report_date")}
            doc.update({"extraction_method": method, "sections": sections})
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=1))
        doc = json.loads(out.read_text())
        entry["processed_path"] = str(out.relative_to(ROOT))
        entry["extraction_method"] = doc["extraction_method"]
        entry["n_sections"] = len(doc["sections"])
        log.info("%s %s %s: %d sections (%s)", entry["ticker"], entry["form"], entry["report_date"],
                 entry["n_sections"], entry["extraction_method"])

    for ticker, cik in sorted({(e["ticker"], e["cik"]) for e in manifest}):
        path = download_companyfacts(client, ticker, cik, paths["xbrl"])
        for e in manifest:
            if e["ticker"] == ticker:
                e["xbrl_path"] = str(path.relative_to(ROOT))

    Path(paths["manifest"]).write_text(json.dumps(manifest, indent=2))
    log.info("Manifest: %d filings -> %s", len(manifest), paths["manifest"])


if __name__ == "__main__":
    main()
