"""Download XBRL company facts (data.sec.gov/api/xbrl/companyfacts).

These JSON files are the numeric ground truth used to generate and score
`numeric_fact` / `comparison` questions. They are NOT indexed in the vector DB.
"""
from __future__ import annotations

import json
from pathlib import Path

from common.sec_client import SECClient

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def download_companyfacts(client: SECClient, ticker: str, cik: int, out_dir: str | Path) -> Path:
    out = Path(out_dir) / f"{ticker}.json"
    if not out.exists():
        facts = client.get_json(COMPANYFACTS_URL.format(cik=cik))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(facts))
    return out


def load_companyfacts(xbrl_dir: str | Path, ticker: str) -> dict:
    return json.loads((Path(xbrl_dir) / f"{ticker}.json").read_text())
