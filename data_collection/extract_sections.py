"""Convert a filing's HTML into structured JSON split by Item (section).

Why a custom extractor instead of edgar-crawler: edgar-crawler is not a pip
package, has its own config/folder layout and its 10-Q support is weaker than
its 10-K support. Our needs are narrow (latest filings of ~6 companies), so a
small, auditable extractor is easier to control and to document:

  1. HTML -> text: drop the hidden inline-XBRL header, turn tables into
     "cell | cell" rows (keeps numbers next to their labels), insert line
     breaks at block elements.
  2. Detect headers with regexes anchored at the start of a line:
       "PART II"  and  "Item 1A. Risk Factors"
  3. Each header opens a segment that runs until the next header. The table
     of contents also contains every header, so for each Item we keep the
     *longest* segment (the real body, not the TOC line).
  4. Fallback: if the structure is not recognised (e.g. filings organised as
     an annual report with a cross-reference index, like JPMorgan's 10-K), we
     split the document on its own title-like headings instead.

10-K sections are named "Item 7"; 10-Q sections are "Part I - Item 2".
"""
from __future__ import annotations

import re
import warnings
from collections import Counter

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

# Inline-XBRL filings are XHTML; the lenient HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = ["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]

PART_RE = re.compile(r"^\s*PART\s+(IV|III|II|I)\b\s*[\.\:\-—–]?\s*(.{0,100})$", re.I)
ITEM_RE = re.compile(r"^\s*ITEM\s*(\d{1,2}[A-C]?)\s*[\.\:\-—–]?\s*(.{0,150})$", re.I)

TITLES_10K = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "[Reserved]",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TITLES_10Q = {
    ("I", "1"): "Financial Statements", ("I", "2"): "Management's Discussion and Analysis",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk", ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings", ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds", ("II", "3"): "Defaults Upon Senior Securities",
    ("II", "4"): "Mine Safety Disclosures", ("II", "5"): "Other Information", ("II", "6"): "Exhibits",
}


# ---------------------------------------------------------------- HTML -> text
def _table_to_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if not cells:
            continue
        row = " | ".join(cells)
        # SEC tables put "$", ")" and "%" in their own cells: glue them back.
        row = re.sub(r"\$\s*\|\s*", "$", row)
        row = re.sub(r"\s*\|\s*\)", ")", row)
        row = re.sub(r"\s*\|\s*%", "%", row)
        rows.append(row)
    return "\n" + "\n".join(rows) + "\n"


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for table in soup.find_all("table"):
        table.replace_with(NavigableString(_table_to_text(table)))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("​", "").replace("’", "'").replace("“", '"').replace("”", '"')
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"\d{1,3}|Table of Contents", ln, re.I)]
    return "\n".join(lines)


# ---------------------------------------------------------------- sectioning
def _section_name(form: str, part: str, item: str) -> str:
    return f"Item {item}" if form.startswith("10-K") else f"Part {part} - Item {item}"


def _section_title(form: str, part: str, item: str, found: str) -> str:
    default = TITLES_10K.get(item) if form.startswith("10-K") else TITLES_10Q.get((part, item))
    found = found.split("|")[0].strip(" .")
    return default or found or ""


def split_by_items(text: str, form: str) -> list[dict]:
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    headers = []  # (char_offset, part, item, found_title)
    part = "I"
    for ln, off in zip(lines, offsets):
        m = PART_RE.match(ln)
        if m and len(ln) < 120:
            part = m.group(1).upper()
            continue
        m = ITEM_RE.match(ln)
        if m and len(ln) < 200:
            headers.append((off, part, m.group(1).upper(), m.group(2)))

    # Consecutive headers with the same Item are merged: some filers (e.g. Microsoft)
    # repeat "PART I / Item 1" as a running header on every page.
    segments: list[dict] = []
    for i, (off, part, item, found) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        name = _section_name(form, part, item)
        if segments and segments[-1]["section"] == name:
            segments[-1]["end"] = end
            continue
        segments.append({"section": name, "title": _section_title(form, part, item, found),
                         "start": off, "end": end})

    best: dict[str, dict] = {}
    for seg in segments:
        seg["text"] = text[seg["start"]:seg.pop("end")].strip()
        if seg["section"] not in best or len(seg["text"]) > len(best[seg["section"]]["text"]):
            best[seg["section"]] = seg
    sections = sorted(best.values(), key=lambda s: s["start"])
    if sections:
        cover = text[: sections[0]["start"]].strip()
        # The cover + TOC precede the first *real* section; keep it (company info, shares outstanding).
        if cover:
            sections.insert(0, {"section": "Cover", "title": "Cover page", "start": 0, "text": cover})
    return sections


def _looks_valid(sections: list[dict], text: str, form: str) -> bool:
    """Structure is plausible if we found the key sections and no section swallowed the document."""
    names = {s["section"] for s in sections}
    key = {"Item 1A", "Item 7", "Item 8"} if form.startswith("10-K") else {"Part I - Item 1", "Part I - Item 2"}
    if not key <= names:
        return False
    biggest = max(len(s["text"]) for s in sections)
    return biggest < 0.8 * len(text)


HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9&',\-/() ]{3,80}$")


def split_by_headings(text: str, target_chars: int = 20000) -> list[dict]:
    """Fallback: split on short title-like lines, then merge into ~target_chars sections.

    Section names come from the first heading of each merged block, prefixed with
    "Report: " so that they are distinguishable from standard Items.
    """
    lines = text.split("\n")
    blocks, cur_title, cur = [], "Cover", []
    for ln in lines:
        words = ln.split()
        is_heading = (HEADING_RE.match(ln) and 1 <= len(words) <= 8 and not ln.endswith(".")
                      and sum(w[0].isupper() for w in words if w[0].isalpha()) >= max(1, len(words) // 2))
        if is_heading and sum(len(x) for x in cur) > 1500:
            blocks.append((cur_title, "\n".join(cur)))
            cur_title, cur = ln, []
        cur.append(ln)
    blocks.append((cur_title, "\n".join(cur)))

    sections, buf_title, buf = [], None, []
    for title, body in blocks:
        if buf_title is None:
            buf_title = title
        buf.append(body)
        if sum(len(b) for b in buf) >= target_chars:
            sections.append((buf_title, "\n".join(buf)))
            buf_title, buf = None, []
    if buf:
        sections.append((buf_title, "\n".join(buf)))

    counts, out = Counter(), []
    for title, body in sections:
        counts[title] += 1
        name = f"Report: {title}" + (f" ({counts[title]})" if counts[title] > 1 else "")
        out.append({"section": name, "title": title, "text": body})
    return out


def extract_sections(html: str | bytes, form: str) -> tuple[list[dict], str]:
    """Return (sections, method) where method is 'items' or 'headings_fallback'."""
    text = html_to_text(html)
    sections = split_by_items(text, form)
    if sections and _looks_valid(sections, text, form):
        return [{k: s[k] for k in ("section", "title", "text")} for s in sections], "items"
    return split_by_headings(text), "headings_fallback"
