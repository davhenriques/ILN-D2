"""Polite HTTP client for SEC EDGAR.

Implements the SEC fair-access rules:
  * descriptive User-Agent (name + email) read from config.yaml;
  * at most `max_requests_per_second` requests per second (global throttle);
  * retries with exponential backoff on 429 / 403 / 5xx;
  * on-disk cache: a URL that was downloaded once is never downloaded again.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


class SECClient:
    def __init__(self, user_agent: str, cache_dir: str | Path, max_rps: float = 8,
                 max_retries: int = 5, backoff_base: float = 2.0):
        if not user_agent or "@" not in user_agent or "email@exemplo" in user_agent:
            raise ValueError("Set sec.user_agent in config.yaml to 'Your Name your@email' (SEC requirement).")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max_rps
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._last = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict) -> "SECClient":
        s = cfg["sec"]
        return cls(s["user_agent"], Path(cfg["paths"]["raw"]) / "http_cache",
                   s["max_requests_per_second"], s["max_retries"], s["backoff_base_seconds"])

    def cache_path(self, url: str) -> Path:
        """Readable cache path: <host>/<path>, with a hash fallback for query strings."""
        p = urlparse(url)
        rel = p.path.lstrip("/") or "index"
        if p.query:
            rel += "_" + hashlib.md5(p.query.encode()).hexdigest()[:8]
        return self.cache_dir / p.netloc / rel

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get_bytes(self, url: str) -> bytes:
        path = self.cache_path(url)
        if path.exists():
            return path.read_bytes()
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException as e:
                status, err = None, e
            else:
                if r.status_code == 200:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r.content)
                    return r.content
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                status, err = r.status_code, None
            if attempt == self.max_retries:
                break
            delay = self.backoff_base * 2 ** attempt
            log.warning("GET %s failed (status=%s, err=%s); retry %d in %.0fs", url, status, err, attempt + 1, delay)
            time.sleep(delay)
        raise RuntimeError(f"Could not download {url} after {self.max_retries} retries")

    def get_json(self, url: str) -> dict:
        return json.loads(self.get_bytes(url))

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode("utf-8", errors="replace")
