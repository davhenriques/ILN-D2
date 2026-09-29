"""Thin wrapper around the Ollama chat API (local, open models)."""
from __future__ import annotations

import time

import httpx
import ollama


class OllamaLLM:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0,
                 num_ctx: int = 8192, num_predict: int = 400, seed: int = 42):
        self.model = model
        self.client = ollama.Client(host=host)
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict, "seed": seed}

    @classmethod
    def from_config(cls, cfg: dict, role: str = "generator", **kw) -> "OllamaLLM":
        m = cfg["models"]
        return cls(m[role], m["ollama_host"], m["temperature"], m["num_ctx"], **kw)

    def chat(self, messages: list[dict], json_format: bool = False) -> tuple[str, float]:
        """Return (text, latency_seconds)."""
        for attempt in range(4):   # survive a brief Ollama restart / dropped connection
            try:
                t0 = time.perf_counter()
                r = self.client.chat(model=self.model, messages=messages, options=self.options,
                                     format="json" if json_format else None)
                return r["message"]["content"].strip(), time.perf_counter() - t0
            except (ConnectionError, httpx.HTTPError):
                if attempt == 3:
                    raise
                time.sleep(5 * 2 ** attempt)
