"""LLM-as-judge with a rubric (correctness and faithfulness, 1-5).

The judge model (config models.judge) is a different model family from the
generator to reduce self-preference bias. The prompt lives in
evaluation/prompts/judge.txt so it can be inspected and versioned.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from agent.llm import OllamaLLM

PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.txt"
MAX_CONTEXT_CHARS = 12000


class Judge:
    def __init__(self, cfg: dict):
        self.llm = OllamaLLM.from_config(cfg, role="judge", num_predict=200)
        self.template = PROMPT_PATH.read_text()

    def score(self, question: str, reference: str, answer: str, context: str | None) -> dict:
        prompt = (self.template.replace("{question}", question).replace("{reference}", reference)
                  .replace("{answer}", answer)
                  .replace("{context}", context[:MAX_CONTEXT_CHARS] if context else "(no context: answer produced without retrieval)"))
        raw, _ = self.llm.chat([{"role": "user", "content": prompt}], json_format=True)
        return parse_judgement(raw, has_context=bool(context))


def parse_judgement(raw: str, has_context: bool) -> dict:
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else {}

    def clip(x):
        try:
            return min(5, max(1, int(x)))
        except (TypeError, ValueError):
            return None

    return {"judge_correctness": clip(d.get("correctness")),
            "judge_faithfulness": clip(d.get("faithfulness")) if has_context else None,
            "judge_explanation": str(d.get("explanation", ""))[:300]}
