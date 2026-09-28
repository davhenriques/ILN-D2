"""Optional web UI (Gradio).  python -m agent.app_gradio  ->  http://127.0.0.1:7860"""
from __future__ import annotations

import gradio as gr

from agent.rag_agent import VARIANTS, RAGAgent
from common.config import load_config, setup_logging

setup_logging()
CFG = load_config()
_AGENTS: dict[str, RAGAgent] = {}


def ask(question: str, variant: str):
    if variant not in _AGENTS:
        _AGENTS[variant] = RAGAgent(CFG, variant)
    r = _AGENTS[variant].answer(question)
    sources = "\n".join(r.sources()) or "(no retrieval)"
    return r.answer, sources, f"{r.latency:.1f}s"


with gr.Blocks(title="FinDocs RAG") as demo:
    gr.Markdown(f"## FinDocs: questions about the 10-K/10-Q filings of {', '.join(CFG['companies']['tickers'])}\n"
                "This agent does **not** provide investment advice.")
    with gr.Row():
        q = gr.Textbox(label="Question", scale=4)
        v = gr.Dropdown(VARIANTS, value=CFG["agent"]["default_variant"], label="Variant", scale=1)
    btn = gr.Button("Ask", variant="primary")
    ans = gr.Textbox(label="Answer", lines=6)
    src = gr.Textbox(label="Sources (* = cited)", lines=6)
    lat = gr.Textbox(label="Latency")
    btn.click(ask, [q, v], [ans, src, lat])
    q.submit(ask, [q, v], [ans, src, lat])

if __name__ == "__main__":
    demo.launch()
