"""Fast unit tests (no network, no models):  pytest -q tests/test_units.py"""
from agent import guardrail
from agent.entities import detect_tickers
from data_collection.extract_sections import split_by_items
from evaluation import metrics as M

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "KO"]


def test_numeric_match_units():
    assert M.numeric_match("Revenue was $94,036 million [1].", 94_036e6)
    assert M.numeric_match("Revenue was $94.0 billion.", 94_036e6)
    assert M.numeric_match("Revenue was $94,036.", 94_036e6)          # table value "in millions"
    assert M.numeric_match("a net loss of $(1,234) million", -1_234e6)
    assert M.numeric_match("diluted EPS of $1.57", 1.57)
    assert not M.numeric_match("Revenue was $85 billion.", 94_036e6)
    assert not M.numeric_match("On June 28, 2025 revenue grew", 28_202)


def test_percent_and_portuguese_numbers():
    assert M.percent_match("an increase of 16.6%", 16.58)
    assert M.numeric_match("a receita foi de 94,0 mil milhões de dólares", 94_036e6)


def test_abstention():
    assert M.is_abstention("I could not find this information in the documents.")
    assert M.is_abstention("Não encontrei essa informação nos documentos.")
    assert not M.is_abstention("Revenue was $5 million.")


def test_retrieval_metrics():
    m = M.retrieval_metrics(["a", "b", "c"], {"b", "z"}, ks=(1, 3))
    assert m["P@1"] == 0 and m["R@3"] == 0.5 and m["MRR"] == 0.5


def test_guardrail():
    assert guardrail.check("Should I buy NVIDIA stock now?")[0]
    assert guardrail.check("Vale a pena investir na Tesla?")[0]
    assert not guardrail.check("How many shares did Apple repurchase in 2025?")[0]


def test_detect_tickers():
    assert detect_tickers("What was Coca-Cola's revenue vs NVIDIA?", TICKERS) == ["NVDA", "KO"]
    assert detect_tickers("Did Apple purchase ...?", TICKERS) == ["AAPL"]   # 'purchase' is not 'chase'


def test_sections_handle_toc_and_running_headers():
    text = "\n".join(["PART I", "Item 1. Business", "Item 1A. Risk Factors", "PART I", "Item 1", "Business body " * 50,
                      "PART I", "Item 1", "more business " * 50, "Item 1A. Risk Factors", "risk " * 80])
    secs = {s["section"]: s["text"] for s in split_by_items(text, "10-K")}
    assert "more business" in secs["Item 1"] and "risk risk" in secs["Item 1A"]
