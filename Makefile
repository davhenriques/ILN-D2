# D2 - RAG agent over SEC filings. Full pipeline:
#   make install && make all
PY ?= .venv/bin/python
SUBSET ?= 40

.PHONY: install models collect index index-experiments questions validation-sample evaluate \
        retrieval-eval experiments report all smoke test cli app clean-results

install:
	python3.13 -m venv .venv && $(PY) -m pip install -r requirements.txt

models:            ## pull the local LLMs (Ollama must be running)
	ollama pull llama3.1:8b && ollama pull qwen2.5:7b

collect:           ## Phase 1: filings + sections + XBRL + manifest
	$(PY) -m data_collection.collect

index:             ## Phase 2: default index (config.yaml)
	$(PY) -m indexing.build_index

index-experiments: ## extra indexes for the chunk-size experiment
	$(PY) -m indexing.build_index --all-experiments

questions:         ## Phase 4.1: questions.jsonl + CSV sample for manual validation
	$(PY) -m evaluation.generate_questions
	$(PY) -m evaluation.export_validation_sample

validation-sample:
	$(PY) -m evaluation.export_validation_sample

evaluate:          ## Phase 4.2-4.3: 4 variants x all questions (+ judge)
	$(PY) -m evaluation.run_all --run-name main

retrieval-eval:    ## retrieval-only metrics for every index x retriever
	$(PY) -m evaluation.retrieval_eval

experiments: index-experiments retrieval-eval  ## chunk size and k (answer level on a subset)
	$(PY) -m evaluation.run_all --variants rag_hybrid_rerank --subset $(SUBSET) --chunk-size 256 --overlap 32 --run-name cs256
	$(PY) -m evaluation.run_all --variants rag_hybrid_rerank --subset $(SUBSET) --run-name cs512
	$(PY) -m evaluation.run_all --variants rag_hybrid_rerank --subset $(SUBSET) --chunk-size 1024 --overlap 128 --embedding embedding_long --run-name cs1024_bge-m3
	$(PY) -m evaluation.run_all --variants rag_hybrid_rerank --subset $(SUBSET) --k 3 --run-name k3
	$(PY) -m evaluation.run_all --variants rag_hybrid_rerank --subset $(SUBSET) --k 10 --run-name k10

report:            ## Phase 4.4: summary.md + figures
	$(PY) -m evaluation.report

all: collect index questions evaluate experiments report

smoke:             ## 1 company, 5 questions, end to end
	$(PY) -m tests.smoke_test

test:
	$(PY) -m pytest -q tests/test_units.py

cli:
	$(PY) -m agent.cli

app:
	$(PY) -m agent.app_gradio
