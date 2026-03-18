PYTHON ?= python

.PHONY: install lint format typecheck test seed-demo api ui ingest splits train eval gss behavior prompt-bench

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src tests

test:
	pytest

seed-demo:
	$(PYTHON) scripts/seed_demo.py

api:
	uvicorn spbce.api.app:app --reload

ui:
	streamlit run src/spbce/ui/app.py

ingest:
	$(PYTHON) scripts/ingest_llm_global_opinions.py

splits:
	$(PYTHON) scripts/build_splits.py --input data/interim/llm_global_opinions.jsonl

train:
	$(PYTHON) scripts/train_baselines.py --input data/interim/llm_global_opinions.jsonl --splits data/processed/splits/group_aware.json
	$(PYTHON) scripts/train_survey_prior.py --input data/interim/llm_global_opinions.jsonl --splits data/processed/splits/group_aware.json

eval:
	$(PYTHON) scripts/evaluate_survey.py --input data/interim/llm_global_opinions.jsonl --splits data/processed/splits/group_aware.json

gss:
	$(PYTHON) scripts/ingest_gss_microdata.py
	$(PYTHON) scripts/build_splits.py --input data/interim/gss_survey_records.jsonl --output-dir data/processed/gss_survey_splits
	$(PYTHON) scripts/build_behavior_splits.py --input data/interim/gss_behavior_proxy_records.jsonl --output-dir data/processed/gss_behavior_splits

behavior:
	$(PYTHON) scripts/train_behavior_model.py --survey-input data/interim/gss_survey_records.jsonl --behavior-input data/interim/gss_behavior_proxy_records.jsonl --split-file data/processed/gss_behavior_splits/behavior_group_aware.json --output-dir data/processed/gss_behavior_artifacts_group
	$(PYTHON) scripts/evaluate_behavior.py --behavior-input data/interim/gss_behavior_proxy_records.jsonl --split-file data/processed/gss_behavior_splits/behavior_group_aware.json --artifact-dir data/processed/gss_behavior_artifacts_group --output reports/benchmarks/gss_behavior_group_eval.json

prompt-bench:
	$(PYTHON) scripts/evaluate_prompt_benchmarks.py --input data/interim/gss_survey_records.jsonl --splits data/processed/gss_survey_splits/group_aware.json --output reports/benchmarks/gss_prompt_benchmark.json --max-records 6

prompt-bench-minimax:
	$(PYTHON) scripts/evaluate_prompt_benchmarks.py --input data/interim/gss_survey_records.jsonl --splits data/processed/gss_survey_splits/group_aware.json --output reports/benchmarks/gss_prompt_benchmark_minimax_m25.json --max-records 6 --llm-provider anthropic_compatible --llm-model MiniMax-M2.5 --env-file minimax_api.env --anthropic-base-url https://api.minimax.io/anthropic --llm-num-samples 4 --llm-max-tokens 256
