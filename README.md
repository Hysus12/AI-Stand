# Survey Prior + Behavior Calibration Engine (SPBCE)

SPBCE is a production-oriented research MVP for predicting survey response distributions, calibrating them against observed survey data, and optionally using those calibrated priors to forecast real-world behavior. The core modeling object is the calibrated response distribution, not synthetic respondents.

## Current milestone

This repository now covers two milestones:

- ingest one real public survey-style dataset end to end
- ingest one stronger public microdata survey source end to end
- normalize records into a canonical schema
- build leakage-safe evaluation splits
- run prompt-only and simple supervised baselines
- evaluate survey fidelity honestly
- define a paired survey-behavior contract
- benchmark human-only vs AI-only vs hybrid outcome models on grouped proxy behavior data
- expose a minimal inference API

The current toy pipeline uses [`Anthropic/llm_global_opinions`](https://huggingface.co/datasets/Anthropic/llm_global_opinions), a public Hugging Face dataset derived from World Values Survey and Pew Global Attitudes data. It is a practical bootstrapping dataset for the survey-fidelity task, but it is not a substitute for broader survey coverage or paired survey-behavior data.

The stronger public microdata path uses the General Social Survey public-use cumulative data. It supports:

- subgroup-aware and temporal survey-fidelity evaluation
- grouped paired survey-behavior proxy records
- behavior validity benchmarking with human-only, AI-only, and hybrid models

## Repository layout

```text
spbce/
  README.md
  pyproject.toml
  configs/
  data/
  docs/
  scripts/
  src/spbce/
  tests/
```

## Quickstart

1. Create a virtual environment with Python 3.11+.
2. Install the project:

```bash
pip install -e .[dev]
```

3. Seed a small local demo artifact:

```bash
python scripts/seed_demo.py
```

4. Run the API:

```bash
uvicorn spbce.api.app:app --reload
```

5. Run the Streamlit demo:

```bash
streamlit run src/spbce/ui/app.py
```

## First milestone pipeline

```bash
python scripts/ingest_llm_global_opinions.py
python scripts/build_splits.py --input data/interim/llm_global_opinions.jsonl
python scripts/train_baselines.py --input data/interim/llm_global_opinions.jsonl --splits data/processed/splits/group_aware.json
python scripts/train_survey_prior.py --input data/interim/llm_global_opinions.jsonl --splits data/processed/splits/group_aware.json
python scripts/evaluate_survey.py --input data/interim/llm_global_opinions.jsonl --splits data/processed/splits/group_aware.json
```

## GSS behavior-validity pipeline

```bash
python scripts/ingest_gss_microdata.py
python scripts/build_splits.py --input data/interim/gss_survey_records.jsonl --output-dir data/processed/gss_survey_splits
python scripts/build_behavior_splits.py --input data/interim/gss_behavior_proxy_records.jsonl --output-dir data/processed/gss_behavior_splits
python scripts/train_baselines.py --input data/interim/gss_survey_records.jsonl --splits data/processed/gss_survey_splits/group_aware.json --output-dir data/processed/gss_survey_artifacts_group
python scripts/train_survey_prior.py --input data/interim/gss_survey_records.jsonl --splits data/processed/gss_survey_splits/group_aware.json --output data/processed/gss_survey_artifacts_group/demo_survey_model.joblib --ood-output data/processed/gss_survey_artifacts_group/demo_ood.joblib
python scripts/train_behavior_model.py --survey-input data/interim/gss_survey_records.jsonl --behavior-input data/interim/gss_behavior_proxy_records.jsonl --split-file data/processed/gss_behavior_splits/behavior_group_aware.json --output-dir data/processed/gss_behavior_artifacts_group
python scripts/evaluate_behavior.py --behavior-input data/interim/gss_behavior_proxy_records.jsonl --split-file data/processed/gss_behavior_splits/behavior_group_aware.json --artifact-dir data/processed/gss_behavior_artifacts_group --output reports/benchmarks/gss_behavior_group_eval.json
python scripts/evaluate_prompt_benchmarks.py --input data/interim/gss_survey_records.jsonl --splits data/processed/gss_survey_splits/group_aware.json --output reports/benchmarks/gss_prompt_benchmark.json --max-records 6
python scripts/evaluate_prompt_benchmarks.py --input data/interim/gss_survey_records.jsonl --splits data/processed/gss_survey_splits/group_aware.json --output reports/benchmarks/gss_prompt_benchmark_minimax_m25.json --max-records 6 --llm-provider anthropic_compatible --llm-model MiniMax-M2.5 --env-file minimax_api.env --anthropic-base-url https://api.minimax.io/anthropic --llm-num-samples 4 --llm-max-tokens 256
```

## Status

- Survey-fidelity path: implemented as a reproducible toy path.
- GSS microdata path: implemented.
- Behavior model: implemented for grouped GSS proxy outcomes.
- Calibration: implemented as a simple temperature scaling layer for the supervised baseline.
- OOD and uncertainty: heuristic first pass only.
- Prompt-only benchmarking: heuristic baseline plus local LLM zero-shot and few-shot persona evaluation.

## MVP inference

Minimal callable engine:

```python
from spbce.inference.mvp import MvpInferenceEngine

engine = MvpInferenceEngine(env_file="api.env")
request = engine.build_request(
    question_text="Can most people be trusted, or do you need to be careful?",
    options=["Can be trusted", "Depends", "Need to be careful"],
    population_text="Adults age 30-44 in the United States",
)
response = engine.predict(request, strategy="heuristic")
```

CLI example:

```bash
python scripts/run_mvp_inference.py \
  --strategy heuristic \
  --question-text "Can most people be trusted, or do you need to be careful?" \
  --options "Can be trusted" "Depends" "Need to be careful" \
  --population-text "Adults age 30-44 in the United States"
```

Supported MVP strategies:

- `heuristic`: current default and current honest production-safe baseline
- `deepseek_direct`: strict-JSON DeepSeek direct probability prediction
- `weighted_hybrid_025`: experimental weighted blend with `llm_weight=0.25`
- `learned_hybrid`: learned combiner backed by a fitted artifact
- `best_hybrid`: deprecated alias for `weighted_hybrid_025`; kept only for backward compatibility and not empirically best

Example learned hybrid invocation:

```bash
python scripts/run_mvp_inference.py \
  --strategy learned_hybrid \
  --learned-combiner-artifact data/processed/artifacts/learned_combiner_v1.joblib \
  --question-text "Can most people be trusted, or do you need to be careful?" \
  --options "Can be trusted" "Depends" "Need to be careful" \
  --population-text "Adults age 30-44 in the United States"
```

## Claims policy

This project does not claim general superiority over human surveys. Any performance claims must be tied to explicit held-out benchmarks, explicit split definitions, and documented domain coverage.
