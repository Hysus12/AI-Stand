# SPBCE Architecture

## Purpose

SPBCE estimates how a target population is likely to answer a survey question, quantifies uncertainty, detects out-of-domain requests, and optionally uses calibrated survey priors to forecast real-world outcomes when paired survey-behavior data exists.

The core truth object is the calibrated response distribution. Synthetic respondents are a downstream presentation layer only.

## System boundaries

### In scope

- canonical schema for survey and paired behavior records
- survey distribution prediction
- calibration of predicted survey distributions
- uncertainty and OOD heuristics
- leakage-safe evaluation
- optional synthetic respondent sampling
- pluggable behavior model interface
- minimal API and UI shell

### Out of scope for first milestone

- state-of-the-art LLM fine-tuning
- strong causal claims
- fully productionized orchestration and monitoring
- any claim of national representativeness without dataset-specific evidence
- robust behavioral forecasting without paired outcome data

## Layered architecture

### Layer 0: Baselines

- majority distribution baseline
- subgroup marginal baseline
- topic baseline
- prompt-only persona-like zero-shot baseline
- simple supervised baseline

These baselines provide honest lower and middle reference points. No model should be advanced without beating prompt-only baselines on held-out, leakage-safe splits.

### Layer 1: Survey prior model

Input:

- question text
- answer options
- population description
- optional context fields

Output:

- probability distribution over answer options

The first milestone uses a simple option-level supervised regressor. Each option becomes a scored row, then scores are rectified and renormalized within question. This keeps the pipeline flexible for variable answer cardinality and is a stable starting point for later encoder-based or generative models.

### Layer 2: Calibration

The first implementation uses scalar temperature scaling over option logits. Later candidates include isotonic calibration, topic-aware residual correction, and subgroup-aware calibration.

### Layer 3: Behavior calibration

This layer is scaffolded behind a stable interface. It accepts:

- calibrated survey prior
- stimulus features
- population features
- market context

Output:

- predicted business or behavioral outcome
- uncertainty proxy

This layer is intentionally not positioned as complete until real paired survey-behavior data is available.

### Layer 4: Synthetic respondent sampler

Synthetic respondents are sampled from calibrated answer distributions, optionally conditioned on lightweight metadata. They are tagged as synthetic artifacts and never presented as observed human data.

## Data flow

1. Ingestion reads source survey data.
2. Canonical normalization maps source examples into `SurveyRecord`.
3. Validation enforces distribution consistency and metadata sanity.
4. Split builder produces group-aware, temporal, held-out-question, held-out-population, and leave-one-domain-out partitions.
5. Baselines and survey prior models train on train partitions only.
6. Calibration fits on validation predictions only.
7. Evaluation reports held-out survey fidelity and confidence alignment.
8. Inference loads trained artifacts, computes OOD heuristics, then serves prediction responses via FastAPI and Streamlit.

## Core modules

- `spbce.schema`: pydantic canonical schema and API contracts
- `spbce.data`: ingestion, serialization, splits
- `spbce.preprocessing`: normalization and feature prep
- `spbce.baselines`: naive and prompt-only baselines
- `spbce.survey_prior`: supervised survey prior models
- `spbce.calibration`: calibration routines
- `spbce.metrics`: survey fidelity and behavior metrics
- `spbce.ood`: novelty heuristics
- `spbce.inference`: runtime pipeline
- `spbce.api`: FastAPI surface
- `spbce.ui`: Streamlit shell

## Initial dataset strategy

The first end-to-end toy path uses `Anthropic/llm_global_opinions` because it offers real public-opinion survey content with question text, answer options, and country-level distributions. It is not sufficient for the full product because:

- it lacks rich respondent microdata
- subgroup structure is limited in the released data
- it does not provide paired behavioral outcomes
- license is non-commercial

It is therefore a bootstrap dataset for the survey-fidelity pipeline, not a deployment dataset for commercial product claims.

## Reliability posture

Every inference response includes:

- predicted distribution
- uncertainty proxy
- OOD flag
- support notes
- calibration notes

The product must prefer abstention or weak-support notes over silent extrapolation.

## What was implemented

- modular repo layout
- canonical schemas
- baseline-friendly data flow
- first-pass calibration and OOD design
- minimal API/UI shells

## Assumptions made

- first milestone can use a public survey-derived dataset with aggregate country-level distributions
- a simple supervised option-level model is adequate for proving the end-to-end path
- heuristic OOD is acceptable before learned uncertainty is available

## Risks and unknowns

- the toy dataset does not represent the intended long-run commercial data regime
- prompt-only baseline quality depends on external zero-shot model availability
- behavior layer validity depends on future paired data quality

## Not implemented yet

- robust paired survey-behavior training
- hierarchical subgroup calibration
- stronger text encoders or decoder models
- benchmark-grade prompt-only LLM evaluation with external APIs

## Exact next step

Implement canonical schema validation and the ingestion/preprocessing path for the first real survey dataset, then build leakage-safe splits around it.
