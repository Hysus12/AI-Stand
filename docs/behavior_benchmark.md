# Behavior Benchmark Definitions

## Goal

Estimate whether AI-derived survey priors can predict held-out behavioral outcomes as well as or better than models that use human survey distributions directly.

## Current public benchmark

Dataset:

- GSS grouped proxy paired records

Target:

- weighted subgroup behavior rate

Outcome variables currently used:

- `boycott`
- `signpet`
- `conoffcl`
- `polfunds`

## Benchmark models

### Human-only

Input:

- observed human survey distributions only

Model:

- ridge regression over flattened grouped survey distributions

### AI-only

Input:

- AI-predicted survey priors for the same questionnaire

Model:

- ridge regression over flattened AI survey-prior distributions

### Hybrid

Input:

- observed human survey distributions
- AI-predicted survey priors

Model:

- ridge regression over concatenated human and AI features

## Leakage policy

- behavior split unit: `group_id`
- survey-prior training records are restricted to behavior-train groups via shared `behavior_group_id`
- validation groups are used for survey-prior calibration only
- test groups are never used for survey-prior fitting or outcome-model fitting

## Metrics

For the current grouped-rate proxy task:

- MAE
- RMSE
- R^2
- Spearman correlation

These metrics compare human-only, AI-only, and hybrid models on actual held-out outcome rates.

## Current interpretation policy

- if AI-only underperforms human-only, report that directly
- if hybrid underperforms human-only, do not present AI as additive
- do not generalize grouped self-reported behavior results to external business outcomes
