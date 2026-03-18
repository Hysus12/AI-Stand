# SPBCE Evaluation Protocol

## Evaluation goals

The evaluation protocol is designed to answer two separate questions:

1. Survey fidelity: how well do model predictions match real human survey response distributions?
2. Behavioral validity: when paired data exists, how well do calibrated survey priors predict downstream outcomes?

These tasks must remain explicitly separated in reporting.

## Non-negotiable rules

- never rely on random row split alone
- never leak study, product, campaign, event, or project groups across train and test
- report held-out uncertainty and OOD diagnostics
- compare against prompt-only and non-neural baselines
- treat synthetic respondents as a visualization layer, not a target

## Canonical split families

### A. Group-aware split

Primary split for first milestone. The split unit is `group_id`, with all rows from the same group assigned to the same fold.

### B. Temporal split

When timestamps exist, train on earlier groups and test on later groups.

### C. Leave-one-domain-out

Hold out an entire domain or source category.

### D. Held-out question split

Hold out unseen questions or question clusters.

### E. Held-out population split

Hold out unseen population signatures.

## First milestone dataset caveat

`Anthropic/llm_global_opinions` does not ship full respondent microdata or rich temporal metadata. For that dataset:

- `group_id` is constructed from source plus question text
- `population` is proxied with country
- temporal split may be unavailable or weak
- held-out-population means held-out country

This is acceptable for a toy first milestone but must be upgraded before product claims.

## Survey fidelity targets

For each test split compute:

- Jensen-Shannon divergence
- safe KL divergence
- option-probability MAE
- option-probability RMSE
- top-option accuracy

Secondary diagnostics:

- confidence versus error correlation
- error by domain
- error by option count
- error by in-domain versus OOD bucket

## Behavioral validity targets

When paired data is available, evaluate on held-out groups:

- human-survey-only model
- AI-based survey-prior model
- hybrid human-plus-AI model

Metric families depend on the outcome:

- rate/probability: log loss, Brier, AUROC, calibration error
- regression: MAE, RMSE, R^2, Spearman
- ranking: Spearman, Kendall, NDCG

## Benchmark order

1. Majority distribution baseline
2. Subgroup marginal baseline
3. Topic baseline
4. Prompt-only baseline
5. Simple supervised baseline
6. Supervised plus calibration
7. Later encoder or decoder models

## Uncertainty and OOD evaluation

The first version uses:

- embedding or TF-IDF centroid distance
- domain novelty
- population novelty
- missing-context penalty
- calibration entropy proxy

We judge the uncertainty layer useful if higher uncertainty is associated with higher held-out error and if OOD examples are enriched in the high-uncertainty bucket.

## Robustness tests

- light paraphrase of question wording
- answer order perturbation
- remove one population field
- perturb one context field
- unseen topic stress test
- unseen domain stress test

Track:

- prediction drift
- confidence drift
- error increase

## Reporting standard

Every benchmark report must include:

- split definition
- number of groups and examples
- domain coverage
- metric table with uncertainty
- limitations and unsupported claims

## What was implemented

- evaluation design for survey fidelity
- leakage-safe split requirements
- explicit baseline stack
- uncertainty and robustness criteria

## Assumptions made

- first milestone can defer behavior metrics until paired data exists
- a single toy dataset is sufficient to validate pipeline mechanics

## Risks and unknowns

- the toy dataset under-represents real deployment complexity
- prompt-only baseline will be weaker than a premium API-based persona baseline

## Not implemented yet

- end-to-end behavior benchmark
- ablation and robustness report automation
- statistical significance testing wrappers

## Exact next step

Implement split builders and metric utilities so this protocol becomes executable rather than aspirational.
