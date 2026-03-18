# Formal Prompt Benchmark v3

Formal prompt benchmark v3 is the next DeepSeek-only candidate benchmark after:

- `formal_prompt_holdout_50.json` (`v1`)
- `formal_prompt_holdout_v2_100.json` (`v2`)

## Scope

This benchmark keeps the evaluation model family fixed:

- `heuristic_prompt_only`
- `deepseek_direct`

It does not add new providers, does not reuse the old 6-row debug slice, and does not change the frozen `v1` or `v2` manifests.

## Construction policy

The v3 manifest is built from:

- source split: `data/processed/gss_survey_splits/held_out_question.json`
- blacklist from prior contamination audit
- blacklist from `formal_prompt_holdout_50.json`

It intentionally does not blacklist `v2` question IDs, because the dataset has no additional held-out question IDs beyond the `v2` coverage once debug and `v1` questions are excluded.

## What v3 expands

Compared with `v2`, the v3 candidate expands:

- more test rows
- more year coverage
- more population-variant coverage

It does **not** expand to more held-out question IDs because the current GSS benchmark pool is exhausted after excluding:

- debug-contaminated questions
- `v1` frozen questions

## Required checks

Before any formal v3 run:

1. validate the frozen manifest with `scripts/validate_formal_prompt_holdout.py`
2. keep few-shot pool train-only
3. keep direct scoring strict-JSON and final-text-only
4. do not use v3 results to retune the DeepSeek path
