# Formal Prompt Benchmark Summary (Frozen Holdout 50)

- Holdout manifest: `data/processed/gss_survey_splits/formal_prompt_holdout_50.json`
- Anti-leakage audit: `reports/audits/formal_prompt_holdout_50_audit.json`
- Contaminated question_id blacklist: `trust`, `natheal`, `natenvir`
- Holdout question_ids: `fair`, `helpful` (25 records each)
- Cost estimate source: official provider pricing as of 2026-03-18; DeepSeek `deepseek-chat` input/output = `$0.28/$0.42` per 1M tokens, OpenAI `gpt-5.1` input/output = `$1.25/$10.00` per 1M tokens.
- Cost limitation: reported cost uses provider usage fields when present and does not model discounts, cached-token pricing, or future pricing changes.

## Core Results

| Provider | Predictor | JS | MAE | RMSE | Top-1 | Final Text | Invalid | JSON Compliance | Cost USD | Avg Latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai_compatible / deepseek-chat | heuristic_prompt_only | 0.0340 | 0.1412 | 0.1560 | 0.44 | 1.00 | 0.00 | 0.00 | 0.0000 | 0.0 |
| openai_compatible / deepseek-chat | llm_zero_shot_persona | 0.3176 | 0.4130 | 0.4530 | 0.18 | 1.00 | 0.00 | 0.00 | 0.0114 | 1014.6 |
| openai_compatible / deepseek-chat | llm_few_shot_persona | 0.1953 | 0.2931 | 0.3356 | 0.46 | 1.00 | 0.00 | 0.00 | 0.0280 | 1055.8 |
| openai_compatible / deepseek-chat | llm_direct_option_probabilities | 0.0532 | 0.1564 | 0.1755 | 0.56 | 1.00 | 0.00 | 1.00 | 0.0248 | 1603.4 |

- Best predictor for `deepseek-chat`: `heuristic_prompt_only` with JS `0.0340`.

| openai / gpt-5.1 | heuristic_prompt_only | 0.0340 | 0.1412 | 0.1560 | 0.44 | 1.00 | 0.00 | 0.00 | 0.0000 | 0.0 |
| openai / gpt-5.1 | llm_zero_shot_persona | 0.4464 | 0.5313 | 0.5728 | 0.02 | 1.00 | 0.00 | 0.00 | 0.1189 | 1127.2 |
| openai / gpt-5.1 | llm_few_shot_persona | 0.1768 | 0.2847 | 0.3214 | 0.44 | 1.00 | 0.00 | 0.00 | 0.2017 | 1116.5 |
| openai / gpt-5.1 | llm_direct_option_probabilities | 0.0672 | 0.1836 | 0.2067 | 0.46 | 1.00 | 0.00 | 1.00 | 0.3242 | 1187.4 |

- Best predictor for `gpt-5.1`: `heuristic_prompt_only` with JS `0.0340`.

## Summary

- Anti-leakage audit passed before both formal runs.
- On this frozen 50-record holdout, neither DeepSeek nor OpenAI direct-probability beat the heuristic baseline on JS divergence.
- DeepSeek direct-probability was more accurate and much cheaper than OpenAI direct-probability.
- OpenAI few-shot persona was the strongest OpenAI LLM baseline, but still worse than the heuristic baseline on JS divergence.