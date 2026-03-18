# Dataset Card: Anthropic `llm_global_opinions`

## Source

- Hugging Face dataset: [Anthropic/llm_global_opinions](https://huggingface.co/datasets/Anthropic/llm_global_opinions)
- Underlying survey sources noted by the dataset card: World Values Survey and Pew Global Attitudes Survey

## Intended role in SPBCE

Bootstrap dataset for the first survey-fidelity milestone. It is suitable for proving the ingestion, normalization, split, baseline, and evaluation stack.

## Access and license

- Publicly downloadable via Hugging Face `datasets`
- License listed on the dataset card: `CC-BY-NC-SA-4.0`
- Commercial product training use is therefore restricted and must not be assumed

## Raw schema

The dataset card describes four columns:

- `question`
- `selections`
- `options`
- `source`

`selections` maps country to response percentages over the listed options.

## Canonical mapping

Each `(question, country)` pair becomes one `SurveyRecord`.

- `dataset_id`: `llm_global_opinions`
- `study_id`: raw `source`
- `group_id`: stable hash of source plus question text
- `domain`: raw `source`
- `country`: country key from `selections`
- `population_text`: `"Adults in {country}"`
- `population_struct.region`: country
- `question_text`: raw `question`
- `options`: raw `options`
- `observed_distribution`: normalized percentages from `selections[country]`

## Coverage strengths

- real public-opinion survey wording
- multiple countries
- direct response distributions
- low ingestion overhead

## Known limitations

- aggregate rather than respondent-level data
- limited subgroup structure in released form
- no paired behavioral outcomes
- unclear field timing for robust temporal holdout
- not representative of customer campaign/product-survey use cases by itself

## Preprocessing choices

- normalize percentages to sum to 1.0
- drop examples where option count and distribution length disagree
- derive canonical IDs via stable hashes
- infer `population_text` from country

## Missingness

- no sample size in the released schema
- no explicit weighting metadata
- no rich subgroup covariates

## SPBCE usage notes

Use this dataset only as a first-pass benchmark source. It is a convenience dataset for the survey-prior problem, not the long-run commercial training corpus.
