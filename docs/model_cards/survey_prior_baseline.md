# Model Card: Simple Supervised Survey Prior Baseline

## Intended use

This model predicts survey response distributions over answer options from question text, option text, and lightweight population metadata. It is intended as a first milestone benchmark, not as the final production model.

## Training data

Current default training path:

- `Anthropic/llm_global_opinions` normalized into `SurveyRecord`

## Model family

- option-level regression model using TF-IDF text features and structured metadata
- per-question score normalization into a valid probability distribution
- optional scalar temperature calibration

## Evaluation splits

- group-aware split required
- held-out question and held-out population split supported
- temporal split only when timestamps are available

## Metrics

- JS divergence
- safe KL divergence
- probability MAE
- probability RMSE
- top-option accuracy

## Known failure modes

- unseen domains and populations
- survey wording very far from training support
- answer options with semantics not seen in training

## Unsupported uses

- national representativeness claims
- behavioral forecasting without paired training data
- synthetic respondent outputs presented as human data

## Calibration notes

The first version uses scalar temperature scaling fitted on validation predictions only.
