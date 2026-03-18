# Paired Survey-Behavior Contract

## Canonical object

```python
PairedSurveyBehaviorRecord = {
  "record_id": str,
  "dataset_id": str,
  "study_id": str,
  "group_id": str,
  "time_start": str | null,
  "time_end": str | null,
  "domain": str,
  "population_text": str,
  "population_struct": {
    "age_band": str | null,
    "gender": str | null,
    "education": str | null,
    "occupation": str | null,
    "income_band": str | null,
    "region": str | null,
    "ideology": str | null,
    "other": dict[str, Any]
  },
  "stimulus_text": str,
  "questionnaire_id": str | null,
  "survey_questions": [
    {
      "question_id": str,
      "question_text": str,
      "question_topic": str | null,
      "question_type": str,
      "options": list[str],
      "option_order": list[int],
      "human_distribution": list[float],
      "sample_size": int | null,
      "weights_available": bool,
      "metadata": dict[str, Any]
    }
  ],
  "survey_distribution_features": dict[str, float] | null,
  "actual_outcome": {
    "outcome_id": str,
    "outcome_type": str,
    "outcome_name": str,
    "outcome_value": float | dict[str, Any],
    "positive_label": str | null,
    "unit": str | null,
    "metadata": dict[str, Any]
  },
  "context_features": {
    "price": float | null,
    "discount": float | null,
    "channel": str | null,
    "exposure": float | null,
    "inventory_constraint": float | null,
    "capacity_constraint": float | null,
    "seasonality": str | null,
    "campaign_type": str | null,
    "brand_name": str | null,
    "brand_strength": float | null,
    "brand_metadata": dict[str, Any],
    "other": dict[str, Any]
  },
  "metadata": dict[str, Any]
}
```

## Contract requirements

- `survey_questions` must contain observed human distributions, not model outputs
- `actual_outcome` must be the held-out behavior target
- `group_id` is the leakage-control unit for behavior evaluation
- `time_start` and `time_end` should be set whenever temporal benchmarking is possible
- context fields may be null, but missingness should be explicit

## Current GSS proxy instantiation

- `stimulus_text`: fixed grouped-proxy benchmark description
- `survey_questions`: grouped attitude distributions
- `actual_outcome`: grouped self-reported behavior rate
- `context_features.channel`: `gss_public_use`
- `context_features.campaign_type`: `survey_behavior_proxy`

## Unsupported implications

This schema supports market-outcome data, but the current public benchmark does not yet instantiate:

- purchase conversion
- CTR or CVR
- registration or attendance logs
- external sales or adoption outcomes
