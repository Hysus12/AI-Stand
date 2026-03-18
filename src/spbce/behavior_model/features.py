from __future__ import annotations

from typing import Any

import pandas as pd

from spbce.schema.api import PredictSurveyRequest, SurveyContext
from spbce.schema.canonical import PairedSurveyBehaviorRecord


def flatten_survey_questions(
    record: PairedSurveyBehaviorRecord, prefix: str, use_ai_distribution: bool = False
) -> dict[str, float]:
    features: dict[str, float] = {}
    for question in record.survey_questions:
        distribution = (
            question.metadata.get("ai_distribution")
            if use_ai_distribution
            else question.human_distribution
        )
        if distribution is None:
            continue
        for option, probability in zip(question.options, distribution, strict=True):
            option_key = option.lower().replace(" ", "_").replace("/", "_")
            features[f"{prefix}__{question.question_id}__{option_key}"] = float(probability)
    return features


def build_behavior_frame(records: list[PairedSurveyBehaviorRecord], mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        outcome_value = record.actual_outcome.outcome_value
        if isinstance(outcome_value, dict):
            raise ValueError("Behavior benchmarking expects scalar outcome_value fields.")
        row: dict[str, Any] = {
            "record_id": record.record_id,
            "group_id": record.group_id,
            "outcome_name": record.actual_outcome.outcome_name,
            "outcome_value": float(outcome_value),
            "seasonality": record.context_features.seasonality or "unknown",
            "campaign_type": record.context_features.campaign_type or "unknown",
            "channel": record.context_features.channel or "unknown",
            "population_signature": record.population_signature(),
        }
        if mode in {"human_only", "hybrid"}:
            row.update(flatten_survey_questions(record, prefix="human"))
        if mode in {"ai_only", "hybrid"}:
            row.update(flatten_survey_questions(record, prefix="ai", use_ai_distribution=True))
        rows.append(row)
    return pd.DataFrame(rows)


def make_survey_request(
    record: PairedSurveyBehaviorRecord, question_index: int
) -> PredictSurveyRequest:
    question = record.survey_questions[question_index]
    return PredictSurveyRequest(
        question_text=question.question_text,
        options=question.options,
        population_text=record.population_text,
        population_struct=record.population_struct,
        context=SurveyContext(
            campaign_type=record.context_features.campaign_type,
            channel=record.context_features.channel,
            time_window=record.time_start,
            market_metadata={"domain": record.domain},
        ),
    )
