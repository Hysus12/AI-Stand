from __future__ import annotations

from dataclasses import dataclass

from spbce.behavior_model.features import build_behavior_frame, make_survey_request
from spbce.behavior_model.models import BehaviorOutcomeModel
from spbce.calibration.temperature import TemperatureScaler
from spbce.metrics.regression import (
    mean_absolute_error,
    r2,
    root_mean_squared_error,
    spearman_correlation,
)
from spbce.schema.canonical import PairedSurveyBehaviorRecord, SurveyRecord
from spbce.survey_prior.simple_supervised import SimpleSupervisedSurveyPrior


@dataclass(slots=True)
class SurveyPriorBundle:
    survey_model: SimpleSupervisedSurveyPrior
    temperature_scaler: TemperatureScaler | None


def filter_behavior_records(
    records: list[PairedSurveyBehaviorRecord], record_ids: list[str]
) -> list[PairedSurveyBehaviorRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def fit_survey_prior_for_behavior(
    survey_records: list[SurveyRecord],
    train_behavior_group_ids: set[str],
    validation_behavior_group_ids: set[str],
) -> SurveyPriorBundle:
    train_records = [
        record
        for record in survey_records
        if record.metadata.get("behavior_group_id") in train_behavior_group_ids
    ]
    validation_records = [
        record
        for record in survey_records
        if record.metadata.get("behavior_group_id") in validation_behavior_group_ids
    ]
    model = SimpleSupervisedSurveyPrior().fit(train_records)
    scaler = None
    if validation_records:
        predictions = model.predict_records(validation_records)
        scaler = TemperatureScaler().fit(
            [predictions[record.record_id] for record in validation_records],
            [record.observed_distribution for record in validation_records],
        )
    return SurveyPriorBundle(survey_model=model, temperature_scaler=scaler)


def attach_ai_predictions(
    records: list[PairedSurveyBehaviorRecord],
    bundle: SurveyPriorBundle,
) -> list[PairedSurveyBehaviorRecord]:
    augmented: list[PairedSurveyBehaviorRecord] = []
    for record in records:
        updated_questions = []
        for question_index, question in enumerate(record.survey_questions):
            request = make_survey_request(record, question_index)
            prediction = bundle.survey_model.predict_proba(request)
            if bundle.temperature_scaler is not None:
                prediction = bundle.temperature_scaler.apply(prediction)
            updated_questions.append(
                question.model_copy(
                    update={
                        "metadata": question.metadata | {"ai_distribution": prediction},
                    }
                )
            )
        augmented.append(record.model_copy(update={"survey_questions": updated_questions}))
    return augmented


def fit_behavior_models(
    train_records: list[PairedSurveyBehaviorRecord],
) -> dict[str, BehaviorOutcomeModel]:
    models: dict[str, BehaviorOutcomeModel] = {}
    for mode in ["human_only", "ai_only", "hybrid"]:
        frame = build_behavior_frame(train_records, mode=mode)
        models[mode] = BehaviorOutcomeModel().fit(frame)
    return models


def evaluate_behavior_models(
    models: dict[str, BehaviorOutcomeModel],
    test_records: list[PairedSurveyBehaviorRecord],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for mode, model in models.items():
        frame = build_behavior_frame(test_records, mode=mode)
        predictions = model.predict(frame)
        actual = []
        for record in test_records:
            outcome_value = record.actual_outcome.outcome_value
            if isinstance(outcome_value, dict):
                raise ValueError("Behavior benchmarking expects scalar outcome_value fields.")
            actual.append(float(outcome_value))
        rows.append(
            {
                "model": mode,
                "mae": mean_absolute_error(actual, predictions),
                "rmse": root_mean_squared_error(actual, predictions),
                "r2": r2(actual, predictions),
                "spearman": spearman_correlation(actual, predictions),
            }
        )
    return rows
