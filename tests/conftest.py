from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.behavior_model.benchmark import (
    attach_ai_predictions,
    fit_behavior_models,
    fit_survey_prior_for_behavior,
)
from spbce.calibration.temperature import TemperatureScaler
from spbce.ood.heuristics import TfidfOodDetector
from spbce.schema.canonical import (
    BehaviorOutcome,
    BehaviorSurveyQuestion,
    ContextFeatures,
    PairedSurveyBehaviorRecord,
    PopulationStruct,
    SurveyRecord,
    stable_hash,
)
from spbce.survey_prior.simple_supervised import SimpleSupervisedSurveyPrior


@pytest.fixture()
def survey_records() -> list[SurveyRecord]:
    raw = [
        ("Do you support clean energy?", ["Yes", "No"], "Taiwan", [0.68, 0.32], "policy"),
        ("Do you support clean energy?", ["Yes", "No"], "Japan", [0.57, 0.43], "policy"),
        (
            "Would you buy this product?",
            ["Yes", "Maybe", "No"],
            "Taiwan",
            [0.22, 0.51, 0.27],
            "consumer",
        ),
        (
            "Would you buy this product?",
            ["Yes", "Maybe", "No"],
            "Japan",
            [0.18, 0.48, 0.34],
            "consumer",
        ),
        (
            "Would you attend this event?",
            ["Yes", "Maybe", "No"],
            "Taiwan",
            [0.30, 0.45, 0.25],
            "events",
        ),
        (
            "Would you attend this event?",
            ["Yes", "Maybe", "No"],
            "Japan",
            [0.24, 0.44, 0.32],
            "events",
        ),
        (
            "How important is low price?",
            ["Very", "Somewhat", "Not"],
            "Taiwan",
            [0.51, 0.34, 0.15],
            "pricing",
        ),
        (
            "How important is low price?",
            ["Very", "Somewhat", "Not"],
            "Japan",
            [0.44, 0.37, 0.19],
            "pricing",
        ),
    ]
    records: list[SurveyRecord] = []
    for question_text, options, country, distribution, domain in raw:
        records.append(
            SurveyRecord(
                record_id=stable_hash(question_text, country),
                dataset_id="test",
                study_id=domain,
                group_id=stable_hash(question_text, domain),
                domain=domain,
                country=country,
                population_text=f"Adults in {country}",
                population_struct=PopulationStruct(region=country),
                question_id=stable_hash(question_text),
                question_text=question_text,
                question_topic=domain,
                question_type="single_choice",
                options=options,
                option_order=list(range(len(options))),
                observed_distribution=distribution,
                metadata={},
            )
        )
    return records


@pytest.fixture()
def demo_artifacts(survey_records: list[SurveyRecord]) -> dict[str, str]:
    model = SimpleSupervisedSurveyPrior().fit(survey_records)
    predictions = model.predict_records(survey_records)
    scaler = TemperatureScaler().fit(
        [predictions[record.record_id] for record in survey_records],
        [record.observed_distribution for record in survey_records],
    )
    prompt = PromptOnlyPersonaBaseline()
    ood = TfidfOodDetector().fit(survey_records)

    output_dir = Path("data/processed/test_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    prompt_path = output_dir / "prompt.joblib"
    ood_path = output_dir / "ood.joblib"
    joblib.dump(
        {"survey_model": model, "temperature_scaler": scaler, "prompt_baseline": prompt},
        model_path,
    )
    joblib.dump(prompt, prompt_path)
    ood.save(str(ood_path))
    return {"model": str(model_path), "prompt": str(prompt_path), "ood": str(ood_path)}


@pytest.fixture()
def behavior_fixture() -> dict[str, object]:
    survey_records: list[SurveyRecord] = []
    behavior_records: list[PairedSurveyBehaviorRecord] = []
    groups = [
        (
            "g1",
            "Adults in Taiwan",
            PopulationStruct(region="Taiwan", gender="female", age_band="18-29"),
        ),
        (
            "g2",
            "Adults in Japan",
            PopulationStruct(region="Japan", gender="female", age_band="18-29"),
        ),
        (
            "g3",
            "Adults in Taiwan",
            PopulationStruct(region="Taiwan", gender="male", age_band="30-44"),
        ),
        (
            "g4",
            "Adults in Japan",
            PopulationStruct(region="Japan", gender="male", age_band="30-44"),
        ),
    ]
    question_bank = [
        ("trust", "Can people be trusted?", ["Yes", "Depends", "No"]),
        ("price", "How important is low price?", ["Very", "Somewhat", "Not"]),
    ]
    survey_distributions = {
        "g1": {"trust": [0.48, 0.22, 0.30], "price": [0.62, 0.25, 0.13]},
        "g2": {"trust": [0.34, 0.28, 0.38], "price": [0.51, 0.31, 0.18]},
        "g3": {"trust": [0.41, 0.20, 0.39], "price": [0.56, 0.27, 0.17]},
        "g4": {"trust": [0.29, 0.24, 0.47], "price": [0.44, 0.33, 0.23]},
    }
    outcomes = {"g1": 0.42, "g2": 0.30, "g3": 0.37, "g4": 0.24}

    for group_id, population_text, population_struct in groups:
        questions: list[BehaviorSurveyQuestion] = []
        for question_id, question_text, options in question_bank:
            distribution = survey_distributions[group_id][question_id]
            survey_records.append(
                SurveyRecord(
                    record_id=stable_hash(group_id, question_id),
                    dataset_id="behavior_test",
                    study_id="behavior_test",
                    group_id=stable_hash(question_id, "2024"),
                    wave_id="2024",
                    time_start="2024-01-01",
                    time_end="2024-12-31",
                    domain="behavior_proxy",
                    country=population_struct.region,
                    language="en",
                    population_text=population_text,
                    population_struct=population_struct,
                    question_id=question_id,
                    question_text=question_text,
                    question_topic="behavior_proxy",
                    question_type="single_choice",
                    options=options,
                    option_order=list(range(len(options))),
                    observed_distribution=distribution,
                    sample_size=120,
                    weights_available=False,
                    metadata={"behavior_group_id": group_id},
                )
            )
            questions.append(
                BehaviorSurveyQuestion(
                    question_id=question_id,
                    question_text=question_text,
                    question_topic="behavior_proxy",
                    question_type="single_choice",
                    options=options,
                    option_order=list(range(len(options))),
                    human_distribution=distribution,
                    sample_size=120,
                    weights_available=False,
                )
            )
        behavior_records.append(
            PairedSurveyBehaviorRecord(
                record_id=stable_hash("behavior", group_id),
                dataset_id="behavior_test",
                study_id="behavior_test",
                group_id=group_id,
                time_start="2024-01-01",
                time_end="2024-12-31",
                domain="behavior_proxy",
                population_text=population_text,
                population_struct=population_struct,
                stimulus_text="Synthetic paired behavior test fixture",
                questionnaire_id="questionnaire_1",
                survey_questions=questions,
                actual_outcome=BehaviorOutcome(
                    outcome_id="purchase_rate",
                    outcome_type="rate",
                    outcome_name="purchase_rate",
                    outcome_value=outcomes[group_id],
                    positive_label="yes",
                    unit="rate",
                ),
                context_features=ContextFeatures(channel="online", campaign_type="proxy"),
            )
        )

    train_group_ids = {"g1", "g2"}
    validation_group_ids = {"g3"}
    bundle = fit_survey_prior_for_behavior(survey_records, train_group_ids, validation_group_ids)
    augmented_records = attach_ai_predictions(behavior_records, bundle)
    models = fit_behavior_models(augmented_records[:3])
    return {
        "survey_records": survey_records,
        "behavior_records": behavior_records,
        "bundle": bundle,
        "augmented_records": augmented_records,
        "models": models,
    }
