from __future__ import annotations

import joblib

from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.calibration.temperature import TemperatureScaler
from spbce.ood.heuristics import TfidfOodDetector
from spbce.schema.canonical import PopulationStruct, SurveyRecord, stable_hash
from spbce.survey_prior.simple_supervised import SimpleSupervisedSurveyPrior
from spbce.utils.io import ensure_parent, write_jsonl


def demo_records() -> list[SurveyRecord]:
    raw = [
        (
            "Do you support renewable energy subsidies?",
            ["Strongly support", "Support", "Oppose"],
            "Taiwan",
            [0.42, 0.38, 0.20],
            "policy",
        ),
        (
            "Do you support renewable energy subsidies?",
            ["Strongly support", "Support", "Oppose"],
            "Japan",
            [0.31, 0.41, 0.28],
            "policy",
        ),
        (
            "How important is affordable pricing when choosing a phone?",
            ["Very important", "Somewhat important", "Not important"],
            "Taiwan",
            [0.55, 0.31, 0.14],
            "consumer",
        ),
        (
            "How important is affordable pricing when choosing a phone?",
            ["Very important", "Somewhat important", "Not important"],
            "Japan",
            [0.48, 0.34, 0.18],
            "consumer",
        ),
        (
            "Would you attend a free community technology event this month?",
            ["Definitely", "Maybe", "No"],
            "Taiwan",
            [0.36, 0.44, 0.20],
            "events",
        ),
        (
            "Would you attend a free community technology event this month?",
            ["Definitely", "Maybe", "No"],
            "Japan",
            [0.27, 0.46, 0.27],
            "events",
        ),
    ]
    records: list[SurveyRecord] = []
    for question_text, options, country, distribution, domain in raw:
        population_text = f"Adults in {country}"
        records.append(
            SurveyRecord(
                record_id=stable_hash(question_text, country),
                dataset_id="demo",
                study_id=domain,
                group_id=stable_hash(question_text, domain),
                domain=domain,
                country=country,
                population_text=population_text,
                population_struct=PopulationStruct(region=country),
                question_id=stable_hash(question_text),
                question_text=question_text,
                question_topic=domain,
                question_type="single_choice",
                options=options,
                option_order=list(range(len(options))),
                observed_distribution=distribution,
                metadata={"seed_demo": True},
            )
        )
    return records


def main() -> None:
    records = demo_records()
    model = SimpleSupervisedSurveyPrior().fit(records)
    predictions = model.predict_records(records)
    scaler = TemperatureScaler().fit(
        [predictions[record.record_id] for record in records],
        [record.observed_distribution for record in records],
    )
    prompt = PromptOnlyPersonaBaseline(backend="heuristic")
    ood = TfidfOodDetector().fit(records)

    ensure_parent("data/processed/artifacts/demo_survey_model.joblib")
    joblib.dump(
        {"survey_model": model, "temperature_scaler": scaler, "prompt_baseline": prompt},
        "data/processed/artifacts/demo_survey_model.joblib",
    )
    joblib.dump(prompt, "data/processed/artifacts/demo_prompt_baseline.joblib")
    ood.save("data/processed/artifacts/demo_ood.joblib")
    write_jsonl(
        "data/interim/demo_records.jsonl",
        [record.model_dump(mode="json") for record in records],
    )
    print("Seeded demo artifacts and records.")


if __name__ == "__main__":
    main()
