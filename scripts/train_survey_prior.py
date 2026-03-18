from __future__ import annotations

import argparse

import joblib

from spbce.calibration.temperature import TemperatureScaler
from spbce.data.datasets import load_survey_records
from spbce.ood.heuristics import TfidfOodDetector
from spbce.schema.canonical import SurveyRecord
from spbce.survey_prior.simple_supervised import SimpleSupervisedSurveyPrior
from spbce.utils.io import ensure_parent, read_json


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", default="data/processed/artifacts/demo_survey_model.joblib")
    parser.add_argument("--ood-output", default="data/processed/artifacts/demo_ood.joblib")
    args = parser.parse_args()

    records = load_survey_records(args.input)
    split_manifest = read_json(args.splits)
    train_records = filter_records(records, split_manifest["resolved_record_ids"]["train"])
    validation_records = filter_records(
        records, split_manifest["resolved_record_ids"]["validation"]
    )

    model = SimpleSupervisedSurveyPrior().fit(train_records)
    scaler = None
    if validation_records:
        validation_predictions = model.predict_records(validation_records)
        validation_truth = {
            record.record_id: record.observed_distribution for record in validation_records
        }
        scaler = TemperatureScaler().fit(
            [validation_predictions[record.record_id] for record in validation_records],
            [validation_truth[record.record_id] for record in validation_records],
        )
    ood = TfidfOodDetector().fit(train_records)

    ensure_parent(args.output)
    ensure_parent(args.ood_output)
    joblib.dump(
        {
            "survey_model": model,
            "temperature_scaler": scaler,
        },
        args.output,
    )
    ood.save(args.ood_output)
    print(f"Saved survey prior artifact to {args.output}")


if __name__ == "__main__":
    main()
