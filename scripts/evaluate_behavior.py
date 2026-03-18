from __future__ import annotations

import argparse

import joblib

from spbce.behavior_model.benchmark import (
    attach_ai_predictions,
    evaluate_behavior_models,
    filter_behavior_records,
)
from spbce.behavior_model.models import BehaviorOutcomeModel
from spbce.data.datasets import load_behavior_records
from spbce.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-input", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--artifact-dir", default="data/processed/behavior_artifacts")
    parser.add_argument("--output", default="reports/benchmarks/behavior_eval.json")
    args = parser.parse_args()

    behavior_records = load_behavior_records(args.behavior_input)
    split_manifest = read_json(args.split_file)
    test_records = filter_behavior_records(
        behavior_records, split_manifest["resolved_record_ids"]["test"]
    )

    survey_bundle = joblib.load(f"{args.artifact_dir}/survey_prior_bundle.joblib")
    augmented_test = attach_ai_predictions(test_records, survey_bundle)
    models = {
        "human_only": BehaviorOutcomeModel.load(
            f"{args.artifact_dir}/human_only_behavior_model.joblib"
        ),
        "ai_only": BehaviorOutcomeModel.load(f"{args.artifact_dir}/ai_only_behavior_model.joblib"),
        "hybrid": BehaviorOutcomeModel.load(f"{args.artifact_dir}/hybrid_behavior_model.joblib"),
    }
    results = evaluate_behavior_models(models, augmented_test)
    write_json(
        args.output,
        {
            "split": split_manifest["strategy"],
            "metrics": ["mae", "rmse", "r2", "spearman"],
            "results": results,
        },
    )
    print(f"Wrote behavior evaluation report to {args.output}")


if __name__ == "__main__":
    main()
