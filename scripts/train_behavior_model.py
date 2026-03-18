from __future__ import annotations

import argparse

import joblib

from spbce.behavior_model.benchmark import (
    attach_ai_predictions,
    filter_behavior_records,
    fit_behavior_models,
    fit_survey_prior_for_behavior,
)
from spbce.data.behavior_splits import write_behavior_split_manifests
from spbce.data.datasets import load_behavior_records, load_survey_records
from spbce.utils.io import ensure_parent, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey-input", required=True)
    parser.add_argument("--behavior-input", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split-output-dir", default="data/processed/behavior_splits")
    parser.add_argument("--strategy", default="behavior_group_aware")
    parser.add_argument("--output-dir", default="data/processed/behavior_artifacts")
    args = parser.parse_args()

    survey_records = load_survey_records(args.survey_input)
    behavior_records = load_behavior_records(args.behavior_input)

    if args.split_file is None:
        write_behavior_split_manifests(behavior_records, args.split_output_dir)
        split_file = f"{args.split_output_dir}/{args.strategy}.json"
    else:
        split_file = args.split_file

    split_manifest = read_json(split_file)
    train_records = filter_behavior_records(
        behavior_records, split_manifest["resolved_record_ids"]["train"]
    )
    validation_records = filter_behavior_records(
        behavior_records, split_manifest["resolved_record_ids"]["validation"]
    )
    test_records = filter_behavior_records(
        behavior_records, split_manifest["resolved_record_ids"]["test"]
    )

    train_group_ids = {record.group_id for record in train_records}
    validation_group_ids = {record.group_id for record in validation_records}

    survey_bundle = fit_survey_prior_for_behavior(
        survey_records=survey_records,
        train_behavior_group_ids=train_group_ids,
        validation_behavior_group_ids=validation_group_ids,
    )

    augmented_train = attach_ai_predictions(train_records, survey_bundle)
    augmented_validation = attach_ai_predictions(validation_records, survey_bundle)
    augmented_test = attach_ai_predictions(test_records, survey_bundle)
    models = fit_behavior_models(augmented_train)

    ensure_parent(f"{args.output_dir}/placeholder.txt")
    joblib.dump(survey_bundle, f"{args.output_dir}/survey_prior_bundle.joblib")
    for mode, model in models.items():
        model.save(f"{args.output_dir}/{mode}_behavior_model.joblib")

    write_json(
        f"{args.output_dir}/behavior_training_manifest.json",
        {
            "split_file": split_file,
            "strategy": split_manifest["strategy"],
            "num_train_records": len(augmented_train),
            "num_validation_records": len(augmented_validation),
            "num_test_records": len(augmented_test),
            "artifacts": {
                "survey_prior_bundle": f"{args.output_dir}/survey_prior_bundle.joblib",
                "human_only": f"{args.output_dir}/human_only_behavior_model.joblib",
                "ai_only": f"{args.output_dir}/ai_only_behavior_model.joblib",
                "hybrid": f"{args.output_dir}/hybrid_behavior_model.joblib",
            },
        },
    )
    print(f"Saved behavior-model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
