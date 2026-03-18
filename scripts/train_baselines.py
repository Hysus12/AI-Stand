from __future__ import annotations

import argparse

import joblib

from spbce.baselines.majority import MajorityDistributionBaseline
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.baselines.subgroup import SubgroupMarginalBaseline
from spbce.baselines.topic import TopicOnlyBaseline
from spbce.data.datasets import load_survey_records
from spbce.schema.canonical import SurveyRecord
from spbce.utils.io import ensure_parent, read_json, write_json


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", default="data/processed/artifacts")
    parser.add_argument("--prompt-backend", default="heuristic")
    args = parser.parse_args()

    records = load_survey_records(args.input)
    split_manifest = read_json(args.splits)
    train_records = filter_records(records, split_manifest["resolved_record_ids"]["train"])

    majority = MajorityDistributionBaseline().fit(train_records)
    subgroup = SubgroupMarginalBaseline().fit(train_records)
    topic = TopicOnlyBaseline().fit(train_records)
    prompt = PromptOnlyPersonaBaseline(backend=args.prompt_backend)

    ensure_parent(f"{args.output_dir}/placeholder.txt")
    joblib.dump(majority, f"{args.output_dir}/majority_baseline.joblib")
    joblib.dump(subgroup, f"{args.output_dir}/subgroup_baseline.joblib")
    joblib.dump(topic, f"{args.output_dir}/topic_baseline.joblib")
    joblib.dump(prompt, f"{args.output_dir}/demo_prompt_baseline.joblib")

    write_json(
        f"{args.output_dir}/baseline_manifest.json",
        {
            "artifacts": {
                "majority": f"{args.output_dir}/majority_baseline.joblib",
                "subgroup": f"{args.output_dir}/subgroup_baseline.joblib",
                "topic": f"{args.output_dir}/topic_baseline.joblib",
                "prompt_only": f"{args.output_dir}/demo_prompt_baseline.joblib",
            }
        },
    )
    print("Saved baseline artifacts")


if __name__ == "__main__":
    main()
