from __future__ import annotations

import argparse
from collections import defaultdict

from spbce.data.datasets import load_survey_records
from spbce.utils.io import read_json, write_json
from spbce.utils.prompt_benchmark import (
    choose_evenly_spaced_records,
    derive_contaminated_question_blacklist,
    question_ids_from_formal_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--formal50-manifest", required=True)
    parser.add_argument("--v2-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-size", type=int, default=72)
    parser.add_argument("--dev-size", type=int, default=24)
    args = parser.parse_args()

    records = load_survey_records(args.input)
    debug_question_ids = derive_contaminated_question_blacklist(args.audit_report)
    excluded_question_ids = set(question_ids_from_formal_manifest(args.formal50_manifest))
    excluded_question_ids.update(question_ids_from_formal_manifest(args.v2_manifest))

    candidate_records = [
        record for record in records if record.question_id not in excluded_question_ids
    ]
    by_question: dict[str, list] = defaultdict(list)
    for record in candidate_records:
        by_question[record.question_id].append(record)
    for question_records in by_question.values():
        question_records.sort(
            key=lambda record: (
                int(record.wave_id or "0"),
                record.population_signature(),
                record.record_id,
            )
        )

    candidate_question_ids = sorted(by_question)
    train_records = []
    dev_records = []
    per_question_train = max(1, args.train_size // max(1, len(candidate_question_ids)))
    per_question_dev = max(1, args.dev_size // max(1, len(candidate_question_ids)))
    for question_id in candidate_question_ids:
        question_records = by_question[question_id]
        split_index = max(1, int(len(question_records) * 0.7))
        train_candidates = question_records[:split_index]
        dev_candidates = question_records[split_index:]
        train_records.extend(choose_evenly_spaced_records(train_candidates, per_question_train))
        dev_records.extend(choose_evenly_spaced_records(dev_candidates, per_question_dev))

    payload = {
        "strategy": "learned_combiner_train_dev_frozen",
        "selection_policy": {
            "excluded_question_ids": sorted(excluded_question_ids),
            "debug_question_ids_used_for_training": debug_question_ids,
            "candidate_question_ids": candidate_question_ids,
            "train_size_requested": args.train_size,
            "dev_size_requested": args.dev_size,
            "selection_method": "per-question temporal-slice plus even-spacing",
        },
        "train_record_ids": [record.record_id for record in train_records],
        "dev_record_ids": [record.record_id for record in dev_records],
        "train_records": [
            {
                "record_id": record.record_id,
                "question_id": record.question_id,
                "wave_id": record.wave_id,
                "population_signature": record.population_signature(),
            }
            for record in train_records
        ],
        "dev_records": [
            {
                "record_id": record.record_id,
                "question_id": record.question_id,
                "wave_id": record.wave_id,
                "population_signature": record.population_signature(),
            }
            for record in dev_records
        ],
    }
    write_json(args.output, payload)
    print(f"Wrote learned combiner split to {args.output}")


if __name__ == "__main__":
    main()
