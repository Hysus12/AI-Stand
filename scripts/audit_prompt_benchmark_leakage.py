from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from spbce.baselines.persona_llm import LocalLlmPersonaBaseline
from spbce.data.datasets import load_survey_records
from spbce.schema.api import PredictSurveyRequest, SurveyContext
from spbce.schema.canonical import SurveyRecord
from spbce.utils.io import read_json, write_json


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def survey_year(record: SurveyRecord) -> int:
    if not record.time_start:
        return 0
    return int(record.time_start[:4])


def record_overlap_summary(
    train_records: list[SurveyRecord],
    validation_records: list[SurveyRecord],
    test_records: list[SurveyRecord],
) -> dict[str, object]:
    train_question_ids = {record.question_id for record in train_records}
    validation_question_ids = {record.question_id for record in validation_records}
    test_question_ids = {record.question_id for record in test_records}
    train_waves = {record.wave_id for record in train_records}
    validation_waves = {record.wave_id for record in validation_records}
    test_waves = {record.wave_id for record in test_records}
    train_populations = {record.population_signature() for record in train_records}
    validation_populations = {record.population_signature() for record in validation_records}
    test_populations = {record.population_signature() for record in test_records}
    train_question_population = {
        (record.question_id, record.population_signature()) for record in train_records
    }
    validation_question_population = {
        (record.question_id, record.population_signature()) for record in validation_records
    }
    test_question_population = {
        (record.question_id, record.population_signature()) for record in test_records
    }
    train_question_text = {normalize_text(record.question_text) for record in train_records}
    validation_question_text = {
        normalize_text(record.question_text) for record in validation_records
    }
    test_question_text = {normalize_text(record.question_text) for record in test_records}

    return {
        "question_id_train_test_overlap_count": len(train_question_ids & test_question_ids),
        "question_id_validation_test_overlap_count": len(
            validation_question_ids & test_question_ids
        ),
        "wave_id_train_test_overlap_count": len(train_waves & test_waves),
        "wave_id_validation_test_overlap_count": len(validation_waves & test_waves),
        "population_signature_train_test_overlap_count": len(train_populations & test_populations),
        "population_signature_validation_test_overlap_count": len(
            validation_populations & test_populations
        ),
        "question_population_train_test_overlap_count": len(
            train_question_population & test_question_population
        ),
        "question_population_validation_test_overlap_count": len(
            validation_question_population & test_question_population
        ),
        "normalized_question_text_train_test_overlap_count": len(
            train_question_text & test_question_text
        ),
        "normalized_question_text_validation_test_overlap_count": len(
            validation_question_text & test_question_text
        ),
        "question_id_train_test_overlap_values": sorted(train_question_ids & test_question_ids),
    }


def lookahead_overlap_summary(
    train_records: list[SurveyRecord], test_records: list[SurveyRecord]
) -> dict[str, int]:
    summary = {
        "train_year_min": min(survey_year(record) for record in train_records),
        "train_year_max": max(survey_year(record) for record in train_records),
        "test_year_min": min(survey_year(record) for record in test_records),
        "test_year_max": max(survey_year(record) for record in test_records),
        "test_records": len(test_records),
        "test_records_with_same_question_population_in_train": 0,
        "test_records_with_future_same_question_population_in_train": 0,
        "test_records_with_past_same_question_population_in_train": 0,
        "test_records_with_future_same_question_in_train": 0,
    }
    for test_record in test_records:
        same_question_population = [
            record
            for record in train_records
            if record.question_id == test_record.question_id
            and record.population_signature() == test_record.population_signature()
        ]
        same_question = [
            record for record in train_records if record.question_id == test_record.question_id
        ]
        if same_question_population:
            summary["test_records_with_same_question_population_in_train"] += 1
            if any(
                survey_year(record) > survey_year(test_record)
                for record in same_question_population
            ):
                summary["test_records_with_future_same_question_population_in_train"] += 1
            if any(
                survey_year(record) < survey_year(test_record)
                for record in same_question_population
            ):
                summary["test_records_with_past_same_question_population_in_train"] += 1
        if any(survey_year(record) > survey_year(test_record) for record in same_question):
            summary["test_records_with_future_same_question_in_train"] += 1
    return summary


def used_test_record_provenance(
    train_records: list[SurveyRecord], used_test_records: list[SurveyRecord]
) -> list[dict[str, object]]:
    persona = LocalLlmPersonaBaseline().fit(train_records)
    rows: list[dict[str, object]] = []
    for record in used_test_records:
        request = PredictSurveyRequest(
            question_text=record.question_text,
            options=record.options,
            population_text=record.population_text,
            population_struct=record.population_struct,
            context=SurveyContext(),
        )
        exemplars = persona._select_exemplars(request)
        same_question_train = [
            candidate for candidate in train_records if candidate.question_id == record.question_id
        ]
        same_wave_train = [
            candidate for candidate in train_records if candidate.wave_id == record.wave_id
        ]
        same_population_train = [
            candidate
            for candidate in train_records
            if candidate.population_signature() == record.population_signature()
        ]
        same_question_population_train = [
            candidate
            for candidate in train_records
            if candidate.question_id == record.question_id
            and candidate.population_signature() == record.population_signature()
        ]
        exact_question_text_train = [
            candidate
            for candidate in train_records
            if normalize_text(candidate.question_text) == normalize_text(record.question_text)
        ]
        rows.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "study_id": record.study_id,
                "split": "test_used_in_max_records",
                "question_id": record.question_id,
                "question_text": record.question_text,
                "wave_id": record.wave_id,
                "time_start": record.time_start,
                "population_text": record.population_text,
                "population_signature": record.population_signature(),
                "same_question_train_count": len(same_question_train),
                "same_wave_train_count": len(same_wave_train),
                "same_population_train_count": len(same_population_train),
                "same_question_and_population_train_count": len(same_question_population_train),
                "exact_question_text_train_count": len(exact_question_text_train),
                "few_shot_exemplars": [
                    {
                        "record_id": exemplar.record_id,
                        "question_id": exemplar.question_id,
                        "question_text": exemplar.question_text,
                        "wave_id": exemplar.wave_id,
                        "time_start": exemplar.time_start,
                        "population_signature": exemplar.population_signature(),
                        "same_question_as_test": exemplar.question_id == record.question_id,
                        "same_population_as_test": (
                            exemplar.population_signature() == record.population_signature()
                        ),
                        "same_wave_as_test": exemplar.wave_id == record.wave_id,
                    }
                    for exemplar in exemplars
                ],
            }
        )
    return rows


def benchmark_report_inventory(reports_dir: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(reports_dir.glob("gss_prompt_benchmark*.json")):
        report = read_json(path)
        inventory.append(
            {
                "report_path": str(path),
                "num_test_records": report.get("num_test_records"),
                "llm_provider": report.get("llm_provider"),
                "llm_model": report.get("llm_model"),
                "result_predictors": [
                    result.get("predictor") for result in report.get("results", [])
                ],
            }
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--reports-dir", default="reports/benchmarks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=6)
    args = parser.parse_args()

    records = load_survey_records(args.input)
    manifest = read_json(args.splits)
    train_records = filter_records(records, manifest["resolved_record_ids"]["train"])
    validation_records = filter_records(records, manifest["resolved_record_ids"]["validation"])
    test_records = filter_records(records, manifest["resolved_record_ids"]["test"])
    used_test_records = test_records[: args.max_records]

    payload = {
        "dataflow": [
            "load survey records from JSONL",
            "load split manifest and resolve train/validation/test record_ids",
            "fit few-shot retriever on train_records only",
            "slice test_records by original record order and max_records",
            "build request from question/options/population/context only",
            "generate LLM outputs and parse final_text only for scoring",
            "compare parsed prediction against observed_distribution after prediction",
        ],
        "summary": {
            "split_strategy": manifest["strategy"],
            "counts": {
                "train_records": len(train_records),
                "validation_records": len(validation_records),
                "test_records": len(test_records),
                "used_test_records_max_records": len(used_test_records),
            },
            "disjointness": {
                "record_ids_train_validation_disjoint": set(
                    manifest["resolved_record_ids"]["train"]
                ).isdisjoint(manifest["resolved_record_ids"]["validation"]),
                "record_ids_train_test_disjoint": set(
                    manifest["resolved_record_ids"]["train"]
                ).isdisjoint(manifest["resolved_record_ids"]["test"]),
                "record_ids_validation_test_disjoint": set(
                    manifest["resolved_record_ids"]["validation"]
                ).isdisjoint(manifest["resolved_record_ids"]["test"]),
                "group_ids_train_validation_disjoint": set(manifest["train_ids"]).isdisjoint(
                    manifest["validation_ids"]
                ),
                "group_ids_train_test_disjoint": set(manifest["train_ids"]).isdisjoint(
                    manifest["test_ids"]
                ),
                "group_ids_validation_test_disjoint": set(manifest["validation_ids"]).isdisjoint(
                    manifest["test_ids"]
                ),
            },
            "overlap_summary": record_overlap_summary(
                train_records=train_records,
                validation_records=validation_records,
                test_records=test_records,
            ),
            "lookahead_overlap": lookahead_overlap_summary(
                train_records=train_records,
                test_records=test_records,
            ),
        },
        "used_test_record_provenance": used_test_record_provenance(
            train_records=train_records,
            used_test_records=used_test_records,
        ),
        "train_question_counter_top10": Counter(
            record.question_id for record in train_records
        ).most_common(10),
        "test_question_counter_top10": Counter(
            record.question_id for record in test_records
        ).most_common(10),
        "benchmark_report_inventory": benchmark_report_inventory(Path(args.reports_dir)),
    }
    write_json(args.output, payload)
    print(f"Wrote prompt benchmark audit report to {args.output}")


if __name__ == "__main__":
    main()
