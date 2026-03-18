from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spbce.schema.canonical import SurveyRecord
from spbce.utils.io import read_json


def normalize_question_text(text: str) -> str:
    return " ".join(text.lower().split())


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def derive_contaminated_question_blacklist(
    audit_report_path: str | Path,
) -> list[str]:
    report = read_json(audit_report_path)
    question_ids = {
        str(row["question_id"])
        for row in report.get("used_test_record_provenance", [])
        if row.get("question_id")
    }
    return sorted(question_ids)


@dataclass(slots=True)
class FewShotPoolSummary:
    pool_size: int
    excluded_records: int
    exclusion_reason_counts: dict[str, int]
    test_question_ids: list[str]
    test_normalized_question_texts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "excluded_records": self.excluded_records,
            "exclusion_reason_counts": self.exclusion_reason_counts,
            "test_question_ids": self.test_question_ids,
            "test_normalized_question_texts": self.test_normalized_question_texts,
        }


def build_few_shot_pool(
    train_records: list[SurveyRecord], test_records: list[SurveyRecord]
) -> tuple[list[SurveyRecord], FewShotPoolSummary]:
    test_question_ids = {record.question_id for record in test_records}
    test_question_texts = {normalize_question_text(record.question_text) for record in test_records}
    test_question_population = {
        (record.question_id, record.population_signature()) for record in test_records
    }

    filtered: list[SurveyRecord] = []
    exclusion_counts: Counter[str] = Counter()
    for record in train_records:
        reasons: list[str] = []
        if record.question_id in test_question_ids:
            reasons.append("same_question_id")
        if normalize_question_text(record.question_text) in test_question_texts:
            reasons.append("same_normalized_question_text")
        if (record.question_id, record.population_signature()) in test_question_population:
            reasons.append("same_question_population_variant")
        if reasons:
            for reason in reasons:
                exclusion_counts[reason] += 1
            continue
        filtered.append(record)
    return filtered, FewShotPoolSummary(
        pool_size=len(filtered),
        excluded_records=len(train_records) - len(filtered),
        exclusion_reason_counts=dict(sorted(exclusion_counts.items())),
        test_question_ids=sorted(test_question_ids),
        test_normalized_question_texts=sorted(test_question_texts),
    )


def choose_evenly_spaced_records(
    records: list[SurveyRecord], sample_size: int
) -> list[SurveyRecord]:
    if sample_size >= len(records):
        return list(records)
    if sample_size <= 0:
        return []
    if sample_size == 1:
        return [records[0]]
    last_index = len(records) - 1
    chosen_indices = {
        round(index * last_index / (sample_size - 1)) for index in range(sample_size)
    }
    chosen = [records[index] for index in sorted(chosen_indices)]
    cursor = 0
    while len(chosen) < sample_size:
        if records[cursor] not in chosen:
            chosen.append(records[cursor])
        cursor += 1
    return chosen[:sample_size]


def build_formal_holdout_manifest(
    records: list[SurveyRecord],
    source_manifest_path: str | Path,
    blacklist_question_ids: list[str],
    output_path: str | Path,
    sample_size: int,
) -> dict[str, Any]:
    source_manifest = read_json(source_manifest_path)
    train_records = filter_records(records, source_manifest["resolved_record_ids"]["train"])
    validation_records = filter_records(
        records, source_manifest["resolved_record_ids"]["validation"]
    )
    candidate_test_records = [
        record
        for record in filter_records(records, source_manifest["resolved_record_ids"]["test"])
        if record.question_id not in set(blacklist_question_ids)
    ]

    by_question: dict[str, list[SurveyRecord]] = defaultdict(list)
    for record in candidate_test_records:
        by_question[record.question_id].append(record)
    for question_records in by_question.values():
        question_records.sort(
            key=lambda record: (
                int(record.wave_id or "0"),
                record.population_signature(),
                record.record_id,
            )
        )

    question_ids = sorted(by_question)
    allocations = {question_id: 0 for question_id in question_ids}
    for _ in range(min(sample_size, len(candidate_test_records))):
        chosen_question = min(
            question_ids,
            key=lambda question_id: (
                allocations[question_id] / max(1, len(by_question[question_id])),
                allocations[question_id],
                question_id,
            ),
        )
        allocations[chosen_question] += 1

    selected: list[SurveyRecord] = []
    for question_id in question_ids:
        selected.extend(
            choose_evenly_spaced_records(by_question[question_id], allocations[question_id])
        )
    selected.sort(
        key=lambda record: (
            record.question_id,
            int(record.wave_id or "0"),
            record.population_signature(),
            record.record_id,
        )
    )

    payload = {
        "strategy": "formal_prompt_holdout_frozen",
        "source_split_strategy": source_manifest["strategy"],
        "source_split_manifest": str(Path(source_manifest_path)),
        "selection_policy": {
            "sample_size_requested": sample_size,
            "selection_method": "question-balanced-even-spacing",
            "question_id_blacklist": blacklist_question_ids,
        },
        "train_record_ids": [record.record_id for record in train_records],
        "validation_record_ids": [record.record_id for record in validation_records],
        "test_record_ids": [record.record_id for record in selected],
        "records": [
            {
                "record_id": record.record_id,
                "source_dataset": record.dataset_id,
                "split": "test",
                "question_id": record.question_id,
                "normalized_question_text": normalize_question_text(record.question_text),
                "population_signature": record.population_signature(),
                "wave_id": record.wave_id,
                "year": int(record.wave_id or "0"),
            }
            for record in selected
        ],
        "available_candidate_test_records": len(candidate_test_records),
        "selected_test_records": len(selected),
    }
    return payload


def validate_formal_holdout(
    records: list[SurveyRecord],
    formal_manifest: dict[str, Any],
    contaminated_question_blacklist: list[str],
) -> dict[str, Any]:
    record_lookup = {record.record_id: record for record in records}
    test_ids = formal_manifest["test_record_ids"]
    missing_test_ids = [record_id for record_id in test_ids if record_id not in record_lookup]
    test_records = [
        record_lookup[record_id] for record_id in test_ids if record_id in record_lookup
    ]
    manifest_record_ids = [row["record_id"] for row in formal_manifest.get("records", [])]
    if manifest_record_ids != test_ids:
        raise ValueError("formal manifest record list does not match test_record_ids ordering")
    blacklisted_hits = [
        record.record_id
        for record in test_records
        if record.question_id in set(contaminated_question_blacklist)
    ]
    train_records = filter_records(records, formal_manifest["train_record_ids"])
    few_shot_pool, pool_summary = build_few_shot_pool(train_records, test_records)

    test_question_ids = {record.question_id for record in test_records}
    test_question_texts = {normalize_question_text(record.question_text) for record in test_records}
    overlap_question_ids = sorted(
        {record.question_id for record in few_shot_pool if record.question_id in test_question_ids}
    )
    overlap_texts = sorted(
        {
            normalize_question_text(record.question_text)
            for record in few_shot_pool
            if normalize_question_text(record.question_text) in test_question_texts
        }
    )
    audit = {
        "status": "pass",
        "formal_manifest_record_count": len(test_records),
        "missing_test_record_ids": missing_test_ids,
        "blacklisted_test_record_ids": blacklisted_hits,
        "few_shot_pool": pool_summary.to_dict(),
        "few_shot_overlap_failures": {
            "same_question_id": overlap_question_ids,
            "same_normalized_question_text": overlap_texts,
        },
        "test_question_counts": dict(
            sorted(Counter(record.question_id for record in test_records).items())
        ),
        "test_year_min": min(int(record.wave_id or "0") for record in test_records),
        "test_year_max": max(int(record.wave_id or "0") for record in test_records),
    }
    if missing_test_ids or blacklisted_hits or overlap_question_ids or overlap_texts:
        audit["status"] = "fail"
    return audit
