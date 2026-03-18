from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from spbce.schema.canonical import SurveyRecord
from spbce.utils.io import write_json


@dataclass(slots=True)
class SplitManifest:
    strategy: str
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "train_ids": self.train_ids,
            "validation_ids": self.validation_ids,
            "test_ids": self.test_ids,
            "metadata": self.metadata,
        }


def _split_group_ids(
    group_ids: list[str], random_state: int, train_size: float, validation_size: float
) -> tuple[list[str], list[str], list[str]]:
    index = list(range(len(group_ids)))
    first_split = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, held_out_idx = next(first_split.split(index, groups=group_ids))
    train_groups = [group_ids[idx] for idx in train_idx]
    held_out_groups = [group_ids[idx] for idx in held_out_idx]

    if len(held_out_groups) == 1:
        return train_groups, [], held_out_groups
    if len(held_out_groups) == 2:
        return train_groups, [held_out_groups[0]], [held_out_groups[1]]

    remaining_fraction = validation_size / max(1e-8, 1.0 - train_size)
    second_split = GroupShuffleSplit(
        n_splits=1, train_size=remaining_fraction, random_state=random_state
    )
    held_out_positions = list(range(len(held_out_groups)))
    validation_idx, test_idx = next(second_split.split(held_out_positions, groups=held_out_groups))
    validation_groups = [held_out_groups[idx] for idx in validation_idx]
    test_groups = [held_out_groups[idx] for idx in test_idx]
    return train_groups, validation_groups, test_groups


def build_group_aware_split(
    records: list[SurveyRecord],
    random_state: int = 42,
    train_size: float = 2 / 3,
    validation_size: float = 1 / 6,
) -> SplitManifest:
    group_ids = sorted({record.group_id for record in records})
    train_groups, validation_groups, test_groups = _split_group_ids(
        group_ids=group_ids,
        random_state=random_state,
        train_size=train_size,
        validation_size=validation_size,
    )
    return SplitManifest(
        strategy="group_aware",
        train_ids=train_groups,
        validation_ids=validation_groups,
        test_ids=test_groups,
        metadata={"unit": "group_id", "num_groups": len(group_ids), "random_state": random_state},
    )


def build_held_out_question_split(
    records: list[SurveyRecord], random_state: int = 42
) -> SplitManifest:
    question_ids = sorted({record.question_id for record in records})
    train_ids, validation_ids, test_ids = _split_group_ids(
        group_ids=question_ids,
        random_state=random_state,
        train_size=2 / 3,
        validation_size=1 / 6,
    )
    return SplitManifest(
        strategy="held_out_question",
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={
            "unit": "question_id",
            "num_questions": len(question_ids),
            "random_state": random_state,
        },
    )


def build_held_out_population_split(
    records: list[SurveyRecord], random_state: int = 42
) -> SplitManifest:
    population_ids = sorted({record.population_signature() for record in records})
    train_ids, validation_ids, test_ids = _split_group_ids(
        group_ids=population_ids,
        random_state=random_state,
        train_size=2 / 3,
        validation_size=1 / 6,
    )
    return SplitManifest(
        strategy="held_out_population",
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={
            "unit": "population_signature",
            "num_populations": len(population_ids),
            "random_state": random_state,
        },
    )


def build_leave_one_domain_out_split(records: list[SurveyRecord]) -> SplitManifest:
    by_domain: dict[str, list[SurveyRecord]] = defaultdict(list)
    for record in records:
        by_domain[record.domain or "unknown"].append(record)
    held_out_domain = min(by_domain.items(), key=lambda item: len(item[1]))[0]
    train_ids = sorted(
        {
            record.group_id
            for domain, group in by_domain.items()
            if domain != held_out_domain
            for record in group
        }
    )
    test_ids = sorted({record.group_id for record in by_domain[held_out_domain]})
    validation_cut = max(1, len(train_ids) // 5)
    validation_ids = train_ids[:validation_cut]
    train_ids = train_ids[validation_cut:]
    return SplitManifest(
        strategy="leave_one_domain_out",
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={"held_out_domain": held_out_domain, "unit": "domain"},
    )


def build_temporal_split(
    records: list[SurveyRecord],
    key_fn: Callable[[SurveyRecord], str] | None = None,
) -> SplitManifest:
    grouped_records: dict[str, list[SurveyRecord]] = defaultdict(list)
    group_year: dict[str, int] = {}
    id_fn = key_fn or (lambda record: record.group_id)
    for record in records:
        if not record.time_start:
            continue
        group_identifier = id_fn(record)
        grouped_records[group_identifier].append(record)
        group_year[group_identifier] = min(
            group_year.get(group_identifier, 9999),
            int(record.time_start[:4]),
        )
    ordered_groups = sorted(group_year, key=lambda group_id: group_year[group_id])
    if len(ordered_groups) < 3:
        return SplitManifest(
            strategy="temporal",
            train_ids=ordered_groups[:-1],
            validation_ids=[],
            test_ids=ordered_groups[-1:],
            metadata={"unit": "time_start_year", "note": "fewer than three temporal groups"},
        )
    train_cut = max(1, int(len(ordered_groups) * 0.7))
    validation_cut = max(train_cut + 1, int(len(ordered_groups) * 0.85))
    return SplitManifest(
        strategy="temporal",
        train_ids=ordered_groups[:train_cut],
        validation_ids=ordered_groups[train_cut:validation_cut],
        test_ids=ordered_groups[validation_cut:],
        metadata={
            "unit": "time_start_year",
            "min_year": group_year[ordered_groups[0]],
            "max_year": group_year[ordered_groups[-1]],
        },
    )


def resolve_record_ids_for_split(
    records: list[SurveyRecord], manifest: SplitManifest
) -> dict[str, list[str]]:
    if manifest.strategy == "held_out_question":
        train_values = set(manifest.train_ids)
        validation_values = set(manifest.validation_ids)
        test_values = set(manifest.test_ids)
        return {
            "train": [record.record_id for record in records if record.question_id in train_values],
            "validation": [
                record.record_id for record in records if record.question_id in validation_values
            ],
            "test": [record.record_id for record in records if record.question_id in test_values],
        }
    if manifest.strategy == "held_out_population":
        train_values = set(manifest.train_ids)
        validation_values = set(manifest.validation_ids)
        test_values = set(manifest.test_ids)
        return {
            "train": [
                record.record_id
                for record in records
                if record.population_signature() in train_values
            ],
            "validation": [
                record.record_id
                for record in records
                if record.population_signature() in validation_values
            ],
            "test": [
                record.record_id
                for record in records
                if record.population_signature() in test_values
            ],
        }
    train_values = set(manifest.train_ids)
    validation_values = set(manifest.validation_ids)
    test_values = set(manifest.test_ids)
    return {
        "train": [record.record_id for record in records if record.group_id in train_values],
        "validation": [
            record.record_id for record in records if record.group_id in validation_values
        ],
        "test": [record.record_id for record in records if record.group_id in test_values],
    }


def write_split_manifests(
    records: list[SurveyRecord], output_dir: str | Path
) -> dict[str, dict[str, object]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifests = [
        build_group_aware_split(records),
        build_temporal_split(records),
        build_held_out_question_split(records),
        build_held_out_population_split(records),
        build_leave_one_domain_out_split(records),
    ]
    serialized: dict[str, dict[str, object]] = {}
    for manifest in manifests:
        payload = manifest.to_dict() | {
            "resolved_record_ids": resolve_record_ids_for_split(records, manifest)
        }
        write_json(output_path / f"{manifest.strategy}.json", payload)
        serialized[manifest.strategy] = payload
    return serialized
