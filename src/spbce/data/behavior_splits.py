from __future__ import annotations

from pathlib import Path

from spbce.data.splits import SplitManifest, _split_group_ids
from spbce.schema.canonical import PairedSurveyBehaviorRecord
from spbce.utils.io import write_json


def build_behavior_group_aware_split(
    records: list[PairedSurveyBehaviorRecord],
    random_state: int = 42,
) -> SplitManifest:
    group_ids = sorted({record.group_id for record in records})
    train_ids, validation_ids, test_ids = _split_group_ids(
        group_ids=group_ids,
        random_state=random_state,
        train_size=2 / 3,
        validation_size=1 / 6,
    )
    return SplitManifest(
        strategy="behavior_group_aware",
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={"unit": "group_id", "num_groups": len(group_ids), "random_state": random_state},
    )


def build_behavior_temporal_split(records: list[PairedSurveyBehaviorRecord]) -> SplitManifest:
    by_group: dict[str, int] = {}
    for record in records:
        if record.time_start:
            by_group[record.group_id] = min(
                by_group.get(record.group_id, 9999), int(record.time_start[:4])
            )
    ordered_groups = sorted(by_group, key=lambda group_id: by_group[group_id])
    if len(ordered_groups) < 3:
        return SplitManifest(
            strategy="behavior_temporal",
            train_ids=ordered_groups[:-1],
            validation_ids=[],
            test_ids=ordered_groups[-1:],
            metadata={"unit": "time_start_year", "note": "fewer than three temporal groups"},
        )
    train_cut = max(1, int(len(ordered_groups) * 0.7))
    validation_cut = max(train_cut + 1, int(len(ordered_groups) * 0.85))
    return SplitManifest(
        strategy="behavior_temporal",
        train_ids=ordered_groups[:train_cut],
        validation_ids=ordered_groups[train_cut:validation_cut],
        test_ids=ordered_groups[validation_cut:],
        metadata={
            "unit": "time_start_year",
            "min_year": by_group[ordered_groups[0]],
            "max_year": by_group[ordered_groups[-1]],
        },
    )


def resolve_behavior_record_ids(
    records: list[PairedSurveyBehaviorRecord], manifest: SplitManifest
) -> dict[str, list[str]]:
    train_groups = set(manifest.train_ids)
    validation_groups = set(manifest.validation_ids)
    test_groups = set(manifest.test_ids)
    return {
        "train": [record.record_id for record in records if record.group_id in train_groups],
        "validation": [
            record.record_id for record in records if record.group_id in validation_groups
        ],
        "test": [record.record_id for record in records if record.group_id in test_groups],
    }


def write_behavior_split_manifests(
    records: list[PairedSurveyBehaviorRecord], output_dir: str | Path
) -> dict[str, dict[str, object]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifests = [
        build_behavior_group_aware_split(records),
        build_behavior_temporal_split(records),
    ]
    serialized: dict[str, dict[str, object]] = {}
    for manifest in manifests:
        payload = manifest.to_dict() | {
            "resolved_record_ids": resolve_behavior_record_ids(records, manifest)
        }
        write_json(output_path / f"{manifest.strategy}.json", payload)
        serialized[manifest.strategy] = payload
    return serialized
