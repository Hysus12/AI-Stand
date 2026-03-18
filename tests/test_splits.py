from __future__ import annotations

from spbce.data.splits import build_group_aware_split, resolve_record_ids_for_split
from spbce.schema.canonical import SurveyRecord


def test_group_split_has_no_group_leakage(survey_records: list[SurveyRecord]) -> None:
    manifest = build_group_aware_split(survey_records, random_state=7)
    resolved = resolve_record_ids_for_split(survey_records, manifest)
    train_ids = set(resolved["train"])
    validation_ids = set(resolved["validation"])
    test_ids = set(resolved["test"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
