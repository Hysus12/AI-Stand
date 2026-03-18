from __future__ import annotations

from spbce.schema.canonical import PairedSurveyBehaviorRecord


def test_paired_behavior_record_schema(behavior_fixture: dict[str, object]) -> None:
    record = behavior_fixture["behavior_records"][0]  # type: ignore[index]
    validated = PairedSurveyBehaviorRecord.model_validate(record.model_dump(mode="json"))
    assert validated.actual_outcome.outcome_type == "rate"
    assert len(validated.survey_questions) == 2
