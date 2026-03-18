from __future__ import annotations

from pathlib import Path

from spbce.schema.canonical import PairedSurveyBehaviorRecord, SurveyRecord
from spbce.utils.io import read_jsonl


def load_survey_records(path: str | Path) -> list[SurveyRecord]:
    return [SurveyRecord.model_validate(row) for row in read_jsonl(path)]


def load_behavior_records(path: str | Path) -> list[PairedSurveyBehaviorRecord]:
    return [PairedSurveyBehaviorRecord.model_validate(row) for row in read_jsonl(path)]
