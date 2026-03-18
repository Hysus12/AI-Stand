from __future__ import annotations

import pytest
from pydantic import ValidationError

from spbce.schema.canonical import PopulationStruct, SurveyRecord


def test_survey_record_normalizes_distribution() -> None:
    record = SurveyRecord(
        record_id="1",
        dataset_id="demo",
        study_id="study",
        group_id="group",
        population_text="Adults in Taiwan",
        population_struct=PopulationStruct(region="Taiwan"),
        question_id="question",
        question_text="Do you agree?",
        question_type="single_choice",
        options=["Yes", "No"],
        option_order=[0, 1],
        observed_distribution=[68, 32],
    )
    assert record.observed_distribution == [0.68, 0.32]


def test_survey_record_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        SurveyRecord(
            record_id="1",
            dataset_id="demo",
            study_id="study",
            group_id="group",
            population_text="Adults in Taiwan",
            population_struct=PopulationStruct(region="Taiwan"),
            question_id="question",
            question_text="Do you agree?",
            question_type="single_choice",
            options=["Yes", "No"],
            option_order=[0, 1],
            observed_distribution=[1.0],
        )
