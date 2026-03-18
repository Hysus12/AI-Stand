from __future__ import annotations

from spbce.ood.heuristics import TfidfOodDetector
from spbce.schema.api import PredictSurveyRequest, SurveyContext
from spbce.schema.canonical import PopulationStruct, SurveyRecord


def test_ood_detector_flags_unsupported_population(survey_records: list[SurveyRecord]) -> None:
    detector = TfidfOodDetector().fit(survey_records)
    assessment = detector.assess(
        PredictSurveyRequest(
            question_text="What is your opinion about asteroid mining regulations?",
            options=["Support", "Oppose"],
            population_text="Adults in Mars Colony One",
            population_struct=PopulationStruct(region="Mars"),
            context=SurveyContext(product_category="space_policy"),
        )
    )
    assert assessment.ood_flag is True
    assert assessment.score >= 0.7
