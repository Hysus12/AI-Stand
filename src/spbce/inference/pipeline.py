from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.behavior_model.base import BehaviorModel
from spbce.calibration.temperature import TemperatureScaler
from spbce.ood.heuristics import TfidfOodDetector
from spbce.schema.api import (
    PredictBehaviorRequest,
    PredictBehaviorResponse,
    PredictSurveyRequest,
    PredictSurveyResponse,
    SampleRespondentsRequest,
    SampleRespondentsResponse,
)
from spbce.survey_prior.simple_supervised import SimpleSupervisedSurveyPrior


@dataclass(slots=True)
class SurveyModelBundle:
    survey_model: SimpleSupervisedSurveyPrior
    prompt_baseline: PromptOnlyPersonaBaseline | None
    temperature_scaler: TemperatureScaler | None
    ood_detector: TfidfOodDetector | None


class SurveyInferencePipeline:
    def __init__(
        self, bundle: SurveyModelBundle, behavior_model: BehaviorModel | None = None
    ) -> None:
        self.bundle = bundle
        self.behavior_model = behavior_model or BehaviorModel()

    @classmethod
    def load(
        cls,
        model_path: str,
        prompt_path: str | None = None,
        ood_path: str | None = None,
    ) -> SurveyInferencePipeline:
        prompt_baseline = None
        temperature_scaler = None
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            survey_model = loaded["survey_model"]
            prompt_baseline = loaded.get("prompt_baseline")
            temperature_scaler = loaded.get("temperature_scaler")
        else:
            survey_model = loaded
        if prompt_baseline is None and prompt_path and Path(prompt_path).exists():
            prompt_baseline = joblib.load(prompt_path)
        ood_detector = (
            TfidfOodDetector.load(ood_path) if ood_path and Path(ood_path).exists() else None
        )
        return cls(
            bundle=SurveyModelBundle(
                survey_model=survey_model,
                prompt_baseline=prompt_baseline,
                temperature_scaler=temperature_scaler,
                ood_detector=ood_detector,
            )
        )

    def predict_survey(self, request: PredictSurveyRequest) -> PredictSurveyResponse:
        distribution = self.bundle.survey_model.predict_proba(request)
        calibration_notes: list[str] = []
        if self.bundle.temperature_scaler is not None:
            distribution = self.bundle.temperature_scaler.apply(distribution)
            calibration_notes.append(
                "Applied scalar temperature scaling with "
                f"T={self.bundle.temperature_scaler.temperature:.2f}."
            )
        uncertainty = float(1.0 - max(distribution))
        support_notes: list[str] = []
        ood_flag = False
        if self.bundle.ood_detector is not None:
            assessment = self.bundle.ood_detector.assess(request)
            uncertainty = max(uncertainty, assessment.score)
            ood_flag = assessment.ood_flag
            support_notes.extend(assessment.notes)
        if self.bundle.prompt_baseline is not None:
            support_notes.append("Prompt-only baseline is available for offline comparison.")
        return PredictSurveyResponse(
            distribution={
                option: float(probability)
                for option, probability in zip(request.options, distribution, strict=True)
            },
            uncertainty=float(min(1.0, uncertainty)),
            ood_flag=ood_flag,
            support_notes=support_notes
            or ["Prediction generated from first-milestone survey prior model."],
            calibration_notes=calibration_notes or ["No calibration layer artifact was loaded."],
        )

    def predict_behavior(self, request: PredictBehaviorRequest) -> PredictBehaviorResponse:
        prediction = self.behavior_model.predict(request)
        survey_response = self.predict_survey(request.survey_payload)
        uncertainty = max(prediction.uncertainty, survey_response.uncertainty)
        return PredictBehaviorResponse(
            predicted_outcome=prediction.predicted_outcome,
            uncertainty=uncertainty,
            ood_flag=survey_response.ood_flag,
            support_notes=prediction.support_notes + survey_response.support_notes,
        )

    def sample_respondents(self, request: SampleRespondentsRequest) -> SampleRespondentsResponse:
        survey_response = self.predict_survey(request.survey_payload)
        options = list(survey_response.distribution)
        probabilities = np.asarray(list(survey_response.distribution.values()), dtype=float)
        draws = np.random.default_rng(42).choice(options, size=request.n, p=probabilities)
        respondents = [
            {
                "respondent_id": f"synthetic_{index}",
                "selected_option": str(choice),
                "synthetic": True,
            }
            for index, choice in enumerate(draws)
        ]
        return SampleRespondentsResponse(
            respondents=respondents,
            sampling_notes=[
                "Synthetic respondents are sampled from the calibrated distribution.",
                "They are presentation artifacts, not observed human records.",
            ],
        )
