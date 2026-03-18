from __future__ import annotations

from functools import lru_cache

import uvicorn
from fastapi import FastAPI

from spbce.inference.mvp import MvpInferenceEngine
from spbce.inference.pipeline import SurveyInferencePipeline
from spbce.schema.api import (
    MvpPredictSurveyRequest,
    MvpPredictSurveyResponse,
    PredictBehaviorRequest,
    PredictBehaviorResponse,
    PredictSurveyRequest,
    PredictSurveyResponse,
    SampleRespondentsRequest,
    SampleRespondentsResponse,
)
from spbce.settings import settings

app = FastAPI(title="SPBCE API", version="0.1.0")


@lru_cache
def get_pipeline() -> SurveyInferencePipeline:
    return SurveyInferencePipeline.load(
        model_path=settings.model_artifact,
        prompt_path=settings.prompt_artifact,
        ood_path=settings.ood_artifact,
    )


@lru_cache
def get_mvp_engine() -> MvpInferenceEngine:
    return MvpInferenceEngine()


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict-survey", response_model=PredictSurveyResponse)
def predict_survey(payload: PredictSurveyRequest) -> PredictSurveyResponse:
    return get_pipeline().predict_survey(payload)


@app.post("/mvp/predict-survey", response_model=MvpPredictSurveyResponse)
def predict_survey_mvp(payload: MvpPredictSurveyRequest) -> MvpPredictSurveyResponse:
    request = PredictSurveyRequest(
        question_text=payload.question_text,
        options=payload.options,
        population_text=payload.population_text,
        population_struct=payload.population_struct,
        context=payload.context,
    )
    return get_mvp_engine().predict(request, strategy=payload.strategy)


@app.post("/predict-behavior", response_model=PredictBehaviorResponse)
def predict_behavior(payload: PredictBehaviorRequest) -> PredictBehaviorResponse:
    return get_pipeline().predict_behavior(payload)


@app.post("/sample-respondents", response_model=SampleRespondentsResponse)
def sample_respondents(payload: SampleRespondentsRequest) -> SampleRespondentsResponse:
    return get_pipeline().sample_respondents(payload)


def run() -> None:
    uvicorn.run("spbce.api.app:app", host="0.0.0.0", port=8000, reload=False)
