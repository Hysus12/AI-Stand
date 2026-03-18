from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spbce.api.app import app, get_mvp_engine, get_pipeline
from spbce.schema.api import MvpCostLatencySummary, MvpPredictSurveyResponse
from spbce.settings import settings


def test_predict_survey_endpoint(
    monkeypatch: pytest.MonkeyPatch, demo_artifacts: dict[str, str]
) -> None:
    monkeypatch.setattr(settings, "model_artifact", demo_artifacts["model"])
    monkeypatch.setattr(settings, "prompt_artifact", demo_artifacts["prompt"])
    monkeypatch.setattr(settings, "ood_artifact", demo_artifacts["ood"])
    get_pipeline.cache_clear()

    client = TestClient(app)
    response = client.post(
        "/predict-survey",
        json={
            "question_text": "Do you support clean energy?",
            "options": ["Yes", "No"],
            "population_text": "Adults in Taiwan",
            "population_struct": {"region": "Taiwan"},
            "context": {"product_category": "policy"},
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert set(payload["distribution"]) == {"Yes", "No"}
    assert abs(sum(payload["distribution"].values()) - 1.0) < 1e-6


def test_sample_respondents_reproducible(
    monkeypatch: pytest.MonkeyPatch, demo_artifacts: dict[str, str]
) -> None:
    monkeypatch.setattr(settings, "model_artifact", demo_artifacts["model"])
    monkeypatch.setattr(settings, "prompt_artifact", demo_artifacts["prompt"])
    monkeypatch.setattr(settings, "ood_artifact", demo_artifacts["ood"])
    get_pipeline.cache_clear()

    client = TestClient(app)
    payload = {
        "survey_payload": {
            "question_text": "Would you buy this product?",
            "options": ["Yes", "Maybe", "No"],
            "population_text": "Adults in Taiwan",
            "population_struct": {"region": "Taiwan"},
            "context": {"product_category": "consumer"},
        },
        "n": 10,
    }
    first = client.post("/sample-respondents", json=payload).json()
    second = client.post("/sample-respondents", json=payload).json()
    assert first["respondents"] == second["respondents"]


def test_mvp_predict_survey_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubMvpEngine:
        def predict(self, request: object, strategy: str | None = None) -> MvpPredictSurveyResponse:
            del request, strategy
            return MvpPredictSurveyResponse(
                distribution={"Yes": 0.7, "No": 0.3},
                top_option="Yes",
                confidence=0.7,
                uncertainty=0.3,
                invalid_flag=False,
                requested_strategy="deepseek_direct",
                actual_strategy_used="deepseek_direct",
                fallback_happened=False,
                fallback_reason=None,
                metadata={"output_type": "llm"},
                cost_latency_summary=MvpCostLatencySummary(
                    total_input_tokens=10,
                    total_output_tokens=5,
                    estimated_api_cost_usd=0.001,
                    total_latency_ms=120.0,
                    average_latency_ms_per_request=120.0,
                    request_count=1,
                    retry_count=0,
                ),
            )

    get_mvp_engine.cache_clear()
    monkeypatch.setattr("spbce.api.app.get_mvp_engine", lambda: StubMvpEngine())

    client = TestClient(app)
    response = client.post(
        "/mvp/predict-survey",
        json={
            "question_text": "Would you buy this?",
            "options": ["Yes", "No"],
            "population_text": "Adults in Taiwan",
            "strategy": "deepseek_direct",
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["actual_strategy_used"] == "deepseek_direct"
    assert payload["fallback_happened"] is False
    assert payload["distribution"]["Yes"] == 0.7
