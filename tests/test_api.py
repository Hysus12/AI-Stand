from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spbce.api.app import app, get_pipeline
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
