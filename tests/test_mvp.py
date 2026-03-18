from __future__ import annotations

import pytest

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.inference.mvp import MvpInferenceEngine


def test_mvp_defaults_to_deepseek_direct_with_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MvpInferenceEngine()

    def failing_sample_distribution(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("OpenAI-compatible request timed out")

    monkeypatch.setattr(
        DirectProbabilityLlmBaseline,
        "sample_distribution",
        failing_sample_distribution,
    )
    request = engine.build_request(
        question_text="Can most people be trusted?",
        options=["Yes", "No"],
        population_text="Adults in the United States",
    )
    response = engine.predict(request)

    assert response.requested_strategy == "deepseek_direct"
    assert response.actual_strategy_used == "heuristic"
    assert response.fallback_happened is True
    assert response.fallback_reason == "timeout"


def test_mvp_direct_success_path_preserves_deepseek_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MvpInferenceEngine()

    def successful_sample_distribution(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {
            "distribution": [0.8, 0.2],
            "scorable": True,
            "json_compliance_rate": 1.0,
            "invalid_output_rate": 0.0,
            "parser_failure_rate": 0.0,
            "final_text_present_rate": 1.0,
            "total_input_tokens": 10,
            "total_output_tokens": 4,
            "estimated_api_cost_usd": 0.001,
            "request_latencies_ms": [100.0],
        }

    monkeypatch.setattr(
        DirectProbabilityLlmBaseline,
        "sample_distribution",
        successful_sample_distribution,
    )
    request = engine.build_request(
        question_text="Can most people be trusted?",
        options=["Yes", "No"],
        population_text="Adults in the United States",
    )
    response = engine.predict(request)

    assert response.requested_strategy == "deepseek_direct"
    assert response.actual_strategy_used == "deepseek_direct"
    assert response.fallback_happened is False
    assert response.distribution["Yes"] == 0.8
