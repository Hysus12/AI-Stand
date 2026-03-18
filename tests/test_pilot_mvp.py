from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.inference.comparison import compare_project_to_reference
from spbce.inference.mvp import MvpInferenceEngine
from spbce.schema.project import ProjectInput


def _stub_distribution(request, few_shot=False):  # type: ignore[no-untyped-def]
    del few_shot
    question = request.question_text.lower()
    population = request.population_text.lower()
    if "sharing client or internal calls" in question and "small agency founders" in population:
        raise RuntimeError("OpenAI-compatible request timed out")

    option_count = len(request.options)
    if "team monthly" in question and "small agency founders" in population:
        base = [0.18, 0.34, 0.28, 0.20][:option_count]
    elif "pro monthly" in question and "startup ops leads" in population:
        base = [0.46, 0.30, 0.16, 0.08][:option_count]
    elif "startup ops leads" in population:
        base = [0.38, 0.31, 0.19, 0.12][:option_count]
    else:
        base = [0.27, 0.33, 0.24, 0.16][:option_count]
    total = float(sum(base))
    distribution = [value / total for value in base]
    return {
        "distribution": distribution,
        "scorable": True,
        "json_compliance_rate": 1.0,
        "invalid_output_rate": 0.0,
        "parser_failure_rate": 0.0,
        "final_text_present_rate": 1.0,
        "total_input_tokens": 120,
        "total_output_tokens": 35,
        "estimated_api_cost_usd": 0.0012,
        "request_latencies_ms": [120.0],
    }


def test_project_schema_enforces_question_bounds() -> None:
    with pytest.raises(ValueError):
        ProjectInput.model_validate(
            {
                "project_id": "bad_project",
                "product_name": "Example",
                "product_brief": "Too few questions",
                "target_segments": [
                    {
                        "segment_id": "s1",
                        "segment_name": "Segment 1",
                        "demographic_description": "Adults",
                    }
                ],
                "survey_questions": [
                    {
                        "question_id": "q1",
                        "question_text": "Would you buy it?",
                        "question_type": "single_choice",
                        "options": ["Yes", "No"],
                    }
                ],
            }
        )


def test_run_project_exports_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DirectProbabilityLlmBaseline,
        "sample_distribution",
        _stub_distribution,
    )
    monkeypatch.setattr(MvpInferenceEngine, "_deepseek_provider_available", lambda self: True)
    engine = MvpInferenceEngine()
    input_path = Path("D:/dev/Gnosis/configs/pilot_mvp_sample.yaml")
    output_dir = Path("D:/dev/Gnosis/data/processed/test_pilot_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_result = engine.run_project_file(
        input_path=input_path,
        output_dir=output_dir,
        synthetic_respondent_count=120,
    )

    assert run_result.project_output.fallback_summary.fallback_count > 0
    assert len(run_result.synthetic_respondents) == 120
    assert (output_dir / "project_result.json").exists()
    assert (output_dir / "synthetic_respondents.csv").exists()
    assert (output_dir / "executive_summary.md").exists()

    project_result = json.loads((output_dir / "project_result.json").read_text(encoding="utf-8"))
    assert project_result["project_id"] == "pilot_nimbusnotes_q1"
    assert project_result["model_route_used"] == "deepseek_direct -> heuristic fallback"
    assert "Most promising segment" in project_result["executive_summary"]


def test_run_project_populates_question_level_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DirectProbabilityLlmBaseline,
        "sample_distribution",
        _stub_distribution,
    )
    monkeypatch.setattr(MvpInferenceEngine, "_deepseek_provider_available", lambda self: True)
    engine = MvpInferenceEngine()
    project = engine.load_project_input("D:/dev/Gnosis/configs/pilot_mvp_sample.yaml")

    run_result = engine.run_project(project)

    assert run_result.project_output.diagnostics.question_level_results
    assert run_result.project_output.recommendations
    assert run_result.project_output.diagnostics.generator_notes


def test_compare_project_to_reference_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DirectProbabilityLlmBaseline,
        "sample_distribution",
        _stub_distribution,
    )
    monkeypatch.setattr(MvpInferenceEngine, "_deepseek_provider_available", lambda self: True)
    engine = MvpInferenceEngine()
    project = engine.load_project_input(
        "D:/dev/Gnosis/configs/samples/sample_en_pew_policy_topline.yaml"
    )
    run_result = engine.run_project(project)
    reference_payload = json.loads(
        Path(
            "D:/dev/Gnosis/data/reference_results/sample_en_pew_policy_topline_reference.json"
        ).read_text(encoding="utf-8")
    )

    comparison_report = compare_project_to_reference(
        project=project,
        project_output=run_result.project_output,
        reference_payload=reference_payload,
    )

    assert comparison_report["summary"]["matched_question_count"] > 0
    assert comparison_report["question_comparisons"]
