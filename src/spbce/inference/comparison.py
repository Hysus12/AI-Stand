from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spbce.metrics.distributions import js_divergence
from spbce.schema.project import ProjectInput, ProjectOutput


def _normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    total = float(sum(distribution.values()))
    if total <= 0:
        return distribution
    return {key: float(value / total) for key, value in distribution.items()}


def _segment_weights(project: ProjectInput) -> dict[str, float]:
    defaults = {
        segment.segment_id: float(segment.estimated_weight or 1.0)
        for segment in project.target_segments
    }
    total = float(sum(defaults.values()))
    return {segment_id: value / total for segment_id, value in defaults.items()}


def _aggregate_ai_distributions(
    project: ProjectInput,
    project_output: ProjectOutput,
) -> dict[str, dict[str, float]]:
    weights = _segment_weights(project)
    aggregated: dict[str, dict[str, float]] = {}
    for segment_result in project_output.per_segment_results:
        segment_weight = weights.get(segment_result.segment_id, 0.0)
        variant_result = segment_result.variant_results[0]
        for question_result in variant_result.question_results:
            bucket = aggregated.setdefault(
                question_result.question_id,
                {option: 0.0 for option in question_result.options},
            )
            for option, probability in question_result.distribution.items():
                bucket[option] = float(bucket.get(option, 0.0) + (segment_weight * probability))
    return {
        question_id: _normalize_distribution(distribution)
        for question_id, distribution in aggregated.items()
    }


def _group_distribution(
    ai_distribution: dict[str, float],
    option_groups: dict[str, list[str]] | None,
) -> dict[str, float]:
    if not option_groups:
        return _normalize_distribution(ai_distribution)
    grouped = {
        group_name: float(
            sum(ai_distribution.get(option, 0.0) for option in options)
        )
        for group_name, options in option_groups.items()
    }
    return _normalize_distribution(grouped)


def compare_project_to_reference(
    project: ProjectInput,
    project_output: ProjectOutput,
    reference_payload: dict[str, Any],
) -> dict[str, Any]:
    ai_aggregate = _aggregate_ai_distributions(project, project_output)
    comparisons: list[dict[str, Any]] = []
    matched_question_ids: set[str] = set()

    for question in reference_payload.get("questions", []):
        question_id = str(question["question_id"])
        ai_distribution = ai_aggregate.get(question_id)
        if ai_distribution is None:
            continue
        matched_question_ids.add(question_id)
        grouped_ai_distribution = _group_distribution(
            ai_distribution=ai_distribution,
            option_groups=question.get("ai_option_groups"),
        )
        reference_distribution = _normalize_distribution(
            {key: float(value) for key, value in question["reference_distribution"].items()}
        )
        ordered_keys = list(reference_distribution)
        ai_values = [float(grouped_ai_distribution.get(key, 0.0)) for key in ordered_keys]
        reference_values = [float(reference_distribution[key]) for key in ordered_keys]
        option_differences = {
            key: float(grouped_ai_distribution.get(key, 0.0) - reference_distribution[key])
            for key in ordered_keys
        }
        comparisons.append(
            {
                "question_id": question_id,
                "question_text": question["question_text"],
                "reference_kind": question.get("reference_kind", "exact"),
                "reference_distribution": reference_distribution,
                "ai_distribution": grouped_ai_distribution,
                "option_differences": option_differences,
                "probability_mae": float(
                    sum(abs(option_differences[key]) for key in option_differences)
                    / len(option_differences)
                ),
                "js_divergence": float(js_divergence(reference_values, ai_values)),
                "top_option_match": (
                    max(reference_distribution, key=reference_distribution.get)
                    == max(grouped_ai_distribution, key=grouped_ai_distribution.get)
                ),
                "notes": question.get("notes"),
            }
        )

    comparisons.sort(key=lambda item: item["js_divergence"])
    unmatched_project_questions = sorted(set(ai_aggregate) - matched_question_ids)
    unmatched_reference_questions = sorted(
        question["question_id"]
        for question in reference_payload.get("questions", [])
        if question["question_id"] not in matched_question_ids
    )
    summary = {
        "matched_question_count": len(comparisons),
        "unmatched_project_questions": unmatched_project_questions,
        "unmatched_reference_questions": unmatched_reference_questions,
        "mean_js_divergence": float(
            sum(item["js_divergence"] for item in comparisons) / len(comparisons)
        )
        if comparisons
        else None,
        "mean_probability_mae": float(
            sum(item["probability_mae"] for item in comparisons) / len(comparisons)
        )
        if comparisons
        else None,
        "closest_questions": [item["question_id"] for item in comparisons[:3]],
        "largest_gap_questions": [item["question_id"] for item in comparisons[-3:]][::-1],
    }
    return {
        "project_id": project.project_id,
        "reference_project_id": reference_payload.get("project_id"),
        "source": reference_payload.get("source", {}),
        "summary": summary,
        "question_comparisons": comparisons,
    }


def render_comparison_summary(comparison_report: dict[str, Any]) -> str:
    summary = comparison_report["summary"]
    lines = [
        f"# Comparison Summary: {comparison_report['project_id']}",
        "",
        f"- Matched questions: {summary['matched_question_count']}",
    ]
    if summary["mean_js_divergence"] is not None:
        lines.append(f"- Mean JS divergence: {summary['mean_js_divergence']:.4f}")
    if summary["mean_probability_mae"] is not None:
        lines.append(f"- Mean probability MAE: {summary['mean_probability_mae']:.4f}")
    if summary["closest_questions"]:
        lines.append(f"- Closest questions: {', '.join(summary['closest_questions'])}")
    if summary["largest_gap_questions"]:
        lines.append(f"- Largest-gap questions: {', '.join(summary['largest_gap_questions'])}")
    if summary["unmatched_project_questions"]:
        lines.append(
            "- Project questions without public reference: "
            + ", ".join(summary["unmatched_project_questions"])
        )
    if summary["unmatched_reference_questions"]:
        lines.append(
            "- Reference questions without AI match: "
            + ", ".join(summary["unmatched_reference_questions"])
        )
    return "\n".join(lines).strip() + "\n"


def export_comparison_report(
    comparison_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "comparison_report.json"
    summary_path = destination / "comparison_summary.md"
    report_path.write_text(
        json.dumps(comparison_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(render_comparison_summary(comparison_report), encoding="utf-8")
    return {
        "comparison_report": str(report_path),
        "comparison_summary": str(summary_path),
    }
