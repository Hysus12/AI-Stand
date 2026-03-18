from __future__ import annotations

import csv
import json
from pathlib import Path

from spbce.schema.project import ProjectRunResult


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_live_run_summary(run_result: ProjectRunResult) -> str:
    diagnostics = run_result.project_output.diagnostics
    api_usage = diagnostics.api_usage
    latency = diagnostics.latency
    output_quality = diagnostics.output_quality
    lines = [
        f"# Live Run Summary: {run_result.project_output.project_id}",
        "",
        "## Route Status",
        (
            f"- Direct success: {output_quality.direct_success_count}/"
            f"{output_quality.total_predictions} "
            f"({output_quality.direct_success_rate:.1%})"
        ),
        f"- Fallback rate: {output_quality.fallback_rate:.1%}",
        f"- Final text present rate: {output_quality.final_text_present_rate:.1%}",
        f"- Invalid output rate: {output_quality.invalid_output_rate:.1%}",
        f"- Parse failure rate: {output_quality.parse_failure_rate:.1%}",
        f"- Response schema compliance: {output_quality.response_schema_compliance:.1%}",
        "",
        "## Usage And Cost",
        f"- Total input tokens: {api_usage.total_input_tokens}",
        f"- Total output tokens: {api_usage.total_output_tokens}",
        f"- Estimated API cost (USD): {api_usage.estimated_api_cost_usd:.6f}",
        (
            f"- Estimated cost per valid direct question (USD): "
            f"{api_usage.estimated_cost_per_valid_question_usd:.6f}"
        ),
        "",
        "## Latency",
        f"- Total runtime seconds: {latency.total_runtime_seconds:.2f}",
        f"- Total request count: {latency.total_request_count}",
        f"- Average latency per request (ms): {latency.average_latency_ms_per_request:.1f}",
        f"- P50 latency (ms): {latency.p50_latency_ms:.1f}",
        f"- P95 latency (ms): {latency.p95_latency_ms:.1f}",
    ]
    if api_usage.usage_notes:
        lines.extend(["", "## Usage Notes"])
        lines.extend(f"- {note}" for note in api_usage.usage_notes)
    if output_quality.fallback_events:
        lines.extend(["", "## Fallback Events"])
        for event in output_quality.fallback_events:
            lines.append(
                "- "
                f"{event['segment_id']} / {event['question_id']} -> "
                f"{event.get('fallback_reason') or 'unspecified'} "
                f"(actual={event.get('actual_strategy_used')})"
            )
    if latency.slowest_requests:
        lines.extend(["", "## Slowest Requests"])
        for record in latency.slowest_requests:
            lines.append(
                "- "
                f"{record['segment_id']} / {record['question_id']}: "
                f"{record['total_latency_ms']:.1f} ms total over {record['request_count']} calls"
            )
    return "\n".join(lines).strip() + "\n"


def export_project_run(
    run_result: ProjectRunResult,
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    project_result_path = destination / "project_result.json"
    synthetic_csv_path = destination / "synthetic_respondents.csv"
    synthetic_jsonl_path = destination / "synthetic_respondents.jsonl"
    executive_summary_path = destination / "executive_summary.md"
    live_run_summary_path = destination / "live_run_summary.md"
    segment_report_path = destination / "segment_report.json"
    recommendations_path = destination / "recommendations.json"
    question_results_path = destination / "question_level_results.json"

    run_result.project_output.synthetic_respondents_path = str(synthetic_csv_path)
    _write_json(project_result_path, run_result.project_output.model_dump(mode="json"))
    _write_json(
        segment_report_path,
        [
            segment.model_dump(mode="json")
            for segment in run_result.project_output.per_segment_results
        ],
    )
    _write_json(
        recommendations_path,
        [
            recommendation.model_dump(mode="json")
            for recommendation in run_result.project_output.recommendations
        ],
    )
    _write_json(
        question_results_path,
        [
            insight.model_dump(mode="json")
            for insight in run_result.project_output.diagnostics.question_level_results
        ],
    )
    executive_summary_path.write_text(
        run_result.project_output.executive_summary,
        encoding="utf-8",
    )
    live_run_summary_path.write_text(
        _render_live_run_summary(run_result),
        encoding="utf-8",
    )

    question_ids: list[str] = []
    if run_result.synthetic_respondents:
        question_ids = sorted(run_result.synthetic_respondents[0].answers)
    fieldnames = [
        "respondent_id",
        "segment_id",
        "segment_name",
        "variant_id",
        "variant_name",
        "latent_profile",
        *question_ids,
    ]
    with synthetic_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for respondent in run_result.synthetic_respondents:
            row = {
                "respondent_id": respondent.respondent_id,
                "segment_id": respondent.segment_id,
                "segment_name": respondent.segment_name,
                "variant_id": respondent.variant_id or "",
                "variant_name": respondent.variant_name or "",
                "latent_profile": respondent.latent_profile,
            }
            row.update(respondent.answers)
            writer.writerow(row)

    with synthetic_jsonl_path.open("w", encoding="utf-8") as handle:
        for respondent in run_result.synthetic_respondents:
            handle.write(
                json.dumps(respondent.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )

    manifest = {
        "project_result": str(project_result_path),
        "synthetic_csv": str(synthetic_csv_path),
        "synthetic_jsonl": str(synthetic_jsonl_path),
        "executive_summary": str(executive_summary_path),
        "live_run_summary": str(live_run_summary_path),
        "segment_report": str(segment_report_path),
        "recommendations": str(recommendations_path),
        "question_level_results": str(question_results_path),
    }
    run_result.export_manifest = manifest
    _write_json(project_result_path, run_result.project_output.model_dump(mode="json"))
    return manifest
